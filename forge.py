#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "httpx",
#     "torch",
#     "diffusers",
#     "transformers",
#     "accelerate",
#     "sentencepiece",
#     "pillow",
# ]
# ///
"""
Forge — GPU arbiter for Pondside.

Routes AI workloads through a single queue so the GPU is never contested.
Proxies chat/embed requests to Ollama. Handles image generation directly
via diffusers. One worker, one GPU, one job at a time.

Usage:
    ./forge.py
    uv run forge.py
"""

import asyncio
import base64
import io
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
FORGE_PORT = int(os.getenv("FORGE_PORT", "8200"))
IMAGE_DIR = Path(os.getenv("FORGE_IMAGE_DIR", "/Pondside/Alpha-Home/images/imagination"))
DEFAULT_IMAGE_MODEL = os.getenv("FORGE_IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")  # placeholder until we pick

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("forge")

# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """Base class for GPU jobs."""
    done: asyncio.Event = field(default_factory=asyncio.Event)
    result: Any = None
    error: Exception | None = None
    created_at: float = field(default_factory=time.time)


@dataclass
class OllamaProxyJob(Job):
    """Forward a request to Ollama."""
    method: str = "POST"
    path: str = ""
    body: dict = field(default_factory=dict)
    headers: dict = field(default_factory=dict)

    async def execute(self, client: httpx.AsyncClient):
        url = f"{OLLAMA_URL}{self.path}"
        log.info(f"forge.proxy → {self.method} {url}")
        resp = await client.request(
            method=self.method,
            url=url,
            json=self.body if self.method in ("POST", "PUT", "PATCH") else None,
            headers={k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")},
            timeout=120.0,
        )
        self.result = {"status_code": resp.status_code, "body": resp.json(), "headers": dict(resp.headers)}


@dataclass
class ImagineJob(Job):
    """Generate an image via diffusers."""
    prompt: str = ""
    negative_prompt: str = ""
    model: str = ""
    steps: int = 20
    width: int = 1024
    height: int = 1024
    seed: int | None = None

    async def execute(self, client: httpx.AsyncClient):
        """
        The heavy lift: unload Ollama's model, load diffusion model,
        generate image, save to disk, unload diffusion model.
        Ollama reloads its model on next request automatically.
        """
        log.info(f"forge.imagine → prompt={self.prompt!r}, model={self.model or DEFAULT_IMAGE_MODEL}")

        # Step 1: Ask Ollama to unload its model
        await _ollama_unload(client)

        # Step 2: Generate image (runs in thread pool to not block event loop)
        loop = asyncio.get_event_loop()
        image_path, gen_time = await loop.run_in_executor(None, self._generate)

        # Step 3: Build result with base64 image for direct context injection
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        self.result = {
            "path": str(image_path),
            "base64": image_b64,
            "prompt": self.prompt,
            "model": self.model or DEFAULT_IMAGE_MODEL,
            "generation_time": gen_time,
            "width": self.width,
            "height": self.height,
        }
        log.info(f"forge.imagine ✓ {image_path} ({gen_time:.1f}s)")

    def _generate(self) -> tuple[Path, float]:
        """Synchronous image generation. Runs in a thread."""
        import torch
        from diffusers import AutoPipelineForText2Image

        model_id = self.model or DEFAULT_IMAGE_MODEL

        t0 = time.time()
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to("cuda")

        generator = None
        if self.seed is not None:
            generator = torch.Generator("cuda").manual_seed(self.seed)

        image = pipe(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt or None,
            num_inference_steps=self.steps,
            width=self.width,
            height=self.height,
            generator=generator,
        ).images[0]

        gen_time = time.time() - t0

        # Save to disk
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{self.prompt[:50].replace(' ', '_').replace('/', '_')}.jpg"
        image_path = IMAGE_DIR / filename
        image.save(image_path, "JPEG", quality=90)

        # Unload the diffusion model to free VRAM
        del pipe
        torch.cuda.empty_cache()

        return image_path, gen_time


# ---------------------------------------------------------------------------
# Ollama VRAM management
# ---------------------------------------------------------------------------


async def _ollama_unload(client: httpx.AsyncClient):
    """Ask Ollama to unload all models from VRAM."""
    try:
        # List running models
        resp = await client.get(f"{OLLAMA_URL}/api/ps", timeout=10.0)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            for model in models:
                model_name = model.get("name", "")
                if model_name:
                    log.info(f"forge.ollama_unload → {model_name}")
                    await client.post(
                        f"{OLLAMA_URL}/api/generate",
                        json={"model": model_name, "keep_alive": 0},
                        timeout=30.0,
                    )
            # Give Ollama a moment to release VRAM
            await asyncio.sleep(2)
    except Exception as e:
        log.warning(f"forge.ollama_unload failed: {e}")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class GPUWorker:
    """Single worker that drains the job queue. One job at a time."""

    def __init__(self):
        self.queue: asyncio.Queue[Job] = asyncio.Queue()
        self.current_job: Job | None = None
        self._client: httpx.AsyncClient | None = None

    async def start(self):
        self._client = httpx.AsyncClient()
        log.info("forge.worker started")
        while True:
            job = await self.queue.get()
            self.current_job = job
            try:
                await job.execute(self._client)
            except Exception as e:
                log.error(f"forge.worker error: {e}", exc_info=True)
                job.error = e
            finally:
                job.done.set()
                self.current_job = None
                self.queue.task_done()

    async def submit(self, job: Job) -> Any:
        """Submit a job and wait for completion."""
        await self.queue.put(job)
        await job.done.wait()
        if job.error:
            raise job.error
        return job.result

    @property
    def status(self) -> dict:
        return {
            "queue_depth": self.queue.qsize(),
            "busy": self.current_job is not None,
            "current_job_type": type(self.current_job).__name__ if self.current_job else None,
        }


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

worker = GPUWorker()


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(worker.start())
    log.info(f"Forge listening on port {FORGE_PORT}")
    yield
    task.cancel()

app = FastAPI(title="Forge", lifespan=lifespan)


# --- Ollama proxy endpoints ---

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    job = OllamaProxyJob(method="POST", path="/v1/chat/completions", body=body)
    result = await worker.submit(job)
    return JSONResponse(content=result["body"], status_code=result["status_code"])


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    job = OllamaProxyJob(method="POST", path="/v1/embeddings", body=body)
    result = await worker.submit(job)
    return JSONResponse(content=result["body"], status_code=result["status_code"])


@app.post("/api/generate")
async def ollama_generate(request: Request):
    """Proxy Ollama's native generate endpoint too."""
    body = await request.json()
    job = OllamaProxyJob(method="POST", path="/api/generate", body=body)
    result = await worker.submit(job)
    return JSONResponse(content=result["body"], status_code=result["status_code"])


@app.post("/api/chat")
async def ollama_chat(request: Request):
    """Proxy Ollama's native chat endpoint."""
    body = await request.json()
    job = OllamaProxyJob(method="POST", path="/api/chat", body=body)
    result = await worker.submit(job)
    return JSONResponse(content=result["body"], status_code=result["status_code"])


@app.post("/api/embeddings")
async def ollama_embeddings(request: Request):
    """Proxy Ollama's native embeddings endpoint."""
    body = await request.json()
    job = OllamaProxyJob(method="POST", path="/api/embeddings", body=body)
    result = await worker.submit(job)
    return JSONResponse(content=result["body"], status_code=result["status_code"])


@app.get("/api/tags")
async def ollama_tags(request: Request):
    """Proxy Ollama's model list endpoint (used for health checks)."""
    job = OllamaProxyJob(method="GET", path="/api/tags")
    result = await worker.submit(job)
    return JSONResponse(content=result["body"], status_code=result["status_code"])


@app.get("/api/ps")
async def ollama_ps(request: Request):
    """Proxy Ollama's running models endpoint."""
    job = OllamaProxyJob(method="GET", path="/api/ps")
    result = await worker.submit(job)
    return JSONResponse(content=result["body"], status_code=result["status_code"])


# --- Imagination endpoint ---

@app.post("/imagine")
async def imagine(request: Request):
    body = await request.json()
    job = ImagineJob(
        prompt=body.get("prompt", ""),
        negative_prompt=body.get("negative_prompt", ""),
        model=body.get("model", ""),
        steps=body.get("steps", 20),
        width=body.get("width", 1024),
        height=body.get("height", 1024),
        seed=body.get("seed"),
    )
    result = await worker.submit(job)
    return JSONResponse(content=result)


# --- Status ---

@app.get("/status")
async def status():
    return {
        "service": "forge",
        "worker": worker.status,
        "ollama_url": OLLAMA_URL,
        "image_dir": str(IMAGE_DIR),
        "default_image_model": DEFAULT_IMAGE_MODEL,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=FORGE_PORT)
