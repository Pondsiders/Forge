#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "httpx",
#     "logfire[fastapi,httpx]",
#     "pillow",
# ]
# ///
"""
Forge — GPU arbiter for Pondside.

Routes AI workloads through a single queue so the GPU is never contested.
Proxies chat/embed requests to Ollama. Delegates image generation to a
subprocess (forge_imagine.py) so VRAM is fully reclaimed on completion.
One worker, one GPU, one job at a time.

Instrumented with Logfire for full observability including gen_ai.* traces.

Usage:
    ./forge.py
    uv run forge.py
"""

import asyncio
import base64
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import logfire
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Context

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
FORGE_PORT = int(os.getenv("FORGE_PORT", "8200"))
IMAGE_DIR = Path(os.getenv("FORGE_IMAGE_DIR", "/Pondside/Alpha-Home/images/imagination"))
DEFAULT_IMAGE_MODEL = os.getenv("FORGE_IMAGE_MODEL", "Tongyi-MAI/Z-Image-Turbo")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("forge")

# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------

logfire.configure(service_name="forge")


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
    trace_context: Context | None = None  # Carries the caller's trace context into the worker
    api_span: Any = None  # The root FastAPI span — gen_ai.* attributes go here


@dataclass
class OllamaProxyJob(Job):
    """Forward a request to Ollama."""
    method: str = "POST"
    path: str = ""
    body: dict = field(default_factory=dict)
    headers: dict = field(default_factory=dict)

    @property
    def _model(self) -> str:
        """Extract model name from request body."""
        return self.body.get("model", "unknown")

    @property
    def _operation(self) -> str:
        """Infer operation type from path."""
        if "embed" in self.path:
            return "embed"
        elif "chat" in self.path:
            return "chat"
        elif "generate" in self.path:
            return "generate"
        elif "tags" in self.path:
            return "list_models"
        elif "ps" in self.path:
            return "list_running"
        return "proxy"

    def _set_api_attr(self, key: str, value: Any):
        """Set an attribute on the root FastAPI span (for the Model Run panel)."""
        if self.api_span and self.api_span.is_recording():
            self.api_span.set_attribute(key, value)

    async def execute(self, client: httpx.AsyncClient):
        url = f"{OLLAMA_URL}{self.path}"

        # Attach gen_ai.* attributes to the root FastAPI span — one click, Model Run panel
        self._set_api_attr("gen_ai.operation.name", self._operation)
        self._set_api_attr("gen_ai.provider.name", "ollama")
        self._set_api_attr("gen_ai.request.model", self._model)

        # Attach input: system instructions and messages (for chat operations)
        # Format: Logfire expects {"role": "...", "parts": [{"type": "text", "content": "..."}]}
        messages = self.body.get("messages", [])
        if messages:
            system_msgs = [m for m in messages if m.get("role") == "system"]
            non_system = [m for m in messages if m.get("role") != "system"]
            if system_msgs:
                self._set_api_attr(
                    "gen_ai.system_instructions",
                    json.dumps([{"type": "text", "content": m.get("content", "")} for m in system_msgs]),
                )
            if non_system:
                self._set_api_attr(
                    "gen_ai.input.messages",
                    json.dumps([
                        {"role": m.get("role"), "parts": [{"type": "text", "content": m.get("content", "")}]}
                        for m in non_system
                    ]),
                )

        # For Ollama native /api/generate (prompt field, no messages array)
        if "prompt" in self.body and not messages:
            self._set_api_attr(
                "gen_ai.input.messages",
                json.dumps([{"role": "user", "parts": [{"type": "text", "content": self.body["prompt"]}]}]),
            )

        # For embeddings, capture the input text
        if self._operation == "embed":
            embed_input = self.body.get("input") or self.body.get("prompt", "")
            if isinstance(embed_input, list):
                preview = embed_input[0][:200] if embed_input else ""
            else:
                preview = str(embed_input)[:200]
            self._set_api_attr(
                "gen_ai.input.messages",
                json.dumps([{"role": "user", "parts": [{"type": "text", "content": preview}]}]),
            )

        with logfire.span(
            "forge.{operation} {model}",
            operation=self._operation,
            model=self._model,
        ):
            resp = await client.request(
                method=self.method,
                url=url,
                json=self.body if self.method in ("POST", "PUT", "PATCH") else None,
                headers={k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")},
                timeout=120.0,
            )

            body = resp.json()
            self.result = {"status_code": resp.status_code, "body": body, "headers": dict(resp.headers)}

            # Extract token usage and attach to the root FastAPI span
            if "prompt_eval_count" in body:
                self._set_api_attr("gen_ai.usage.input_tokens", body["prompt_eval_count"])
            if "eval_count" in body:
                self._set_api_attr("gen_ai.usage.output_tokens", body["eval_count"])
            if "total_duration" in body:
                self._set_api_attr("forge.total_duration_ns", body["total_duration"])
            if "model" in body:
                self._set_api_attr("gen_ai.response.model", body["model"])

            # Attach output message (for chat operations)
            # Format: {"role": "...", "parts": [{"type": "text", "content": "..."}]}
            if "message" in body:
                self._set_api_attr(
                    "gen_ai.output.messages",
                    json.dumps([{"role": body["message"].get("role", "assistant"),
                                 "parts": [{"type": "text", "content": body["message"].get("content", "")}]}]),
                )
            elif "response" in body:
                # /api/generate returns response as a string
                self._set_api_attr(
                    "gen_ai.output.messages",
                    json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": body["response"]}]}]),
                )
            elif "embedding" in body or "embeddings" in body:
                self._set_api_attr(
                    "gen_ai.output.messages",
                    json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": "[embedding vector]"}]}]),
                )


@dataclass
class ImagineJob(Job):
    """Generate an image via a subprocess.

    Delegates to forge_imagine.py which loads the model, generates,
    saves to disk, and exits. Process exit guarantees full VRAM cleanup —
    no ghost allocations, no "sometimes," no anterograde amnesia at 2 AM.
    """
    prompt: str = ""
    negative_prompt: str = ""
    model: str = ""
    steps: int = 20
    width: int = 1024
    height: int = 1024
    seed: int | None = None

    # Path to the subprocess script (sibling of this file)
    IMAGINE_SCRIPT: Path = Path(__file__).parent / "forge_imagine.py"

    def _set_api_attr(self, key: str, value: Any):
        """Set an attribute on the root FastAPI span."""
        if self.api_span and self.api_span.is_recording():
            self.api_span.set_attribute(key, value)

    async def execute(self, client: httpx.AsyncClient):
        """
        Unload Ollama, spawn subprocess for generation, read result.
        Ollama reloads its model on next request automatically.
        """
        model_id = self.model or DEFAULT_IMAGE_MODEL

        # Attach gen_ai.* to root FastAPI span
        self._set_api_attr("gen_ai.operation.name", "imagine")
        self._set_api_attr("gen_ai.provider.name", "diffusers")
        self._set_api_attr("gen_ai.request.model", model_id)
        self._set_api_attr("gen_ai.input.messages", json.dumps([
            {"role": "user", "parts": [{"type": "text", "content": self.prompt}]}
        ]))

        with logfire.span(
            "forge.imagine {model}",
            model=model_id,
            prompt_preview=self.prompt[:80],
        ) as span:
            # Step 1: Ask Ollama to unload its model to free VRAM
            with logfire.span("forge.imagine.ollama_unload"):
                await _ollama_unload(client)

            # Step 2: Generate image in subprocess
            with logfire.span(
                "forge.imagine.subprocess",
                steps=self.steps,
                width=self.width,
                height=self.height,
            ):
                # Propagate trace context so subprocess spans nest here
                from opentelemetry.propagate import inject as otel_inject
                carrier: dict[str, str] = {}
                otel_inject(carrier)

                # Build job spec
                spec = json.dumps({
                    "prompt": self.prompt,
                    "negative_prompt": self.negative_prompt,
                    "model": model_id,
                    "steps": self.steps,
                    "width": self.width,
                    "height": self.height,
                    "seed": self.seed,
                    "image_dir": str(IMAGE_DIR),
                })

                # Spawn — subprocess gets full env plus traceparent
                env = {**os.environ, "TRACEPARENT": carrier.get("traceparent", "")}
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, str(self.IMAGINE_SCRIPT),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )
                stdout, stderr = await proc.communicate(input=spec.encode())

                # Log subprocess output (model loading progress, etc.)
                if stderr:
                    for line in stderr.decode().strip().split("\n"):
                        if line.strip():
                            log.info(f"[imagine] {line}")

                if proc.returncode != 0:
                    raise RuntimeError(
                        f"forge_imagine.py failed (exit {proc.returncode}): "
                        f"{stderr.decode()[-500:]}"
                    )

                # Parse result — subprocess is dead, VRAM is free
                # Take last non-empty line (libraries may print to stdout)
                stdout_lines = [l for l in stdout.decode().strip().split("\n") if l.strip()]
                result = json.loads(stdout_lines[-1])
                image_path = Path(result["path"])
                gen_time = result["generation_time"]

            span.set_attribute("forge.generation_time_s", gen_time)
            span.set_attribute("forge.image_path", str(image_path))

            # Attach output to root FastAPI span
            self._set_api_attr("forge.generation_time_s", gen_time)
            self._set_api_attr(
                "gen_ai.output.messages",
                json.dumps([{"role": "assistant", "parts": [
                    {"type": "text", "content": f"[image: {image_path.name}] ({gen_time:.1f}s)"}
                ]}]),
            )

            # Step 3: Build result with base64 image for direct context injection
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")

            self.result = {
                "path": str(image_path),
                "base64": image_b64,
                "prompt": self.prompt,
                "model": model_id,
                "generation_time": gen_time,
                "width": self.width,
                "height": self.height,
            }


# ---------------------------------------------------------------------------
# Ollama VRAM management
# ---------------------------------------------------------------------------


async def _ollama_unload(client: httpx.AsyncClient):
    """Ask Ollama to unload all models from VRAM."""
    try:
        resp = await client.get(f"{OLLAMA_URL}/api/ps", timeout=10.0)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            for model in models:
                model_name = model.get("name", "")
                if model_name:
                    logfire.info("forge.ollama_unload → {model}", model=model_name)
                    await client.post(
                        f"{OLLAMA_URL}/api/generate",
                        json={"model": model_name, "keep_alive": 0},
                        timeout=30.0,
                    )
            await asyncio.sleep(2)
    except Exception as e:
        logfire.warning("forge.ollama_unload failed: {error}", error=str(e))


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class GPUWorker:
    """Single worker that drains the job queue. One job at a time."""

    def __init__(self):
        self.queue: asyncio.Queue[Job] = asyncio.Queue()
        self.current_job: Job | None = None
        self._client: httpx.AsyncClient | None = None
        self._jobs_completed: int = 0

    async def start(self):
        self._client = httpx.AsyncClient()
        logfire.instrument_httpx(self._client)
        logfire.info("forge.worker started")
        while True:
            job = await self.queue.get()
            self.current_job = job
            queue_wait = time.time() - job.created_at
            try:
                # Attach to the caller's trace context so worker spans
                # nest inside the HTTP request span (the bar tab pattern)
                ctx = job.trace_context or otel_context.get_current()
                token = otel_context.attach(ctx)
                try:
                    with logfire.span(
                        "forge.worker.job",
                        job_type=type(job).__name__,
                        queue_wait_s=round(queue_wait, 3),
                        queue_depth=self.queue.qsize(),
                    ):
                        await job.execute(self._client)
                        self._jobs_completed += 1
                finally:
                    otel_context.detach(token)
            except Exception as e:
                logfire.error("forge.worker error: {error}", error=str(e))
                job.error = e
            finally:
                job.done.set()
                self.current_job = None
                self.queue.task_done()

    async def submit(self, job: Job) -> Any:
        """Submit a job and wait for completion."""
        # Capture the caller's trace context so the worker can nest under it
        job.trace_context = otel_context.get_current()
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
            "jobs_completed": self._jobs_completed,
        }


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

worker = GPUWorker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(worker.start())
    logfire.info("Forge listening on port {port}", port=FORGE_PORT)
    yield
    task.cancel()

app = FastAPI(title="Forge", lifespan=lifespan)
logfire.instrument_fastapi(app)


# --- Helpers ---

def _capture_api_span() -> Any:
    """Grab the current FastAPI span so gen_ai.* attributes land on the root."""
    return trace.get_current_span()


# --- Ollama proxy endpoints ---

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    job = OllamaProxyJob(method="POST", path="/v1/chat/completions", body=body, api_span=_capture_api_span())
    result = await worker.submit(job)
    return JSONResponse(content=result["body"], status_code=result["status_code"])


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    job = OllamaProxyJob(method="POST", path="/v1/embeddings", body=body, api_span=_capture_api_span())
    result = await worker.submit(job)
    return JSONResponse(content=result["body"], status_code=result["status_code"])


@app.post("/api/generate")
async def ollama_generate(request: Request):
    """Proxy Ollama's native generate endpoint."""
    body = await request.json()
    job = OllamaProxyJob(method="POST", path="/api/generate", body=body, api_span=_capture_api_span())
    result = await worker.submit(job)
    return JSONResponse(content=result["body"], status_code=result["status_code"])


@app.post("/api/chat")
async def ollama_chat(request: Request):
    """Proxy Ollama's native chat endpoint."""
    body = await request.json()
    job = OllamaProxyJob(method="POST", path="/api/chat", body=body, api_span=_capture_api_span())
    result = await worker.submit(job)
    return JSONResponse(content=result["body"], status_code=result["status_code"])


@app.post("/api/embeddings")
async def ollama_embeddings(request: Request):
    """Proxy Ollama's native embeddings endpoint."""
    body = await request.json()
    job = OllamaProxyJob(method="POST", path="/api/embeddings", body=body, api_span=_capture_api_span())
    result = await worker.submit(job)
    return JSONResponse(content=result["body"], status_code=result["status_code"])


@app.get("/api/tags")
async def ollama_tags(request: Request):
    """Proxy Ollama's model list endpoint (used for health checks)."""
    job = OllamaProxyJob(method="GET", path="/api/tags", api_span=_capture_api_span())
    result = await worker.submit(job)
    return JSONResponse(content=result["body"], status_code=result["status_code"])


@app.get("/api/ps")
async def ollama_ps(request: Request):
    """Proxy Ollama's running models endpoint."""
    job = OllamaProxyJob(method="GET", path="/api/ps", api_span=_capture_api_span())
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
        api_span=_capture_api_span(),
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
