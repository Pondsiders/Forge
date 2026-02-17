#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "logfire",
#     "torch",
#     "diffusers",
#     "transformers",
#     "accelerate",
#     "bitsandbytes",
#     "sentencepiece",
#     "pillow",
# ]
# ///
"""
Subprocess image generator for Forge.

Runs in a separate process to guarantee VRAM cleanup on exit.
When this process exits, the OS reclaims all GPU memory — no cleanup needed,
no "sometimes," no ghost allocations.

Receives job spec as JSON on stdin, outputs result as JSON on stdout.
All logging goes to stderr. Inherits trace context from TRACEPARENT env var.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Logging to stderr — stdout is reserved for the JSON result
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("forge-imagine")


def main():
    # Read job spec from stdin
    spec = json.loads(sys.stdin.read())

    prompt = spec["prompt"]
    negative_prompt = spec.get("negative_prompt", "")
    model_id = spec["model"]
    steps = spec.get("steps", 20)
    width = spec.get("width", 1024)
    height = spec.get("height", 1024)
    seed = spec.get("seed")
    image_dir = Path(spec.get("image_dir", "/Pondside/Alpha-Home/images/imagination"))

    # --- Observability: connect to parent trace ---
    span_ctx = None
    try:
        import logfire

        logfire.configure(
            service_name="forge-imagine",
            console=False,
            distributed_tracing=True,  # We intentionally propagate from Forge
        )

        traceparent = os.environ.get("TRACEPARENT", "")
        if traceparent:
            from opentelemetry import context
            from opentelemetry.propagate import extract

            ctx = extract({"traceparent": traceparent})
            context.attach(ctx)

        span_ctx = logfire.span(
            "forge.imagine.generate",
            model=model_id,
            prompt_preview=prompt[:80],
            steps=steps,
            width=width,
            height=height,
        )
    except Exception as e:
        log.warning(f"Logfire setup failed (non-fatal): {e}")

    # --- Import heavy deps (only in this subprocess) ---
    import torch
    from diffusers import DiffusionPipeline

    # --- Generate ---
    t0 = time.time()

    if span_ctx:
        span_ctx.__enter__()

    try:
        log.info(f"Loading model {model_id}...")
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        )
        pipe.to("cuda")
        load_time = time.time() - t0
        log.info(f"Model loaded in {load_time:.1f}s")

        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)

        gen_kwargs: dict = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "width": width,
            "height": height,
        }
        if negative_prompt:
            gen_kwargs["negative_prompt"] = negative_prompt
        if generator:
            gen_kwargs["generator"] = generator

        log.info(f"Generating image ({steps} steps, {width}x{height})...")
        image = pipe(**gen_kwargs).images[0]

        gen_time = time.time() - t0

        # Save to disk
        image_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_prompt = prompt[:50].replace(" ", "_").replace("/", "_")
        filename = f"{timestamp}_{safe_prompt}.jpg"
        image_path = image_dir / filename
        image.save(image_path, "JPEG", quality=90)

        log.info(f"Saved {image_path} ({gen_time:.1f}s total)")

        if span_ctx:
            span_ctx.span.set_attribute("forge.generation_time_s", gen_time)
            span_ctx.span.set_attribute("forge.image_path", str(image_path))
            span_ctx.__exit__(None, None, None)

    except Exception as e:
        if span_ctx:
            span_ctx.__exit__(type(e), e, e.__traceback__)
        raise

    # Output result as JSON to stdout
    # No VRAM cleanup needed — process exit handles everything
    result = {
        "path": str(image_path),
        "generation_time": gen_time,
        "model": model_id,
        "width": width,
        "height": height,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
