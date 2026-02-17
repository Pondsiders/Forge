# Forge — GPU arbiter for Pondside
# PyTorch base image includes CUDA 12.6, cuDNN 9, Python 3.12, torch.
# We just add the application-level deps and forge.py.

FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel

# Application dependencies (torch is already in the base image)
RUN pip install --no-cache-dir \
        fastapi \
        uvicorn \
        httpx \
        "logfire[fastapi,httpx]" \
        diffusers \
        transformers \
        accelerate \
        bitsandbytes \
        sentencepiece \
        pillow

WORKDIR /app
COPY forge.py forge_imagine.py ./

# HF cache lives on a mounted volume — don't bake models into the image
ENV HF_HOME=/cache/huggingface

# Forge listens on 8200
EXPOSE 8200

CMD ["python", "forge.py"]
