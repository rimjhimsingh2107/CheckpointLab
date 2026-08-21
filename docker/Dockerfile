# CheckpointLab worker image.
#
# One image, used for every role: each DDP rank is a container running
# training/train.py in single-rank worker mode. That mode already exists --
# Phase 4 added it so the controller could supervise ranks as individual OS
# processes -- and a container is just a stronger boundary around the same
# thing. Nothing in the training code is container-aware.
#
# CPU-only on purpose. The default Linux torch wheel bundles CUDA and is
# ~2.5 GB; this project trains on gloo across CPU workers, so the CPU wheel
# is both correct and roughly an order of magnitude smaller. For real GPU
# runs you would swap the base image for an nvidia/cuda one and drop the
# --index-url.

FROM python:3.13-slim AS base

# libgomp1: torch's CPU kernels are OpenMP-parallel and the slim image
# doesn't ship the runtime. Without it `import torch` fails with a bare
# "libgomp.so.1: cannot open shared object file", which is an unhelpful
# way to discover this.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies before source, so editing a .py file doesn't invalidate the
# torch download layer. torch is ~200 MB even as the CPU wheel; rebuilding
# that on every code change makes the image useless for iteration.
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY training/    training/
COPY checkpoint/  checkpoint/
COPY controller/  controller/
COPY failures/    failures/
COPY metrics/     metrics/
COPY experiments/ experiments/
COPY tests/       tests/
COPY pytest.ini   ./

# Imports in this project are rooted at the repo, and the worker is launched
# as `python training/train.py` rather than as a module, so make the root
# importable rather than relying on cwd.
ENV PYTHONPATH=/app

# Written to by the workers; declared so a bind-mount or volume lands
# somewhere that already exists with the right ownership.
RUN mkdir -p /app/results /app/data

# No ENTRYPOINT that assumes a role. Which rank a container is, and whether
# it's a worker or the controller, is entirely the compose command's business.
CMD ["python", "-c", "import torch; print(f'CheckpointLab image OK — torch {torch.__version__}, cuda={torch.cuda.is_available()}')"]
