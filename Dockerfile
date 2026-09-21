# syntax=docker/dockerfile:1
#
# Laya typed-decision service.
#
# CPU (default):
#   docker build -t laya:cpu .
# NVIDIA:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 -t laya:cu124 .
# AMD/ROCm:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/rocm6.2 -t laya:rocm .
#
# Weights are NOT baked in: mount a volume at /models (HF_HOME) so the ~1.7GB
# checkpoint is downloaded once and survives rebuilds.

FROM python:3.12-slim AS base

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # transformers deadlocks probing TF/JAX during laya.load(); disable both.
    USE_TF=0 \
    USE_JAX=0 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    HF_HOME=/models

RUN pip install --index-url "${TORCH_INDEX}" torch

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN useradd --create-home --uid 1000 laya \
 && mkdir -p /models && chown laya:laya /models

WORKDIR /srv


FROM base AS runtime

COPY --chown=laya:laya app/ /srv/app/
COPY --chown=laya:laya scripts/ /srv/scripts/

USER laya

ENV LAYA_CHECKPOINT=convaiinnovations/laya \
    LAYA_DEVICE=auto \
    TORCH_NUM_THREADS=4 \
    OMP_NUM_THREADS=4 \
    PORT=8000

EXPOSE 8000

# Unhealthy until the checkpoint is actually loaded. start_period covers the
# ~3 min first-run download; afterwards the cache volume makes it ~60s.
HEALTHCHECK --interval=10s --timeout=5s --start-period=600s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz',timeout=4).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
