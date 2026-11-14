# Local Voice Studio — Linux container for CPU or NVIDIA CUDA inference.
#
# CUDA build (default):
#   docker build -t local-voice-studio .
# CPU build:
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
#                -t local-voice-studio:cpu .
#
# Model weights are deliberately not baked into the image. Download them into
# the mounted /models volume once with:
#   docker compose run --rm api python -m tts.setup_models
FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/huggingface \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 curl git tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-core.txt requirements-api.txt requirements-agent.txt requirements-transliteration.txt requirements.txt ./

# Install the accelerator-matched PyTorch wheel first so the requirements pins
# are satisfied without downloading a second multi-gigabyte wheel.
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url "${TORCH_INDEX_URL}" \
        torch==2.11.0 torchaudio==2.11.0 \
    && python -m pip install -r requirements-api.txt -r requirements-agent.txt \
    && python -m pip install --no-deps -r requirements-transliteration.txt \
    && python -m pip install pydload pandas

COPY api ./api
COPY agent ./agent
COPY tts ./tts
COPY eval ./eval
COPY static ./static
COPY app.py voice_lab_app.py ./

# Fail the build if the pinned F5-TTS source cannot import under the pinned
# NumPy/datasets/pyarrow set. This catches the pyarrow NumPy-2 mismatch before
# an image is published.
RUN python -c "from f5_tts.api import F5TTS; import api.main; import agent.server; print('F5-TTS and service imports OK')"

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data /app/cache /app/outputs /app/voices \
       /app/experiments/chatterbox_telugu/models /models/huggingface \
    && chown -R app:app /app /models

USER app
EXPOSE 8000 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
