#!/usr/bin/env bash
set -Eeuo pipefail

cd /app

hf_home="${HF_HOME:-/models/huggingface}"
marker="${hf_home}/.indicf5-ready"
token_file="${hf_home}/token"

mkdir -p /app/references /app/outputs /app/cache /app/data "$hf_home"

needs_models=0
if [[ ! -f "$marker" ]]; then
    needs_models=1
fi
if [[ "${VOICE_LAB_FORCE_MODELS:-0}" == "1" ]]; then
    needs_models=1
fi

if [[ "$needs_models" == "1" && "${VOICE_LAB_SKIP_MODELS:-0}" != "1" ]]; then
    if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" && ! -s "$token_file" ]]; then
        cat >&2 <<'EOF'

ERROR: IndicF5 weights are not cached yet and no Hugging Face token was provided.

One-time setup:
  1. Sign in at https://huggingface.co/ai4bharat/IndicF5
  2. Accept the model terms.
  3. Pass a token when starting the container. For example:

       PowerShell:
         $env:HF_TOKEN = "hf_..."
         docker compose -f docker-compose.voice-lab.yml up --build

       macOS/Linux:
         export HF_TOKEN=hf_...
         docker compose -f docker-compose.voice-lab.yml up --build

The token is only needed for the first download. The weights are stored in the
hf_models Docker volume, so later starts work without a token or network.

EOF
        exit 3
    fi

    echo "==> Downloading the gated IndicF5 and Vocos weights into ${hf_home}"
    echo "==> If this reports 401/403, accept the model terms and check the token."
    if ! python -m tts.setup_models --skip-transliterator; then
        cat >&2 <<'EOF'

ERROR: The model download failed.

Check that:
  - HF_TOKEN is valid;
  - the account has accepted the terms at
    https://huggingface.co/ai4bharat/IndicF5;
  - the machine can reach huggingface.co.

Rerun the same command after fixing access. Partial downloads are resumed.
EOF
        exit 3
    fi
    touch "$marker"
    echo "==> Model assets are cached. Restarting will not need the token."
fi

if [[ "${VOICE_LAB_SKIP_MODELS:-0}" == "1" && ! -f "$marker" ]]; then
    echo "WARNING: VOICE_LAB_SKIP_MODELS=1 and no model cache was found; the UI will start but generation will fail." >&2
fi

echo "==> Starting Telugu Voice Lab on http://127.0.0.1:${VOICE_LAB_PORT:-7860}"
exec python voice_lab_app.py
