#!/usr/bin/env bash
# One-command setup and launcher for the local IndicF5 Telugu Voice Lab.
#
# The model weights are gated on Hugging Face.  After accepting the model terms
# once, export HF_TOKEN (or log in with `hf auth login`) and run this script.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${TTS_VENV:-$ROOT_DIR/.venv}"
PYTHON_REQUEST="${TTS_PYTHON:-}"
PORT="${VOICE_LAB_PORT:-7860}"
DEVICE="${TTS_DEVICE:-auto}"
NO_BROWSER=0
SETUP_ONLY=0
SKIP_INSTALL=0
SKIP_MODELS=0

log() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage: ./run.sh [options]

Creates/reuses .venv, installs the pinned runtime, caches the gated IndicF5
weights, and starts the Telugu Voice Lab at http://127.0.0.1:7860.

Options:
  --no-browser       Do not try to open a browser (useful on servers/SSH).
  --port PORT        Voice Lab port (default: 7860).
  --device DEVICE    auto, cpu, mps, cuda, or cuda:N (default: auto).
  --setup-only       Install dependencies and cache weights, then exit.
  --skip-install     Reuse an existing virtualenv; do not run pip.
  --skip-models      Reuse cached weights; do not contact Hugging Face.
  --venv PATH        Virtualenv directory (default: .venv).
  --python PATH      Python 3.11 or 3.12 executable to use.
  -h, --help         Show this message.

Environment:
  HF_TOKEN           Hugging Face token with access to ai4bharat/IndicF5.
                     A token saved by `hf auth login` is also used.
  TTS_VENV           Override the virtualenv directory.
  TTS_PYTHON         Override the Python executable.
  VOICE_LAB_PORT     Override the UI port.
  TTS_DEVICE         Override the inference device.
  TTS_TORCH_INDEX_URL
                     Override the PyTorch wheel index (for example a different
                     CUDA wheel index).

One-time prerequisite:
  Accept the terms at https://huggingface.co/ai4bharat/IndicF5, then either
  `export HF_TOKEN=hf_...` or run `hf auth login`.  The script does not bypass
  gated-model access.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --no-browser) NO_BROWSER=1; shift ;;
        --port)
            [ "$#" -ge 2 ] || die "--port requires a value"
            PORT="$2"; shift 2 ;;
        --device)
            [ "$#" -ge 2 ] || die "--device requires a value"
            DEVICE="$2"; shift 2 ;;
        --setup-only) SETUP_ONLY=1; shift ;;
        --skip-install) SKIP_INSTALL=1; shift ;;
        --skip-models) SKIP_MODELS=1; shift ;;
        --venv)
            [ "$#" -ge 2 ] || die "--venv requires a path"
            VENV_DIR="$2"; shift 2 ;;
        --python)
            [ "$#" -ge 2 ] || die "--python requires a path"
            PYTHON_REQUEST="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1 (use --help)" ;;
    esac
done

case "$PORT" in
    ''|*[!0-9]*) die "PORT must be a number, got: $PORT" ;;
esac
[ "$PORT" -ge 1 ] && [ "$PORT" -le 65535 ] || die "PORT must be between 1 and 65535"
if ! [[ "$DEVICE" =~ ^(auto|cpu|mps|cuda(:[0-9]+)?)$ ]]; then
    die "DEVICE must be auto, cpu, mps, cuda, or cuda:N (got: $DEVICE)"
fi

OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
    Darwin)
        case "$ARCH" in
            arm64|x86_64) ;;
            *) die "Unsupported macOS architecture: $ARCH" ;;
        esac ;;
    Linux)
        case "$ARCH" in
            x86_64|aarch64|arm64) ;;
            *) die "Unsupported Linux architecture: $ARCH" ;;
        esac ;;
    *)
        die "This launcher supports macOS and Linux. On Windows, use WSL2 and run it inside the Linux environment." ;;
esac

command -v git >/dev/null 2>&1 || die "git is required to install the pinned IndicF5/F5-TTS source package."

# The venv + PyTorch + weights need several GB.  Check before downloading.
AVAILABLE_KB="$(df -Pk "$ROOT_DIR" 2>/dev/null | awk 'NR==2 {print $4}' || true)"
if [ -n "$AVAILABLE_KB" ] && [ "$AVAILABLE_KB" -lt 5242880 ]; then
    die "At least 5 GB of free disk space is required; found approximately $((AVAILABLE_KB / 1024 / 1024)) GB."
fi

venv_python_ok() {
    "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] in ((3, 11), (3, 12)) else 1)' >/dev/null 2>&1
}

if [ -n "$PYTHON_REQUEST" ]; then
    if [ -x "$PYTHON_REQUEST" ]; then
        PYTHON="$PYTHON_REQUEST"
    elif command -v "$PYTHON_REQUEST" >/dev/null 2>&1; then
        PYTHON="$(command -v "$PYTHON_REQUEST")"
    else
        die "Python executable not found: $PYTHON_REQUEST"
    fi
    venv_python_ok "$PYTHON" || die "Python 3.11 or 3.12 is required; $PYTHON is not supported."
else
    PYTHON=""
    for candidate in python3.12 python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            candidate_path="$(command -v "$candidate")"
            if venv_python_ok "$candidate_path"; then
                PYTHON="$candidate_path"
                break
            fi
        fi
    done
    [ -n "$PYTHON" ] || die "Python 3.11 or 3.12 was not found. Install python3.12 (macOS: brew install python@3.12; Ubuntu: sudo apt install python3.12 python3.12-venv) or pass --python PATH."
fi

log "Using Python: $("$PYTHON" --version 2>&1)"

# FFmpeg and its shared libraries are needed by the audio decoder (and by
# TorchCodec).  Try the common safe package-manager paths; otherwise explain.
if ! command -v ffmpeg >/dev/null 2>&1; then
    log "FFmpeg is missing; attempting to install it."
    if [ "$OS" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
        brew install ffmpeg
    elif [ "$OS" = "Linux" ] && command -v apt-get >/dev/null 2>&1; then
        if [ "$(id -u)" -eq 0 ]; then
            apt-get update
            apt-get install -y ffmpeg libsndfile1
        elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
            sudo -n apt-get update
            sudo -n apt-get install -y ffmpeg libsndfile1
        else
            die "FFmpeg is required. Install it with: sudo apt-get install ffmpeg libsndfile1"
        fi
    else
        die "FFmpeg is required. Install it with Homebrew (macOS) or your Linux package manager, then rerun."
    fi
fi
command -v ffmpeg >/dev/null 2>&1 || die "FFmpeg installation did not produce an executable. Install it manually and rerun."

VENV_PYTHON="$VENV_DIR/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
    [ "$SKIP_INSTALL" -eq 0 ] || die "$VENV_DIR has no usable Python. Remove --skip-install or choose another --venv."
    log "Creating virtualenv at $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR" || die "Could not create the virtualenv. On Debian/Ubuntu install python3-venv."
fi
venv_python_ok "$VENV_PYTHON" || die "$VENV_DIR uses an unsupported Python. Remove it or set --venv to a fresh path."

MARKER="$VENV_DIR/.tts-quickstart-ready-v1"
NEED_INSTALL=0
if [ "$SKIP_INSTALL" -eq 0 ]; then
    if [ ! -f "$MARKER" ]; then
        NEED_INSTALL=1
    else
        for requirement in "$ROOT_DIR/requirements.txt" "$ROOT_DIR/requirements-core.txt" "$ROOT_DIR/requirements-api.txt"; do
            if [ -f "$requirement" ] && [ "$requirement" -nt "$MARKER" ]; then
                NEED_INSTALL=1
                break
            fi
        done
    fi
fi

if [ "$SKIP_INSTALL" -eq 1 ] && ! "$VENV_PYTHON" -c 'import torch, gradio, f5_tts, soundfile, torchcodec' >/dev/null 2>&1; then
    die "The existing virtualenv is incomplete. Rerun without --skip-install."
fi

if [ "$NEED_INSTALL" -eq 1 ]; then
    log "Installing/updating Python packages (this can take several minutes)..."
    export PIP_NO_INPUT=1
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

    TORCH_INDEX="${TTS_TORCH_INDEX_URL:-}"
    if [ -z "$TORCH_INDEX" ]; then
        if [ "$OS" = "Linux" ]; then
            if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
                TORCH_INDEX="https://download.pytorch.org/whl/cu128"
                log "NVIDIA GPU detected; installing CUDA PyTorch wheels."
            else
                TORCH_INDEX="https://download.pytorch.org/whl/cpu"
                log "No NVIDIA GPU detected; installing CPU PyTorch wheels."
            fi
        fi
    fi

    if [ -n "$TORCH_INDEX" ]; then
        "$VENV_PYTHON" -m pip install --index-url "$TORCH_INDEX" torch==2.11.0 torchaudio==2.11.0
    else
        "$VENV_PYTHON" -m pip install torch==2.11.0 torchaudio==2.11.0
    fi
    "$VENV_PYTHON" -m pip install -r "$ROOT_DIR/requirements.txt"

    if ! "$VENV_PYTHON" - <<'PY'
import torch
import torchaudio
import torchcodec
import gradio
import f5_tts
import soundfile
print(f"PyTorch {torch.__version__}, torchaudio {torchaudio.__version__}, Gradio {gradio.__version__}")
PY
    then
        die "The installed runtime failed its import check. See the pip/FFmpeg output above."
    fi
    touch "$MARKER"
fi

mkdir -p "$ROOT_DIR/outputs" "$ROOT_DIR/references" "$ROOT_DIR/cache"

if [ "$SKIP_MODELS" -eq 0 ]; then
    log "Caching IndicF5 and Vocos weights (first run: about 1.5 GB)..."
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    if ! "$VENV_PYTHON" -m tts.setup_models --skip-transliterator; then
        cat >&2 <<'EOF'

Hugging Face model setup failed.

The IndicF5 model is gated. One-time setup is required:
  1. Sign in at https://huggingface.co/ai4bharat/IndicF5
  2. Accept the model terms.
  3. Export a token with access: export HF_TOKEN=hf_...
     Or run: hf auth login
  4. Rerun: ./run.sh

The script never bypasses the gated-model agreement and never prints the token.
If the weights are already cached, you can rerun with --skip-models.
EOF
        exit 3
    fi
fi

if [ "$SETUP_ONLY" -eq 1 ]; then
    log "Setup complete. Model assets and the Python environment are ready."
    exit 0
fi

export PATH="$VENV_DIR/bin:$PATH"
export GRADIO_ANALYTICS_ENABLED=False
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TTS_DEVICE="$DEVICE"
export VOICE_LAB_PORT="$PORT"
export PYTHONUNBUFFERED=1
if [ "$NO_BROWSER" -eq 1 ]; then
    export VOICE_LAB_NO_BROWSER=1
fi

log "Starting Telugu Voice Lab on http://127.0.0.1:$PORT"
printf '%s\n' "First generation loads IndicF5 and is slower; later generations reuse the loaded model."
printf '%s\n' "Press Ctrl-C to stop."
exec "$VENV_PYTHON" "$ROOT_DIR/voice_lab_app.py"
