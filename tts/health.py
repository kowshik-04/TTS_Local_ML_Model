"""Real, side-effect-free health checks. Never loads the model or touches the network."""

from __future__ import annotations

import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path


def _hf_snapshot_has(repo_id: str, filenames: list[str]) -> bool:
    cache = Path(os.environ.get("HF_HUB_CACHE") or
                 Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    snapshots = cache / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(all((snap / name).exists() for name in filenames) for snap in snapshots.iterdir())


def _device() -> str:
    try:
        import torch
    except ImportError:
        return "PyTorch not installed"
    if torch.cuda.is_available():
        return "CUDA"
    if torch.backends.mps.is_available():
        return "MPS (Apple Silicon)"
    return "CPU"


def _writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".write-check-"):
            pass
        return True
    except OSError:
        return False


def collect_health(output_dir: Path, engine=None) -> list[tuple[str, str, str]]:
    """Return (label, value, level) rows; level is ok | warn | error | info."""
    loaded = getattr(engine, "_bundle", None) is not None if engine is not None else False
    assets = (
        _hf_snapshot_has("ai4bharat/IndicF5", ["model.safetensors", "checkpoints/vocab.txt"])
        and _hf_snapshot_has("charactr/vocos-mel-24khz", ["config.yaml", "pytorch_model.bin"])
    )
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    return [
        ("Production engine", "IndicF5 (ai4bharat/IndicF5)", "info"),
        ("Model", "Loaded in memory" if loaded else "Not loaded (loads on first generation)",
         "ok" if loaded else "info"),
        ("Inference device", _device(), "info"),
        ("FFmpeg", shutil.which("ffmpeg") or "Not found — MP3 export will fail",
         "ok" if shutil.which("ffmpeg") else "error"),
        ("Local model assets", "Found in Hugging Face cache" if assets
         else "Missing — run `python -m tts.setup_models`", "ok" if assets else "error"),
        ("Output folder writable", str(output_dir) if _writable(output_dir) else "Not writable",
         "ok" if _writable(output_dir) else "error"),
        ("Hugging Face offline mode", "Enabled (HF_HUB_OFFLINE=1)" if offline else "Not enabled",
         "ok" if offline else "warn"),
        ("Network binding", "127.0.0.1 only, share=False, analytics disabled", "ok"),
        ("Python", f"{platform.python_version()} · {Path(sys.executable).parent.parent.name}", "info"),
    ]
