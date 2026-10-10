from __future__ import annotations

import threading
import warnings
from pathlib import Path
from typing import Callable

import numpy as np

from tts.audio_utils import prepared_reference
from tts.chunking import split_text
from tts.text_normalizer import normalize_text


REPO_ID = "ai4bharat/IndicF5"
VOCODER_REPO_ID = "charactr/vocos-mel-24khz"


def _select_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class IndicF5Engine:
    """Lazy-loaded wrapper retaining the project's validated direct DiT loader."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._bundle = None

    def _load(self):
        with self._lock:
            if self._bundle is not None:
                return self._bundle
            try:
                import torch
                from huggingface_hub import hf_hub_download
                from safetensors.torch import load_file
                from f5_tts.model import DiT
                from f5_tts.infer.utils_infer import load_model, load_vocoder
            except ImportError as error:
                raise RuntimeError(
                    "IndicF5 dependencies are incomplete. Install requirements.txt in .venv."
                ) from error

            try:
                vocab_path = hf_hub_download(
                    REPO_ID, filename="checkpoints/vocab.txt", local_files_only=True
                )
                checkpoint_path = hf_hub_download(
                    REPO_ID, filename="model.safetensors", local_files_only=True
                )
                vocoder_config = hf_hub_download(
                    VOCODER_REPO_ID, filename="config.yaml", local_files_only=True
                )
                hf_hub_download(
                    VOCODER_REPO_ID, filename="pytorch_model.bin", local_files_only=True
                )
            except Exception as error:
                raise RuntimeError(
                    "Required IndicF5/Vocos model files are not cached locally. "
                    "Run `python -m tts.setup_models` while online, then use the app offline."
                ) from error

            device = _select_device(torch)
            vocoder = load_vocoder(
                vocoder_name="vocos",
                is_local=True,
                local_path=str(Path(vocoder_config).parent),
                device=device,
            )
            model = load_model(
                DiT,
                dict(
                    dim=1024,
                    depth=22,
                    heads=16,
                    ff_mult=2,
                    text_dim=512,
                    conv_layers=4,
                ),
                mel_spec_type="vocos",
                vocab_file=vocab_path,
                device=device,
            )
            state_dict = load_file(checkpoint_path, device=device)
            state_dict = {
                key.replace("ema_model._orig_mod.", ""): value
                for key, value in state_dict.items()
                if key.startswith("ema_model.")
            }
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    "IndicF5 checkpoint mismatch: "
                    f"{len(missing)} missing and {len(unexpected)} unexpected keys."
                )
            model.eval()
            self._bundle = (torch, model, vocoder, device)
            return self._bundle

    def generate(
        self,
        reference_path: str | Path,
        reference_text: str,
        target_text: str,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, int]:
        reference_text = normalize_text(reference_text)
        target_text = normalize_text(target_text)
        if not reference_text:
            raise ValueError("Enter the transcript spoken in the reference recording.")
        if not target_text:
            raise ValueError("Enter the text to synthesize.")
        torch, model, vocoder, device = self._load()
        try:
            from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
        except ImportError as error:
            raise RuntimeError("The installed F5-TTS package is missing inference utilities.") from error

        with prepared_reference(reference_path) as (reference_copy, _audio_info):
            prepared_path = None
            try:
                ref_audio, ref_text = preprocess_ref_audio_text(
                    str(reference_copy),
                    reference_text,
                    device=device,
                )
                prepared_path = Path(ref_audio)
                chunks = split_text(target_text)
                if not chunks:
                    raise ValueError("No speakable text remained after normalization.")
                rendered: list[np.ndarray] = []
                sample_rate = None
                with torch.inference_mode():
                    for index, (chunk, pause_seconds) in enumerate(chunks):
                        audio, sample_rate, _ = infer_process(
                            ref_audio,
                            ref_text,
                            chunk,
                            model,
                            vocoder,
                            mel_spec_type="vocos",
                            device=device,
                        )
                        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                        if audio.size == 0 or not np.isfinite(audio).all():
                            raise RuntimeError(f"Synthesis returned invalid audio for chunk {index + 1}.")
                        rendered.append(audio)
                        if pause_seconds:
                            pause = np.zeros(int(sample_rate * pause_seconds), dtype=np.float32)
                            rendered.append(pause)
                        if progress:
                            progress(index + 1, len(chunks))
                if sample_rate is None:
                    raise RuntimeError("Synthesis did not return a sample rate.")
                return np.concatenate(rendered), int(sample_rate)
            finally:
                if prepared_path is not None:
                    prepared_path.unlink(missing_ok=True)


def format_device() -> str:
    try:
        import torch
    except ImportError:
        return "unavailable"
    return _select_device(torch)
