"""Optional, experimental Telugu/English Chatterbox engine.

This engine is disabled by default.  It exists because the validated IndicF5
checkpoint is Telugu-only on paper, while the community Chatterbox-Telugu
fine-tune claims Telugu + English + code-switching with reference-audio cloning.
Those quality claims have not been independently certified in this repository.
Do not route production traffic here until eval/benchmark.py and listening tests
pass for your reference voices.

License: the fine-tune is CC-BY-4.0 (attribution required); upstream Chatterbox
components are MIT.  The model card is the authoritative source.
"""

from __future__ import annotations

import inspect
import logging
import threading
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from tts.audio_utils import (
    AudioValidationError,
    prepare_reference_cached,
    resample_audio,
)
from tts.chatterbox_compat import (
    EXPECTED_T3_MODEL,
    LICENSE,
    MODEL_ID,
    CompatibilityError,
    compatibility_scope,
    loader_model_dir,
    preflight,
)
from tts.chunking import split_text
from tts.config import ChatterboxOptions
from tts.languages import LANG_ENGLISH, LANG_MIXED, LANG_ROMANIZED, LANG_TELUGU, canonical_language
from tts.text_normalizer import normalize_text


LOGGER = logging.getLogger("local-voice-studio.chatterbox")


def _select_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _filter_kwargs(function, kwargs: dict) -> dict:
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def _to_numpy(audio) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    array = np.asarray(audio, dtype=np.float32).squeeze()
    if array.ndim != 1:
        raise RuntimeError("Chatterbox returned non-mono audio.")
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("Chatterbox returned empty or invalid audio.")
    return np.ascontiguousarray(array, dtype=np.float32)


def _language_id(language: str) -> str:
    language = canonical_language(language)
    if language == LANG_ENGLISH:
        return "en"
    # Telugu and the fine-tune's Telugu-English mode both use the Telugu token.
    return "te"


class ChatterboxEngine:
    engine_name = "chatterbox"
    engine_version = "chatterbox-telugu-poc-r1"
    experimental = True

    def __init__(
        self,
        *,
        model_dir: str | Path,
        device: str = "auto",
        options: ChatterboxOptions | None = None,
        reference_cache_dir: str | Path | None = None,
        max_reference_seconds: float = 12.0,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device_request = device
        self.options = options or ChatterboxOptions()
        self.reference_cache_dir = (
            Path(reference_cache_dir)
            if reference_cache_dir is not None
            else self.model_dir.parent / ".reference-cache"
        )
        self.max_reference_seconds = float(max_reference_seconds)
        self._lock = threading.RLock()
        self._generate_lock = threading.Lock()
        self._bundle = None
        self._compat_stack: ExitStack | None = None
        self._load_seconds: float | None = None

    @property
    def loaded(self) -> bool:
        return self._bundle is not None

    def preflight(self) -> dict:
        return preflight(self.model_dir)

    def download(self, token: str | None = None) -> Path:
        from huggingface_hub import snapshot_download

        self.model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(MODEL_ID, local_dir=self.model_dir, token=token)
        return self.model_dir

    def _load(self):
        with self._lock:
            if self._bundle is not None:
                return self._bundle
            import time

            started = time.perf_counter()
            try:
                import torch
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            except ImportError as error:
                raise RuntimeError(
                    "The optional Chatterbox engine is enabled but chatterbox-tts is not "
                    "installed in this environment. Install experiments/chatterbox_telugu/"
                    "requirements.txt or disable TTS_ENABLE_CHATTERBOX."
                ) from error
            try:
                metadata = preflight(self.model_dir)
            except CompatibilityError as error:
                raise RuntimeError(
                    f"Chatterbox-Telugu is not ready: {error}. Download it with "
                    "`python -m tts.setup_models --chatterbox`."
                ) from error
            device = _select_device(torch) if self.device_request == "auto" else self.device_request
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("Chatterbox CUDA was requested but CUDA is unavailable.")
            if device == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("Chatterbox MPS was requested but MPS is unavailable.")
            stack = ExitStack()
            try:
                stack.enter_context(compatibility_scope())
                LOGGER.info("Loading Chatterbox-Telugu from %s on %s", self.model_dir, device)
                parameters = inspect.signature(
                    ChatterboxMultilingualTTS.from_local
                ).parameters
                kwargs = {"device": device}
                if "t3_model" in parameters:
                    kwargs["t3_model"] = EXPECTED_T3_MODEL
                with loader_model_dir(self.model_dir) as loader_dir:
                    model = ChatterboxMultilingualTTS.from_local(loader_dir, **kwargs)
            except Exception:
                stack.close()
                raise
            self._compat_stack = stack
            self._bundle = (torch, model, device, metadata)
            self._load_seconds = time.perf_counter() - started
            LOGGER.info("Chatterbox-Telugu loaded in %.1fs", self._load_seconds)
            return self._bundle

    def load(self):
        return self._load()

    def warmup(self) -> dict:
        self._load()
        return self.info()

    def info(self) -> dict:
        bundle = self._bundle
        return {
            "engine": self.engine_name,
            "experimental": True,
            "loaded": bundle is not None,
            "device": bundle[2] if bundle else self.device_request,
            "model": MODEL_ID,
            "license": LICENSE,
            "load_seconds": self._load_seconds,
            "uses_reference_transcript": False,
            "note": "Community fine-tune; quality and English/Tenglish parity are not certified.",
        }

    def _reference(self, reference_path: str | Path) -> Path:
        path, info = prepare_reference_cached(
            reference_path,
            self.reference_cache_dir,
            min_seconds=1.0,
            max_seconds=self.max_reference_seconds,
        )
        if info.duration_seconds > self.max_reference_seconds:
            raise AudioValidationError(
                f"Reference is {info.duration_seconds:.1f}s; the maximum is "
                f"{self.max_reference_seconds:.0f}s."
            )
        return path

    def _chunks(self, text: str) -> list[tuple[str, float]]:
        chunks = split_text(text, max_bytes=420)
        if not chunks:
            raise ValueError("No speakable text remained after normalization.")
        return chunks

    def _generate_chunk(
        self,
        torch,
        model,
        device: str,
        reference_audio: Path,
        text: str,
        language: str,
        options: ChatterboxOptions,
    ) -> tuple[np.ndarray, int]:
        kwargs = _filter_kwargs(
            model.generate,
            {
                "language_id": _language_id(language),
                "audio_prompt_path": str(reference_audio),
                "exaggeration": options.exaggeration,
                "cfg_weight": options.cfg_weight,
                "temperature": options.temperature,
                "repetition_penalty": options.repetition_penalty,
                "min_p": options.min_p,
                "top_p": options.top_p,
            },
        )
        with torch.inference_mode():
            audio = model.generate(text, **kwargs)
        array = _to_numpy(audio)
        sample_rate = int(getattr(model, "sr", 24_000))
        if sample_rate != 24_000:
            array = resample_audio(array, sample_rate, 24_000)
            sample_rate = 24_000
        return array, sample_rate

    def generate(
        self,
        reference_path: str | Path,
        reference_text: str,
        target_text: str,
        language: str = LANG_TELUGU,
        progress: Callable[[int, int], None] | None = None,
        *,
        options: ChatterboxOptions | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[np.ndarray, int]:
        target_text = normalize_text(target_text)
        if not target_text:
            raise ValueError("Enter the text to synthesize.")
        options = options or self.options
        torch, model, device, _metadata = self._load()
        with self._generate_lock:
            reference_audio = self._reference(reference_path)
            chunks = self._chunks(target_text)
            rendered: list[np.ndarray] = []
            sample_rate = 24_000
            for index, (chunk, pause_seconds) in enumerate(chunks):
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Generation cancelled.")
                audio, sample_rate = self._generate_chunk(
                    torch, model, device, reference_audio, chunk, language, options
                )
                rendered.append(audio)
                if pause_seconds:
                    rendered.append(np.zeros(int(sample_rate * pause_seconds), dtype=np.float32))
                if progress:
                    progress(index + 1, len(chunks))
            return np.concatenate(rendered).astype(np.float32, copy=False), sample_rate

    def generate_stream(
        self,
        reference_path: str | Path,
        reference_text: str,
        target_text: str,
        language: str = LANG_TELUGU,
        progress: Callable[[int, int], None] | None = None,
        *,
        options: ChatterboxOptions | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[tuple[np.ndarray, int, int, int]]:
        target_text = normalize_text(target_text)
        if not target_text:
            raise ValueError("Enter the text to synthesize.")
        options = options or self.options
        torch, model, device, _metadata = self._load()
        self._generate_lock.acquire()
        try:
            reference_audio = self._reference(reference_path)
            chunks = self._chunks(target_text)
            for index, (chunk, pause_seconds) in enumerate(chunks):
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Generation cancelled.")
                audio, sample_rate = self._generate_chunk(
                    torch, model, device, reference_audio, chunk, language, options
                )
                if pause_seconds:
                    audio = np.concatenate(
                        (
                            audio,
                            np.zeros(int(sample_rate * pause_seconds), dtype=np.float32),
                        )
                    )
                if progress:
                    progress(index + 1, len(chunks))
                yield audio, sample_rate, index + 1, len(chunks)
        finally:
            self._generate_lock.release()
