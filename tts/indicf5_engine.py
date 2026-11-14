from __future__ import annotations

import inspect
import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from tts.audio_utils import (
    AudioValidationError,
    prepared_reference,
    prepare_reference_cached,
    resample_audio,
)
from tts.chunking import split_for_indicf5
from tts.config import InferenceOptions, PROFILES
from tts.text_normalizer import normalize_text


REPO_ID = "ai4bharat/IndicF5"
VOCODER_REPO_ID = "charactr/vocos-mel-24khz"
LOGGER = logging.getLogger("local-voice-studio.indicf5")


def _select_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_device(torch, requested: str) -> str:
    requested = (requested or "auto").lower()
    if requested == "auto":
        return _select_device(torch)
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"TTS_DEVICE={requested} requested, but CUDA is unavailable.")
        return requested
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("TTS_DEVICE=mps requested, but MPS is unavailable.")
        return "mps"
    if requested == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported device: {requested}")


def _filter_kwargs(function, kwargs: dict) -> dict:
    """Pass only arguments supported by the installed F5 version."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return kwargs
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


class IndicF5Engine:
    """Lazy-loaded wrapper retaining the project's validated direct DiT loader.

    The model is loaded once and all generation is serialized.  The wrapper is
    hardware-agnostic: Apple MPS, CUDA and CPU are selected at runtime.
    """

    engine_name = "indicf5"
    engine_version = "indicf5-r2"

    def __init__(
        self,
        *,
        device: str = "auto",
        options: InferenceOptions | None = None,
        reference_cache_dir: str | Path | None = None,
        max_reference_seconds: float = 12.0,
        max_chunk_bytes: int = 420,
        inference_threads: int = 0,
    ) -> None:
        self._lock = threading.RLock()
        self._generate_lock = threading.Lock()
        self._bundle = None
        self.device_request = device
        self.options = options or PROFILES["quality"]
        self.reference_cache_dir = Path(reference_cache_dir) if reference_cache_dir else None
        self.max_reference_seconds = float(max_reference_seconds)
        self.max_chunk_bytes = int(max_chunk_bytes)
        self.inference_threads = int(inference_threads)
        self._load_seconds: float | None = None

    @property
    def loaded(self) -> bool:
        return self._bundle is not None

    def _load_model(self, load_model, DiT, checkpoint_path: Path, vocab_path: str, device: str):
        config = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
        )
        parameters = inspect.signature(load_model).parameters
        requires_checkpoint = (
            "ckpt_path" in parameters
            and parameters["ckpt_path"].default is inspect.Parameter.empty
        )
        if requires_checkpoint:
            # Newer F5 releases load the checkpoint inside load_model.  This branch
            # exists only for a fresh environment; the validated baseline uses the
            # manual EMA path below.
            return load_model(
                DiT,
                config,
                ckpt_path=str(checkpoint_path),
                mel_spec_type="vocos",
                vocab_file=str(vocab_path),
                device=device,
            )
        return load_model(
            DiT,
            config,
            mel_spec_type="vocos",
            vocab_file=str(vocab_path),
            device=device,
        )

    def _load(self):
        with self._lock:
            if self._bundle is not None:
                return self._bundle
            import time

            started = time.perf_counter()
            try:
                import torch
                from huggingface_hub import hf_hub_download
                from safetensors.torch import load_file
                from f5_tts.model import DiT
                from f5_tts.infer.utils_infer import load_model, load_vocoder
            except ImportError as error:
                raise RuntimeError(
                    "IndicF5 dependencies are incomplete. Install requirements.txt in the inference "
                    "environment (torch, torchaudio, transformers, f5-tts, vocos)."
                ) from error

            if self.inference_threads > 0:
                torch.set_num_threads(self.inference_threads)

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
                    "Required IndicF5/Vocos model files are not cached locally. Accept the "
                    "gated model terms on Hugging Face, run `python -m tts.setup_models` while "
                    "online, then use the app offline."
                ) from error

            device = _resolve_device(torch, self.device_request)
            LOGGER.info("Loading IndicF5 on %s", device)
            vocoder = load_vocoder(
                vocoder_name="vocos",
                is_local=True,
                local_path=str(Path(vocoder_config).parent),
                device=device,
            )
            model = self._load_model(load_model, DiT, Path(checkpoint_path), vocab_path, device)
            parameters = inspect.signature(load_model).parameters
            requires_checkpoint = (
                "ckpt_path" in parameters
                and parameters["ckpt_path"].default is inspect.Parameter.empty
            )
            if not requires_checkpoint:
                state_dict = load_file(checkpoint_path, device=device)
                state_dict = {
                    key.replace("ema_model._orig_mod.", ""): value
                    for key, value in state_dict.items()
                    if key.startswith("ema_model.")
                }
                if not state_dict:
                    raise RuntimeError(
                        "IndicF5 checkpoint does not contain the expected EMA weights."
                    )
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    raise RuntimeError(
                        "IndicF5 checkpoint mismatch: "
                        f"{len(missing)} missing and {len(unexpected)} unexpected keys."
                    )
            model.eval()
            self._bundle = (torch, model, vocoder, device)
            self._load_seconds = time.perf_counter() - started
            LOGGER.info("IndicF5 loaded in %.1fs on %s", self._load_seconds, device)
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
            "loaded": bundle is not None,
            "device": bundle[3] if bundle else self.device_request,
            "profile": next(
                (name for name, value in PROFILES.items() if value == self.options),
                "custom",
            ),
            "load_seconds": self._load_seconds,
            "reference_cache": str(self.reference_cache_dir) if self.reference_cache_dir else None,
        }

    @contextmanager
    def _reference(self, reference_path: str | Path):
        if self.reference_cache_dir is not None:
            path, info = prepare_reference_cached(
                reference_path,
                self.reference_cache_dir,
                min_seconds=1.0,
                max_seconds=self.max_reference_seconds,
            )
            yield path, info
            return
        with prepared_reference(reference_path) as (path, info):
            if info.duration_seconds > self.max_reference_seconds:
                raise AudioValidationError(
                    f"Reference is {info.duration_seconds:.1f}s; the aligned maximum is "
                    f"{self.max_reference_seconds:.0f}s."
                )
            yield path, info

    def _prepare_reference(self, torch, device: str, reference_path: Path, reference_text: str):
        try:
            from f5_tts.infer.utils_infer import preprocess_ref_audio_text
        except ImportError as error:
            raise RuntimeError("The installed F5-TTS package is missing inference utilities.") from error
        kwargs = _filter_kwargs(
            preprocess_ref_audio_text,
            {"show_info": lambda *args, **kwargs: None, "device": device},
        )
        result = preprocess_ref_audio_text(str(reference_path), reference_text, **kwargs)
        if not isinstance(result, tuple) or len(result) < 2:
            raise RuntimeError("F5 reference preprocessing returned an unexpected result.")
        ref_audio, ref_text = result[0], result[1]
        if isinstance(ref_text, str):
            ref_text = normalize_text(ref_text)
        return str(ref_audio), ref_text

    def _infer_chunk(
        self,
        torch,
        model,
        vocoder,
        device: str,
        ref_audio: str,
        ref_text: str,
        chunk: str,
        options: InferenceOptions,
    ) -> tuple[np.ndarray, int]:
        try:
            from f5_tts.infer.utils_infer import infer_process
        except ImportError as error:
            raise RuntimeError("The installed F5-TTS package is missing inference utilities.") from error
        kwargs = _filter_kwargs(
            infer_process,
            {
                "mel_spec_type": "vocos",
                "device": device,
                "nfe_step": options.nfe_step,
                "cfg_strength": options.cfg_strength,
                "sway_sampling_coef": options.sway_sampling_coef,
                "speed": options.speed,
                "target_rms": options.target_rms,
                "cross_fade_duration": options.cross_fade_duration,
            },
        )
        with torch.inference_mode():
            result = infer_process(ref_audio, ref_text, chunk, model, vocoder, **kwargs)
        if not isinstance(result, tuple) or len(result) < 2:
            raise RuntimeError("F5 inference returned an unexpected result.")
        audio, sample_rate = result[0], int(result[1])
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError("Synthesis returned empty or invalid audio.")
        if sample_rate != 24_000:
            audio = resample_audio(audio, sample_rate, 24_000)
            sample_rate = 24_000
        return audio, sample_rate

    def _chunks(self, target_text: str, ref_text: str, ref_seconds: float, options: InferenceOptions):
        chunks = split_for_indicf5(
            target_text,
            ref_text,
            ref_seconds,
            speed=options.speed,
            max_bytes=self.max_chunk_bytes,
        )
        if not chunks:
            raise ValueError("No speakable text remained after normalization.")
        return chunks

    def generate(
        self,
        reference_path: str | Path,
        reference_text: str,
        target_text: str,
        progress: Callable[[int, int], None] | None = None,
        *,
        options: InferenceOptions | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[np.ndarray, int]:
        reference_text = normalize_text(reference_text)
        target_text = normalize_text(target_text)
        if not reference_text:
            raise ValueError("Enter the transcript spoken in the reference recording.")
        if not target_text:
            raise ValueError("Enter the text to synthesize.")
        options = options or self.options
        options.validate()
        torch, model, vocoder, device = self._load()
        with self._generate_lock:
            with self._reference(reference_path) as (reference_copy, reference_info):
                ref_audio, ref_text = self._prepare_reference(
                    torch, device, reference_copy, reference_text
                )
                chunks = self._chunks(target_text, ref_text, reference_info.duration_seconds, options)
                rendered: list[np.ndarray] = []
                sample_rate = 24_000
                for index, (chunk, pause_seconds) in enumerate(chunks):
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError("Generation cancelled.")
                    audio, sample_rate = self._infer_chunk(
                        torch, model, vocoder, device, ref_audio, ref_text, chunk, options
                    )
                    rendered.append(audio)
                    if pause_seconds:
                        rendered.append(
                            np.zeros(int(sample_rate * pause_seconds), dtype=np.float32)
                        )
                    if progress:
                        progress(index + 1, len(chunks))
                return np.concatenate(rendered).astype(np.float32, copy=False), sample_rate

    def generate_stream(
        self,
        reference_path: str | Path,
        reference_text: str,
        target_text: str,
        progress: Callable[[int, int], None] | None = None,
        *,
        options: InferenceOptions | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[tuple[np.ndarray, int, int, int]]:
        """Yield (raw_chunk, sample_rate, index, total) as each sentence completes.

        The caller is responsible for post-processing each raw chunk.  A trailing
        pause is included in the chunk after its speech.
        """
        reference_text = normalize_text(reference_text)
        target_text = normalize_text(target_text)
        if not reference_text:
            raise ValueError("Enter the transcript spoken in the reference recording.")
        if not target_text:
            raise ValueError("Enter the text to synthesize.")
        options = options or self.options
        options.validate()
        torch, model, vocoder, device = self._load()
        self._generate_lock.acquire()
        try:
            with self._reference(reference_path) as (reference_copy, reference_info):
                ref_audio, ref_text = self._prepare_reference(
                    torch, device, reference_copy, reference_text
                )
                chunks = self._chunks(target_text, ref_text, reference_info.duration_seconds, options)
                for index, (chunk, pause_seconds) in enumerate(chunks):
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError("Generation cancelled.")
                    audio, sample_rate = self._infer_chunk(
                        torch, model, vocoder, device, ref_audio, ref_text, chunk, options
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


def format_device() -> str:
    try:
        import torch
    except ImportError:
        return "unavailable"
    try:
        return _select_device(torch)
    except Exception:
        return "unavailable"
