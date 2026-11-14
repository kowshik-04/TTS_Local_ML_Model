"""Hardware-independent TTS service used by the HTTP API, agent and UI.

This layer owns text preparation, voice resolution, exact-output caching,
post-processing and stream planning.  It deliberately has no FastAPI/Gradio
imports so it can be tested without a web server or model weights.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from tts.audio_utils import AudioValidationError, prepare_reference_cached
from tts.cache import CACHE_VERSION, AudioCache, make_cache_key
from tts.config import PROFILES, InferenceOptions, Settings
from tts.engine_router import EngineRouter
from tts.languages import LANG_ENGLISH, LANG_MIXED, LANG_ROMANIZED, LANG_TELUGU, canonical_language
from tts.postprocess import (
    PostProcessConfig,
    postprocess_speech,
    postprocess_stream_chunk,
)
from tts.text_frontend import FrontendResult, prepare_speech_text
from tts.voice_store import VoiceError, VoiceProfile, VoiceStore


LOGGER = logging.getLogger("local-voice-studio.service")
SERVICE_VERSION = "tts-service-r1"


class TTSError(ValueError):
    """User-facing synthesis error with a stable code for the API layer."""

    def __init__(self, message: str, code: str = "invalid_request") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class VoiceHandle:
    voice_id: str
    name: str
    path: Path
    transcript: str
    fingerprint: str
    language: str = LANG_TELUGU
    warnings: tuple[str, ...] = ()
    experimental: bool = False


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    language: str = LANG_TELUGU
    voice_id: str | None = None
    # None means "use the deployment default" (TTS_ENGINE).  An explicit
    # "auto" still means language-based routing.
    engine: str | None = None
    profile: str | None = None
    speed: float | None = None
    use_cache: bool = True
    # Optional per-request reference instead of a stored voice profile.
    reference_path: str | None = None
    reference_transcript: str | None = None
    reference_name: str = "Custom reference"
    owner_key_id: str | None = None


@dataclass(frozen=True)
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    normalized_text: str
    engine: str
    voice_id: str
    cache_hit: bool
    generation_seconds: float
    audio_seconds: float
    warnings: tuple[str, ...] = ()
    experimental: bool = False
    options: InferenceOptions | None = None


@dataclass
class StreamPlan:
    request: SpeechRequest
    normalized: FrontendResult
    engine: str
    voice: VoiceHandle
    options: InferenceOptions | None
    cache_key: str | None
    cached_audio: np.ndarray | None = None
    cached_sample_rate: int = 24_000
    warnings: tuple[str, ...] = field(default_factory=tuple)
    experimental: bool = False


@dataclass(frozen=True)
class StreamChunk:
    audio: np.ndarray
    sample_rate: int
    index: int
    total: int
    is_last: bool


def _voice_fingerprint(path: Path, transcript: str) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as error:
        raise TTSError(f"Reference audio is unavailable: {error}", "reference_unavailable") from error
    digest.update(b"\x00")
    digest.update(transcript.strip().encode("utf-8"))
    return digest.hexdigest()


def _options_for(settings: Settings, profile: str | None, speed: float | None) -> InferenceOptions:
    if profile is None:
        options = settings.options
    else:
        profile = profile.lower()
        if profile == "custom":
            options = settings.options
        elif profile in PROFILES:
            options = PROFILES[profile]
        else:
            raise TTSError(
                f"Unknown profile {profile!r}; use quality, balanced, realtime or custom.",
                "invalid_profile",
            )
    if speed is not None:
        if not 0.5 <= float(speed) <= 2.0:
            raise TTSError("speed must be between 0.5 and 2.0.", "invalid_speed")
        options = replace(options, speed=float(speed))
    options.validate()
    return options


class TTSService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        router: EngineRouter | None = None,
        voices: VoiceStore | None = None,
        cache: AudioCache | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.settings.validate()
        self.router = router or EngineRouter(settings=self.settings)
        self.voices = voices or VoiceStore(
            self.settings.voices_dir,
            min_seconds=self.settings.min_voice_seconds,
            max_seconds=self.settings.max_aligned_reference_seconds,
            reference_cache_dir=self.settings.reference_cache_dir,
        )
        self.cache = cache or (
            AudioCache(self.settings.cache_dir / "audio", self.settings.cache_max_bytes)
            if self.settings.cache_enabled
            else None
        )
        self.postprocess = PostProcessConfig(
            target_rms_dbfs=self.settings.target_rms_dbfs,
            peak_dbfs=self.settings.peak_dbfs,
            highpass_hz=self.settings.highpass_hz,
            fade_ms=self.settings.fade_ms,
            tail_ms=self.settings.tail_ms,
        )
        self._queue_lock = threading.BoundedSemaphore(max(1, self.settings.api_concurrency))

    # ---------- preparation ----------
    def _normalize(self, text: str, language: str) -> FrontendResult:
        if not text or not text.strip():
            raise TTSError("Enter text to synthesize.", "empty_text")
        if len(text) > self.settings.max_text_chars:
            raise TTSError(
                f"Text is too long ({len(text)} characters); the limit is "
                f"{self.settings.max_text_chars}.",
                "text_too_long",
            )
        try:
            language = canonical_language(language, self.settings.default_language)
        except ValueError as error:
            raise TTSError(str(error), "invalid_language") from error
        if language == LANG_ROMANIZED:
            try:
                from tts.transliteration import romanized_telugu_to_script

                transliterated = romanized_telugu_to_script(text)
            except (ImportError, RuntimeError, ValueError) as error:
                raise TTSError(
                    f"Romanized Telugu conversion is unavailable: {error}",
                    "transliteration_unavailable",
                ) from error
            result = prepare_speech_text(transliterated, LANG_TELUGU)
            return FrontendResult(
                result.text,
                LANG_ROMANIZED,
                result.warnings + ("Romanized input was converted from IndicXlit output.",),
                result.replaced_markup,
            )
        try:
            return prepare_speech_text(text, language)
        except ValueError as error:
            raise TTSError(str(error), "text_normalization_failed") from error

    def _resolve_voice(self, request: SpeechRequest) -> VoiceHandle:
        if request.reference_path:
            transcript = (request.reference_transcript or "").strip()
            if not transcript:
                raise TTSError(
                    "reference_transcript is required with a custom reference audio file.",
                    "reference_transcript_required",
                )
            path = Path(request.reference_path).expanduser().resolve()
            try:
                prepared_path, info = prepare_reference_cached(
                    path,
                    self.settings.reference_cache_dir,
                    min_seconds=1.0,
                    max_seconds=self.settings.max_aligned_reference_seconds,
                )
            except AudioValidationError as error:
                raise TTSError(str(error), "invalid_reference") from error
            return VoiceHandle(
                voice_id="custom",
                name=request.reference_name or "Custom reference",
                path=prepared_path,
                transcript=transcript,
                fingerprint=_voice_fingerprint(prepared_path, transcript),
                warnings=tuple(info.warnings) + (
                    "Per-request references are not stored; use /v1/voices for production.",
                ),
            )
        if not request.voice_id:
            raise TTSError(
                "Provide voice_id or reference audio with reference_transcript.",
                "voice_required",
            )
        try:
            profile = self.voices.get(request.voice_id, owner_key_id=request.owner_key_id)
        except VoiceError as error:
            raise TTSError(str(error), "voice_not_found") from error
        return self._from_profile(profile)

    def _from_profile(self, profile: VoiceProfile) -> VoiceHandle:
        return VoiceHandle(
            voice_id=profile.voice_id,
            name=profile.name,
            path=profile.wav_path,
            transcript=profile.transcript,
            fingerprint=profile.fingerprint,
            language=profile.language,
            warnings=profile.warnings,
            experimental=profile.engine_hint == "chatterbox",
        )

    def _prepare(self, request: SpeechRequest) -> tuple[FrontendResult, str, VoiceHandle, InferenceOptions | None]:
        try:
            language = canonical_language(request.language, self.settings.default_language)
        except ValueError as error:
            raise TTSError(str(error), "invalid_language") from error
        normalized = self._normalize(request.text, language)
        voice = self._resolve_voice(request)
        # Romanized input is spoken as Telugu after transliteration.
        route_language = LANG_TELUGU if language == LANG_ROMANIZED else normalized.language
        requested_engine = request.engine or self.settings.engine
        try:
            engine = self.router.resolve_engine(route_language, requested_engine)
        except ValueError as error:
            raise TTSError(str(error), "engine_unavailable") from error
        LOGGER.info(
            "TTS route: requested=%s resolved=%s language=%s voice=%s",
            requested_engine,
            engine,
            route_language,
            voice.voice_id,
        )
        options = _options_for(self.settings, request.profile, request.speed) if engine == "indicf5" else None
        warnings = list(normalized.warnings) + list(voice.warnings)
        experimental = engine == "chatterbox" or route_language == LANG_MIXED
        if engine == "chatterbox":
            warnings.append(
                "Chatterbox-Telugu is an optional community fine-tune; quality, English "
                "pronunciation and cross-language speaker similarity require listening review."
            )
        return normalized, engine, voice, options

    def _cache_key(self, normalized: str, engine: str, voice: VoiceHandle, options: InferenceOptions | None) -> str:
        postprocess = self.postprocess
        postprocess_version = ":".join(
            (
                CACHE_VERSION,
                f"rms={postprocess.target_rms_dbfs}",
                f"peak={postprocess.peak_dbfs}",
                f"hp={postprocess.highpass_hz}",
                f"fade={postprocess.fade_ms}",
                f"tail={postprocess.tail_ms}",
                f"trim={int(postprocess.trim_edges)}",
            )
        )
        return make_cache_key(
            engine=engine,
            voice_fingerprint=voice.fingerprint,
            text=normalized,
            options=(options.__dict__ if options else {"chatterbox": self.settings.chatterbox_options.__dict__}),
            sample_rate=24_000,
            postprocess_version=postprocess_version,
        )

    def _ensure_engine_ready(self, engine: str) -> None:
        try:
            self.router.ensure_ready(engine)
        except Exception as error:
            raise TTSError(
                f"The {engine} engine could not be initialized: {error}",
                "engine_initialization_failed",
            ) from error

    def _generation_error(self, engine: str, error: Exception) -> TTSError:
        if "cancel" in str(error).lower():
            return TTSError("Generation cancelled.", "cancelled")
        LOGGER.exception("%s generation failed", engine)
        return TTSError(f"The {engine} engine failed during generation: {error}", "generation_failed")

    # ---------- non-streaming ----------
    def synthesize(
        self,
        request: SpeechRequest,
        *,
        progress: Callable[[int, int], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> SynthesisResult:
        started = time.perf_counter()
        normalized, engine, voice, options = self._prepare(request)
        cache_key = self._cache_key(normalized.text, engine, voice, options)
        if self.cache is not None and request.use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                audio, entry = cached
                return SynthesisResult(
                    audio=audio,
                    sample_rate=entry.sample_rate,
                    normalized_text=normalized.text,
                    engine=engine,
                    voice_id=voice.voice_id,
                    cache_hit=True,
                    generation_seconds=time.perf_counter() - started,
                    audio_seconds=entry.duration_seconds,
                    warnings=tuple(normalized.warnings) + tuple(voice.warnings),
                    experimental=engine == "chatterbox" or normalized.language == LANG_MIXED,
                    options=options,
                )
        # Load the model before acquiring the queue so missing weights or a
        # broken installation fail as a normal request error, not as a later
        # generation failure.
        self._ensure_engine_ready(engine)
        acquired = self._queue_lock.acquire(timeout=self.settings.api_queue_timeout_seconds)
        if not acquired:
            raise TTSError(
                "The synthesis queue is full. Retry shortly or increase TTS_API_CONCURRENCY.",
                "queue_timeout",
            )
        try:
            raw, sample_rate = self.router.generate(
                normalized.language if normalized.language != LANG_ROMANIZED else LANG_TELUGU,
                str(voice.path),
                voice.transcript,
                normalized.text,
                progress=progress,
                engine=engine,
                options=options,
                cancel_event=cancel_event,
            )
            audio = postprocess_speech(raw, sample_rate, self.postprocess)
        except TTSError:
            raise
        except Exception as error:
            raise self._generation_error(engine, error) from error
        finally:
            self._queue_lock.release()
        audio_seconds = len(audio) / sample_rate
        if self.cache is not None and request.use_cache:
            try:
                self.cache.put(
                    cache_key,
                    audio,
                    sample_rate,
                    engine=engine,
                    voice_fingerprint=voice.fingerprint,
                    text_chars=len(normalized.text),
                )
            except Exception:
                LOGGER.exception("Could not write the audio cache; continuing without it")
        warnings = tuple(normalized.warnings) + tuple(voice.warnings)
        if engine == "chatterbox":
            warnings += (
                "Chatterbox-Telugu is experimental; listen to every voice before production use.",
            )
        return SynthesisResult(
            audio=audio,
            sample_rate=sample_rate,
            normalized_text=normalized.text,
            engine=engine,
            voice_id=voice.voice_id,
            cache_hit=False,
            generation_seconds=time.perf_counter() - started,
            audio_seconds=audio_seconds,
            warnings=warnings,
            experimental=engine == "chatterbox" or normalized.language == LANG_MIXED,
            options=options,
        )

    # ---------- streaming ----------
    def plan_stream(self, request: SpeechRequest) -> StreamPlan:
        normalized, engine, voice, options = self._prepare(request)
        cache_key = self._cache_key(normalized.text, engine, voice, options)
        plan = StreamPlan(
            request=request,
            normalized=normalized,
            engine=engine,
            voice=voice,
            options=options,
            cache_key=cache_key,
            warnings=tuple(normalized.warnings) + tuple(voice.warnings),
            experimental=engine == "chatterbox" or normalized.language == LANG_MIXED,
        )
        if self.cache is not None and request.use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                plan.cached_audio, entry = cached
                plan.cached_sample_rate = entry.sample_rate
                plan.warnings += ("Cache hit: exact audio reused.",)
        if plan.cached_audio is None:
            # This runs before the HTTP response headers are sent.  A gated
            # checkpoint or missing dependency therefore becomes a 503 instead
            # of an empty 200 stream that the agent would silently play.
            self._ensure_engine_ready(engine)
        return plan

    def stream(
        self,
        plan: StreamPlan,
        *,
        progress: Callable[[int, int], None] | None = None,
        cancel_event: threading.Event | None = None,
        collect_for_cache: bool = True,
    ) -> Iterator[StreamChunk]:
        if plan.cached_audio is not None:
            yield StreamChunk(plan.cached_audio, plan.cached_sample_rate, 1, 1, True)
            return
        acquired = self._queue_lock.acquire(timeout=self.settings.api_queue_timeout_seconds)
        if not acquired:
            raise TTSError(
                "The synthesis queue is full. Retry shortly or increase TTS_API_CONCURRENCY.",
                "queue_timeout",
            )
        collected: list[np.ndarray] = []
        sample_rate = 24_000
        try:
            chunks = self.router.generate_stream(
                plan.normalized.language if plan.normalized.language != LANG_ROMANIZED else LANG_TELUGU,
                str(plan.voice.path),
                plan.voice.transcript,
                plan.normalized.text,
                progress=progress,
                engine=plan.engine,
                options=plan.options,
                cancel_event=cancel_event,
            )
            total = 0
            index = 0
            for raw, sample_rate, index, total in chunks:
                if cancel_event is not None and cancel_event.is_set():
                    raise TTSError("Generation cancelled.", "cancelled")
                audio = postprocess_stream_chunk(raw, sample_rate, self.postprocess)
                if collect_for_cache:
                    collected.append(audio)
                yield StreamChunk(audio, sample_rate, index, total, index >= total)
        except TTSError:
            raise
        except Exception as error:
            raise self._generation_error(plan.engine, error) from error
        finally:
            self._queue_lock.release()
        if (
            collect_for_cache
            and collected
            and self.cache is not None
            and plan.cache_key is not None
            and (cancel_event is None or not cancel_event.is_set())
        ):
            try:
                audio = np.concatenate(collected)
                self.cache.put(
                    plan.cache_key,
                    audio,
                    sample_rate,
                    engine=plan.engine,
                    voice_fingerprint=plan.voice.fingerprint,
                    text_chars=len(plan.normalized.text),
                )
            except Exception:
                LOGGER.exception("Could not write the streaming audio cache; continuing")

    # ---------- text helpers ----------
    def transliterate(self, text: str) -> str:
        try:
            from tts.transliteration import romanized_telugu_to_script

            return romanized_telugu_to_script(text)
        except (ImportError, RuntimeError, ValueError) as error:
            raise TTSError(f"Romanized Telugu conversion is unavailable: {error}", "transliteration_unavailable") from error

    def preview(self, text: str, language: str = LANG_TELUGU) -> dict:
        normalized = self._normalize(text, language)
        from tts.chunking import split_text

        chunks = split_text(normalized.text, max_bytes=self.settings.max_chunk_bytes)
        return {
            "normalized_text": normalized.text,
            "language": normalized.language,
            "warnings": list(normalized.warnings),
            "chunks": [{"text": chunk, "pause_seconds": pause} for chunk, pause in chunks],
        }

    def capabilities(self) -> dict:
        return {
            "service_version": SERVICE_VERSION,
            "profile": self.settings.profile,
            "default_engine": self.settings.engine,
            "routing": (
                "Requests that omit engine use TTS_ENGINE. engine=auto routes "
                "Telugu and Telugu-English to IndicF5, and English to Chatterbox "
                "only when TTS_ENABLE_CHATTERBOX=1. Failures never trigger a "
                "silent engine switch."
            ),
            "languages": self.router.capabilities(),
            "engines": self.router.info(),
            "limits": {
                "max_text_chars": self.settings.max_text_chars,
                "max_reference_seconds": self.settings.max_aligned_reference_seconds,
            },
        }

    def health(self) -> dict:
        return {
            "service": "ok",
            "engine": self.router.info(),
            "voices": len(self.voices.list()),
            "cache": self.cache.stats() if self.cache is not None else {"enabled": False},
        }
