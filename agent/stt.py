"""Pluggable speech-to-text backends for the voice agent.

Local ``faster-whisper`` is the default.  An OpenAI-compatible hosted
transcription endpoint can be selected without changing the session pipeline.
Heavy dependencies are imported lazily so the agent's protocol tests and the
TTS API do not require a Whisper installation.
"""

from __future__ import annotations

import asyncio
import io
import threading
from math import gcd
from typing import Protocol

import httpx
import numpy as np
from scipy.signal import resample_poly


class STTError(RuntimeError):
    pass


class SpeechToText(Protocol):
    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...


def _resample(audio: np.ndarray, source_rate: int, target_rate: int = 16_000) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    common = gcd(source_rate, target_rate)
    result = resample_poly(np.asarray(audio, dtype=np.float32), target_rate // common, source_rate // common)
    return np.asarray(result, dtype=np.float32)


class FasterWhisperSTT:
    """Lazy local faster-whisper backend (CTranslate2, CPU or CUDA)."""

    def __init__(
        self,
        model: str = "large-v3",
        *,
        device: str = "auto",
        compute_type: str = "auto",
        language: str = "te",
    ) -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = None if language in {"", "auto", "Auto"} else language
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel
            except ImportError as error:
                raise STTError(
                    "faster-whisper is not installed. Install agent/requirements.txt or "
                    "set AGENT_STT_PROVIDER=openai."
                ) from error
            compute_type = "default" if self.compute_type in {"", "auto"} else self.compute_type
            try:
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=compute_type,
                )
            except Exception as error:
                raise STTError(
                    f"Could not load faster-whisper model {self.model_name!r}: {error}"
                ) from error
            return self._model

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        model = self._load()
        prepared = _resample(audio, 16_000)
        if prepared.size == 0:
            return ""
        try:
            segments, _ = model.transcribe(
                prepared,
                language=self.language,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
                temperature=0.0,
            )
            return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        except Exception as error:
            raise STTError(f"Local transcription failed: {error}") from error

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        prepared = _resample(audio, sample_rate) if sample_rate != 16_000 else np.asarray(audio, dtype=np.float32)
        return await asyncio.to_thread(self._transcribe_sync, prepared)


class OpenAICompatibleSTT:
    """Hosted OpenAI-compatible ``/audio/transcriptions`` backend."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str,
        model: str = "whisper-1",
        language: str = "te",
    ) -> None:
        if not base_url:
            raise STTError("AGENT_STT_BASE_URL is required for the openai STT provider.")
        if not api_key:
            raise STTError("AGENT_STT_API_KEY is required for the openai STT provider.")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.language = None if language in {"", "auto"} else language

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        import soundfile as sf

        prepared = _resample(audio, sample_rate)
        buffer = io.BytesIO()
        sf.write(buffer, prepared, 16_000, format="WAV", subtype="PCM_16")
        data = {
            "model": self.model,
            "response_format": "json",
            "temperature": "0",
        }
        if self.language:
            data["language"] = self.language
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    data=data,
                    files={"file": ("utterance.wav", buffer.getvalue(), "audio/wav")},
                )
        except httpx.HTTPError as error:
            raise STTError(f"Hosted transcription request failed: {error}") from error
        if response.status_code >= 400:
            detail = response.text.strip()[:500]
            raise STTError(f"Hosted transcription returned HTTP {response.status_code}: {detail}")
        try:
            payload = response.json()
            return str(payload.get("text", "")).strip()
        except ValueError as error:
            raise STTError("Hosted transcription returned invalid JSON.") from error


def build_stt(settings) -> SpeechToText:
    if settings.agent_stt_provider == "openai":
        return OpenAICompatibleSTT(
            settings.agent_stt_base_url,
            api_key=settings.agent_stt_api_key,
            model=settings.agent_stt_model,
            language=settings.agent_stt_language,
        )
    return FasterWhisperSTT(
        settings.agent_stt_model,
        device=settings.agent_stt_device,
        compute_type=settings.agent_stt_compute_type,
        language=settings.agent_stt_language,
    )
