"""FastAPI request/response models.

The API is intentionally explicit about experimental capabilities.  Unknown
fields are rejected so a typo cannot silently change a paid request.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


EngineName = Literal["auto", "indicf5", "chatterbox"]
ProfileName = Literal["quality", "balanced", "realtime", "custom"]
AudioFormat = Literal["wav", "mp3", "opus", "pcm"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class TTSRequest(StrictModel):
    text: str = Field(..., min_length=1, max_length=100_000)
    language: str = Field("Telugu", examples=["Telugu", "English", "Telugu + English", "Romanized Telugu"])
    voice_id: str = Field(..., min_length=3, max_length=64)
    # Omitted engine uses the deployment default (TTS_ENGINE), which is
    # normally `auto`.  Explicit `auto` still means language-based routing.
    engine: EngineName | None = None
    profile: ProfileName | None = None
    speed: float | None = Field(None, ge=0.5, le=2.0)
    use_cache: bool = True
    format: AudioFormat = "wav"


class TTSStreamRequest(StrictModel):
    text: str = Field(..., min_length=1, max_length=100_000)
    language: str = Field("Telugu")
    voice_id: str = Field(..., min_length=3, max_length=64)
    engine: EngineName | None = None
    profile: ProfileName | None = None
    speed: float | None = Field(None, ge=0.5, le=2.0)
    use_cache: bool = True


class TransliterateRequest(StrictModel):
    text: str = Field(..., min_length=1, max_length=20_000)


class TransliterateResponse(StrictModel):
    telugu: str
    warning: str = (
        "IndicXlit output is ambiguous; review and edit the Telugu script before synthesis."
    )


class PreviewRequest(StrictModel):
    text: str = Field(..., min_length=1, max_length=100_000)
    language: str = Field("Telugu")


class PreviewChunk(StrictModel):
    text: str
    pause_seconds: float


class PreviewResponse(StrictModel):
    normalized_text: str
    language: str
    warnings: list[str]
    chunks: list[PreviewChunk]


class VoiceResponse(StrictModel):
    voice_id: str
    name: str
    language: str
    duration_seconds: float
    sample_rate: int
    source_sha256: str
    consent: bool
    created_at: str
    warnings: list[str] = Field(default_factory=list)
    engine_hint: str = "indicf5"
    transcript: str | None = None


class VoiceListResponse(StrictModel):
    voices: list[VoiceResponse]


class UsageResponse(StrictModel):
    period: str
    chars: int
    audio_ms: int
    requests: int
    monthly_char_quota: int
    chars_remaining: int | None


class MeResponse(StrictModel):
    key_id: str
    name: str
    scopes: list[str]
    usage: UsageResponse


class ErrorBody(StrictModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(StrictModel):
    error: ErrorBody


class CapabilitiesResponse(StrictModel):
    service_version: str
    profile: str
    default_engine: str = "auto"
    routing: str = ""
    languages: list[dict]
    engines: dict
    limits: dict
