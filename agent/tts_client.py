"""Streaming client for the local TTS API's raw PCM endpoint."""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx


LOGGER = logging.getLogger("local-voice-studio.agent.tts")


class TTSClientError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 0, code: str = "tts_error") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class TTSClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        voice_id: str = "",
        language: str = "Telugu",
        engine: str = "auto",
        timeout_seconds: float = 180.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not base_url:
            raise TTSClientError("AGENT_TTS_URL is required.")
        engine = (engine or "auto").lower()
        if engine not in {"auto", "indicf5", "chatterbox"}:
            raise TTSClientError(
                "AGENT_TTS_ENGINE must be auto, indicf5 or chatterbox.",
                code="invalid_engine",
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.voice_id = voice_id
        self.language = language
        self.engine = engine
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    @property
    def stream_url(self) -> str:
        if self.base_url.endswith("/v1/tts/stream"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/tts/stream"
        return f"{self.base_url}/v1/tts/stream"

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "audio/L16"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def stream_pcm(
        self,
        text: str,
        *,
        voice_id: str | None = None,
        language: str | None = None,
        engine: str | None = None,
    ) -> AsyncIterator[bytes]:
        voice = voice_id or self.voice_id
        if not voice:
            raise TTSClientError("A voice_id is required for agent speech.")
        requested_engine = (engine or self.engine or "auto").lower()
        body = {
            "text": text,
            "voice_id": voice,
            "language": language or self.language,
            "engine": requested_engine,
            "use_cache": True,
        }
        timeout = httpx.Timeout(self.timeout_seconds, connect=15.0)
        try:
            async with httpx.AsyncClient(
                timeout=timeout, transport=self._transport
            ) as client:
                async with client.stream(
                    "POST",
                    self.stream_url,
                    headers=self._headers(),
                    json=body,
                ) as response:
                    if response.status_code >= 400:
                        raw = (await response.aread()).decode("utf-8", errors="replace")
                        code = "tts_http_error"
                        message = raw[:500]
                        try:
                            payload = json.loads(raw)
                            code = str((payload.get("error") or {}).get("code", code))
                            message = str((payload.get("error") or {}).get("message", message))
                        except (ValueError, AttributeError):
                            pass
                        raise TTSClientError(
                            f"TTS returned HTTP {response.status_code}: {message}",
                            status_code=response.status_code,
                            code=code,
                        )
                    resolved_engine = response.headers.get("x-engine", "unknown")
                    LOGGER.info(
                        "TTS stream accepted: requested_engine=%s resolved_engine=%s "
                        "language=%s voice=%s",
                        requested_engine,
                        resolved_engine,
                        body["language"],
                        voice,
                    )
                    received = 0
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            received += len(chunk)
                            yield chunk
                    if received == 0:
                        raise TTSClientError(
                            "TTS stream returned no audio; inference may have failed.",
                            status_code=response.status_code,
                            code="empty_audio",
                        )
        except httpx.HTTPError as error:
            raise TTSClientError(f"TTS request failed: {error}") from error

    async def synthesize_pcm(self, text: str, **kwargs) -> bytes:
        parts = [chunk async for chunk in self.stream_pcm(text, **kwargs)]
        return b"".join(parts)


def build_tts_client(settings) -> TTSClient:
    return TTSClient(
        settings.agent_tts_url,
        api_key=settings.agent_tts_api_key,
        voice_id=settings.agent_voice_id,
        language=settings.default_language,
        engine=settings.agent_tts_engine,
    )
