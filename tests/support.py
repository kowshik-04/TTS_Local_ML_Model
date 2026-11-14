"""Shared fakes and temporary-service helpers for the test suite."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import soundfile as sf

from tts.cache import AudioCache
from tts.config import Settings
from tts.engine_router import EngineRouter
from tts.service import TTSService
from tts.voice_store import VoiceStore


def tone(seconds=3.2, rate=24_000, amplitude=0.2):
    samples = np.arange(int(rate * seconds), dtype=np.float32) / rate
    return (amplitude * np.sin(2 * np.pi * 180 * samples)).astype(np.float32)


class FakeEngine:
    engine_name = "indicf5"

    def __init__(self):
        self.calls = []
        self.stream_calls = []
        self.load_calls = 0
        self.load_error = None
        self.generate_error = None
        self.cancel_on_call = False
        self.loaded = True

    def load(self):
        self.load_calls += 1
        if self.load_error is not None:
            raise self.load_error
        return True

    def generate(self, reference_path, reference_text, target_text, progress=None, *, options=None, cancel_event=None):
        self.calls.append(
            {
                "reference_path": reference_path,
                "reference_text": reference_text,
                "target_text": target_text,
                "options": options,
            }
        )
        if self.cancel_on_call or (cancel_event is not None and cancel_event.is_set()):
            raise RuntimeError("Generation cancelled.")
        if self.generate_error is not None:
            raise self.generate_error
        return tone(0.5), 24_000

    def generate_stream(self, reference_path, reference_text, target_text, progress=None, *, options=None, cancel_event=None):
        self.stream_calls.append(target_text)
        for index in range(2):
            if self.cancel_on_call or (cancel_event is not None and cancel_event.is_set()):
                raise RuntimeError("Generation cancelled.")
            if self.generate_error is not None:
                raise self.generate_error
            yield tone(0.3), 24_000, index + 1, 2

    def info(self):
        return {"engine": "indicf5", "loaded": True, "device": "cpu"}

    def warmup(self):
        return self.info()


def make_settings(directory: Path, **overrides) -> Settings:
    values = dict(
        root=directory,
        output_dir=directory / "outputs",
        voices_dir=directory / "voices",
        cache_dir=directory / "cache",
        reference_cache_dir=directory / "cache" / "references",
        warmup=False,
        cache_enabled=True,
        api_db=directory / "data" / "accounts.sqlite3",
        api_static_key="test-key",
        api_concurrency=1,
        min_voice_seconds=3.0,
        max_text_chars=500,
        max_aligned_reference_seconds=12.0,
    )
    values.update(overrides)
    return Settings(**values)


def make_service(directory: Path, engine=None, chatterbox=None, settings_engine=None, **overrides):
    if settings_engine is not None:
        overrides["engine"] = settings_engine
    settings = make_settings(directory, **overrides)
    engine = engine or FakeEngine()
    router = EngineRouter(engine, settings=settings, chatterbox=chatterbox)
    voices = VoiceStore(
        settings.voices_dir,
        min_seconds=settings.min_voice_seconds,
        max_seconds=settings.max_aligned_reference_seconds,
        reference_cache_dir=settings.reference_cache_dir,
    )
    cache = AudioCache(settings.cache_dir / "audio", settings.cache_max_bytes)
    service = TTSService(settings, router=router, voices=voices, cache=cache)
    return service, engine


def create_voice(service: TTSService, directory: Path, owner="owner-a"):
    source = directory / "reference.wav"
    sf.write(source, tone(3.2, rate=48_000), 48_000, subtype="PCM_16")
    return service.voices.create(
        "Test Voice",
        source,
        "నమస్కారం. ఇది ఒక పరీక్ష వాక్యం.",
        consent=True,
        owner_key_id=owner,
    )


# --- Voice-agent fakes -------------------------------------------------------


class FakeSTT:
    def __init__(self, text="నమస్కారం, ఎలా ఉన్నారు?", error=None):
        self.text = text
        self.error = error
        self.calls = []

    async def transcribe(self, audio, sample_rate):
        self.calls.append({"length": len(audio), "sample_rate": sample_rate})
        if self.error:
            raise self.error
        return self.text


class FakeLLM:
    def __init__(
        self,
        tokens=None,
        *,
        delay=0.0,
        error=None,
        block_after_first=False,
        block_seconds=30.0,
    ):
        self.tokens = list(tokens or ["నమస్కారం, ", "ఎలా ఉన్నారు?", " ధన్యవాదాలు."])
        self.delay = delay
        self.error = error
        self.block_after_first = block_after_first
        self.block_seconds = block_seconds
        self.calls = []

    async def stream(self, messages):
        self.calls.append([dict(message) for message in messages])
        if self.error:
            raise self.error
        for index, token in enumerate(self.tokens):
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.block_after_first and index == 1:
                await asyncio.sleep(self.block_seconds)
            yield token


class FakeTTSClient:
    def __init__(
        self,
        chunks=None,
        *,
        delay=0.0,
        error=None,
        block_after_first=False,
        block_seconds=30.0,
    ):
        self.chunks = list(chunks or [b"\x01\x00" * 240, b"\x02\x00" * 240])
        self.delay = delay
        self.error = error
        self.block_after_first = block_after_first
        self.block_seconds = block_seconds
        self.calls = []
        self.started = asyncio.Event()

    async def stream_pcm(self, text, *, voice_id=None, language=None):
        self.calls.append({"text": text, "voice_id": voice_id, "language": language})
        self.started.set()
        if self.error:
            raise self.error
        for index, chunk in enumerate(self.chunks):
            if index:
                if self.block_after_first:
                    await asyncio.sleep(self.block_seconds)
                if self.delay:
                    await asyncio.sleep(self.delay)
            yield chunk
