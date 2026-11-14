"""Dependency-free energy VAD with pre-roll and endpointing.

This is intentionally simple and predictable: it is a good default for a
low-latency demo and for tests, but it is not a replacement for a noise-robust
neural VAD in a noisy room.  The ``VadSegmenter`` protocol boundary keeps that
upgrade local.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VadConfig:
    sample_rate: int = 16_000
    frame_ms: int = 20
    speech_dbfs: float = -42.0
    silence_ms: int = 650
    min_speech_ms: int = 250
    max_utterance_seconds: float = 25.0
    pre_roll_ms: int = 200
    tail_ms: int = 120

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if not 10 <= self.frame_ms <= 100 or self.frame_ms % 10:
            raise ValueError("frame_ms must be a multiple of 10 between 10 and 100.")
        if self.silence_ms < self.frame_ms * 2:
            raise ValueError("silence_ms must cover at least two frames.")
        if self.min_speech_ms < self.frame_ms:
            raise ValueError("min_speech_ms must cover at least one frame.")
        if self.max_utterance_seconds <= 0:
            raise ValueError("max_utterance_seconds must be positive.")

    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)

    @property
    def frame_bytes(self) -> int:
        return self.frame_samples * 2

    @property
    def min_speech_frames(self) -> int:
        return max(1, math.ceil(self.min_speech_ms / self.frame_ms))

    @property
    def silence_frames(self) -> int:
        return max(1, math.ceil(self.silence_ms / self.frame_ms))

    @property
    def pre_roll_frames(self) -> int:
        return max(0, math.ceil(self.pre_roll_ms / self.frame_ms))


@dataclass(frozen=True)
class VadEvent:
    kind: str  # "speech_start" | "utterance"
    audio: bytes = b""
    duration_seconds: float = 0.0


def pcm16_to_float(data: bytes) -> np.ndarray:
    if len(data) % 2:
        raise ValueError("PCM16 data must contain an even number of bytes.")
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(audio: np.ndarray) -> bytes:
    array = np.asarray(audio, dtype=np.float32).reshape(-1)
    return (np.clip(array, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _frame_dbfs(frame: np.ndarray) -> float:
    if frame.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1e-6))


class VadSegmenter:
    """Stream PCM16 frames and emit speech-start/utterance boundaries.

    ``feed_pcm16`` accepts arbitrary byte boundaries; incomplete frames are
    buffered.  ``flush`` ends the current utterance when the client explicitly
    stops talking.
    """

    def __init__(self, config: VadConfig | None = None) -> None:
        self.config = config or VadConfig()
        self._pending = bytearray()
        self._pre_roll: deque[bytes] = deque(maxlen=self.config.pre_roll_frames)
        self._frames: list[bytes] = []
        self._speech_run = 0
        self._silence_run = 0
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._pending.clear()
        self._pre_roll.clear()
        self._frames.clear()
        self._speech_run = 0
        self._silence_run = 0
        self._active = False

    def feed_pcm16(self, data: bytes) -> list[VadEvent]:
        if not data:
            return []
        self._pending.extend(data)
        events: list[VadEvent] = []
        frame_bytes = self.config.frame_bytes
        while len(self._pending) >= frame_bytes:
            frame = bytes(self._pending[:frame_bytes])
            del self._pending[:frame_bytes]
            events.extend(self._process_frame(frame))
        return events

    def flush(self) -> list[VadEvent]:
        if self._pending:
            frame = bytes(self._pending)
            frame += b"\x00" * (self.config.frame_bytes - len(frame))
            self._pending.clear()
            self._process_frame(frame)
        if not self._active:
            return []
        audio = self._utterance_audio()
        self._reset_utterance()
        return [VadEvent("utterance", audio, len(audio) / 2 / self.config.sample_rate)]

    def _process_frame(self, frame: bytes) -> list[VadEvent]:
        events: list[VadEvent] = []
        samples = pcm16_to_float(frame)
        is_speech = _frame_dbfs(samples) >= self.config.speech_dbfs
        if not self._active:
            self._pre_roll.append(frame)
            if is_speech:
                self._speech_run += 1
            else:
                self._speech_run = 0
            if self._speech_run >= self.config.min_speech_frames:
                self._active = True
                self._frames = list(self._pre_roll)
                self._pre_roll.clear()
                self._silence_run = 0
                self._speech_run = 0
                events.append(VadEvent("speech_start"))
            return events

        self._frames.append(frame)
        if is_speech:
            self._silence_run = 0
        else:
            self._silence_run += 1
        max_frames = max(
            1,
            int(self.config.max_utterance_seconds * 1000 / self.config.frame_ms),
        )
        if self._silence_run >= self.config.silence_frames or len(self._frames) >= max_frames:
            audio = self._utterance_audio()
            self._reset_utterance()
            events.append(VadEvent("utterance", audio, len(audio) / 2 / self.config.sample_rate))
        return events

    def _utterance_audio(self) -> bytes:
        frames = list(self._frames)
        # Keep a short natural tail, but avoid sending the full endpointing
        # silence to the recognizer after a long pause.
        tail_frames = max(1, math.ceil(self.config.tail_ms / self.config.frame_ms))
        while len(frames) > tail_frames:
            last = pcm16_to_float(frames[-1])
            if _frame_dbfs(last) >= self.config.speech_dbfs:
                break
            frames.pop()
        return b"".join(frames)

    def _reset_utterance(self) -> None:
        self._frames = []
        self._pre_roll.clear()
        self._speech_run = 0
        self._silence_run = 0
        self._active = False
