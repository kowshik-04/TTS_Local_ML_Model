"""Safe, conservative audio post-processing for speech output.

The goal is consistency and click-free delivery, not "enhancement" that changes
the speaker's timbre.  We remove DC/rumble, trim only edge silence, normalize
speech level, and peak-limit by attenuation.  No compressor, EQ curve, denoiser
or vocoder replacement is applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt


@dataclass(frozen=True)
class PostProcessConfig:
    target_rms_dbfs: float = -20.0
    peak_dbfs: float = -1.5
    highpass_hz: float = 55.0
    fade_ms: float = 8.0
    tail_ms: float = 80.0
    trim_edges: bool = True
    trim_threshold_dbfs: float = -45.0
    trim_keep_ms: float = 30.0
    max_gain_db: float = 12.0
    min_gain_db: float = -9.0


def _db_to_amplitude(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _amplitude_to_db(amplitude: float) -> float:
    return float(20.0 * np.log10(max(amplitude, 1e-12)))


def _mono_float(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 2:
        array = np.mean(array, axis=1, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError("Audio must be mono or a 2-D channel matrix.")
    if array.size == 0:
        raise ValueError("Audio is empty.")
    if not np.isfinite(array).all():
        raise ValueError("Audio contains non-finite samples.")
    return array


def _highpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0 or sample_rate <= cutoff_hz * 4:
        return audio
    sos = butter(2, cutoff_hz / (sample_rate / 2.0), btype="highpass", output="sos")
    return sosfilt(sos, audio).astype(np.float32, copy=False)


def _trim_edge_silence(audio: np.ndarray, sample_rate: int, config: PostProcessConfig) -> np.ndarray:
    if not config.trim_edges or audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return audio
    threshold = peak * _db_to_amplitude(config.trim_threshold_dbfs)
    above = np.flatnonzero(np.abs(audio) >= threshold)
    if above.size == 0:
        return audio
    keep = int(sample_rate * config.trim_keep_ms / 1000.0)
    start = max(0, int(above[0]) - keep)
    end = min(audio.size, int(above[-1]) + keep + 1)
    return audio[start:end]


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def _apply_fades(audio: np.ndarray, sample_rate: int, fade_ms: float) -> np.ndarray:
    length = min(int(sample_rate * fade_ms / 1000.0), audio.size // 2)
    if length <= 1:
        return audio
    ramp = np.linspace(0.0, 1.0, length, dtype=np.float32)
    audio = audio.copy()
    audio[:length] *= ramp
    audio[-length:] *= ramp[::-1]
    return audio


def _limit_peak(audio: np.ndarray, peak_target: float) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > peak_target and peak > 0:
        audio = audio * (peak_target / peak)
    return audio.astype(np.float32, copy=False)


def postprocess_speech(
    audio: np.ndarray,
    sample_rate: int,
    config: PostProcessConfig | None = None,
) -> np.ndarray:
    """Return mono float32 speech at the same sample rate."""
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    config = config or PostProcessConfig()
    working = _mono_float(audio)
    working = working - np.mean(working, dtype=np.float64).astype(np.float32)
    working = _highpass(working, sample_rate, config.highpass_hz)
    working = _trim_edge_silence(working, sample_rate, config)
    if working.size == 0:
        raise ValueError("Audio is silent after trimming.")

    speech_rms = _rms(working)
    if speech_rms < 1e-6:
        raise ValueError("Audio is silent after post-processing.")
    target_rms = _db_to_amplitude(config.target_rms_dbfs)
    if speech_rms > 0:
        gain_db = _amplitude_to_db(target_rms / speech_rms)
        gain_db = float(np.clip(gain_db, config.min_gain_db, config.max_gain_db))
        working = working * _db_to_amplitude(gain_db)

    working = _apply_fades(working, sample_rate, config.fade_ms)
    working = _limit_peak(working, _db_to_amplitude(config.peak_dbfs))
    if config.tail_ms > 0:
        tail = np.zeros(int(sample_rate * config.tail_ms / 1000.0), dtype=np.float32)
        working = np.concatenate((working, tail))
    return np.ascontiguousarray(working, dtype=np.float32)


def postprocess_stream_chunk(
    audio: np.ndarray,
    sample_rate: int,
    config: PostProcessConfig | None = None,
) -> np.ndarray:
    """Conservative per-chunk processing for first-audio streaming.

    Whole-utterance loudness normalization cannot be known before the last chunk.
    Streaming therefore uses a peak guard and fades only; clients that need
    exact loudness consistency should request the non-streaming endpoint.
    """
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    config = config or PostProcessConfig()
    working = _mono_float(audio)
    working = working - np.mean(working, dtype=np.float64).astype(np.float32)
    working = _highpass(working, sample_rate, config.highpass_hz)
    if _rms(working) < 1e-6:
        raise ValueError("Streaming chunk is silent.")
    working = _apply_fades(working, sample_rate, config.fade_ms)
    working = _limit_peak(working, _db_to_amplitude(config.peak_dbfs))
    return np.ascontiguousarray(working, dtype=np.float32)


def to_pcm16(audio: np.ndarray) -> bytes:
    """Little-endian signed 16-bit PCM, clipped safely."""
    array = np.asarray(audio, dtype=np.float32).reshape(-1)
    array = np.clip(array, -1.0, 1.0)
    return (array * 32767.0).astype("<i2", copy=False).tobytes()


def from_pcm16(data: bytes) -> np.ndarray:
    """Decode little-endian signed 16-bit PCM into float32 [-1, 1]."""
    if len(data) % 2:
        raise ValueError("PCM16 payload must contain an even number of bytes.")
    array = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    return np.ascontiguousarray(array)
