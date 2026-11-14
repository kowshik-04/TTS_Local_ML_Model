from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from math import gcd
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 24_000
MAX_REFERENCE_SECONDS = 60
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
REFERENCE_PREP_VERSION = "2026-10-r1"


class AudioValidationError(ValueError):
    pass


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    duration_seconds: float
    peak: float
    rms: float
    warnings: tuple[str, ...] = ()
    source_sample_rate: int | None = None
    source_duration_seconds: float | None = None
    trimmed_seconds: float | None = None
    cache_key: str | None = None

    def as_cache_dict(self) -> dict:
        data = asdict(self)
        data["warnings"] = list(self.warnings)
        return data

    @classmethod
    def from_cache_dict(cls, data: dict) -> "AudioInfo":
        return cls(
            sample_rate=int(data["sample_rate"]),
            duration_seconds=float(data["duration_seconds"]),
            peak=float(data["peak"]),
            rms=float(data["rms"]),
            warnings=tuple(str(item) for item in data.get("warnings", [])),
            source_sample_rate=(
                int(data["source_sample_rate"]) if data.get("source_sample_rate") is not None else None
            ),
            source_duration_seconds=(
                float(data["source_duration_seconds"])
                if data.get("source_duration_seconds") is not None
                else None
            ),
            trimmed_seconds=(
                float(data["trimmed_seconds"]) if data.get("trimmed_seconds") is not None else None
            ),
            cache_key=data.get("cache_key"),
        )


def _ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioValidationError(
            "FFmpeg is required to decode audio and export MP3/Opus. "
            "Install it with `brew install ffmpeg` (macOS), your Linux package manager, "
            "or `winget install --id Gyan.FFmpeg.Shared -e` (Windows; shared build), then restart."
        )
    return ffmpeg


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_audio_file(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode any FFmpeg-supported file to mono float32 without touching the source."""
    path = Path(path)
    ffmpeg = _ffmpeg()
    if not path.is_file():
        raise AudioValidationError("The selected audio file is unavailable.")
    if path.stat().st_size == 0:
        raise AudioValidationError("The selected audio file is empty.")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise AudioValidationError("Audio must be smaller than 100 MB.")
    command = [
        ffmpeg, "-nostdin", "-v", "error", "-i", str(path),
        "-map", "0:a:0", "-ac", "1", "-c:a", "pcm_f32le", "-f", "wav", "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=120)
    except subprocess.TimeoutExpired as error:
        raise AudioValidationError("Audio decoding took too long; try a shorter recording.") from error
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise AudioValidationError(
            "The audio file could not be decoded. Choose a valid WAV, MP3, M4A, FLAC or OGG file."
            + (f" Details: {detail[-500:]}" if detail else "")
        )
    try:
        audio, sample_rate = sf.read(io.BytesIO(result.stdout), dtype="float32", always_2d=False)
    except (RuntimeError, ValueError) as error:
        raise AudioValidationError("The decoded audio has an invalid or unsupported format.") from error
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if audio.ndim != 1 or sample_rate <= 0 or audio.size == 0:
        raise AudioValidationError("The recording contains no usable audio.")
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


# Backwards-compatible private alias used by older modules/tests.
_decode_audio = decode_audio_file


def validate_audio(audio: np.ndarray, sample_rate: int) -> AudioInfo:
    if sample_rate <= 0 or audio.ndim != 1 or audio.size == 0:
        raise AudioValidationError("The recording contains no usable audio.")
    if not np.isfinite(audio).all():
        raise AudioValidationError("The recording contains invalid numeric audio samples.")
    duration = len(audio) / sample_rate
    if duration < 1.0:
        raise AudioValidationError("Reference audio must be at least 1 second long.")
    if duration > MAX_REFERENCE_SECONDS:
        raise AudioValidationError(
            f"Reference audio is {duration:.1f} seconds; the maximum is "
            f"{MAX_REFERENCE_SECONDS} seconds. A clear 8–12 second clip is recommended."
        )
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if peak < 0.002 or rms < 0.0008:
        raise AudioValidationError(
            "The recording is silent or too quiet. Record a clear voice sample and try again."
        )
    clipped_ratio = float(np.mean(np.abs(audio) >= 0.999))
    if clipped_ratio > 0.01:
        raise AudioValidationError(
            "The recording is heavily clipped. Lower the microphone gain and record it again."
        )
    warnings = ()
    if clipped_ratio > 0.001:
        warnings = ("Some samples are near clipping; a lower microphone gain may improve quality.",)
    return AudioInfo(sample_rate, duration, peak, rms, warnings)


def trim_edge_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_dbfs: float = -45.0,
    keep_ms: float = 50.0,
) -> np.ndarray:
    """Trim only leading/trailing silence; never edit the spoken transcript region."""
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return audio
    threshold = peak * (10.0 ** (threshold_dbfs / 20.0))
    above = np.flatnonzero(np.abs(audio) >= threshold)
    if above.size == 0:
        return audio
    keep = int(sample_rate * keep_ms / 1000.0)
    start = max(0, int(above[0]) - keep)
    end = min(audio.size, int(above[-1]) + keep + 1)
    return np.asarray(audio[start:end], dtype=np.float32)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive.")
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    common = gcd(int(source_rate), int(target_rate))
    resampled = resample_poly(
        np.asarray(audio, dtype=np.float32),
        target_rate // common,
        source_rate // common,
    )
    return np.asarray(resampled, dtype=np.float32)


def prepare_reference_audio(
    path: str | Path,
    *,
    min_seconds: float = 1.0,
    max_seconds: float = 12.0,
    trim_silence: bool = True,
) -> tuple[np.ndarray, AudioInfo]:
    """Decode, validate, trim edge silence, enforce alignment limit, resample to 24 kHz."""
    source = Path(path)
    audio, source_rate = decode_audio_file(source)
    source_info = validate_audio(audio, source_rate)
    source_duration = len(audio) / source_rate
    if trim_silence:
        audio = trim_edge_silence(audio, source_rate)
    trimmed_duration = len(audio) / source_rate
    if trimmed_duration < min_seconds:
        raise AudioValidationError(
            f"Reference audio has only {trimmed_duration:.1f}s of speech after trimming; "
            f"at least {min_seconds:.0f}s is required."
        )
    if trimmed_duration > max_seconds:
        raise AudioValidationError(
            f"Reference audio is {trimmed_duration:.1f}s after trimming; the safe maximum is "
            f"{max_seconds:.0f}s because IndicF5 clips long references while keeping the full "
            "transcript, which misaligns the voice. Record an 8–12 second clip and enter its "
            "exact transcript."
        )
    audio = resample_audio(audio, source_rate)
    output_info = AudioInfo(
        sample_rate=TARGET_SAMPLE_RATE,
        duration_seconds=len(audio) / TARGET_SAMPLE_RATE,
        peak=float(np.max(np.abs(audio))),
        rms=float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
        warnings=source_info.warnings,
        source_sample_rate=source_rate,
        source_duration_seconds=source_duration,
        trimmed_seconds=max(0.0, source_duration - trimmed_duration),
    )
    return np.asarray(audio, dtype=np.float32), output_info


def _write_wav_atomic(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=".wav-", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
    try:
        sf.write(temp_path, audio, sample_rate, format="WAV", subtype="PCM_24")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


_REFERENCE_LOCKS_GUARD = threading.Lock()
_REFERENCE_LOCKS: dict[str, threading.Lock] = {}


def _reference_lock(key: str) -> threading.Lock:
    with _REFERENCE_LOCKS_GUARD:
        lock = _REFERENCE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _REFERENCE_LOCKS[key] = lock
        return lock


def prepare_reference_cached(
    path: str | Path,
    cache_dir: str | Path,
    *,
    min_seconds: float = 1.0,
    max_seconds: float = 12.0,
    version: str = REFERENCE_PREP_VERSION,
) -> tuple[Path, AudioInfo]:
    """Return a stable, preprocessed 24 kHz reference WAV plus metadata.

    A stable path matters: F5's own reference cache is keyed by file hash, and the
    vocoder/preprocessor can then reuse its silence analysis on every request.
    """
    source = Path(path)
    if not source.is_file():
        raise AudioValidationError("The selected reference audio file is unavailable.")
    source_hash = _file_sha256(source)
    key = hashlib.sha256(
        f"{source_hash}:{min_seconds}:{max_seconds}:{version}".encode("utf-8")
    ).hexdigest()[:32]
    cache_dir = Path(cache_dir)
    wav_path = cache_dir / f"{key}.wav"
    json_path = cache_dir / f"{key}.json"
    with _reference_lock(key):
        if wav_path.is_file() and json_path.is_file():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                info = AudioInfo.from_cache_dict(data)
                return wav_path, replace(info, cache_key=key)
            except (OSError, ValueError, KeyError):
                # Corrupt sidecar: rebuild it rather than failing a request.
                wav_path.unlink(missing_ok=True)
                json_path.unlink(missing_ok=True)
        audio, info = prepare_reference_audio(
            source, min_seconds=min_seconds, max_seconds=max_seconds, trim_silence=True
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        _write_wav_atomic(wav_path, audio, TARGET_SAMPLE_RATE)
        payload = replace(info, cache_key=key).as_cache_dict()
        temp_json = json_path.with_suffix(".json.tmp")
        temp_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_json, json_path)
        return wav_path, replace(info, cache_key=key)


@contextmanager
def prepared_reference(path: str | Path) -> Iterator[tuple[Path, AudioInfo]]:
    """Backward-compatible temporary 24 kHz copy.

    This does not trim speech or enforce the 12 s alignment limit; the engine and
    voice store use :func:`prepare_reference_cached` instead.
    """
    source = Path(path)
    audio, source_rate = decode_audio_file(source)
    info = validate_audio(audio, source_rate)
    source_duration = len(audio) / source_rate
    audio = resample_audio(audio, source_rate)
    output_info = AudioInfo(
        TARGET_SAMPLE_RATE,
        len(audio) / TARGET_SAMPLE_RATE,
        float(np.max(np.abs(audio))),
        float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
        info.warnings,
        source_sample_rate=source_rate,
        source_duration_seconds=source_duration,
        trimmed_seconds=0.0,
    )
    with tempfile.TemporaryDirectory(prefix="local-tts-reference-") as temp_dir:
        prepared_path = Path(temp_dir) / "reference_mono_24khz.wav"
        sf.write(prepared_path, audio, TARGET_SAMPLE_RATE, subtype="PCM_24")
        yield prepared_path, output_info


def _validate_output(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("Synthesis returned empty or invalid audio.")
    peak = float(np.max(np.abs(array)))
    rms = float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))
    if rms < 1e-5:
        raise RuntimeError("Synthesis returned audio that is too quiet to use.")
    if peak > 1.05:
        raise RuntimeError("Synthesis returned clipped audio; no output was saved.")
    return array


def encode_audio_bytes(audio: np.ndarray, sample_rate: int, audio_format: str = "wav") -> tuple[bytes, str]:
    """Encode speech in memory. Returns (payload, media_type)."""
    array = _validate_output(audio)
    audio_format = (audio_format or "wav").lower()
    if audio_format in {"pcm", "pcm16", "pcm_s16le"}:
        clipped = np.clip(array, -1.0, 1.0)
        return (clipped * 32767.0).astype("<i2").tobytes(), "audio/L16"
    if audio_format == "wav":
        buffer = io.BytesIO()
        sf.write(buffer, array, sample_rate, format="WAV", subtype="PCM_24")
        return buffer.getvalue(), "audio/wav"
    if audio_format in {"mp3", "mpeg"}:
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, array, sample_rate, format="WAV", subtype="PCM_24")
        command = [
            _ffmpeg(), "-nostdin", "-v", "error", "-i", "pipe:0",
            "-codec:a", "libmp3lame", "-b:a", "192k", "-f", "mp3", "pipe:1",
        ]
        result = subprocess.run(command, input=wav_buffer.getvalue(), capture_output=True, check=False, timeout=180)
        if result.returncode != 0 or not result.stdout:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"MP3 export failed: {detail[-500:] or 'FFmpeg returned no output.'}")
        return result.stdout, "audio/mpeg"
    if audio_format in {"opus", "ogg"}:
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, array, sample_rate, format="WAV", subtype="PCM_24")
        command = [
            _ffmpeg(), "-nostdin", "-v", "error", "-i", "pipe:0",
            "-codec:a", "libopus", "-b:a", "64k", "-vbr", "on", "-f", "ogg", "pipe:1",
        ]
        result = subprocess.run(command, input=wav_buffer.getvalue(), capture_output=True, check=False, timeout=180)
        if result.returncode != 0 or not result.stdout:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Opus export failed: {detail[-500:] or 'FFmpeg returned no output.'}")
        return result.stdout, "audio/ogg"
    raise ValueError(f"Unsupported output format {audio_format!r}; use wav, mp3, opus or pcm.")


def save_outputs(audio: np.ndarray, sample_rate: int, output_dir: Path) -> tuple[Path, Path]:
    array = _validate_output(audio)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"voice_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    wav_path, mp3_path = output_dir / f"{stem}.wav", output_dir / f"{stem}.mp3"
    with tempfile.TemporaryDirectory(prefix=".tts-staging-", dir=output_dir) as temp_dir:
        staged_wav = Path(temp_dir) / wav_path.name
        staged_mp3 = Path(temp_dir) / mp3_path.name
        sf.write(staged_wav, array, sample_rate, subtype="PCM_24")
        command = [
            _ffmpeg(), "-nostdin", "-v", "error", "-y", "-i", str(staged_wav),
            "-codec:a", "libmp3lame", "-b:a", "192k", str(staged_mp3),
        ]
        result = subprocess.run(command, capture_output=True, check=False, timeout=180)
        if result.returncode != 0 or not staged_mp3.is_file():
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"MP3 export failed: {detail[-500:] or 'FFmpeg returned no output.'}")
        os.replace(staged_wav, wav_path)
        os.replace(staged_mp3, mp3_path)
    return wav_path, mp3_path


def clear_generated_outputs(output_dir: Path) -> int:
    removed = 0
    if not output_dir.exists():
        return removed
    for path in output_dir.iterdir():
        if path.is_file() and path.name.startswith("voice_") and path.suffix in {".wav", ".mp3"}:
            path.unlink()
            removed += 1
    return removed
