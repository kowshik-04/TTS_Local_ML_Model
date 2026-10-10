from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 24_000
MAX_REFERENCE_SECONDS = 60
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


class AudioValidationError(ValueError):
    pass


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    duration_seconds: float
    peak: float
    rms: float
    warnings: tuple[str, ...] = ()


def _decode_audio(path: Path) -> tuple[np.ndarray, int]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioValidationError(
            "FFmpeg is required to decode WAV/MP3 and other supported uploads. "
            "Install it with `brew install ffmpeg`, then restart the app."
        )
    if not path.is_file():
        raise AudioValidationError("The selected reference audio file is unavailable.")
    if path.stat().st_size == 0:
        raise AudioValidationError("The selected reference audio file is empty.")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise AudioValidationError("Reference audio must be smaller than 100 MB.")

    command = [
        ffmpeg, "-nostdin", "-v", "error", "-i", str(path),
        "-map", "0:a:0", "-ac", "1", "-c:a", "pcm_f32le", "-f", "wav", "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=90)
    except subprocess.TimeoutExpired as error:
        raise AudioValidationError("Audio decoding took too long; try a shorter recording.") from error
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise AudioValidationError(
            "The audio file could not be decoded. Choose a valid WAV or MP3 recording."
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
            f"{MAX_REFERENCE_SECONDS} seconds. A clear 10–15 second clip is recommended."
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


@contextmanager
def prepared_reference(path: str | Path) -> Iterator[tuple[Path, AudioInfo]]:
    """Decode to a private working copy and resample with an anti-aliased polyphase filter."""
    source = Path(path)
    audio, source_rate = _decode_audio(source)
    info = validate_audio(audio, source_rate)
    if source_rate != TARGET_SAMPLE_RATE:
        common = gcd(source_rate, TARGET_SAMPLE_RATE)
        audio = resample_poly(
            audio,
            TARGET_SAMPLE_RATE // common,
            source_rate // common,
        ).astype(np.float32, copy=False)
    output_info = AudioInfo(
        TARGET_SAMPLE_RATE,
        len(audio) / TARGET_SAMPLE_RATE,
        float(np.max(np.abs(audio))),
        float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
        info.warnings,
    )
    with tempfile.TemporaryDirectory(prefix="local-tts-reference-") as temp_dir:
        prepared_path = Path(temp_dir) / "reference_mono_24khz.wav"
        sf.write(prepared_path, audio, TARGET_SAMPLE_RATE, subtype="PCM_24")
        yield prepared_path, output_info


def save_outputs(audio: np.ndarray, sample_rate: int, output_dir: Path) -> tuple[Path, Path]:
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeError("Synthesis returned empty or invalid audio.")
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms < 1e-5:
        raise RuntimeError("Synthesis returned audio that is too quiet to use.")
    if peak > 1.05:
        raise RuntimeError("Synthesis returned clipped audio; no output files were saved.")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"voice_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    wav_path, mp3_path = output_dir / f"{stem}.wav", output_dir / f"{stem}.mp3"
    with tempfile.TemporaryDirectory(prefix=".tts-staging-", dir=output_dir) as temp_dir:
        staged_wav = Path(temp_dir) / wav_path.name
        staged_mp3 = Path(temp_dir) / mp3_path.name
        sf.write(staged_wav, audio, sample_rate, subtype="PCM_24")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("FFmpeg is required for MP3 export; install it with `brew install ffmpeg`.")
        command = [
            ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(staged_wav),
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
