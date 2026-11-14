"""Small, thread-safe exact-output cache for repeated agent phrases.

The key includes the reference-voice fingerprint, normalized text, engine,
sampler options and post-processing version.  A cache hit returns the exact WAV
that would be generated again, so it is safe to use for greetings, prompts and
repeated agent turns.  Cache entries are never shared across different voices.

Writes use unique temporary files and atomic renames, so concurrent processes
do not corrupt an entry; cache eviction is last-writer-wins per process rather
than a globally coordinated LRU.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


CACHE_VERSION = "tts-cache-r1"
_CACHE_KEY = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CacheEntry:
    key: str
    audio_path: Path
    sample_rate: int
    duration_seconds: float
    engine: str
    voice_fingerprint: str
    created_at: float
    size_bytes: int


def make_cache_key(
    *,
    engine: str,
    voice_fingerprint: str,
    text: str,
    options: dict,
    sample_rate: int,
    postprocess_version: str = CACHE_VERSION,
) -> str:
    payload = json.dumps(
        {
            "engine": engine,
            "voice": voice_fingerprint,
            "text": text,
            "options": options,
            "sample_rate": sample_rate,
            "postprocess": postprocess_version,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AudioCache:
    def __init__(self, root: str | Path, max_bytes: int = 10 * 1024 * 1024 * 1024) -> None:
        self.root = Path(root)
        self.max_bytes = int(max_bytes)
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, key: str) -> tuple[Path, Path]:
        if not _CACHE_KEY.fullmatch(key or ""):
            raise ValueError("Invalid cache key.")
        return self.root / f"{key}.wav", self.root / f"{key}.json"

    def get(self, key: str) -> tuple[np.ndarray, CacheEntry] | None:
        wav_path, json_path = self._paths(key)
        if not (wav_path.is_file() and json_path.is_file()):
            return None
        try:
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
        except (OSError, ValueError, RuntimeError):
            self._delete(key)
            return None
        if audio.ndim != 1 or not np.isfinite(audio).all() or audio.size == 0:
            self._delete(key)
            return None
        if int(metadata.get("sample_rate", 0)) != int(sample_rate):
            self._delete(key)
            return None
        entry = CacheEntry(
            key=key,
            audio_path=wav_path,
            sample_rate=int(sample_rate),
            duration_seconds=float(len(audio) / sample_rate),
            engine=str(metadata.get("engine", "")),
            voice_fingerprint=str(metadata.get("voice_fingerprint", "")),
            created_at=float(metadata.get("created_at", 0.0)),
            size_bytes=wav_path.stat().st_size,
        )
        try:
            os.utime(wav_path, None)
        except OSError:
            pass
        return np.ascontiguousarray(audio, dtype=np.float32), entry

    def put(
        self,
        key: str,
        audio: np.ndarray,
        sample_rate: int,
        *,
        engine: str,
        voice_fingerprint: str,
        text_chars: int = 0,
    ) -> CacheEntry:
        array = np.asarray(audio, dtype=np.float32).reshape(-1)
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError("Cannot cache empty or invalid audio.")
        wav_path, json_path = self._paths(key)
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock:
            suffix = uuid.uuid4().hex
            temp_wav = wav_path.with_name(f".{wav_path.name}.{suffix}.tmp")
            sf.write(temp_wav, array, sample_rate, format="WAV", subtype="PCM_24")
            os.replace(temp_wav, wav_path)
            payload = {
                "key": key,
                "sample_rate": int(sample_rate),
                "duration_seconds": float(len(array) / sample_rate),
                "engine": engine,
                "voice_fingerprint": voice_fingerprint,
                "text_chars": int(text_chars),
                "created_at": time.time(),
            }
            temp_json = json_path.with_name(f".{json_path.name}.{suffix}.tmp")
            temp_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temp_json, json_path)
            self._evict_if_needed()
        return CacheEntry(
            key=key,
            audio_path=wav_path,
            sample_rate=sample_rate,
            duration_seconds=len(array) / sample_rate,
            engine=engine,
            voice_fingerprint=voice_fingerprint,
            created_at=payload["created_at"],
            size_bytes=wav_path.stat().st_size,
        )

    def _delete(self, key: str) -> None:
        wav_path, json_path = self._paths(key)
        wav_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)

    def _total_bytes(self) -> int:
        total = 0
        for path in self.root.glob("*.wav"):
            try:
                total += path.stat().st_size
            except OSError:
                pass
        return total

    def _evict_if_needed(self) -> None:
        if self.max_bytes <= 0:
            return
        if self._total_bytes() <= self.max_bytes:
            return
        entries: list[tuple[float, Path]] = []
        for path in self.root.glob("*.wav"):
            try:
                entries.append((path.stat().st_mtime, path))
            except OSError:
                pass
        entries.sort()
        for _, path in entries:
            if self._total_bytes() <= self.max_bytes:
                break
            self._delete(path.stem)

    def stats(self) -> dict:
        with self._lock:
            files = list(self.root.glob("*.wav"))
            total = sum(path.stat().st_size for path in files if path.is_file())
            return {"entries": len(files), "bytes": total, "max_bytes": self.max_bytes}

    def clear(self) -> int:
        with self._lock:
            count = 0
            for path in list(self.root.glob("*.wav")) + list(self.root.glob("*.json")):
                path.unlink(missing_ok=True)
                count += 1
            return count
