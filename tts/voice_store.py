"""Persistent, API-safe custom voice profiles.

A profile owns a validated 24 kHz mono reference WAV and the exact transcript
that matches it.  Recording length is capped at 12 seconds so IndicF5's internal
reference clipping cannot silently misalign the transcript.  Profiles are never
overwritten and deletion is restricted to this module's own directory format.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from tts.audio_utils import AudioValidationError, prepare_reference_cached
from tts.text_normalizer import normalize_text


OWNER = "local-voice-studio"
VOICE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
SLUG_INVALID = re.compile(r"[^0-9A-Za-z_-]+")
MAX_TRANSCRIPT_CHARS = 5000
PROFILE_VERSION = 1


class VoiceError(ValueError):
    pass


@dataclass(frozen=True)
class VoiceProfile:
    voice_id: str
    name: str
    directory: Path
    wav_path: Path
    transcript_path: Path
    transcript: str
    language: str
    duration_seconds: float
    sample_rate: int
    source_sha256: str
    consent: bool
    created_at: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    engine_hint: str = "indicf5"
    owner_key_id: str = "local"

    @property
    def valid(self) -> bool:
        return self.wav_path.is_file() and self.transcript_path.is_file() and bool(self.transcript.strip())

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        with open(self.wav_path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        digest.update(b"\x00")
        digest.update(self.transcript.encode("utf-8"))
        return digest.hexdigest()

    def as_dict(self, *, include_transcript: bool = True) -> dict:
        data = {
            "voice_id": self.voice_id,
            "name": self.name,
            "language": self.language,
            "duration_seconds": round(self.duration_seconds, 3),
            "sample_rate": self.sample_rate,
            "source_sha256": self.source_sha256,
            "consent": self.consent,
            "created_at": self.created_at,
            "warnings": list(self.warnings),
            "engine_hint": self.engine_hint,
            "owner_key_id": self.owner_key_id,
        }
        if include_transcript:
            data["transcript"] = self.transcript
        return data


def _slug(name: str) -> str:
    slug = SLUG_INVALID.sub("_", (name or "").strip()).strip("_-").lower()
    return slug[:32]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".profile-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class VoiceStore:
    def __init__(
        self,
        root: str | Path,
        *,
        min_seconds: float = 3.0,
        max_seconds: float = 12.0,
        reference_cache_dir: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.min_seconds = float(min_seconds)
        self.max_seconds = float(max_seconds)
        self.reference_cache_dir = Path(reference_cache_dir or (self.root / ".prepared"))

    def _dir(self, voice_id: str) -> Path:
        if not VOICE_ID.fullmatch(voice_id or ""):
            raise VoiceError("Invalid voice id.")
        path = (self.root / voice_id).resolve()
        if path.parent != self.root.resolve():
            raise VoiceError("Voice path escapes the voice directory.")
        return path

    def _profile_from_dir(self, directory: Path) -> VoiceProfile | None:
        profile_path = directory / "profile.json"
        if not profile_path.is_file():
            return None
        try:
            data = json.loads(profile_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return None
            if data.get("owner") != OWNER or data.get("version") != PROFILE_VERSION:
                return None
            voice_id = str(data["voice_id"])
            if voice_id != directory.name or not VOICE_ID.fullmatch(voice_id):
                return None
            transcript_path = directory / "reference.txt"
            try:
                transcript = transcript_path.read_text(encoding="utf-8").strip()
            except OSError:
                transcript = ""
            profile = VoiceProfile(
                voice_id=voice_id,
                name=str(data.get("name", voice_id)),
                directory=directory,
                wav_path=directory / "reference.wav",
                transcript_path=transcript_path,
                transcript=transcript,
                language=str(data.get("language", "Telugu")),
                duration_seconds=float(data.get("duration_seconds", 0.0)),
                sample_rate=int(data.get("sample_rate", 24000)),
                source_sha256=str(data.get("source_sha256", "")),
                consent=bool(data.get("consent", False)),
                created_at=str(data.get("created_at", "")),
                warnings=tuple(str(item) for item in data.get("warnings", [])),
                engine_hint=str(data.get("engine_hint", "indicf5")),
                owner_key_id=str(data.get("owner_key_id", "local")),
            )
            # The on-disk filename and the profile metadata must agree.  This
            # prevents a copied/edited profile from claiming another voice id.
            if profile.voice_id != directory.name:
                return None
            return profile
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def get(
        self,
        voice_id: str,
        *,
        owner_key_id: str | None = None,
        allow_admin: bool = False,
    ) -> VoiceProfile:
        profile = self._profile_from_dir(self._dir(voice_id))
        if profile is None or not profile.valid:
            raise VoiceError(f"Voice profile {voice_id!r} was not found or is incomplete.")
        if not profile.consent:
            raise VoiceError("Voice profile does not contain recorded consent.")
        if owner_key_id is not None and profile.owner_key_id != owner_key_id and not allow_admin:
            raise VoiceError(f"Voice profile {voice_id!r} was not found.")
        return profile

    def list(
        self,
        *,
        include_transcript: bool = False,
        owner_key_id: str | None = None,
        allow_admin: bool = False,
    ) -> list[dict]:
        profiles: list[VoiceProfile] = []
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir() or directory.name.startswith(".") or directory.is_symlink():
                continue
            profile = self._profile_from_dir(directory)
            if profile is None or not profile.valid:
                continue
            if owner_key_id is not None and profile.owner_key_id != owner_key_id and not allow_admin:
                continue
            profiles.append(profile)
        profiles.sort(key=lambda item: item.created_at, reverse=True)
        return [profile.as_dict(include_transcript=include_transcript) for profile in profiles]

    def find_duplicate(
        self, source_sha256: str, transcript: str, owner_key_id: str = "local"
    ) -> VoiceProfile | None:
        transcript = normalize_text(transcript)
        for directory in self.root.iterdir():
            if not directory.is_dir() or directory.is_symlink():
                continue
            profile = self._profile_from_dir(directory)
            if (
                profile is not None
                and profile.source_sha256 == source_sha256
                and profile.owner_key_id == owner_key_id
                and normalize_text(profile.transcript) == transcript
                and profile.valid
                and profile.consent
            ):
                return profile
        return None

    def create(
        self,
        name: str,
        audio_path: str | Path,
        transcript: str,
        *,
        language: str = "Telugu",
        consent: bool = False,
        engine_hint: str = "indicf5",
        owner_key_id: str = "local",
    ) -> VoiceProfile:
        if not consent:
            raise VoiceError(
                "Explicit consent is required. Confirm that you own the voice or have "
                "written permission to clone it."
            )
        name = (name or "").strip()
        slug = _slug(name)
        if not slug:
            raise VoiceError("Enter a name using letters or numbers.")
        transcript = normalize_text(transcript)
        if not transcript:
            raise VoiceError("A transcript of the words spoken in the recording is required.")
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            raise VoiceError("Transcript is too long.")
        if not audio_path:
            raise VoiceError("Upload or record reference audio first.")
        source = Path(audio_path)
        if not source.is_file():
            raise VoiceError("Reference audio file was not found.")
        source_bytes = source.read_bytes()
        source_sha = _sha256(source_bytes)
        duplicate = self.find_duplicate(source_sha, transcript, owner_key_id)
        if duplicate is not None:
            return duplicate
        try:
            prepared_path, info = prepare_reference_cached(
                source,
                self.reference_cache_dir,
                min_seconds=self.min_seconds,
                max_seconds=self.max_seconds,
            )
        except AudioValidationError as error:
            raise VoiceError(str(error)) from error
        voice_id = f"v_{slug}_{source_sha[:8]}"
        if len(voice_id) > 64:
            voice_id = f"v_{slug[:40]}_{source_sha[:8]}"
        for attempt in range(20):
            candidate = voice_id if attempt == 0 else f"{voice_id[:55]}_{attempt}"
            try:
                directory = self._dir(candidate)
                directory.mkdir(parents=False, exist_ok=False)
                break
            except FileExistsError:
                continue
        else:
            raise VoiceError("Could not allocate a unique voice id; try again.")
        try:
            shutil.copyfile(prepared_path, directory / "reference.wav")
            (directory / "reference.txt").write_text(transcript, encoding="utf-8")
            payload = {
                "owner": OWNER,
                "version": PROFILE_VERSION,
                "voice_id": candidate,
                "name": name,
                "language": language,
                "duration_seconds": info.duration_seconds,
                "sample_rate": info.sample_rate,
                "source_sha256": source_sha,
                "transcript_sha256": _sha256(transcript.encode("utf-8")),
                "consent": True,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "warnings": list(info.warnings),
                "engine_hint": engine_hint,
                "owner_key_id": owner_key_id,
                "preparation": "mono 24 kHz, edge silence trimmed, max 12 s",
            }
            _atomic_json(directory / "profile.json", payload)
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        profile = self._profile_from_dir(directory)
        if profile is None:
            shutil.rmtree(directory, ignore_errors=True)
            raise VoiceError("Voice profile could not be written.")
        return profile

    def delete(
        self,
        voice_id: str,
        *,
        owner_key_id: str | None = None,
        allow_admin: bool = False,
    ) -> None:
        directory = self._dir(voice_id)
        if directory.is_symlink() or not directory.is_dir():
            raise VoiceError("Voice profile was not found.")
        profile = self._profile_from_dir(directory)
        if profile is None:
            raise VoiceError("Refusing to delete a directory that is not an app-owned voice profile.")
        if owner_key_id is not None and profile.owner_key_id != owner_key_id and not allow_admin:
            raise VoiceError("Voice profile was not found.")
        if not profile.consent:
            raise VoiceError("Refusing to delete a profile without its consent record.")
        shutil.rmtree(directory)
