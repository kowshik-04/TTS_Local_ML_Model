"""Safe discovery and creation of reference voices stored as ``NN_name.wav`` + ``NN_name.txt``."""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf

from tts.audio_utils import AudioValidationError, prepared_reference


_PAIR_NAME = re.compile(r"^(?P<number>\d{2,})_(?P<slug>[^/\\]+)$")
_SLUG_INVALID = re.compile(r"[^0-9A-Za-z_-]+")
MAX_TRANSCRIPT_CHARS = 5_000
MIN_SECONDS, MAX_SECONDS = 1.0, 60.0


class ReferenceError(ValueError):
    pass


@dataclass(frozen=True)
class ReferenceInfo:
    stem: str
    number: int
    label: str
    wav_path: Path
    txt_path: Path
    transcript: str = ""
    duration_seconds: float | None = None
    sample_rate: int | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    error: str | None = None

    @property
    def has_transcript(self) -> bool:
        return bool(self.transcript.strip())

    @property
    def valid(self) -> bool:
        return self.error is None and self.has_transcript


def _label(slug: str) -> str:
    return slug.replace("_", " ").replace("-", " ").strip().title() or slug


def _inspect(wav_path: Path, txt_path: Path, number: int, slug: str) -> ReferenceInfo:
    warnings: list[str] = []
    error = None
    duration = rate = None
    transcript = ""
    try:
        info = sf.info(str(wav_path))
        duration, rate = float(info.duration), int(info.samplerate)
        if duration < MIN_SECONDS:
            error = "Audio is shorter than 1 second."
        elif duration > MAX_SECONDS:
            error = f"Audio is longer than {MAX_SECONDS:.0f} seconds."
    except (RuntimeError, OSError) as exc:
        error = f"Audio cannot be read: {exc}"
    if not txt_path.is_file():
        warnings.append("Transcript file is missing.")
    else:
        try:
            transcript = txt_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            warnings.append("Transcript could not be read as UTF-8.")
        else:
            if not transcript:
                warnings.append("Transcript file is empty.")
    return ReferenceInfo(
        stem=wav_path.stem, number=number, label=_label(slug), wav_path=wav_path,
        txt_path=txt_path, transcript=transcript, duration_seconds=duration,
        sample_rate=rate, warnings=tuple(warnings), error=error,
    )


def discover_references(reference_dir: Path) -> list[ReferenceInfo]:
    """Top-level ``NN_name.wav`` files only; ``processed/`` is never listed."""
    reference_dir = Path(reference_dir)
    if not reference_dir.is_dir():
        return []
    found = []
    for wav_path in reference_dir.glob("*.wav"):
        match = _PAIR_NAME.match(wav_path.stem)
        if not match or not wav_path.is_file():
            continue
        found.append(_inspect(
            wav_path, wav_path.with_suffix(".txt"),
            int(match["number"]), match["slug"],
        ))
    return sorted(found, key=lambda item: (item.number, item.stem))


def next_reference_number(reference_dir: Path) -> int:
    numbers = [0]
    reference_dir = Path(reference_dir)
    if reference_dir.is_dir():
        for path in reference_dir.iterdir():
            match = _PAIR_NAME.match(path.stem)
            if match:
                numbers.append(int(match["number"]))
    return max(numbers) + 1


def slugify(name: str) -> str:
    slug = _SLUG_INVALID.sub("_", (name or "").strip()).strip("_-").lower()
    return slug[:40]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_reference(reference_dir: Path, name: str, audio_path: str | Path,
                   transcript: str) -> ReferenceInfo:
    """Create a new numbered pair; never overwrites. Stored audio is validated mono 24 kHz WAV."""
    reference_dir = Path(reference_dir)
    slug = slugify(name)
    if not slug:
        raise ReferenceError("Enter a name using letters or numbers.")
    transcript = (transcript or "").strip()
    if not transcript:
        raise ReferenceError("A transcript of the spoken words is required.")
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        raise ReferenceError("Transcript is too long.")
    if not audio_path:
        raise ReferenceError("Upload or record reference audio first.")
    try:
        with prepared_reference(audio_path) as (working_copy, _info):
            reference_dir.mkdir(parents=True, exist_ok=True)
            for _attempt in range(50):
                number = next_reference_number(reference_dir)
                stem = f"{number:02d}_{slug}"
                wav_path = reference_dir / f"{stem}.wav"
                txt_path = reference_dir / f"{stem}.txt"
                try:
                    # "xb" fails if the file exists, so concurrent saves cannot overwrite.
                    with open(wav_path, "xb") as wav_out:
                        with open(working_copy, "rb") as source:
                            shutil.copyfileobj(source, wav_out)
                except FileExistsError:
                    continue
                try:
                    with open(txt_path, "x", encoding="utf-8") as txt_out:
                        txt_out.write(transcript)
                except FileExistsError:
                    wav_path.unlink(missing_ok=True)
                    continue
                return _inspect(wav_path, txt_path, number, slug)
    except AudioValidationError as exc:
        raise ReferenceError(str(exc)) from exc
    raise ReferenceError("Could not allocate a free reference number; try again.")
