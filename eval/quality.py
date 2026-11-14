#!/usr/bin/env python3
"""Offline speaker-similarity and ASR quality evaluation.

This module is intentionally outside the serving path.  It evaluates a
manifest of generated audio against one consented (or explicitly synthetic)
reference recording and writes per-sample and aggregate measurements.

It does not impose pass/fail thresholds.  Speaker-embedding cosine similarity is
a model-dependent proxy for perceptual speaker identity, not proof of it.  ASR
word/character error rates are affected by the recognizer's Telugu ability,
normalization choices and code-switch handling.  Human listening remains the
authority for pronunciation, naturalness and identity.

Manifest schema::

    {
      "reference": {
        "audio": "reference.wav",
        "transcript": "optional exact transcript",
        "consent_confirmed": true,
        "synthetic": false,
        "notes": "how consent was obtained"
      },
      "samples": [
        {
          "id": "te001",
          "audio": "te001.wav",
          "expected_text": "నమస్కారం.",
          "language": "te",
          "engine": "chatterbox",
          "model_id": "shankarpandala/chatterbox-telugu",
          "model_revision": "optional"
        }
      ]
    }

Paths may be absolute or relative to the manifest file.  Set
``reference.synthetic`` to true only for a non-human test signal; the report
then labels the similarity score as pipeline-only evidence.

Optional dependencies::

    pip install -r requirements-quality.txt

Examples::

    python eval/quality.py --manifest eval/quality-manifest.json --dry-run
    python eval/quality.py --manifest eval/quality-manifest.json \
        --asr-model large-v3 --device cpu

``--offline`` sets the Hugging Face/Transformers offline variables before the
optional libraries are imported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tts.audio_utils import AudioValidationError, decode_audio_file  # noqa: E402


QUALITY_VERSION = "1.0"
DEFAULT_ASR_MODEL = "large-v3"
DEFAULT_SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_CACHE_DIR = ROOT / ".cache" / "quality-models"
SHORT_CLIP_SECONDS = 2.0
SYNTHETIC_WARNING = (
    "The reference is marked synthetic. Speaker-similarity numbers demonstrate "
    "the metric pipeline only; they are not evidence of human identity preservation."
)


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    audio_path: Path
    expected_text: str
    language: str = "te"
    engine: str = ""
    model_id: str = ""
    model_revision: str = ""


@dataclass(frozen=True)
class QualityManifest:
    path: Path
    reference_audio: Path
    reference_transcript: str
    consent_confirmed: bool
    synthetic: bool
    reference_notes: str
    samples: tuple[EvaluationSample, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_manifest(path: str | Path) -> QualityManifest:
    """Load and validate a quality manifest without importing model packages."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read manifest: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Manifest must be a JSON object.")
    reference = payload.get("reference")
    if not isinstance(reference, dict):
        raise ValueError("Manifest must contain a 'reference' object.")
    audio_value = reference.get("audio")
    if not isinstance(audio_value, str) or not audio_value.strip():
        raise ValueError("reference.audio is required.")
    reference_audio = _resolve(path.parent, audio_value)
    if not reference_audio.is_file():
        raise ValueError(f"Reference audio not found: {reference_audio}")
    synthetic = bool(reference.get("synthetic", False))
    consent = bool(reference.get("consent_confirmed", False))
    if not synthetic and not consent:
        raise ValueError(
            "reference.consent_confirmed must be true for human audio, or "
            "reference.synthetic must be true for a non-human test signal."
        )
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("Manifest must contain a non-empty 'samples' list.")
    samples: list[EvaluationSample] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, dict):
            raise ValueError(f"samples[{index}] must be an object.")
        sample_id = str(raw.get("id") or "").strip()
        if not sample_id:
            raise ValueError(f"samples[{index}].id is required.")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample id: {sample_id}")
        seen.add(sample_id)
        audio = raw.get("audio")
        if not isinstance(audio, str) or not audio.strip():
            raise ValueError(f"samples[{index}].audio is required.")
        sample_audio = _resolve(path.parent, audio)
        if not sample_audio.is_file():
            raise ValueError(f"Sample audio not found: {sample_audio}")
        expected_text = str(raw.get("expected_text") or "").strip()
        if not expected_text:
            raise ValueError(f"samples[{index}].expected_text is required.")
        samples.append(
            EvaluationSample(
                sample_id=sample_id,
                audio_path=sample_audio,
                expected_text=expected_text,
                language=str(raw.get("language") or "te").strip(),
                engine=str(raw.get("engine") or "").strip(),
                model_id=str(raw.get("model_id") or "").strip(),
                model_revision=str(raw.get("model_revision") or "").strip(),
            )
        )
    return QualityManifest(
        path=path,
        reference_audio=reference_audio,
        reference_transcript=str(reference.get("transcript") or "").strip(),
        consent_confirmed=consent,
        synthetic=synthetic,
        reference_notes=str(reference.get("notes") or "").strip(),
        samples=tuple(samples),
    )


def asr_language(value: str) -> str:
    """Map project language labels to a Whisper language code."""
    value = value.strip().lower().replace("_", "-")
    if value.startswith("en"):
        return "en"
    # Telugu, Tenglish, romanized Telugu and mixed Telugu-English all need the
    # Telugu acoustic/linguistic hint for Whisper; the text may contain Latin.
    return "te"


def normalize_text(text: str) -> str:
    """Normalize for ASR comparison while preserving Telugu combining marks."""
    value = unicodedata.normalize("NFC", text).casefold()
    pieces: list[str] = []
    for character in value:
        category = unicodedata.category(character)
        if category.startswith("P") or category.startswith("S"):
            pieces.append(" ")
        else:
            pieces.append(character)
    return " ".join("".join(pieces).split())


def edit_distance(reference: Iterable[str], hypothesis: Iterable[str]) -> int:
    """Levenshtein distance over arbitrary hashable units."""
    reference = list(reference)
    hypothesis = list(hypothesis)
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for row, reference_unit in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_unit in enumerate(hypothesis, start=1):
            cost = 0 if reference_unit == hypothesis_unit else 1
            current.append(
                min(
                    current[column - 1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word error rate after :func:`normalize_text`; empty references score 1."""
    reference_words = normalize_text(reference).split()
    hypothesis_words = normalize_text(hypothesis).split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return edit_distance(reference_words, hypothesis_words) / len(reference_words)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Character error rate after normalization, excluding whitespace."""
    reference_chars = list(normalize_text(reference).replace(" ", ""))
    hypothesis_chars = list(normalize_text(hypothesis).replace(" ", ""))
    if not reference_chars:
        return 0.0 if not hypothesis_chars else 1.0
    return edit_distance(reference_chars, hypothesis_chars) / len(reference_chars)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"Embedding dimensions differ: {left.shape} vs {right.shape}")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("Cannot compute cosine similarity for a zero embedding.")
    value = float(np.dot(left, right) / (left_norm * right_norm))
    return max(-1.0, min(1.0, value))


def audio_statistics(path: str | Path) -> dict[str, Any]:
    """Decode through the project audio path and report technical signal facts."""
    path = Path(path)
    audio, sample_rate = decode_audio_file(path)
    if audio.size == 0:
        raise AudioValidationError("Decoded audio is empty.")
    if not np.isfinite(audio).all():
        raise AudioValidationError("Decoded audio contains non-finite samples.")
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    clipped = float(np.mean(np.abs(audio) >= 0.999))
    near_silent = float(np.mean(np.abs(audio) < 1e-4))
    channels = None
    try:
        info = sf.info(str(path))
        channels = int(info.channels)
    except (RuntimeError, ValueError):
        channels = 1
    return {
        "sample_rate": int(sample_rate),
        "channels": channels,
        "duration_seconds": round(audio.size / sample_rate, 4),
        "sample_count": int(audio.size),
        "peak": round(peak, 6),
        "rms": round(rms, 6),
        "clipped_sample_fraction": round(clipped, 8),
        "near_silent_sample_fraction": round(near_silent, 6),
        "finite": True,
        "file_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


class SpeakerEmbedder:
    """Lazy SpeechBrain ECAPA embedder (downloads its public model on first use)."""

    def __init__(
        self,
        model_id: str = DEFAULT_SPEAKER_MODEL,
        *,
        device: str = "auto",
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
    ) -> None:
        try:
            import torch
            import torchaudio
            from speechbrain.inference.speaker import SpeakerRecognition
        except ImportError as error:
            raise RuntimeError(
                "Speaker similarity needs speechbrain, torch and torchaudio. "
                "Install requirements-quality.txt."
            ) from error
        self._torch = torch
        self._torchaudio = torchaudio
        self.model_id = model_id
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # SpeechBrain 1.0.x still passes the removed ``use_auth_token`` argument
        # to huggingface_hub.  Bridge that call for the lifetime of this one
        # model load; do not alter the installed packages.
        import huggingface_hub

        original_download = huggingface_hub.hf_hub_download
        original_parameters = inspect.signature(original_download).parameters

        def compatible_download(*args, **kwargs):
            if "use_auth_token" in kwargs and "token" in original_parameters:
                kwargs["token"] = kwargs.pop("use_auth_token")
            elif "use_auth_token" in kwargs:
                kwargs.pop("use_auth_token")
            return original_download(*args, **kwargs)

        huggingface_hub.hf_hub_download = compatible_download
        try:
            self._model = SpeakerRecognition.from_hparams(
                source=model_id,
                savedir=str(cache_dir / model_id.replace("/", "--")),
                run_opts={"device": device},
            )
        finally:
            huggingface_hub.hf_hub_download = original_download

    def embed(self, path: str | Path) -> np.ndarray:
        import numpy as np
        import soundfile as sf

        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        if audio.size == 0:
            raise RuntimeError(f"Audio is empty: {path}")
        mono = audio.mean(axis=1, dtype=np.float32)
        if sample_rate != 16_000:
            import torch
            import torchaudio.functional as F

            waveform = torch.from_numpy(mono).unsqueeze(0)
            waveform = F.resample(waveform, sample_rate, 16_000)
            mono = waveform.squeeze(0).numpy()
        with self._torch.no_grad():
            waveform = self._torch.from_numpy(np.asarray(mono, dtype=np.float32)).unsqueeze(0)
            embedding = self._model.encode_batch(waveform)
        return embedding.detach().cpu().numpy().reshape(-1)


class SpeechRecognizer:
    """Lazy faster-whisper recognizer; CPU is the portable default."""

    def __init__(
        self,
        model: str = DEFAULT_ASR_MODEL,
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise RuntimeError(
                "ASR WER/CER needs faster-whisper. Install requirements-quality.txt."
            ) from error
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model
        self.model = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
            download_root=str(cache_dir / "faster-whisper"),
        )

    def transcribe(self, path: str | Path, language: str = "te") -> str:
        segments, _info = self.model.transcribe(
            str(path),
            language=language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip())


def _dependency_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def evaluate_manifest(
    manifest: QualityManifest,
    *,
    expected_sample_rate: int | None = 24_000,
    embedder: SpeakerEmbedder | None = None,
    recognizer: SpeechRecognizer | None = None,
    speaker_model: str = DEFAULT_SPEAKER_MODEL,
    asr_model: str = DEFAULT_ASR_MODEL,
) -> dict[str, Any]:
    """Evaluate all samples; optional models make this testable without weights."""
    reference_stats = audio_statistics(manifest.reference_audio)
    if expected_sample_rate and reference_stats["sample_rate"] != expected_sample_rate:
        reference_stats["sample_rate_mismatch"] = True
    reference_embedding = embedder.embed(manifest.reference_audio) if embedder else None
    records: list[dict[str, Any]] = []
    failures = 0
    for sample in manifest.samples:
        record: dict[str, Any] = {
            "id": sample.sample_id,
            "language": sample.language,
            "engine": sample.engine,
            "model_id": sample.model_id,
            "model_revision": sample.model_revision,
            "expected_text": sample.expected_text,
            "audio": str(sample.audio_path),
            "status": "ok",
        }
        started = time.perf_counter()
        try:
            stats = audio_statistics(sample.audio_path)
            if expected_sample_rate and stats["sample_rate"] != expected_sample_rate:
                stats["sample_rate_expected"] = expected_sample_rate
                stats["sample_rate_mismatch"] = True
            record.update(stats)
            if embedder is not None:
                if reference_embedding is None:
                    raise RuntimeError("Reference embedding was not initialized.")
                record["speaker_similarity"] = round(
                    cosine_similarity(reference_embedding, embedder.embed(sample.audio_path)),
                    6,
                )
            if recognizer is not None:
                hypothesis = recognizer.transcribe(
                    sample.audio_path, asr_language(sample.language)
                )
                record["asr_hypothesis"] = hypothesis
                record["asr_normalized_hypothesis"] = normalize_text(hypothesis)
                record["asr_normalized_expected"] = normalize_text(sample.expected_text)
                record["wer"] = round(word_error_rate(sample.expected_text, hypothesis), 6)
                record["cer"] = round(character_error_rate(sample.expected_text, hypothesis), 6)
        except Exception as error:
            failures += 1
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"
        record["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        records.append(record)

    def values(key: str) -> list[float]:
        return [
            float(record[key])
            for record in records
            if record.get("status") == "ok" and isinstance(record.get(key), (int, float))
        ]

    def summary(key: str) -> dict[str, Any]:
        items = values(key)
        if not items:
            return {"count": 0, "min": None, "median": None, "max": None}
        return {
            "count": len(items),
            "min": round(min(items), 6),
            "median": round(statistics.median(items), 6),
            "max": round(max(items), 6),
        }

    short_samples = [
        record
        for record in records
        if record.get("status") == "ok"
        and isinstance(record.get("duration_seconds"), (int, float))
        and float(record["duration_seconds"]) < SHORT_CLIP_SECONDS
    ]
    warnings = [SYNTHETIC_WARNING] if manifest.synthetic else []
    if short_samples:
        warnings.append(
            f"{len(short_samples)} sample(s) are shorter than {SHORT_CLIP_SECONDS:g}s "
            "("
            + ", ".join(str(record["id"]) for record in short_samples)
            + "). Speaker similarity, WER and CER are unstable for very short "
            "utterances and can produce ASR hallucinations; treat those values as "
            "outliers, not quality evidence."
        )
    if embedder is not None:
        warnings.append(
            "Speaker-embedding cosine similarity is a model-dependent proxy, not "
            "proof of perceptual identity. Listen to the samples."
        )
    if recognizer is not None:
        warnings.append(
            "ASR error rates depend on the recognizer, normalization and language "
            "coverage. Telugu and code-switched recognition can be unreliable; "
            "WER/CER above 1.0 can occur when the recognizer inserts or invents text."
        )

    return {
        "quality_version": QUALITY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": str(manifest.path),
        "reference": {
            "audio": str(manifest.reference_audio),
            "transcript": manifest.reference_transcript,
            "consent_confirmed": manifest.consent_confirmed,
            "synthetic": manifest.synthetic,
            "notes": manifest.reference_notes,
            "stats": reference_stats,
        },
        "models": {
            "speaker": {
                "id": speaker_model,
                "enabled": embedder is not None,
                "package_version": _dependency_version("speechbrain"),
            },
            "asr": {
                "id": asr_model,
                "enabled": recognizer is not None,
                "package_version": _dependency_version("faster-whisper"),
            },
        },
        "sample_count": len(records),
        "ok_count": len(records) - failures,
        "failure_count": failures,
        "samples": records,
        "summary": {
            "speaker_similarity": summary("speaker_similarity"),
            "wer": summary("wer"),
            "cer": summary("cer"),
            "duration_seconds": summary("duration_seconds"),
            "peak": summary("peak"),
            "rms": summary("rms"),
        },
        "warnings": warnings,
        "short_clip_seconds": SHORT_CLIP_SECONDS,
        "short_clip_count": len(short_samples),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "packages": {
                name: _dependency_version(name)
                for name in ("torch", "torchaudio", "speechbrain", "faster-whisper")
            },
        },
    }


def _summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Speech quality evaluation",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Manifest: `{report['manifest']}`",
        f"- Samples: {report['sample_count']} ({report['ok_count']} evaluated, "
        f"{report['failure_count']} failed)",
        f"- Reference: `{report['reference']['audio']}`"
        + (" · **synthetic test signal**" if report["reference"]["synthetic"] else ""),
        *(
            [f"- Reference notes: {report['reference']['notes']}"]
            if report["reference"].get("notes")
            else []
        ),
        f"- Speaker model: `{report['models']['speaker']['id']}`"
        + ("" if report["models"]["speaker"]["enabled"] else " (disabled)"),
        f"- ASR model: `{report['models']['asr']['id']}`"
        + ("" if report["models"]["asr"]["enabled"] else " (disabled)"),
        "",
        "## Summary",
        "",
        "| Metric | Count | Min | Median | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ("speaker_similarity", "wer", "cer", "duration_seconds", "peak", "rms"):
        item = report["summary"].get(key, {})
        lines.append(
            f"| {key} | {item.get('count', 0)} | {item.get('min')} | "
            f"{item.get('median')} | {item.get('max')} |"
        )
    lines.extend(["", "## Samples", ""])
    lines.append("| ID | Engine | Language | Duration | Similarity | WER | CER | Status |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|")
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    for record in report["samples"]:
        lines.append(
            f"| `{cell(record['id'])}` | {cell(record.get('engine') or '—')} | "
            f"{cell(record.get('language') or '—')} | {cell(record.get('duration_seconds', '—'))} | "
            f"{cell(record.get('speaker_similarity', '—'))} | {cell(record.get('wer', '—'))} | "
            f"{cell(record.get('cer', '—'))} | {cell(record['status'])} |"
        )
        if record.get("asr_hypothesis"):
            lines.append(
                f"|  |  |  |  |  |  |  | ASR: {cell(record['asr_hypothesis'])} |"
            )
    lines.extend(["", "## Interpretation", ""])
    for warning in report["warnings"]:
        lines.append(f"- {warning}")
    lines.extend(
        [
            "- No universal similarity or error-rate threshold is asserted.",
            "- Technical signal checks do not establish pronunciation or naturalness.",
            "- ASR WER/CER are measurements of one recognizer, not pronunciation "
            "certification. Values above 1.0 mean the recognizer inserted or invented text.",
            "- If the reference is synthetic, no score in this report demonstrates "
            "human voice cloning or human speaker identity preservation.",
            "- Short clips can be dominated by endpointing, ASR hallucination and "
            "embedding instability; inspect the short-clip warning above before "
            "citing any individual score.",
            "- This script is an offline evaluation tool, not part of the serving path.",
            "",
        ]
    )
    if report["failure_count"]:
        lines.extend(["## Failures", ""])
        for record in report["samples"]:
            if record["status"] == "failed":
                lines.append(f"- `{record['id']}`: {record.get('error')}")
        lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "eval" / "quality-results")
    parser.add_argument("--device", default="auto", help="Speaker model device: auto, cpu, mps, cuda.")
    parser.add_argument("--speaker-model", default=DEFAULT_SPEAKER_MODEL)
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    parser.add_argument("--asr-device", default="cpu")
    parser.add_argument("--asr-compute-type", default="int8")
    parser.add_argument("--expected-sample-rate", type=int, default=24_000)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--no-speaker", action="store_true")
    parser.add_argument("--no-asr", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        manifest = load_manifest(args.manifest)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(
            f"{len(manifest.samples)} sample(s); reference={manifest.reference_audio}; "
            f"synthetic={manifest.synthetic}; consent={manifest.consent_confirmed}",
            file=sys.stderr,
        )
        for sample in manifest.samples:
            print(
                f"{sample.sample_id}\t{sample.language}\t{sample.engine}\t"
                f"{sample.audio_path}",
                file=sys.stderr,
            )
        return 0

    embedder = None
    recognizer = None
    try:
        if not args.no_speaker:
            embedder = SpeakerEmbedder(
                args.speaker_model,
                device=args.device,
                cache_dir=args.model_cache,
            )
        if not args.no_asr:
            recognizer = SpeechRecognizer(
                args.asr_model,
                device=args.asr_device,
                compute_type=args.asr_compute_type,
                cache_dir=args.model_cache,
            )
        report = evaluate_manifest(
            manifest,
            expected_sample_rate=args.expected_sample_rate,
            embedder=embedder,
            recognizer=recognizer,
            speaker_model=args.speaker_model,
            asr_model=args.asr_model,
        )
    except Exception as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir / run_id
    _write_json(output_dir / "quality.json", report)
    (output_dir / "quality.md").write_text(_summary_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "run_dir": str(output_dir),
                "samples": report["sample_count"],
                "failed": report["failure_count"],
                "speaker_similarity": report["summary"]["speaker_similarity"],
                "wer": report["summary"]["wer"],
                "cer": report["summary"]["cer"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if report["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
