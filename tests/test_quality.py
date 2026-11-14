import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.quality import (
    QualityManifest,
    _summary_markdown,
    asr_language,
    audio_statistics,
    character_error_rate,
    cosine_similarity,
    evaluate_manifest,
    load_manifest,
    normalize_text,
    word_error_rate,
)


def _write_wav(path: Path, *, sample_rate: int = 24_000, seconds: float = 0.5) -> None:
    samples = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    signal = 0.2 * np.sin(2 * np.pi * 220.0 * samples)
    sf.write(path, signal, sample_rate, subtype="PCM_16")


def _manifest_payload(tmp: Path, *, synthetic: bool = True, consent: bool = True) -> dict:
    reference = tmp / "reference.wav"
    sample = tmp / "sample.wav"
    _write_wav(reference)
    _write_wav(sample)
    return {
        "reference": {
            "audio": "reference.wav",
            "transcript": "reference",
            "synthetic": synthetic,
            "consent_confirmed": consent,
            "notes": "synthetic test voice; pipeline only",
        },
        "samples": [
            {
                "id": "te001",
                "audio": "sample.wav",
                "expected_text": "నమస్కారం",
                "language": "te",
                "engine": "test",
                "model_id": "test/model",
            }
        ],
    }


class ManifestTests(unittest.TestCase):
    def test_load_relative_paths_and_synthetic_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(json.dumps(_manifest_payload(tmp)), encoding="utf-8")
            manifest = load_manifest(manifest_path)
        self.assertTrue(manifest.synthetic)
        self.assertTrue(manifest.reference_audio.is_absolute())
        self.assertEqual(manifest.samples[0].language, "te")

    def test_human_reference_requires_consent(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(
                json.dumps(_manifest_payload(tmp, synthetic=False, consent=False)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "consent_confirmed"):
                load_manifest(manifest_path)

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            payload = _manifest_payload(tmp)
            payload["samples"].append(dict(payload["samples"][0]))
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate sample id"):
                load_manifest(manifest_path)


class TextMetricTests(unittest.TestCase):
    def test_normalization_preserves_telugu_marks(self):
        source = "నమస్కారం, world!"
        normalized = normalize_text(source)
        self.assertEqual(normalized, "నమస్కారం world")
        self.assertIn("్", normalized)
        self.assertIn("ా", normalized)

    def test_wer_and_cer(self):
        self.assertEqual(word_error_rate("hello, world!", "Hello world"), 0.0)
        self.assertAlmostEqual(word_error_rate("hello world", "hello there world"), 0.5)
        self.assertEqual(character_error_rate("abc", "abc"), 0.0)
        self.assertAlmostEqual(character_error_rate("abc", "axc"), 1 / 3)
        self.assertEqual(word_error_rate("", ""), 0.0)
        self.assertEqual(word_error_rate("", "unexpected"), 1.0)

    def test_asr_language_mapping(self):
        self.assertEqual(asr_language("te"), "te")
        self.assertEqual(asr_language("Telugu"), "te")
        self.assertEqual(asr_language("mixed"), "te")
        self.assertEqual(asr_language("tenglish"), "te")
        self.assertEqual(asr_language("en"), "en")
        self.assertEqual(asr_language("English"), "en")

    def test_cosine_similarity(self):
        self.assertAlmostEqual(cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])), 1.0)
        self.assertAlmostEqual(cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 0.0)
        with self.assertRaisesRegex(ValueError, "zero"):
            cosine_similarity(np.zeros(2), np.ones(2))


class AudioStatisticsTests(unittest.TestCase):
    def test_reports_technical_signal_facts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tone.wav"
            _write_wav(path, seconds=0.25)
            stats = audio_statistics(path)
        self.assertEqual(stats["sample_rate"], 24_000)
        self.assertEqual(stats["channels"], 1)
        self.assertAlmostEqual(stats["duration_seconds"], 0.25, places=2)
        self.assertGreater(stats["peak"], 0.1)
        self.assertEqual(stats["finite"], True)
        self.assertEqual(stats["clipped_sample_fraction"], 0.0)


class EvaluationTests(unittest.TestCase):
    def test_evaluate_with_injected_models(self):
        class FakeEmbedder:
            def embed(self, path):
                # Same reference/sample pair gets a high but non-identical score.
                return np.array([1.0, 1.0, 0.0], dtype=np.float32)

        class FakeRecognizer:
            def transcribe(self, path, language):
                self.language = language
                return "నమస్కారం"

        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(json.dumps(_manifest_payload(tmp)), encoding="utf-8")
            manifest = load_manifest(manifest_path)
            report = evaluate_manifest(
                manifest,
                embedder=FakeEmbedder(),
                recognizer=FakeRecognizer(),
                expected_sample_rate=24_000,
            )
        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(report["failure_count"], 0)
        self.assertAlmostEqual(report["samples"][0]["speaker_similarity"], 1.0)
        self.assertEqual(report["samples"][0]["wer"], 0.0)
        self.assertEqual(report["samples"][0]["cer"], 0.0)
        self.assertTrue(any("synthetic" in warning.lower() for warning in report["warnings"]))
        self.assertTrue(any("shorter than" in warning for warning in report["warnings"]))
        self.assertEqual(report["short_clip_count"], 1)
        self.assertEqual(report["summary"]["speaker_similarity"]["count"], 1)
        self.assertEqual(
            report["reference"]["notes"], "synthetic test voice; pipeline only"
        )
        markdown = _summary_markdown(report)
        self.assertIn("Reference notes: synthetic test voice; pipeline only", markdown)
        self.assertIn("metric pipeline only", markdown.lower())
        self.assertIn("Values above 1.0", markdown)
        self.assertIn("Short clips", markdown)

    def test_failed_sample_is_reported_not_raised(self):
        class BrokenRecognizer:
            def transcribe(self, path, language):
                raise RuntimeError("recognizer unavailable")

        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(json.dumps(_manifest_payload(tmp)), encoding="utf-8")
            manifest = load_manifest(manifest_path)
            report = evaluate_manifest(
                manifest,
                recognizer=BrokenRecognizer(),
                expected_sample_rate=24_000,
            )
        self.assertEqual(report["failure_count"], 1)
        self.assertEqual(report["samples"][0]["status"], "failed")
        self.assertIn("recognizer unavailable", report["samples"][0]["error"])

    def test_empty_sample_list_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            payload = _manifest_payload(tmp)
            payload["samples"] = []
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty"):
                load_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main()
