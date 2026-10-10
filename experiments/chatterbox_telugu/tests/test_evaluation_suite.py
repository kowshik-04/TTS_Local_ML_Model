import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_corpus import (
    CASES,
    MIXED_TELUGU_ENGLISH,
    PURE_TELUGU,
    REFERENCE_ROOT,
    REFERENCE_TRANSCRIPTS,
)
from evaluate_suite import _run_cases
from tts.chunking import split_text
from tts.text_normalizer import normalize_text


class EvaluationSuiteTests(unittest.TestCase):
    def test_corpus_has_separate_fixed_telugu_and_mixed_suites(self):
        self.assertGreaterEqual(len(PURE_TELUGU), 30)
        self.assertGreaterEqual(len(CASES), 30)
        self.assertTrue(all(case.language_id == "te" for case in CASES))
        self.assertTrue(all(case.case_id.startswith("te") for case in PURE_TELUGU))
        self.assertTrue(all(case.case_id.startswith("mix") for case in MIXED_TELUGU_ENGLISH))
        self.assertEqual(len({case.case_id for case in CASES}), len(CASES))
        categories = {case.category for case in CASES}
        self.assertTrue(
            {
                "conversation",
                "formal",
                "vowels_aspiration",
                "conjuncts",
                "similar_sounding_words",
                "punctuation",
                "numbers",
                "names",
                "long_sentence",
                "mixed_technical",
            }.issubset(categories)
        )
        self.assertTrue(
            all(case.reference_id in REFERENCE_TRANSCRIPTS for case in CASES)
        )
        for reference_id, transcript in REFERENCE_TRANSCRIPTS.items():
            disk_transcript = (
                REFERENCE_ROOT / f"{reference_id}.txt"
            ).read_text(encoding="utf-8").strip()
            self.assertEqual(transcript, disk_transcript)
        mixed_text = " ".join(case.text for case in MIXED_TELUGU_ENGLISH)
        for term in ("Red Hat", "Linux", "Kubernetes"):
            self.assertIn(term, mixed_text)

    def test_normalizer_changes_spacing_only_and_preserves_telugu_unicode(self):
        source = "  అమ్మా,   మీరు  ఎప్పుడు వస్తారు?\r\n\r\n  క్షమించండి!  "
        normalized = normalize_text(source)
        self.assertEqual(normalized, "అమ్మా, మీరు ఎప్పుడు వస్తారు?\n\nక్షమించండి!")
        self.assertEqual(
            [part for part in normalized.replace("\n", " ").split(" ") if part],
            ["అమ్మా,", "మీరు", "ఎప్పుడు", "వస్తారు?", "క్షమించండి!"],
        )
        self.assertEqual(normalize_text(" \n\t "), "")

    def test_chunking_preserves_words_order_punctuation_and_unicode(self):
        source = (
            "అమ్మా, మీరు ఎప్పుడు వస్తారు? క్షమించండి! "
            + " ".join(["స్వాతంత్ర్యం"] * 90)
            + "।"
        )
        normalized = normalize_text(source)
        chunks = split_text(normalized, max_bytes=96)
        chunk_text = " ".join(chunk for chunk, _ in chunks)
        tokens = lambda value: re.findall(r"\w+|[^\w\s]", value)
        self.assertGreater(len(chunks), 2)
        self.assertEqual(tokens(chunk_text), tokens(normalized))
        self.assertTrue(all(len(chunk.encode("utf-8")) <= 96 for chunk, _ in chunks))
        self.assertEqual(sum(chunk.count("స్వాతంత్ర్యం") for chunk, _ in chunks), 90)

    def test_chatterbox_receives_the_exact_corpus_text(self):
        source = "అమ్మా, మీరు ఎప్పుడు వస్తారు?"
        case = {
            "case_id": "te-test",
            "category": "test",
            "text": source,
            "language_id": "te",
            "reference_id": "01_conversation",
        }
        model = Mock()
        model.sr = 24_000
        model.tokenizer.preprocess_text.return_value = source
        model.generate.return_value = np.ones(48_000, dtype=np.float32) * 0.1
        with tempfile.TemporaryDirectory() as directory:
            engine_dir = Path(directory) / "chatterbox"
            engine_dir.mkdir()
            result_path = engine_dir / "cases.jsonl"
            _run_cases(
                [case],
                "chatterbox",
                engine_dir,
                result_path,
                model,
                "cpu",
                punc_norm=lambda text: text,
            )
            result = json.loads(result_path.read_text(encoding="utf-8").strip())
            self.assertEqual(model.generate.call_args.args[0], source)
            self.assertEqual(result["input_text"], source)
            self.assertEqual(result["text_passed_to_generate"], source)
            self.assertEqual(result["text_after_punctuation_normalization"], source)
            self.assertEqual(result["model_preprocessed_text"], source)
            self.assertEqual(result["status"], "succeeded")

    def test_indicf5_receives_original_target_and_records_normalized_form(self):
        source = "  అమ్మా,   మీరు  ఎప్పుడు వస్తారు? "
        case = {
            "case_id": "te-test",
            "category": "test",
            "text": source,
            "language_id": "te",
            "reference_id": "01_conversation",
        }
        model = Mock()
        model.generate.return_value = (np.ones(48_000, dtype=np.float32) * 0.1, 24_000)
        with tempfile.TemporaryDirectory() as directory:
            engine_dir = Path(directory) / "indicf5"
            engine_dir.mkdir()
            result_path = engine_dir / "cases.jsonl"
            _run_cases(
                [case],
                "indicf5",
                engine_dir,
                result_path,
                model,
                "cpu",
                normalize_text=normalize_text,
                split_text=split_text,
            )
            result = json.loads(result_path.read_text(encoding="utf-8").strip())
            self.assertEqual(model.generate.call_args.args[2], source)
            self.assertEqual(result["input_text"], source)
            self.assertEqual(result["text_passed_to_generate"], source)
            self.assertEqual(
                result["model_preprocessed_text"],
                "అమ్మా, మీరు ఎప్పుడు వస్తారు?",
            )
            self.assertEqual(result["status"], "succeeded")


if __name__ == "__main__":
    unittest.main()
