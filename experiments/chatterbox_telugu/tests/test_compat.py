import unittest
from pathlib import Path

from chatterbox import mtl_tts
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.tokenizers import MTLTokenizer

from compat import EXPECTED_VOCAB_SIZE, compatibility_scope, preflight


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "chatterbox-telugu"


class CompatibilityTests(unittest.TestCase):
    def test_preflight_matches_telugu_checkpoint_vocabulary(self):
        result = preflight(MODEL_DIR)
        self.assertEqual(result["vocab_size"], EXPECTED_VOCAB_SIZE)
        self.assertEqual(result["telugu_language_token_id"], EXPECTED_VOCAB_SIZE - 1)

    def test_language_tokens_are_within_checkpoint_vocabulary(self):
        tokenizer = MTLTokenizer(
            str(MODEL_DIR / "grapheme_mtl_merged_expanded_v1.json")
        )
        for language, text in (
            ("te", "నమస్కారం! ఇది తెలుగు పరీక్ష."),
            ("en", "Hello! This is an English test."),
        ):
            with self.subTest(language=language):
                token_ids = tokenizer.text_to_tokens(text, language_id=language)
                self.assertGreaterEqual(int(token_ids.min()), 0)
                self.assertLess(int(token_ids.max()), EXPECTED_VOCAB_SIZE)
                if language == "te":
                    self.assertIn(EXPECTED_VOCAB_SIZE - 1, token_ids.tolist()[0])

    def test_compatibility_scope_restores_exact_class_descriptor_on_error(self):
        original_descriptor = T3Config.__dict__["multilingual"]
        original_language = mtl_tts.SUPPORTED_LANGUAGES.get("te")
        self.assertEqual(T3Config.multilingual().text_tokens_dict_size, 2454)

        with self.assertRaisesRegex(RuntimeError, "scope probe"):
            with compatibility_scope():
                self.assertEqual(T3Config.multilingual().text_tokens_dict_size, 2521)
                self.assertEqual(mtl_tts.SUPPORTED_LANGUAGES["te"], "Telugu")
                raise RuntimeError("scope probe")

        self.assertIs(T3Config.__dict__["multilingual"], original_descriptor)
        self.assertEqual(T3Config.multilingual().text_tokens_dict_size, 2454)
        self.assertEqual(mtl_tts.SUPPORTED_LANGUAGES.get("te"), original_language)


if __name__ == "__main__":
    unittest.main()
