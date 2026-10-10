import unittest
from unittest.mock import patch

from tts.chunking import split_text
from tts.text_normalizer import normalize_text
from tts.transliteration import romanized_telugu_to_script


class FakeTransliterator:
    def translit_sentence(self, text):
        return {"te": f"తెలుగు({text})"}


class TextProcessingTests(unittest.TestCase):
    def test_normalizer_preserves_words_and_paragraphs(self):
        text = "  Linux   Kubernetes  \n\n  తెలుగు వాక్యం  "
        self.assertEqual(normalize_text(text), "Linux Kubernetes\n\nతెలుగు వాక్యం")

    def test_chunking_respects_utf8_limit_and_keeps_words(self):
        text = " ".join(["తెలుగు"] * 90) + ".\n\nNext paragraph."
        chunks = split_text(text, max_bytes=90)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(chunk.encode("utf-8")) <= 90 for chunk, _ in chunks))
        self.assertEqual(" ".join(chunk for chunk, _ in chunks).replace(" .", "."), text.replace("\n\n", " "))
        self.assertEqual(chunks[-1][1], 0.0)

    def test_romanized_conversion_preserves_english_and_spacing(self):
        with patch("tts.transliteration._get_engine", return_value=FakeTransliterator()):
            result = romanized_telugu_to_script("Nenu Red Hat lo, Linux vadutanu.")
        self.assertEqual(result, "తెలుగు(Nenu) Red Hat తెలుగు(lo), Linux తెలుగు(vadutanu).")

    def test_empty_romanized_text_is_rejected(self):
        with self.assertRaises(ValueError):
            romanized_telugu_to_script(" \n ")


if __name__ == "__main__":
    unittest.main()
