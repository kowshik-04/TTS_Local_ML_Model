import unittest

from tenglish import preview_tenglish, text_for_synthesis


class TenglishPreviewTests(unittest.TestCase):
    @staticmethod
    def fake_transliterator(words):
        mapping = {
            "nenu": "నేను",
            "chalaa": "చాలా",
            "baagunnaanu": "బాగున్నాను",
            "lo": "లో",
            "mariyu": "మరియు",
            "vadutanu": "వాడుతాను",
            "malli": "మళ్ళీ",
            "khana": "ఖానా",
        }
        return [mapping[word.lower()] for word in words]

    def test_transliterates_tenglish_and_preserves_english_terms(self):
        source = "Nenu Red Hat lo Linux mariyu Kubernetes vadutanu."
        result = preview_tenglish(source, self.fake_transliterator)
        self.assertEqual(result.source_text, source)
        self.assertEqual(
            result.converted_text,
            "నేను Red Hat లో Linux మరియు Kubernetes వాడుతాను.",
        )

    def test_long_vowels_aspiration_punctuation_numbers_and_repetition(self):
        source = "Nenu chalaa baagunnaanu, malli malli khana! 2026."
        result = preview_tenglish(source, self.fake_transliterator)
        self.assertEqual(
            result.converted_text,
            "నేను చాలా బాగున్నాను, మళ్ళీ మళ్ళీ ఖానా! 2026.",
        )

    def test_existing_telugu_unicode_is_preserved(self):
        source = "Nenu తెలుగు మళ్ళీ APIతో."
        result = preview_tenglish(source, self.fake_transliterator)
        self.assertEqual(result.converted_text, "నేను తెలుగు మళ్ళీ APIతో.")

    def test_names_acronyms_versions_and_technical_terms_are_preserved(self):
        source = "Kowshik, API v2 lo Kubernetes!"
        result = preview_tenglish(source, self.fake_transliterator)
        self.assertEqual(result.converted_text, "Kowshik, API v2 లో Kubernetes!")

    def test_empty_input_is_rejected_before_transliterator_runs(self):
        def unexpected_transliterator(words):
            self.fail("Translator should not run for empty input.")

        with self.assertRaises(ValueError):
            preview_tenglish(" \n\t", unexpected_transliterator)

    def test_punctuation_spacing_and_long_paragraph_are_not_dropped(self):
        source = "Nenu, chalaa! " + " ".join(["malli"] * 120) + "?"
        result = preview_tenglish(source, self.fake_transliterator)
        self.assertTrue(result.converted_text.startswith("నేను, చాలా! "))
        self.assertTrue(result.converted_text.endswith("?"))
        self.assertEqual(result.converted_text.count("మళ్ళీ"), 120)

    def test_missing_word_translation_fails_instead_of_silently_dropping(self):
        with self.assertRaisesRegex(RuntimeError, "omitted"):
            preview_tenglish("Nenu", lambda words: [])

    def test_exact_preview_text_is_selected_for_synthesis(self):
        edited_preview = "నేను, చాలా!"
        self.assertIs(text_for_synthesis("Nenu, chalaa!", edited_preview), edited_preview)
        self.assertEqual(text_for_synthesis("Nenu", None), "Nenu")


if __name__ == "__main__":
    unittest.main()
