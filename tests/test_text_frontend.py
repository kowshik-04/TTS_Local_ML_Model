import unittest

from tts.languages import LANG_ENGLISH, LANG_MIXED, LANG_ROMANIZED, LANG_TELUGU
from tts.text_frontend import (
    FrontendResult,
    number_to_english,
    number_to_telugu,
    prepare_speech_text,
    split_segments,
)


class TextFrontendTests(unittest.TestCase):
    def test_telugu_cardinals_use_indian_grouping(self):
        self.assertEqual(number_to_telugu(0), "సున్నా")
        self.assertEqual(number_to_telugu(21), "ఇరవై ఒకటి")
        self.assertEqual(number_to_telugu(105), "నూట ఐదు")
        self.assertEqual(number_to_telugu(1_000), "వెయ్యి")
        self.assertEqual(number_to_telugu(2_345), "రెండు వేల మూడు వందల నలభై ఐదు")
        self.assertEqual(number_to_telugu(100_000), "లక్ష")
        self.assertEqual(number_to_telugu(10_000_000), "కోటి")

    def test_english_cardinals(self):
        self.assertEqual(number_to_english(0), "zero")
        self.assertEqual(number_to_english(21), "twenty-one")
        self.assertEqual(number_to_english(105), "one hundred five")
        self.assertEqual(number_to_english(1_000_001), "one million one")

    def test_identifiers_are_not_read_as_quantities(self):
        result = prepare_speech_text(
            "COVID-19, GPT-4, 5G, MP3, H100 and B2B are identifiers.",
            LANG_ENGLISH,
        )
        for token in ("COVID-19", "GPT-4", "5G", "MP3", "H100", "B2B"):
            self.assertIn(token, result.text)
        self.assertNotIn("minus", result.text.lower())
        self.assertNotIn("grams", result.text.lower())
        self.assertNotIn("nine", result.text.lower())

    def test_units_percent_and_currency(self):
        result = prepare_speech_text("100 Mbps, 5 km and 3.5%", LANG_ENGLISH)
        self.assertIn("one hundred megabits per second", result.text)
        self.assertIn("five kilometers", result.text)
        self.assertIn("three point five percent", result.text)

        telugu = prepare_speech_text("5 కి.మీ కాదు; 5 km మరియు 12.5%", LANG_TELUGU)
        self.assertIn("ఐదు కిలోమీటర్లు", telugu.text)
        self.assertIn("పన్నెండు పాయింట్ ఐదు శాతం", telugu.text)

        rupees = prepare_speech_text("₹1,250.50", LANG_TELUGU)
        self.assertIn("రూపాయలు", rupees.text)
        self.assertIn("పైసలు", rupees.text)
        self.assertIn("వెయ్యి", rupees.text)

        english_rupees = prepare_speech_text("Rs. 500", LANG_ENGLISH)
        self.assertIn("five hundred rupees", english_rupees.text)

    def test_dates_and_times(self):
        english = prepare_speech_text("On 10/12/2024 at 3:05 pm.", LANG_ENGLISH)
        self.assertIn("December tenth two thousand twenty-four", english.text)
        self.assertIn("three oh five p m", english.text)

        telugu = prepare_speech_text("2024-12-10 న సాయంత్రం 8:05", LANG_TELUGU)
        self.assertIn("పది డిసెంబర్ రెండు వేల ఇరవై నాలుగు", telugu.text)
        self.assertIn("ఎనిమిది గంటల ఐదు నిమిషాలు", telugu.text)

        invalid_date = prepare_speech_text("99/99/2024", LANG_ENGLISH)
        self.assertNotIn("December", invalid_date.text)
        self.assertIn("ninety-nine", invalid_date.text)

    def test_ordinals_fractions_and_long_digit_runs(self):
        english = prepare_speech_text(
            "The 21st, 22nd, 112th and 3/4 values; call 9876543210.", LANG_ENGLISH
        )
        self.assertIn("twenty-first", english.text)
        self.assertIn("twenty-second", english.text)
        self.assertIn("one hundred twelfth", english.text)
        self.assertIn("three quarters", english.text)
        self.assertIn("nine eight seven six five four three two one zero", english.text)

        telugu = prepare_speech_text("21వ, 100వ, 3/4", LANG_TELUGU)
        self.assertIn("ఇరవై ఒకటవ", telugu.text)
        self.assertIn("వందవ", telugu.text)
        self.assertIn("ముప్పావు", telugu.text)

    def test_markup_urls_email_and_emoji_are_removed(self):
        result = prepare_speech_text(
            "# Heading\n**Namaste** [docs](https://example.test) "
            "https://example.test a@example.test 🎉",
            LANG_TELUGU,
        )
        self.assertTrue(result.replaced_markup)
        self.assertTrue(any("Markup" in warning for warning in result.warnings))
        self.assertTrue(any("Emoji" in warning for warning in result.warnings))
        self.assertIn("Namaste", result.text)
        self.assertIn("docs", result.text)
        self.assertIn("లింక్", result.text)
        self.assertIn("ఇమెయిల్", result.text)
        self.assertNotIn("https://", result.text)
        self.assertNotIn("**", result.text)
        self.assertNotIn("🎉", result.text)

    def test_warnings_are_explicit(self):
        romanized = prepare_speech_text("namaste ela unnaru", LANG_ROMANIZED)
        self.assertTrue(any("Romanized" in warning for warning in romanized.warnings))

        wrong_script = prepare_speech_text("hello world", LANG_TELUGU)
        self.assertTrue(any("no Telugu-script" in warning for warning in wrong_script.warnings))

        mixed = prepare_speech_text("hello, ఎలా ఉన్నారు?", LANG_MIXED)
        self.assertTrue(any("experimental" in warning for warning in mixed.warnings))
        self.assertFalse(mixed.text.endswith("."))

    def test_terminal_punctuation_and_paragraphs(self):
        result = prepare_speech_text("first line\n\nsecond line", LANG_ENGLISH)
        self.assertTrue(result.text.endswith("."))
        self.assertIn("\n\n", result.text)
        self.assertEqual(
            split_segments(result.text),
            ["first line.", "second line."],
        )

    def test_empty_and_unspeakable_text_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare_speech_text("   ", LANG_TELUGU)
        with self.assertRaises(ValueError):
            prepare_speech_text("🎉🎉", LANG_TELUGU)

    def test_result_is_immutable(self):
        result = prepare_speech_text("నమస్కారం", LANG_TELUGU)
        self.assertIsInstance(result, FrontendResult)
        with self.assertRaises(Exception):
            result.text = "changed"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
