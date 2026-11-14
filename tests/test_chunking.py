import unittest

from tts.chunking import indicf5_max_bytes, split_for_indicf5, split_text


class ChunkingTests(unittest.TestCase):
    def test_splits_at_sentence_and_paragraph_boundaries(self):
        chunks = split_text("First sentence. Second sentence!\n\nThird paragraph?", max_bytes=400)
        self.assertEqual([chunk for chunk, _ in chunks], [
            "First sentence. Second sentence!",
            "Third paragraph?",
        ])
        self.assertGreater(chunks[0][1], 0.0)
        self.assertEqual(chunks[-1][1], 0.0)

    def test_abbreviations_do_not_create_sentence_breaks(self):
        chunks = split_text("Dr. Rao met Prof. Rao at 5 p.m. today.", max_bytes=400)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][0], "Dr. Rao met Prof. Rao at 5 p.m. today.")

    def test_utf8_byte_limit_is_respected_for_telugu(self):
        text = " ".join(["నమస్కారం"] * 40)
        chunks = split_text(text, max_bytes=64)
        self.assertGreater(len(chunks), 1)
        for chunk, _ in chunks:
            self.assertLessEqual(len(chunk.encode("utf-8")), 64)
            self.assertTrue(chunk)

    def test_long_words_fail_instead_of_exceeding_the_budget(self):
        with self.assertRaises(ValueError):
            split_text("x" * 100, max_bytes=32)

    def test_indicf5_budget_is_bounded(self):
        self.assertGreaterEqual(indicf5_max_bytes("వాక్యం", 10.0), 64)
        self.assertLessEqual(indicf5_max_bytes("short", 1.0, speed=2.0), 420)
        chunks = split_for_indicf5(
            "ఒక వాక్యం. రెండవ వాక్యం. మూడవ వాక్యం.",
            "ఇది ఒక సుదీర్ఘమైన సూచన వాక్యం.",
            10.0,
            max_bytes=64,
        )
        self.assertTrue(chunks)
        for chunk, _ in chunks:
            self.assertLessEqual(len(chunk.encode("utf-8")), 64)

    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(split_text("   "), [])


if __name__ == "__main__":
    unittest.main()
