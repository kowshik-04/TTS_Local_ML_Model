import unittest

from agent.session import SentenceBuffer


class SentenceBufferTests(unittest.TestCase):
    def test_first_phrase_can_end_at_comma(self):
        buffer = SentenceBuffer(first_min_chars=4)
        self.assertEqual(buffer.feed("Hello, "), ["Hello,"])
        self.assertEqual(buffer.feed("this is a test. Next"), ["this is a test."])
        self.assertEqual(buffer.feed(" sentence."), ["Next sentence."])

    def test_later_commas_do_not_split(self):
        buffer = SentenceBuffer(first_min_chars=4)
        self.assertEqual(buffer.feed("Hello, "), ["Hello,"])
        self.assertEqual(buffer.feed("world, this is still one phrase."), ["world, this is still one phrase."])

    def test_short_first_comma_is_ignored(self):
        buffer = SentenceBuffer(first_min_chars=10)
        self.assertEqual(buffer.feed("Hi, there."), ["Hi, there."])

    def test_abbreviations_are_not_sentence_ends(self):
        buffer = SentenceBuffer(min_chars=1, first_min_chars=1)
        self.assertEqual(buffer.feed("Dr. Rao is here. Next."), ["Dr. Rao is here.", "Next."])
        self.assertEqual(buffer.feed(" No. 5 is ready."), ["No. 5 is ready."])

    def test_inline_decimal_is_not_a_boundary(self):
        buffer = SentenceBuffer()
        self.assertEqual(buffer.feed("The value is 3.14 today."), ["The value is 3.14 today."])

    def test_telugu_danda_and_question(self):
        buffer = SentenceBuffer(min_chars=4)
        self.assertEqual(buffer.feed("నమస్కారం। ఎలా ఉన్నారు?"), ["నమస్కారం।", "ఎలా ఉన్నారు?"])

    def test_quotes_and_brackets_are_kept(self):
        buffer = SentenceBuffer(min_chars=4)
        self.assertEqual(buffer.feed('He said "come in." Then left.'), ['He said "come in."', "Then left."])

    def test_max_length_splits_on_word_boundary(self):
        buffer = SentenceBuffer(min_chars=4, max_chars=20, first_min_chars=4)
        self.assertEqual(buffer.feed("one two three four five"), ["one two three four"])
        self.assertEqual(buffer.flush(), ["five"])

    def test_long_unbroken_token_is_hard_split(self):
        buffer = SentenceBuffer(min_chars=4, max_chars=20)
        self.assertEqual(buffer.feed("abcdefghijklmnopqrstuvwxyz"), ["abcdefghijklmnopqrst"])
        self.assertEqual(buffer.flush(), ["uvwxyz"])

    def test_short_sentences_wait_for_min_chars(self):
        buffer = SentenceBuffer(min_chars=8)
        self.assertEqual(buffer.feed("Hi. There is more."), ["Hi. There is more."])

    def test_flush_returns_and_clears_remainder(self):
        buffer = SentenceBuffer()
        self.assertEqual(buffer.feed("unfinished phrase"), [])
        self.assertEqual(buffer.flush(), ["unfinished phrase"])
        self.assertEqual(buffer.flush(), [])
        self.assertEqual(buffer.buffered, "")

    def test_invalid_limits(self):
        with self.assertRaises(ValueError):
            SentenceBuffer(min_chars=0)
        with self.assertRaises(ValueError):
            SentenceBuffer(min_chars=10, max_chars=5)

    def test_token_by_token_streaming(self):
        buffer = SentenceBuffer(min_chars=4, first_min_chars=4)
        output = []
        for token in ["నమ", "స్కా", "రం, ", "ఎలా ", "ఉన్నా", "రు?", " బాగు", "న్నాను."]:
            output.extend(buffer.feed(token))
        output.extend(buffer.flush())
        self.assertEqual(output, ["నమస్కారం,", "ఎలా ఉన్నారు?", "బాగున్నాను."])


if __name__ == "__main__":
    unittest.main()
