import unittest

from eval.benchmark import Trial, _error_details, _percentile, _stats
from eval.corpus import CASES, ENGLISH, MIXED, TELUGU, select_cases


class CorpusTests(unittest.TestCase):
    def test_ids_are_unique_and_cover_all_languages(self):
        ids = [case.case_id for case in CASES]
        self.assertEqual(len(ids), len(set(ids)))
        languages = {case.language for case in CASES}
        self.assertEqual(languages, {TELUGU, MIXED, ENGLISH})
        self.assertGreaterEqual(len([case for case in CASES if case.language == TELUGU]), 30)

    def test_select_cases(self):
        self.assertEqual(len(select_cases(["te001", "mix001"])), 2)
        with self.assertRaises(ValueError):
            select_cases(["missing"])


class BenchmarkMathTests(unittest.TestCase):
    def test_percentile(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertEqual(_percentile(values, 0.5), 30.0)
        self.assertEqual(_percentile(values, 0.95), 48.0)
        self.assertIsNone(_percentile([], 0.5))

    def test_stats_ignore_failures_for_latency(self):
        trials = [
            Trial("a", "conversation", TELUGU, "sync", 1, True, wall_ms=100, ttfb_ms=100, audio_ms=2000, rtf=1.0),
            Trial("a", "conversation", TELUGU, "sync", 2, True, wall_ms=300, ttfb_ms=300, audio_ms=4000, rtf=2.0),
            Trial("b", "conversation", TELUGU, "sync", 1, False, error="boom"),
        ]
        stats = _stats(trials)
        self.assertEqual(stats["trials"], 3)
        self.assertEqual(stats["successful"], 2)
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(stats["wall_ms_p50"], 200.0)
        self.assertEqual(stats["audio_seconds"], 6.0)
        self.assertAlmostEqual(stats["audio_seconds_per_wall_second"], 6000 / 400)

    def test_error_details_parses_api_error(self):
        body = b'{"error":{"code":"rate_limited","message":"slow down"}}'
        self.assertEqual(_error_details(429, body), ("rate_limited", "slow down"))
        self.assertEqual(_error_details(500, b"oops")[1], "oops")


if __name__ == "__main__":
    unittest.main()
