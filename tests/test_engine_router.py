import unittest

from tts.engine_router import EngineRouter, LANG_ENGLISH, LANG_MIXED, LANG_TELUGU


class FakeEngine:
    def __init__(self):
        self.calls = []

    def generate(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return [0.0], 24_000


class EngineRouterTests(unittest.TestCase):
    def test_telugu_and_mixed_route_to_indicf5(self):
        engine = FakeEngine()
        router = EngineRouter(engine)
        for language in (LANG_TELUGU, LANG_MIXED):
            self.assertEqual(router.generate(language, "ref.wav", "వాక్యం", "నమస్కారం")[1], 24_000)
        self.assertEqual(len(engine.calls), 2)

    def test_english_fails_instead_of_falling_back_to_a_preset_voice(self):
        router = EngineRouter(FakeEngine())
        with self.assertRaisesRegex(ValueError, "not enabled"):
            router.generate(LANG_ENGLISH, "ref.wav", "hello", "hello")


if __name__ == "__main__":
    unittest.main()
