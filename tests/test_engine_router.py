import unittest

from tts.config import Settings
from tts.engine_router import (
    EngineRouter,
    LANG_ENGLISH,
    LANG_MIXED,
    LANG_ROMANIZED,
    LANG_TELUGU,
)


class FakeEngine:
    engine_name = "indicf5"

    def __init__(self, *, load_error=None):
        self.calls = []
        self.load_calls = 0
        self.load_error = load_error

    def load(self):
        self.load_calls += 1
        if self.load_error is not None:
            raise self.load_error
        return True

    def warmup(self):
        self.load_calls += 1
        if self.load_error is not None:
            raise self.load_error
        return self.info()

    def generate(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return [0.0], 24_000

    def info(self):
        return {"engine": self.engine_name, "loaded": True, "device": "cpu"}


class FakeChatterbox(FakeEngine):
    engine_name = "chatterbox"


def make_router(*, chatterbox=None):
    return EngineRouter(
        FakeEngine(),
        settings=Settings(enable_chatterbox=chatterbox is not None),
        chatterbox=chatterbox,
    )


class EngineRouterTests(unittest.TestCase):
    def test_telugu_and_tenglish_default_to_indicf5_even_with_chatterbox(self):
        chatterbox = FakeChatterbox()
        router = make_router(chatterbox=chatterbox)
        self.assertEqual(router.resolve_engine(LANG_TELUGU, "auto"), "indicf5")
        self.assertEqual(router.resolve_engine(LANG_MIXED, "auto"), "indicf5")
        self.assertEqual(router.resolve_engine(LANG_TELUGU, "indicf5"), "indicf5")
        self.assertEqual(router.resolve_engine(LANG_MIXED, "indicf5"), "indicf5")

    def test_tenglish_can_explicitly_use_chatterbox(self):
        chatterbox = FakeChatterbox()
        router = make_router(chatterbox=chatterbox)
        self.assertEqual(router.resolve_engine(LANG_MIXED, "chatterbox"), "chatterbox")
        self.assertEqual(router.resolve_engine(LANG_TELUGU, "chatterbox"), "chatterbox")

    def test_english_auto_uses_chatterbox_only_when_enabled(self):
        chatterbox = FakeChatterbox()
        router = make_router(chatterbox=chatterbox)
        self.assertEqual(router.resolve_engine(LANG_ENGLISH, "auto"), "chatterbox")

        disabled = make_router()
        with self.assertRaisesRegex(ValueError, "not enabled"):
            disabled.resolve_engine(LANG_ENGLISH, "auto")

    def test_explicit_chatterbox_without_enablement_is_rejected(self):
        router = make_router()
        with self.assertRaisesRegex(ValueError, "not enabled"):
            router.resolve_engine(LANG_TELUGU, "chatterbox")

    def test_indicf5_never_claims_english(self):
        router = make_router(chatterbox=FakeChatterbox())
        with self.assertRaisesRegex(ValueError, "English-only"):
            router.resolve_engine(LANG_ENGLISH, "indicf5")

    def test_romanized_requires_preview_and_unknown_engine_is_rejected(self):
        router = make_router()
        with self.assertRaisesRegex(ValueError, "preview"):
            router.generate(LANG_ROMANIZED, "ref.wav", "ref", "text")
        with self.assertRaisesRegex(ValueError, "Unsupported engine"):
            router.resolve_engine(LANG_TELUGU, "cloud")

    def test_ensure_ready_loads_selected_engine_and_propagates_failure(self):
        chatterbox = FakeChatterbox()
        router = make_router(chatterbox=chatterbox)
        router.ensure_ready("indicf5")
        router.ensure_ready("chatterbox")
        self.assertEqual(router.indicf5.load_calls, 1)
        self.assertEqual(chatterbox.load_calls, 1)

        broken = FakeChatterbox(load_error=RuntimeError("weights missing"))
        broken_router = make_router(chatterbox=broken)
        with self.assertRaisesRegex(RuntimeError, "weights missing"):
            broken_router.ensure_ready("chatterbox")

    def test_capabilities_keep_indicf5_as_tenglish_default(self):
        router = make_router(chatterbox=FakeChatterbox())
        rows = {row["language"]: row for row in router.capabilities()}
        self.assertEqual(rows[LANG_MIXED]["engine"], "indicf5")
        self.assertEqual(rows[LANG_MIXED]["optional_engine"], "chatterbox")

    def test_warmup_respects_an_explicit_engine_and_rejects_disabled_chatterbox(self):
        chatterbox = FakeChatterbox()
        router = make_router(chatterbox=chatterbox)
        self.assertEqual(router.warmup("chatterbox"), {"chatterbox": chatterbox.info()})
        self.assertEqual(router.indicf5.load_calls, 0)
        self.assertEqual(chatterbox.load_calls, 1)

        disabled = make_router()
        with self.assertRaisesRegex(RuntimeError, "not enabled"):
            disabled.warmup("chatterbox")


if __name__ == "__main__":
    unittest.main()
