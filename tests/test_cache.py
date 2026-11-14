import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from tts.cache import AudioCache, make_cache_key


def _key(text="hello", voice="v1", options=None):
    return make_cache_key(
        engine="indicf5",
        voice_fingerprint=voice,
        text=text,
        options=options or {"nfe_step": 32},
        sample_rate=24_000,
    )


class AudioCacheTests(unittest.TestCase):
    def test_key_is_deterministic_and_sensitive_to_inputs(self):
        first = _key()
        self.assertEqual(first, _key())
        self.assertNotEqual(first, _key(text="hello!"))
        self.assertNotEqual(first, _key(voice="v2"))
        self.assertNotEqual(first, _key(options={"nfe_step": 16}))

    def test_put_get_and_clear_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = AudioCache(directory, max_bytes=1_000_000)
            key = _key()
            audio = np.sin(np.linspace(0, 20, 24_000)).astype(np.float32) * 0.1
            cache.put(
                key,
                audio,
                24_000,
                engine="indicf5",
                voice_fingerprint="v1",
                text_chars=5,
            )
            loaded, entry = cache.get(key)
            self.assertEqual(entry.sample_rate, 24_000)
            self.assertEqual(entry.engine, "indicf5")
            self.assertTrue(np.allclose(loaded, audio, atol=1e-5))
            self.assertEqual(cache.stats()["entries"], 1)
            self.assertGreaterEqual(cache.clear(), 2)
            self.assertIsNone(cache.get(key))

    def test_corrupt_entry_is_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = AudioCache(directory)
            key = _key()
            audio = np.ones(24_000, dtype=np.float32) * 0.05
            cache.put(key, audio, 24_000, engine="indicf5", voice_fingerprint="v1")
            json_path = Path(directory) / f"{key}.json"
            json_path.write_text("{not-json", encoding="utf-8")
            self.assertIsNone(cache.get(key))
            self.assertFalse((Path(directory) / f"{key}.wav").exists())

    def test_invalid_keys_cannot_escape_the_cache_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = AudioCache(directory)
            with self.assertRaises(ValueError):
                cache.get("../outside")
            with self.assertRaises(ValueError):
                cache.put(
                    "../../outside",
                    np.ones(100, dtype=np.float32) * 0.1,
                    24_000,
                    engine="indicf5",
                    voice_fingerprint="v1",
                )

    def test_eviction_keeps_total_size_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = np.ones(24_000, dtype=np.float32) * 0.05
            first = _key(text="first")
            second = _key(text="second")
            cache = AudioCache(directory, max_bytes=100_000)
            cache.put(first, audio, 24_000, engine="indicf5", voice_fingerprint="v1")
            time.sleep(0.02)
            cache.put(second, audio, 24_000, engine="indicf5", voice_fingerprint="v1")
            stats = cache.stats()
            self.assertLessEqual(stats["bytes"], 100_000)
            self.assertIsNotNone(cache.get(second))
            self.assertIsNone(cache.get(first))

    def test_rejects_empty_or_non_finite_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = AudioCache(directory)
            with self.assertRaises(ValueError):
                cache.put(_key(), np.zeros(0, dtype=np.float32), 24_000, engine="x", voice_fingerprint="v")
            with self.assertRaises(ValueError):
                cache.put(
                    _key(), np.array([0.1, np.nan], dtype=np.float32), 24_000,
                    engine="x", voice_fingerprint="v",
                )

    def test_sidecar_metadata_is_json(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = AudioCache(directory)
            key = _key()
            cache.put(
                key, np.ones(100, dtype=np.float32) * 0.1, 24_000,
                engine="indicf5", voice_fingerprint="v1", text_chars=5,
            )
            metadata = json.loads((Path(directory) / f"{key}.json").read_text())
            self.assertEqual(metadata["key"], key)
            self.assertEqual(metadata["text_chars"], 5)


if __name__ == "__main__":
    unittest.main()
