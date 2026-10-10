import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from tts.history import APP_OWNER, delete_item, discover_history, read_metadata, write_metadata
from tts.library import ReferenceError, discover_references, next_reference_number, save_reference
from tts.health import collect_health


def sine(path, seconds=4, rate=24000):
    t = np.arange(int(seconds * rate)) / rate
    sf.write(path, (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"), rate)


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_discover_pairs_and_issues(self):
        sine(self.dir / "01_ok.wav")
        (self.dir / "01_ok.txt").write_text("నమస్కారం", encoding="utf-8")
        sine(self.dir / "02_notext.wav")
        (self.dir / "03_bad.wav").write_bytes(b"junk")
        (self.dir / "03_bad.txt").write_text("x", encoding="utf-8")
        (self.dir / "processed").mkdir()
        sine(self.dir / "processed" / "01_ok.wav")
        refs = {r.stem: r for r in discover_references(self.dir)}
        self.assertEqual(set(refs), {"01_ok", "02_notext", "03_bad"})
        self.assertTrue(refs["01_ok"].valid)
        self.assertEqual(refs["01_ok"].transcript, "నమస్కారం")
        self.assertFalse(refs["02_notext"].valid)
        self.assertFalse(refs["03_bad"].valid)

    def test_save_never_overwrites(self):
        sine(self.dir / "01_a.wav")
        (self.dir / "01_a.txt").write_text("a", encoding="utf-8")
        src = self.dir / "src.wav"
        sine(src)
        self.assertEqual(next_reference_number(self.dir), 2)
        info = save_reference(self.dir, "New Voice", src, "తెలుగు")
        self.assertTrue(info.wav_path.name.startswith("02_"))
        self.assertEqual((self.dir / "01_a.txt").read_text(encoding="utf-8"), "a")
        with self.assertRaises(ReferenceError):
            save_reference(self.dir, "x", src, "")


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_legacy_and_owned(self):
        sine(self.dir / "voice_legacy.wav")
        sine(self.dir / "voice_new.wav")
        write_metadata(self.dir / "voice_new.wav", {"text": "తెలుగు"})
        items = {i.stem: i for i in discover_history(self.dir)}
        self.assertTrue(items["voice_legacy"].legacy)
        self.assertFalse(items["voice_new"].legacy)
        self.assertEqual(items["voice_new"].text, "తెలుగు")
        self.assertEqual(len(discover_history(self.dir, "తెలుగు")), 1)

    def test_sidecar_not_overwritten_and_malformed(self):
        sine(self.dir / "voice_a.wav")
        write_metadata(self.dir / "voice_a.wav", {"text": "one"})
        with self.assertRaises(Exception):
            write_metadata(self.dir / "voice_a.wav", {"text": "two"})
        self.assertEqual(read_metadata(self.dir / "voice_a.json")[0]["text"], "one")
        (self.dir / "voice_b.json").write_text("{bad", encoding="utf-8")
        self.assertTrue(read_metadata(self.dir / "voice_b.json")[1])

    def test_delete_rules(self):
        sine(self.dir / "voice_legacy.wav")
        sine(self.dir / "voice_new.wav")
        write_metadata(self.dir / "voice_new.wav", {"text": "x"})
        with self.assertRaises(PermissionError):
            delete_item(self.dir, "voice_legacy")
        with self.assertRaises(ValueError):
            delete_item(self.dir, "../voice_new")
        (self.dir / "voice_foreign.json").write_text(json.dumps({"owner": "other"}), encoding="utf-8")
        with self.assertRaises(PermissionError):
            delete_item(self.dir, "voice_foreign")
        self.assertTrue((self.dir / "voice_legacy.wav").exists())
        delete_item(self.dir, "voice_new")
        self.assertFalse((self.dir / "voice_new.wav").exists())
        self.assertTrue((self.dir / "voice_legacy.wav").exists())


class HealthTests(unittest.TestCase):
    def test_rows_without_model(self):
        with tempfile.TemporaryDirectory() as d:
            rows = collect_health(Path(d))
        self.assertTrue(rows)
        for label, value, level in rows:
            self.assertIn(level, {"ok", "warn", "error", "info"})


if __name__ == "__main__":
    unittest.main()
