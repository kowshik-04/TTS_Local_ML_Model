import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from tts.voice_store import VoiceError, VoiceStore


def _write_reference(path: Path, seconds: float = 3.2, rate: int = 48_000) -> Path:
    samples = np.arange(int(rate * seconds), dtype=np.float32) / rate
    audio = 0.08 * np.sin(2 * np.pi * 180 * samples)
    sf.write(path, audio.astype(np.float32), rate, subtype="PCM_16")
    return path


class VoiceStoreTests(unittest.TestCase):
    def make_store(self, directory: str) -> VoiceStore:
        return VoiceStore(Path(directory) / "voices", min_seconds=3.0, max_seconds=12.0)

    def test_create_prepares_24khz_profile_and_persists_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            source = _write_reference(Path(directory) / "source.wav")
            profile = store.create(
                "My Voice", source, "నమస్కారం, ఇది పరీక్ష.",
                consent=True,
                owner_key_id="owner-a",
            )
            self.assertTrue(profile.voice_id.startswith("v_my_voice_"))
            self.assertEqual(profile.sample_rate, 24_000)
            self.assertLessEqual(profile.duration_seconds, 12.0)
            self.assertEqual(profile.transcript, "నమస్కారం, ఇది పరీక్ష.")
            self.assertEqual(profile.owner_key_id, "owner-a")
            self.assertTrue(profile.valid)
            self.assertTrue((profile.directory / "profile.json").is_file())
            info = sf.info(profile.wav_path)
            self.assertEqual(info.samplerate, 24_000)
            self.assertEqual(info.channels, 1)

            loaded = store.get(profile.voice_id, owner_key_id="owner-a")
            self.assertEqual(loaded.fingerprint, profile.fingerprint)
            self.assertIn("My Voice", [item["name"] for item in store.list(owner_key_id="owner-a")])

    def test_consent_and_transcript_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            source = _write_reference(Path(directory) / "source.wav")
            with self.assertRaisesRegex(VoiceError, "consent"):
                store.create("Voice", source, "text")
            with self.assertRaisesRegex(VoiceError, "transcript"):
                store.create("Voice", source, "   ", consent=True)

    def test_reference_longer_than_alignment_limit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            source = _write_reference(Path(directory) / "long.wav", seconds=13.0)
            with self.assertRaisesRegex(VoiceError, "safe maximum"):
                store.create("Long", source, "text", consent=True)

    def test_duplicate_is_reused_for_the_same_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            source = _write_reference(Path(directory) / "source.wav")
            first = store.create("Voice", source, "same words", consent=True, owner_key_id="a")
            second = store.create("Voice copy", source, " same words ", consent=True, owner_key_id="a")
            self.assertEqual(first.voice_id, second.voice_id)
            third = store.create("Voice", source, "same words", consent=True, owner_key_id="b")
            self.assertNotEqual(first.voice_id, third.voice_id)

    def test_ownership_is_enforced_but_admins_can_access(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            source = _write_reference(Path(directory) / "source.wav")
            profile = store.create("Owner voice", source, "text", consent=True, owner_key_id="a")
            self.assertEqual(store.list(owner_key_id="b"), [])
            with self.assertRaises(VoiceError):
                store.get(profile.voice_id, owner_key_id="b")
            self.assertEqual(
                store.get(profile.voice_id, owner_key_id="b", allow_admin=True).voice_id,
                profile.voice_id,
            )
            with self.assertRaises(VoiceError):
                store.delete(profile.voice_id, owner_key_id="b")
            store.delete(profile.voice_id, owner_key_id="b", allow_admin=True)
            self.assertFalse(profile.directory.exists())

    def test_malformed_or_foreign_profiles_are_not_listed_or_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "voices"
            root.mkdir()
            malformed = root / "v_malformed_deadbeef"
            malformed.mkdir()
            (malformed / "profile.json").write_text(
                json.dumps({"owner": "local-voice-studio", "version": 1}), encoding="utf-8"
            )
            foreign = root / "v_foreign_deadbeef"
            foreign.mkdir()
            (foreign / "profile.json").write_text(
                json.dumps({"owner": "another-app", "version": 1}), encoding="utf-8"
            )
            store = self.make_store(directory)
            self.assertEqual(store.list(), [])
            with self.assertRaises(VoiceError):
                store.get("v_malformed_deadbeef")
            with self.assertRaisesRegex(VoiceError, "not an app-owned"):
                store.delete("v_foreign_deadbeef")
            self.assertTrue(foreign.exists())

    def test_invalid_voice_ids_are_rejected_before_touching_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            for voice_id in ("../outside", "a/b", "has space", "", ".hidden"):
                with self.assertRaises(VoiceError):
                    store.get(voice_id)

    def test_missing_source_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            with self.assertRaisesRegex(VoiceError, "not found"):
                store.create("Voice", Path(directory) / "missing.wav", "text", consent=True)


if __name__ == "__main__":
    unittest.main()
