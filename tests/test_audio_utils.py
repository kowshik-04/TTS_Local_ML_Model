import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from tts.audio_utils import (
    AudioValidationError,
    clear_generated_outputs,
    prepared_reference,
    save_outputs,
    validate_audio,
)


class AudioUtilsTests(unittest.TestCase):
    def test_rejects_silent_recording(self):
        with self.assertRaises(AudioValidationError):
            validate_audio(np.zeros(24_000, dtype=np.float32), 24_000)

    def test_resamples_stereo_source_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "reference.wav"
            rate = 48_000
            samples = np.arange(rate * 2, dtype=np.float32) / rate
            stereo = np.column_stack((0.1 * np.sin(2 * np.pi * 220 * samples),) * 2)
            sf.write(source, stereo, rate)
            before = source.read_bytes()
            with prepared_reference(source) as (working_copy, info):
                converted, converted_rate = sf.read(working_copy)
                self.assertEqual(converted_rate, 24_000)
                self.assertEqual(converted.ndim, 1)
                self.assertEqual(len(converted), 48_000)
                self.assertEqual(info.duration_seconds, 2.0)
            self.assertEqual(source.read_bytes(), before)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for output-format tests")
    def test_decodes_mp3_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "reference.wav"
            mp3 = Path(directory) / "reference.mp3"
            samples = 0.1 * np.sin(np.linspace(0, 800, 48_000)).astype(np.float32)
            sf.write(source, samples, 48_000)
            subprocess.run(
                [shutil.which("ffmpeg"), "-nostdin", "-v", "error", "-y", "-i",
                 str(source), "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3)],
                check=True,
            )
            with prepared_reference(mp3) as (working_copy, info):
                self.assertEqual(sf.info(working_copy).samplerate, 24_000)
                self.assertEqual(info.duration_seconds, 1.0)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for output-format tests")
    def test_wav_mp3_export_and_scoped_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            wav_path, mp3_path = save_outputs(
                0.1 * np.sin(np.linspace(0, 100, 24_000)).astype(np.float32),
                24_000,
                output_dir,
            )
            self.assertTrue(wav_path.is_file())
            self.assertTrue(mp3_path.is_file())
            sf.info(wav_path)
            user_file = output_dir / "reference.wav"
            user_file.write_bytes(b"keep")
            self.assertEqual(clear_generated_outputs(output_dir), 2)
            self.assertTrue(user_file.is_file())


if __name__ == "__main__":
    unittest.main()
