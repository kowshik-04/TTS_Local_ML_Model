import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from tts.postprocess import (
    PostProcessConfig,
    from_pcm16,
    postprocess_speech,
    postprocess_stream_chunk,
    to_pcm16,
)


def _tone(seconds=2.0, rate=24_000, frequency=220.0, amplitude=0.05):
    samples = np.arange(int(rate * seconds), dtype=np.float32) / rate
    return (amplitude * np.sin(2 * np.pi * frequency * samples)).astype(np.float32)


class PostProcessTests(unittest.TestCase):
    def test_output_is_finite_peak_limited_and_tail_padded(self):
        audio = _tone()
        output = postprocess_speech(audio, 24_000, PostProcessConfig())
        self.assertEqual(output.ndim, 1)
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(np.isfinite(output).all())
        peak_limit = 10 ** (-1.5 / 20.0)
        self.assertLessEqual(float(np.max(np.abs(output))), peak_limit + 1e-6)
        self.assertGreater(len(output), len(audio) * 0.9)
        self.assertLess(len(output), len(audio) + 24_000 * 0.2)

    def test_dc_offset_is_removed(self):
        audio = _tone(1.0, amplitude=0.02) + 0.1
        output = postprocess_speech(audio, 24_000)
        self.assertLess(abs(float(np.mean(output))), 0.001)

    def test_silence_is_rejected(self):
        with self.assertRaises(ValueError):
            postprocess_speech(np.zeros(24_000, dtype=np.float32), 24_000)

    def test_stereo_input_is_downmixed(self):
        mono = _tone(1.0)
        stereo = np.column_stack((mono, mono))
        output = postprocess_speech(stereo, 24_000)
        self.assertEqual(output.ndim, 1)
        self.assertGreater(float(np.max(np.abs(output))), 0.0)

    def test_stream_chunk_does_not_add_tail(self):
        audio = _tone(0.4)
        output = postprocess_stream_chunk(audio, 24_000)
        self.assertLessEqual(len(output), len(audio))
        self.assertLessEqual(float(np.max(np.abs(output))), 10 ** (-1.5 / 20.0) + 1e-6)

    def test_pcm16_round_trip(self):
        audio = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        decoded = from_pcm16(to_pcm16(audio))
        self.assertEqual(decoded.dtype, np.float32)
        np.testing.assert_allclose(decoded, audio, atol=1e-4)

    def test_pcm16_rejects_odd_length(self):
        with self.assertRaises(ValueError):
            from_pcm16(b"\x00")


if __name__ == "__main__":
    unittest.main()
