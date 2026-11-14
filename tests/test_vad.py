import unittest

import numpy as np

from agent.vad import VadConfig, VadSegmenter, float_to_pcm16, pcm16_to_float


SAMPLE_RATE = 16_000
FRAME_SAMPLES = 320  # 20 ms
SPEECH = np.full(FRAME_SAMPLES, 0.09, dtype=np.float32)
SILENCE = np.zeros(FRAME_SAMPLES, dtype=np.float32)


def frames(*values):
    return b"".join(float_to_pcm16(value) for value in values)


def config(**overrides):
    values = dict(
        sample_rate=SAMPLE_RATE,
        frame_ms=20,
        speech_dbfs=-42.0,
        silence_ms=100,
        min_speech_ms=40,
        max_utterance_seconds=5.0,
        pre_roll_ms=40,
        tail_ms=20,
    )
    values.update(overrides)
    return VadConfig(**values)


class VadConfigTests(unittest.TestCase):
    def test_frame_derived_values(self):
        cfg = config()
        self.assertEqual(cfg.frame_samples, 320)
        self.assertEqual(cfg.frame_bytes, 640)
        self.assertEqual(cfg.min_speech_frames, 2)
        self.assertEqual(cfg.silence_frames, 5)
        self.assertEqual(cfg.pre_roll_frames, 2)

    def test_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            config(sample_rate=0)
        with self.assertRaises(ValueError):
            config(frame_ms=15)
        with self.assertRaises(ValueError):
            config(frame_ms=110)
        with self.assertRaises(ValueError):
            config(silence_ms=20)
        with self.assertRaises(ValueError):
            config(min_speech_ms=10)
        with self.assertRaises(ValueError):
            config(max_utterance_seconds=0)


class PcmConversionTests(unittest.TestCase):
    def test_round_trip(self):
        original = np.linspace(-0.8, 0.8, 1000, dtype=np.float32)
        restored = pcm16_to_float(float_to_pcm16(original))
        self.assertEqual(restored.shape, original.shape)
        self.assertLess(float(np.max(np.abs(restored - original))), 1e-3)

    def test_clips_out_of_range(self):
        data = float_to_pcm16(np.array([-3.0, 3.0], dtype=np.float32))
        values = np.frombuffer(data, dtype="<i2")
        self.assertEqual(values.tolist(), [-32767, 32767])

    def test_rejects_odd_byte_count(self):
        with self.assertRaises(ValueError):
            pcm16_to_float(b"\x00\x00\x01")


class VadSegmenterTests(unittest.TestCase):
    def test_speech_start_and_endpoint(self):
        vad = VadSegmenter(config())
        self.assertEqual(vad.feed_pcm16(frames(SILENCE, SILENCE)), [])
        events = vad.feed_pcm16(frames(SPEECH, SPEECH))
        self.assertEqual([event.kind for event in events], ["speech_start"])
        self.assertTrue(vad.active)

        events = vad.feed_pcm16(frames(*([SILENCE] * 5)))
        self.assertEqual([event.kind for event in events], ["utterance"])
        self.assertFalse(vad.active)
        self.assertGreater(events[0].duration_seconds, 0)
        self.assertEqual(len(events[0].audio) % 2, 0)

    def test_pre_roll_is_included(self):
        vad = VadSegmenter(config(pre_roll_ms=60))
        events = vad.feed_pcm16(frames(SILENCE, SPEECH, SPEECH))
        self.assertEqual([event.kind for event in events], ["speech_start"])
        events = vad.feed_pcm16(frames(*([SILENCE] * 5)))
        audio = pcm16_to_float(events[0].audio)
        # The pre-roll keeps the frame of silence before speech onset.  The
        # long endpointing silence is trimmed to one frame, which is then
        # dropped when speech frames precede it.
        self.assertEqual(audio.size, 3 * FRAME_SAMPLES)
        self.assertTrue(np.all(audio[:FRAME_SAMPLES] == 0))
        self.assertGreater(float(np.max(np.abs(audio[FRAME_SAMPLES:]))), 0)

    def test_accepts_arbitrary_byte_boundaries(self):
        data = frames(SILENCE, SPEECH, SPEECH, *([SILENCE] * 5))
        vad = VadSegmenter(config())
        events = []
        for start in range(0, len(data), 317):
            events.extend(vad.feed_pcm16(data[start:start + 317]))
        self.assertEqual([event.kind for event in events], ["speech_start", "utterance"])

    def test_flush_ends_active_utterance(self):
        vad = VadSegmenter(config())
        vad.feed_pcm16(frames(SPEECH, SPEECH))
        self.assertTrue(vad.active)
        events = vad.flush()
        self.assertEqual([event.kind for event in events], ["utterance"])
        self.assertFalse(vad.active)
        self.assertGreater(events[0].duration_seconds, 0)
        self.assertEqual(vad.flush(), [])

    def test_flush_is_noop_when_idle(self):
        vad = VadSegmenter(config())
        self.assertEqual(vad.flush(), [])

    def test_flush_pads_partial_frame(self):
        vad = VadSegmenter(config())
        vad.feed_pcm16(frames(SPEECH, SPEECH))
        self.assertTrue(vad.active)
        events = vad.flush()
        self.assertEqual(len(events), 1)
        self.assertEqual(len(events[0].audio) % 2, 0)

    def test_max_utterance_forces_endpoint(self):
        vad = VadSegmenter(
            config(max_utterance_seconds=0.1, silence_ms=10_000, tail_ms=0)
        )
        events = vad.feed_pcm16(frames(*([SPEECH] * 2)))
        self.assertEqual([event.kind for event in events], ["speech_start"])
        events = vad.feed_pcm16(frames(*([SPEECH] * 4)))
        self.assertEqual([event.kind for event in events], ["utterance"])
        self.assertFalse(vad.active)

    def test_reset_discards_state(self):
        vad = VadSegmenter(config())
        vad.feed_pcm16(frames(SPEECH, SPEECH, SILENCE))
        self.assertTrue(vad.active)
        vad.reset()
        self.assertFalse(vad.active)
        self.assertEqual(vad.flush(), [])

    def test_empty_feed_is_noop(self):
        vad = VadSegmenter(config())
        self.assertEqual(vad.feed_pcm16(b""), [])

    def test_start_is_debounced(self):
        vad = VadSegmenter(config())
        self.assertEqual(vad.feed_pcm16(frames(SPEECH)), [])
        self.assertFalse(vad.active)

    def test_short_burst_does_not_trigger(self):
        vad = VadSegmenter(config(silence_ms=60))
        self.assertEqual(vad.feed_pcm16(frames(SPEECH, SILENCE, SILENCE, SILENCE)), [])
        self.assertFalse(vad.active)


if __name__ == "__main__":
    unittest.main()
