import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from tests.support import FakeEngine, create_voice, make_service, tone
from tts.service import SpeechRequest, TTSError


class TTSServiceTests(unittest.TestCase):
    def test_synthesize_normalizes_postprocesses_and_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            service, engine = make_service(directory)
            voice = create_voice(service, directory)
            request = SpeechRequest(
                text="100 Mbps మరియు 5G", language="Telugu",
                voice_id=voice.voice_id, owner_key_id="owner-a",
            )
            first = service.synthesize(request)
            self.assertFalse(first.cache_hit)
            self.assertEqual(first.engine, "indicf5")
            self.assertEqual(first.voice_id, voice.voice_id)
            self.assertIn("వంద మెగాబిట్స్", first.normalized_text)
            self.assertIn("5G", first.normalized_text)
            self.assertTrue(np.isfinite(first.audio).all())
            self.assertEqual(first.sample_rate, 24_000)
            self.assertGreater(first.audio_seconds, 0.4)
            self.assertEqual(len(engine.calls), 1)

            second = service.synthesize(request)
            self.assertTrue(second.cache_hit)
            self.assertEqual(len(engine.calls), 1)
            np.testing.assert_allclose(first.audio, second.audio, atol=1e-5)

            third = service.synthesize(
                SpeechRequest(
                    text=request.text, language=request.language, voice_id=voice.voice_id,
                    owner_key_id="owner-a", use_cache=False,
                )
            )
            self.assertFalse(third.cache_hit)
            self.assertEqual(len(engine.calls), 2)

    def test_profiles_and_speed_reach_the_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            service, engine = make_service(directory)
            voice = create_voice(service, directory)
            request = SpeechRequest(
                text="నమస్కారం", language="Telugu", voice_id=voice.voice_id,
                owner_key_id="owner-a", profile="realtime", speed=1.2, use_cache=False,
            )
            result = service.synthesize(request)
            options = engine.calls[-1]["options"]
            self.assertEqual(options.nfe_step, 16)
            self.assertEqual(options.speed, 1.2)
            self.assertEqual(result.options.nfe_step, 16)

    def test_invalid_language_profile_and_speed_are_structured_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            service, _ = make_service(directory)
            voice = create_voice(service, directory)
            with self.assertRaisesRegex(TTSError, "Unsupported language") as caught:
                service.synthesize(
                    SpeechRequest(text="hello", language="Klingon", voice_id=voice.voice_id)
                )
            self.assertEqual(caught.exception.code, "invalid_language")
            with self.assertRaises(TTSError) as caught:
                service.synthesize(
                    SpeechRequest(text="నమస్కారం", language="Telugu", voice_id=voice.voice_id,
                                  profile="warp", owner_key_id="owner-a")
                )
            self.assertEqual(caught.exception.code, "invalid_profile")
            with self.assertRaises(TTSError) as caught:
                service.synthesize(
                    SpeechRequest(text="నమస్కారం", language="Telugu", voice_id=voice.voice_id,
                                  speed=3.0, owner_key_id="owner-a")
                )
            self.assertEqual(caught.exception.code, "invalid_speed")

    def test_text_limits_and_voice_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            service, _ = make_service(directory, max_text_chars=20)
            voice = create_voice(service, directory)
            with self.assertRaises(TTSError) as caught:
                service.synthesize(
                    SpeechRequest(text="x" * 21, language="English", voice_id=voice.voice_id)
                )
            self.assertEqual(caught.exception.code, "text_too_long")
            with self.assertRaises(TTSError) as caught:
                service.synthesize(SpeechRequest(text="నమస్కారం", voice_id=None))
            self.assertEqual(caught.exception.code, "voice_required")

    def test_voice_ownership_is_enforced_by_the_service(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            service, _ = make_service(directory)
            voice = create_voice(service, directory, owner="owner-a")
            with self.assertRaises(TTSError) as caught:
                service.synthesize(
                    SpeechRequest(text="నమస్కారం", voice_id=voice.voice_id, owner_key_id="owner-b")
                )
            self.assertEqual(caught.exception.code, "voice_not_found")

    def test_custom_reference_requires_a_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            service, _ = make_service(directory)
            source = directory / "custom.wav"
            sf.write(source, tone(3.0), 24_000, subtype="PCM_16")
            with self.assertRaises(TTSError) as caught:
                service.synthesize(
                    SpeechRequest(text="నమస్కారం", reference_path=str(source))
                )
            self.assertEqual(caught.exception.code, "reference_transcript_required")
            result = service.synthesize(
                SpeechRequest(
                    text="నమస్కారం", reference_path=str(source),
                    reference_transcript="నమస్కారం", use_cache=False,
                )
            )
            self.assertEqual(result.voice_id, "custom")
            self.assertTrue(any("not stored" in warning for warning in result.warnings))

    def test_cancel_event_becomes_a_structured_cancelled_error(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            engine = FakeEngine()
            engine.cancel_on_call = True
            service, _ = make_service(directory, engine=engine)
            voice = create_voice(service, directory)
            with self.assertRaises(TTSError) as caught:
                service.synthesize(
                    SpeechRequest(
                        text="నమస్కారం", voice_id=voice.voice_id, owner_key_id="owner-a",
                        use_cache=False,
                    )
                )
            self.assertEqual(caught.exception.code, "cancelled")

    def test_queue_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            service, _ = make_service(directory, api_queue_timeout_seconds=0.01)
            voice = create_voice(service, directory)
            service._queue_lock.acquire()
            try:
                with self.assertRaises(TTSError) as caught:
                    service.synthesize(
                        SpeechRequest(
                            text="నమస్కారం", voice_id=voice.voice_id,
                            owner_key_id="owner-a", use_cache=False,
                        )
                    )
                self.assertEqual(caught.exception.code, "queue_timeout")
            finally:
                service._queue_lock.release()

    def test_streaming_yields_chunks_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            service, engine = make_service(directory)
            voice = create_voice(service, directory)
            request = SpeechRequest(
                text="నమస్కారం. ఇది రెండవ వాక్యం.", voice_id=voice.voice_id,
                owner_key_id="owner-a", use_cache=True,
            )
            plan = service.plan_stream(request)
            self.assertIsNone(plan.cached_audio)
            chunks = list(service.stream(plan))
            self.assertEqual(len(chunks), 2)
            self.assertEqual([chunk.index for chunk in chunks], [1, 2])
            self.assertTrue(chunks[-1].is_last)
            self.assertEqual(chunks[0].sample_rate, 24_000)

            cached_plan = service.plan_stream(request)
            self.assertIsNotNone(cached_plan.cached_audio)
            cached_chunks = list(service.stream(cached_plan))
            self.assertEqual(len(cached_chunks), 1)
            self.assertTrue(cached_chunks[0].is_last)
            self.assertEqual(len(engine.stream_calls), 1)

    def test_streaming_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            service, _ = make_service(directory)
            voice = create_voice(service, directory)
            plan = service.plan_stream(
                SpeechRequest(
                    text="నమస్కారం", voice_id=voice.voice_id, owner_key_id="owner-a",
                    use_cache=False,
                )
            )
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(TTSError) as caught:
                list(service.stream(plan, cancel_event=cancel))
            self.assertEqual(caught.exception.code, "cancelled")

    def test_missing_or_broken_engine_fails_before_streaming(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            engine = FakeEngine()
            engine.load_error = RuntimeError("weights are not cached")
            service, _ = make_service(directory, engine=engine)
            voice = create_voice(service, directory)
            request = SpeechRequest(
                text="నమస్కారం", voice_id=voice.voice_id, owner_key_id="owner-a",
                use_cache=False,
            )
            with self.assertRaises(TTSError) as caught:
                service.synthesize(request)
            self.assertEqual(caught.exception.code, "engine_initialization_failed")
            self.assertIn("weights are not cached", str(caught.exception))
            with self.assertRaises(TTSError) as caught:
                service.plan_stream(request)
            self.assertEqual(caught.exception.code, "engine_initialization_failed")

    def test_generation_failure_is_structured_and_not_silent(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            engine = FakeEngine()
            engine.generate_error = RuntimeError("decoder exploded")
            service, _ = make_service(directory, engine=engine)
            voice = create_voice(service, directory)
            request = SpeechRequest(
                text="నమస్కారం", voice_id=voice.voice_id, owner_key_id="owner-a",
                use_cache=False,
            )
            with self.assertRaises(TTSError) as caught:
                service.synthesize(request)
            self.assertEqual(caught.exception.code, "generation_failed")
            self.assertIn("decoder exploded", str(caught.exception))

            stream_engine = FakeEngine()
            stream_engine.generate_error = RuntimeError("stream decoder exploded")
            stream_service, _ = make_service(directory, engine=stream_engine)
            stream_voice = create_voice(stream_service, directory)
            plan = stream_service.plan_stream(
                SpeechRequest(
                    text="నమస్కారం", voice_id=stream_voice.voice_id,
                    owner_key_id="owner-a", use_cache=False,
                )
            )
            with self.assertRaises(TTSError) as caught:
                list(stream_service.stream(plan))
            self.assertEqual(caught.exception.code, "generation_failed")

    def test_omitted_engine_uses_the_deployment_default(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            indicf5 = FakeEngine()
            chatterbox = FakeEngine()
            chatterbox.engine_name = "chatterbox"
            service, _ = make_service(
                directory,
                engine=indicf5,
                chatterbox=chatterbox,
                settings_engine="chatterbox",
            )
            voice = create_voice(service, directory)
            result = service.synthesize(
                SpeechRequest(
                    text="Hello", language="English", voice_id=voice.voice_id,
                    owner_key_id="owner-a", use_cache=False,
                )
            )
            self.assertEqual(result.engine, "chatterbox")
            self.assertEqual(len(chatterbox.calls), 1)
            self.assertEqual(len(indicf5.calls), 0)

    def test_preview_and_capabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            service, _ = make_service(directory)
            preview = service.preview("100 Mbps and 5G", "English")
            self.assertIn("one hundred megabits per second", preview["normalized_text"])
            self.assertIn("5G", preview["normalized_text"])
            self.assertTrue(preview["chunks"])
            capabilities = service.capabilities()
            self.assertIn("limits", capabilities)
            self.assertEqual(capabilities["profile"], "quality")
            self.assertEqual(service.health()["service"], "ok")


if __name__ == "__main__":
    unittest.main()
