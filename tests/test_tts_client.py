import json
import unittest

import httpx

from agent.tts_client import TTSClient, TTSClientError


class TTSClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_configured_engine_and_resolved_header_is_observable(self):
        calls = []

        async def handler(request):
            calls.append(json.loads(request.content))
            return httpx.Response(
                200,
                headers={"X-Engine": "chatterbox"},
                content=b"\x01\x00" * 8,
            )

        client = TTSClient(
            "http://tts.test",
            voice_id="voice-a",
            engine="chatterbox",
            transport=httpx.MockTransport(handler),
        )
        audio = await client.synthesize_pcm("hello", language="English")
        self.assertEqual(audio, b"\x01\x00" * 8)
        self.assertEqual(calls[0]["engine"], "chatterbox")
        self.assertEqual(calls[0]["language"], "English")

    async def test_default_engine_is_auto(self):
        bodies = []

        async def handler(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, content=b"\x01\x00")

        client = TTSClient(
            "http://tts.test",
            voice_id="voice-a",
            transport=httpx.MockTransport(handler),
        )
        await client.synthesize_pcm("test")
        self.assertEqual(bodies[0]["engine"], "auto")

    async def test_http_error_preserves_stable_code(self):
        async def handler(request):
            return httpx.Response(
                503,
                json={
                    "error": {
                        "code": "engine_initialization_failed",
                        "message": "weights are not cached",
                    }
                },
            )

        client = TTSClient(
            "http://tts.test",
            voice_id="voice-a",
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaises(TTSClientError) as context:
            await client.synthesize_pcm("test")
        self.assertEqual(context.exception.status_code, 503)
        self.assertEqual(context.exception.code, "engine_initialization_failed")
        self.assertIn("weights are not cached", str(context.exception))

    async def test_empty_audio_is_an_error_not_success(self):
        async def handler(request):
            return httpx.Response(200, headers={"X-Engine": "indicf5"}, content=b"")

        client = TTSClient(
            "http://tts.test",
            voice_id="voice-a",
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaises(TTSClientError) as context:
            await client.synthesize_pcm("test")
        self.assertEqual(context.exception.code, "empty_audio")
        self.assertIn("no audio", str(context.exception))

    def test_invalid_engine_is_rejected(self):
        with self.assertRaises(TTSClientError) as context:
            TTSClient("http://tts.test", engine="cloud")
        self.assertEqual(context.exception.code, "invalid_engine")


if __name__ == "__main__":
    unittest.main()
