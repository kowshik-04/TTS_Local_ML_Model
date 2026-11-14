import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from api.accounts import AccountStore
from api.auth import AuthManager
from api.main import create_app
from tests.support import FakeEngine, make_service, tone


def _wav_bytes(seconds=3.2, rate=24_000):
    buffer = io.BytesIO()
    sf.write(buffer, tone(seconds, rate=rate), rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.directory = Path(self._temp.name)
        self.service, self.engine = make_service(self.directory)
        self.settings = self.service.settings
        self.accounts = AccountStore(self.settings.api_db)
        self.auth = AuthManager(self.settings, self.accounts)
        self.app = create_app(
            self.settings,
            service=self.service,
            accounts=self.accounts,
            auth=self.auth,
        )
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.client.__enter__()
        self.key, self.account = self.accounts.create_key(
            "test", scopes=("tts", "voices"), monthly_char_quota=100_000,
            rate_limit_per_min=10_000,
        )
        self.headers = {"Authorization": f"Bearer {self.key}"}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self._temp.cleanup()

    def create_voice(self, name="API Voice", consent=True, headers=None):
        response = self.client.post(
            "/v1/voices",
            headers=headers or self.headers,
            data={
                "name": name,
                "transcript": "నమస్కారం. ఇది ఒక పరీక్ష.",
                "consent": "true" if consent else "false",
                "language": "Telugu",
                "engine_hint": "indicf5",
            },
            files={"audio": ("reference.wav", _wav_bytes(), "audio/wav")},
        )
        return response

    def synthesis_payload(self, voice_id, **overrides):
        payload = {
            "text": "నమస్కారం. 100 Mbps మరియు 5G.",
            "language": "Telugu",
            "voice_id": voice_id,
            "format": "wav",
        }
        payload.update(overrides)
        return payload


class ApiAuthTests(ApiTestCase):
    def test_health_is_public_but_tts_requires_a_key(self):
        self.assertEqual(self.client.get("/healthz").status_code, 200)
        response = self.client.get("/v1/me")
        self.assertEqual(response.status_code, 401)
        body = response.json()
        self.assertEqual(body["error"]["code"], "unauthorized")
        self.assertTrue(response.headers["X-Request-Id"].startswith("req_") or
                        response.headers["X-Request-Id"])

    def test_me_capabilities_and_preview(self):
        response = self.client.get("/v1/me", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["key_id"], self.account.key_id)
        self.assertEqual(response.json()["usage"]["chars"], 0)

        capabilities = self.client.get("/v1/capabilities", headers=self.headers)
        self.assertEqual(capabilities.status_code, 200)
        self.assertIn("languages", capabilities.json())
        self.assertEqual(capabilities.json()["default_engine"], "auto")
        self.assertIn("Failures never trigger", capabilities.json()["routing"])

        preview = self.client.post(
            "/v1/preview",
            headers=self.headers,
            json={"text": "100 Mbps and 5G", "language": "English"},
        )
        self.assertEqual(preview.status_code, 200)
        self.assertIn("one hundred megabits per second", preview.json()["normalized_text"])
        self.assertIn("5G", preview.json()["normalized_text"])

    def test_rate_limit_returns_429_with_retry_after(self):
        limited_key, _ = self.accounts.create_key(
            "limited", scopes=("tts",), monthly_char_quota=1000, rate_limit_per_min=1
        )
        headers = {"Authorization": f"Bearer {limited_key}"}
        self.assertEqual(self.client.get("/v1/me", headers=headers).status_code, 200)
        response = self.client.get("/v1/me", headers=headers)
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["error"]["code"], "rate_limited")
        self.assertIn("Retry-After", response.headers)

    def test_readyz_follows_an_explicitly_configured_chatterbox_engine(self):
        class ChatterboxFake(FakeEngine):
            def info(self):
                return {"engine": "chatterbox", "loaded": True, "device": "cpu"}

        service, _ = make_service(
            self.directory / "ready-chatterbox",
            settings_engine="chatterbox",
            chatterbox=ChatterboxFake(),
            warmup=True,
        )
        accounts = AccountStore(service.settings.api_db)
        app = create_app(
            service.settings,
            service=service,
            accounts=accounts,
            auth=AuthManager(service.settings, accounts),
        )
        with TestClient(app) as client:
            response = client.get("/readyz")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["model_loaded"])
        self.assertEqual(body["configured_engine"], "chatterbox")
        self.assertEqual(body["readiness_engine"], "chatterbox")
        self.assertIsNone(body["warmup_error"])

    def test_readyz_is_not_ready_when_configured_chatterbox_is_disabled(self):
        service, _ = make_service(
            self.directory / "missing-chatterbox",
            settings_engine="chatterbox",
            warmup=True,
        )
        accounts = AccountStore(service.settings.api_db)
        app = create_app(
            service.settings,
            service=service,
            accounts=accounts,
            auth=AuthManager(service.settings, accounts),
        )
        with TestClient(app) as client:
            response = client.get("/readyz")
        self.assertEqual(response.status_code, 503, response.text)
        body = response.json()
        self.assertFalse(body["model_loaded"])
        self.assertEqual(body["readiness_engine"], "chatterbox")
        self.assertIn("not enabled", body["warmup_error"])

    def test_metrics_requires_admin(self):
        normal = self.client.get("/metrics", headers=self.headers)
        self.assertEqual(normal.status_code, 403)
        admin_key, _ = self.accounts.create_key("admin", scopes=("tts", "voices", "admin"))
        response = self.client.get("/metrics", headers={"Authorization": f"Bearer {admin_key}"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("tts_http_requests_total", response.text)


class ApiTTSTests(ApiTestCase):
    def test_wav_synthesis_headers_quota_and_cache(self):
        voice = self.create_voice().json()
        payload = self.synthesis_payload(voice["voice_id"])
        first = self.client.post("/v1/tts", headers=self.headers, json=payload)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.headers["content-type"], "audio/wav")
        self.assertEqual(first.headers["x-engine"], "indicf5")
        self.assertEqual(first.headers["x-voice-id"], voice["voice_id"])
        self.assertEqual(first.headers["x-cache"], "miss")
        self.assertEqual(first.headers["x-audio-format"], "wav")
        self.assertEqual(first.headers["x-sample-rate"], "24000")
        self.assertGreater(int(first.headers["x-audio-duration-ms"]), 0)
        audio, rate = sf.read(io.BytesIO(first.content), dtype="float32")
        self.assertEqual(rate, 24_000)
        self.assertGreater(len(audio), 0)
        self.assertTrue(np.isfinite(audio).all())

        second = self.client.post("/v1/tts", headers=self.headers, json=payload)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.headers["x-cache"], "hit")

        usage = self.client.get("/v1/me", headers=self.headers).json()["usage"]
        self.assertEqual(usage["chars"], len(payload["text"]) * 2)
        self.assertEqual(usage["requests"], 2)
        self.assertGreater(usage["audio_ms"], 0)

    def test_quota_is_enforced_and_refunded_on_failure(self):
        voice = self.create_voice().json()
        small_key, _ = self.accounts.create_key(
            "small", scopes=("tts",), monthly_char_quota=5, rate_limit_per_min=1000
        )
        payload = self.synthesis_payload(voice["voice_id"])
        response = self.client.post(
            "/v1/tts", headers={"Authorization": f"Bearer {small_key}"}, json=payload
        )
        self.assertEqual(response.status_code, 402)
        self.assertEqual(response.json()["error"]["code"], "quota_exceeded")

        self.engine.cancel_on_call = True
        failed = self.client.post("/v1/tts", headers=self.headers, json=payload)
        self.assertEqual(failed.status_code, 499)
        self.assertEqual(failed.json()["error"]["code"], "cancelled")
        usage = self.client.get("/v1/me", headers=self.headers).json()["usage"]
        self.assertEqual(usage["chars"], 0)

    def test_validation_and_service_errors_are_json(self):
        voice = self.create_voice().json()
        invalid = self.client.post(
            "/v1/tts",
            headers=self.headers,
            json=self.synthesis_payload(voice["voice_id"], language="Klingon"),
        )
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(invalid.json()["error"]["code"], "invalid_language")

        unknown = self.client.post(
            "/v1/tts",
            headers=self.headers,
            json=self.synthesis_payload("v_missing_123"),
        )
        self.assertEqual(unknown.status_code, 404)
        self.assertEqual(unknown.json()["error"]["code"], "voice_not_found")

        extra = self.client.post(
            "/v1/tts",
            headers=self.headers,
            json=self.synthesis_payload(voice["voice_id"], typo=True),
        )
        self.assertEqual(extra.status_code, 422)
        self.assertEqual(extra.json()["error"]["code"], "validation_error")

        english = self.client.post(
            "/v1/tts",
            headers=self.headers,
            json=self.synthesis_payload(voice["voice_id"], language="English", text="hello"),
        )
        self.assertEqual(english.status_code, 422)
        self.assertEqual(english.json()["error"]["code"], "engine_unavailable")

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for MP3 export")
    def test_mp3_export(self):
        voice = self.create_voice().json()
        response = self.client.post(
            "/v1/tts",
            headers=self.headers,
            json=self.synthesis_payload(voice["voice_id"], format="mp3"),
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "audio/mpeg")
        self.assertGreater(len(response.content), 100)

    def test_streaming_preflight_failure_returns_error_before_pcm_headers(self):
        voice = self.create_voice().json()
        self.engine.load_error = RuntimeError("weights are not cached")
        payload = self.synthesis_payload(voice["voice_id"])
        payload.pop("format")
        response = self.client.post("/v1/tts/stream", headers=self.headers, json=payload)
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(
            response.json()["error"]["code"], "engine_initialization_failed"
        )

    def test_generation_failure_is_json_not_empty_audio(self):
        voice = self.create_voice().json()
        self.engine.generate_error = RuntimeError("decoder exploded")
        response = self.client.post(
            "/v1/tts",
            headers=self.headers,
            json=self.synthesis_payload(voice["voice_id"]),
        )
        self.assertEqual(response.status_code, 500, response.text)
        self.assertEqual(response.json()["error"]["code"], "generation_failed")

    def test_streaming_generation_failure_returns_json_before_pcm_headers(self):
        voice = self.create_voice().json()
        self.engine.generate_error = RuntimeError("decoder exploded")
        payload = self.synthesis_payload(voice["voice_id"])
        payload.pop("format")
        response = self.client.post("/v1/tts/stream", headers=self.headers, json=payload)
        self.assertEqual(response.status_code, 500, response.text)
        self.assertEqual(response.json()["error"]["code"], "generation_failed")

    def test_streaming_pcm_and_cache(self):
        voice = self.create_voice().json()
        payload = self.synthesis_payload(voice["voice_id"])
        payload.pop("format")
        first = self.client.post("/v1/tts/stream", headers=self.headers, json=payload)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.headers["content-type"], "audio/L16")
        self.assertEqual(first.headers["x-audio-format"], "pcm_s16le")
        self.assertEqual(first.headers["x-cache"], "miss")
        self.assertGreater(len(first.content), 0)
        self.assertEqual(len(first.content) % 2, 0)

        second = self.client.post("/v1/tts/stream", headers=self.headers, json=payload)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.headers["x-cache"], "hit")
        self.assertGreater(len(second.content), 0)

    def test_websocket_streaming(self):
        voice = self.create_voice().json()
        with self.client.websocket_connect("/v1/tts/ws", headers=self.headers) as websocket:
            websocket.send_json({
                "type": "start",
                "text": "నమస్కారం. ఇది ఒక పరీక్ష.",
                "voice_id": voice["voice_id"],
                "language": "Telugu",
            })
            ready = websocket.receive_json()
            self.assertEqual(ready["type"], "ready")
            self.assertEqual(ready["audio_format"], "pcm_s16le")
            self.assertEqual(ready["sample_rate"], 24_000)
            audio = bytearray()
            done = None
            while done is None:
                message = websocket.receive()
                if message.get("bytes") is not None:
                    audio.extend(message["bytes"])
                elif message.get("text") is not None:
                    payload = json.loads(message["text"])
                    if payload.get("type") == "done":
                        done = payload
            self.assertGreater(len(audio), 0)
            self.assertFalse(done["cancelled"])

    def test_websocket_rejects_missing_key(self):
        with self.client.websocket_connect("/v1/tts/ws") as websocket:
            websocket.send_json({"type": "start", "text": "hello", "voice_id": "v_missing_123"})
            with self.assertRaises(Exception):
                websocket.receive_json()


class ApiVoiceTests(ApiTestCase):
    def test_voice_crud_and_consent(self):
        refused = self.create_voice(consent=False)
        self.assertEqual(refused.status_code, 400)
        self.assertIn("consent", refused.text.lower())

        created = self.create_voice()
        self.assertEqual(created.status_code, 200, created.text)
        voice = created.json()
        self.assertTrue(voice["consent"])
        self.assertEqual(voice["language"], "Telugu")
        self.assertIn("transcript", voice)
        self.assertNotIn("owner_key_id", voice)

        listed = self.client.get("/v1/voices", headers=self.headers)
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.json()["voices"]), 1)
        fetched = self.client.get(f"/v1/voices/{voice['voice_id']}", headers=self.headers)
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.json()["voice_id"], voice["voice_id"])

        deleted = self.client.delete(f"/v1/voices/{voice['voice_id']}", headers=self.headers)
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(self.client.get("/v1/voices", headers=self.headers).json()["voices"], [])

    def test_voice_ownership_is_enforced(self):
        voice = self.create_voice().json()
        other_key, _ = self.accounts.create_key("other", scopes=("tts", "voices"))
        other_headers = {"Authorization": f"Bearer {other_key}"}
        self.assertEqual(
            self.client.get(f"/v1/voices/{voice['voice_id']}", headers=other_headers).status_code,
            404,
        )
        self.assertEqual(
            self.client.delete(f"/v1/voices/{voice['voice_id']}", headers=other_headers).status_code,
            404,
        )
        admin_key, _ = self.accounts.create_key("admin2", scopes=("tts", "voices", "admin"))
        admin_headers = {"Authorization": f"Bearer {admin_key}"}
        self.assertEqual(
            self.client.get(f"/v1/voices/{voice['voice_id']}", headers=admin_headers).status_code,
            200,
        )

    def test_invalid_voice_upload_fields(self):
        response = self.client.post(
            "/v1/voices",
            headers=self.headers,
            data={
                "name": "Bad", "transcript": "text", "consent": "true",
                "language": "Klingon", "engine_hint": "indicf5",
            },
            files={"audio": ("reference.wav", _wav_bytes(), "audio/wav")},
        )
        self.assertEqual(response.status_code, 400)
        response = self.client.post(
            "/v1/voices",
            headers=self.headers,
            data={
                "name": "Bad", "transcript": "text", "consent": "true",
                "language": "Telugu", "engine_hint": "unknown",
            },
            files={"audio": ("reference.wav", _wav_bytes(), "audio/wav")},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
