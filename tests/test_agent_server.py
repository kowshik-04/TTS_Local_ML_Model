import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from agent.server import create_agent_app
from agent.tts_client import TTSClientError
from agent.vad import float_to_pcm16
from tests.support import FakeLLM, FakeSTT, FakeTTSClient, make_settings
import numpy as np


def frame(value, samples=320):
    return float_to_pcm16(np.full(samples, value, dtype=np.float32))


SILENCE = frame(0.0)
SPEECH = frame(0.09)


def receive_message(ws):
    return ws.receive()


def parse_event(message):
    text = message.get("text")
    if not text:
        return None
    return json.loads(text)


def receive_until(ws, target_type, *, limit=300):
    events = []
    audio_bytes = 0
    for _ in range(limit):
        message = receive_message(ws)
        if message.get("type") == "websocket.close":
            break
        payload = message.get("bytes")
        if payload:
            audio_bytes += len(payload)
            continue
        event = parse_event(message)
        if not event:
            continue
        events.append(event)
        if event.get("type") == target_type:
            return events, audio_bytes
    raise AssertionError(
        f"Did not receive {target_type!r}; got {[event.get('type') for event in events]}"
    )


class AgentServerTestCase(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.directory = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)

    def make_client(self, *, stt=None, llm=None, tts=None, require_api_key=None, **overrides):
        values = dict(
            agent_voice_id="voice-a",
            agent_vad_min_speech_ms=40,
            agent_vad_silence_ms=100,
            agent_vad_pre_roll_ms=40,
        )
        values.update(overrides)
        settings = make_settings(self.directory, **values)
        app = create_agent_app(
            settings,
            stt=stt or FakeSTT(),
            llm=llm or FakeLLM(),
            tts=tts or FakeTTSClient(),
            require_api_key=require_api_key,
        )
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def start_session(self, ws, *, api_key=None, voice_id="voice-a", language="Telugu"):
        payload = {"type": "start", "language": language, "sample_rate": 16_000}
        if voice_id:
            payload["voice_id"] = voice_id
        if api_key:
            payload["api_key"] = api_key
        ws.send_json(payload)

    def test_index_serves_bundled_demo(self):
        client = self.make_client()
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Local Voice Studio", response.text)
        self.assertIn("/v1/agent/ws", response.text)

    def test_healthz(self):
        client = self.make_client()
        response = client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["input_sample_rate"], 16_000)
        self.assertEqual(body["output_sample_rate"], 24_000)
        self.assertEqual(body["tts_engine"], "auto")
        self.assertTrue(body["voice_configured"])

    def test_start_say_turn_and_ping(self):
        client = self.make_client()
        with client.websocket_connect("/v1/agent/ws") as ws:
            self.start_session(ws)
            ready = ws.receive_json()
            self.assertEqual(ready["type"], "ready")
            self.assertEqual(ready["audio_format"], "pcm_s16le")
            self.assertEqual(ready["voice_id"], "voice-a")
            self.assertEqual(ready["tts_engine"], "auto")

            ws.send_json({"type": "say", "text": "నమస్కారం"})
            events, audio_bytes = receive_until(ws, "turn_end")
            kinds = [event["type"] for event in events]
            self.assertIn("assistant_delta", kinds)
            self.assertIn("phrase", kinds)
            self.assertIn("audio_end", kinds)
            self.assertGreater(audio_bytes, 0)
            self.assertIn("ధన్యవాదాలు", events[-1]["text"])

            listening = ws.receive_json()
            self.assertEqual(listening["type"], "listening")

            ws.send_json({"type": "ping"})
            self.assertEqual(ws.receive_json(), {"type": "pong"})

    def test_tts_failure_is_reported_with_stable_detail(self):
        tts = FakeTTSClient(
            error=TTSClientError(
                "The indicf5 engine could not be initialized: gated weights",
                status_code=503,
                code="engine_initialization_failed",
            )
        )
        client = self.make_client(tts=tts)
        with client.websocket_connect("/v1/agent/ws") as ws:
            self.start_session(ws)
            self.assertEqual(ws.receive_json()["type"], "ready")
            ws.send_json({"type": "say", "text": "నమస్కారం"})
            events, audio_bytes = receive_until(ws, "error")
        self.assertEqual(audio_bytes, 0)
        error = events[-1]
        self.assertEqual(error["code"], "tts_failed")
        self.assertEqual(error["tts_code"], "engine_initialization_failed")
        self.assertEqual(error["status_code"], 503)
        self.assertIn("gated weights", error["message"])

    def test_empty_llm_response_is_reported_over_websocket(self):
        class EmptyLLM:
            async def stream(self, messages):
                if False:
                    yield ""

        client = self.make_client(llm=EmptyLLM())
        with client.websocket_connect("/v1/agent/ws") as ws:
            self.start_session(ws)
            self.assertEqual(ws.receive_json()["type"], "ready")
            ws.send_json({"type": "say", "text": "నమస్కారం"})
            events, audio_bytes = receive_until(ws, "error")
            self.assertEqual(audio_bytes, 0)
            error = events[-1]
            self.assertEqual(error["code"], "empty_llm_response")
            self.assertIn("no speakable text", error["message"])
            self.assertEqual(ws.receive_json()["type"], "listening")

    def test_configured_agent_tts_engine_is_visible(self):
        client = self.make_client(agent_tts_engine="chatterbox")
        health = client.get("/healthz").json()
        self.assertEqual(health["tts_engine"], "chatterbox")
        with client.websocket_connect("/v1/agent/ws") as ws:
            self.start_session(ws)
            self.assertEqual(ws.receive_json()["tts_engine"], "chatterbox")

    def test_binary_vad_turn_end_to_end(self):
        stt = FakeSTT("ఇది ఒక పరీక్ష.")
        client = self.make_client(stt=stt)
        with client.websocket_connect("/v1/agent/ws") as ws:
            self.start_session(ws)
            self.assertEqual(ws.receive_json()["type"], "ready")

            ws.send_bytes(SILENCE * 2)
            ws.send_bytes(SPEECH * 3)
            speech = ws.receive_json()
            self.assertEqual(speech["type"], "speech_start")
            self.assertFalse(speech["interrupted"])

            ws.send_bytes(SILENCE * 6)
            events, audio_bytes = receive_until(ws, "turn_end")
            kinds = [event["type"] for event in events]
            self.assertIn("vad_end", kinds)
            self.assertIn("transcript", kinds)
            self.assertGreater(audio_bytes, 0)
            self.assertEqual(stt.calls[0]["sample_rate"], 16_000)
            self.assertGreater(stt.calls[0]["length"], 0)
            self.assertEqual(events[-1]["type"], "turn_end")

    def test_interrupt_stops_active_turn(self):
        llm = FakeLLM(block_after_first=True, block_seconds=30.0)
        client = self.make_client(llm=llm)
        with client.websocket_connect("/v1/agent/ws") as ws:
            self.start_session(ws)
            self.assertEqual(ws.receive_json()["type"], "ready")

            ws.send_json({"type": "say", "text": "ఒక పొడవైన ప్రశ్న"})
            events, _ = receive_until(ws, "assistant_delta")
            ws.send_json({"type": "interrupt"})
            interrupted_events, _ = receive_until(ws, "interrupted")
            all_events = events + interrupted_events
            self.assertNotIn("turn_end", [event["type"] for event in all_events])

            ws.send_json({"type": "reset"})
            self.assertEqual(ws.receive_json()["type"], "reset")

    def test_reset_and_unknown_command(self):
        client = self.make_client()
        with client.websocket_connect("/v1/agent/ws") as ws:
            self.start_session(ws)
            self.assertEqual(ws.receive_json()["type"], "ready")
            ws.send_json({"type": "reset"})
            self.assertEqual(ws.receive_json()["type"], "reset")
            ws.send_json({"type": "mystery"})
            error = ws.receive_json()
            self.assertEqual(error["type"], "error")
            self.assertEqual(error["code"], "unknown_command")
            ws.send_text("{not json")
            error = ws.receive_json()
            self.assertEqual(error["code"], "invalid_json")

    def test_requires_agent_key_when_configured(self):
        client = self.make_client(agent_api_key="secret")
        with client.websocket_connect("/v1/agent/ws") as ws:
            self.start_session(ws)
            with self.assertRaises(WebSocketDisconnect) as context:
                ws.receive_json()
            self.assertEqual(context.exception.code, 4401)

    def test_accepts_agent_key_from_header(self):
        client = self.make_client(agent_api_key="secret")
        with client.websocket_connect(
            "/v1/agent/ws", headers={"Authorization": "Bearer secret"}
        ) as ws:
            self.start_session(ws)
            self.assertEqual(ws.receive_json()["type"], "ready")

    def test_rejects_invalid_start_settings(self):
        client = self.make_client()
        with client.websocket_connect("/v1/agent/ws") as ws:
            self.start_session(ws, language="Klingon")
            error = ws.receive_json()
            self.assertEqual(error["code"], "invalid_language")
            with self.assertRaises(WebSocketDisconnect):
                ws.receive_json()

        with client.websocket_connect("/v1/agent/ws") as ws:
            ws.send_json({"type": "start", "voice_id": "voice-a", "sample_rate": 8000})
            error = ws.receive_json()
            self.assertEqual(error["code"], "unsupported_sample_rate")
            with self.assertRaises(WebSocketDisconnect):
                ws.receive_json()

    def test_first_message_must_be_start(self):
        client = self.make_client()
        with client.websocket_connect("/v1/agent/ws") as ws:
            ws.send_json({"type": "ping"})
            with self.assertRaises(WebSocketDisconnect) as context:
                ws.receive_json()
            self.assertEqual(context.exception.code, 4400)

    def test_voice_required_without_configured_voice(self):
        client = self.make_client(agent_voice_id="")
        with client.websocket_connect("/v1/agent/ws") as ws:
            self.start_session(ws, voice_id="")
            error = ws.receive_json()
            self.assertEqual(error["code"], "voice_required")


if __name__ == "__main__":
    unittest.main()
