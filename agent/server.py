"""FastAPI WebSocket bridge for the real-time Telugu voice agent.

Protocol (JSON text events plus binary PCM16 frames):

Client -> server
  {"type":"start","voice_id":"...","language":"Telugu","api_key":"...",
   "sample_rate":16000}
  <binary PCM16 mono frames at the declared sample rate>
  {"type":"interrupt"} | {"type":"end_utterance"} | {"type":"reset"}
  {"type":"say","text":"..."}   # text-mode/test-mode turn
  {"type":"stop"}

Server -> client
  {"type":"ready", ...}
  {"type":"speech_start"} / {"type":"vad_end","duration_seconds":...}
  {"type":"transcript","text":"..."}
  {"type":"assistant_delta","text":"..."}
  {"type":"phrase","text":"..."}
  <binary PCM16 mono frames at output_sample_rate>
  {"type":"audio_end","text":"...","index":1}
  {"type":"turn_end","text":"..."}
  {"type":"interrupted"} / {"type":"listening"} / {"type":"error",...}
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from agent.llm import build_llm
from agent.session import ConversationSession
from agent.stt import build_stt
from agent.tts_client import build_tts_client
from agent.vad import VadConfig, VadSegmenter
from tts.config import Settings, get_settings
from tts.languages import canonical_language


LOGGER = logging.getLogger("local-voice-studio.agent")
AGENT_VERSION = "0.2.0"


def _vad_config(settings: Settings) -> VadConfig:
    return VadConfig(
        sample_rate=settings.agent_input_sample_rate,
        frame_ms=settings.agent_vad_frame_ms,
        speech_dbfs=settings.agent_vad_speech_dbfs,
        silence_ms=settings.agent_vad_silence_ms,
        min_speech_ms=settings.agent_vad_min_speech_ms,
        max_utterance_seconds=settings.agent_vad_max_utterance_seconds,
        pre_roll_ms=settings.agent_vad_pre_roll_ms,
    )


def create_agent_app(
    settings: Settings | None = None,
    *,
    stt=None,
    llm=None,
    tts=None,
    require_api_key: bool | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    stt = stt or build_stt(settings)
    llm = llm or build_llm(settings)
    tts = tts or build_tts_client(settings)
    require_key = bool(settings.agent_api_key) if require_api_key is None else require_api_key
    vad_config = _vad_config(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        LOGGER.info(
            "Voice agent ready: input=%d Hz output=%d Hz stt=%s llm=%s tts_engine=%s",
            settings.agent_input_sample_rate,
            settings.agent_output_sample_rate,
            settings.agent_stt_provider,
            settings.agent_llm_model,
            settings.agent_tts_engine,
        )
        if not require_key:
            LOGGER.warning("AGENT_API_KEY is not set; the agent WebSocket has no local gatekeeper.")
        yield

    app = FastAPI(
        title="Local Voice Studio Agent",
        version=AGENT_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
    )
    app.state.settings = settings
    app.state.stt = stt
    app.state.llm = llm
    app.state.tts = tts
    app.state.vad_config = vad_config

    @app.get("/healthz", tags=["system"])
    async def healthz():
        return {
            "status": "ok",
            "version": AGENT_VERSION,
            "input_sample_rate": settings.agent_input_sample_rate,
            "output_sample_rate": settings.agent_output_sample_rate,
            "stt_provider": settings.agent_stt_provider,
            "llm_model": settings.agent_llm_model,
            "tts_engine": settings.agent_tts_engine,
            "voice_configured": bool(settings.agent_voice_id),
        }

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index():
        page = Path(__file__).parent / "static" / "index.html"
        if page.is_file():
            return HTMLResponse(page.read_text(encoding="utf-8"))
        return HTMLResponse(
            "<h1>Local Voice Studio Agent</h1>"
            "<p>WebSocket endpoint: <code>/v1/agent/ws</code></p>"
        )

    @app.websocket("/v1/agent/ws")
    async def agent_websocket(websocket: WebSocket):
        await websocket.accept()
        send_lock = asyncio.Lock()

        async def send_json(payload: dict) -> None:
            async with send_lock:
                await websocket.send_json(payload)

        async def send_audio(payload: bytes) -> None:
            async with send_lock:
                await websocket.send_bytes(payload)

        try:
            start = await asyncio.wait_for(websocket.receive_json(), timeout=20.0)
        except (asyncio.TimeoutError, WebSocketDisconnect):
            await websocket.close(code=4400, reason="Expected a start message.")
            return
        if not isinstance(start, dict) or start.get("type") not in {"start", "say"}:
            await websocket.close(code=4400, reason="First message must be type=start.")
            return

        authorization = websocket.headers.get("authorization", "")
        header_key = ""
        if authorization.lower().startswith("bearer "):
            header_key = authorization[7:].strip()
        header_key = header_key or websocket.headers.get("x-api-key", "").strip()
        supplied_key = str(start.get("api_key") or header_key)
        if require_key and not secrets.compare_digest(
            supplied_key, settings.agent_api_key
        ):
            await websocket.close(code=4401, reason="Missing or invalid agent API key.")
            return

        try:
            language = canonical_language(start.get("language") or settings.default_language)
        except ValueError as error:
            await send_json({"type": "error", "code": "invalid_language", "message": str(error)})
            await websocket.close(code=4400)
            return
        voice_id = str(start.get("voice_id") or settings.agent_voice_id or "")
        if not voice_id:
            await send_json(
                {"type": "error", "code": "voice_required", "message": "Provide voice_id or set AGENT_VOICE_ID."}
            )
            await websocket.close(code=4400)
            return
        try:
            requested_rate = int(start.get("sample_rate") or settings.agent_input_sample_rate)
        except (TypeError, ValueError):
            await send_json({"type": "error", "code": "invalid_sample_rate"})
            await websocket.close(code=4400)
            return
        if requested_rate != settings.agent_input_sample_rate:
            await send_json(
                {
                    "type": "error",
                    "code": "unsupported_sample_rate",
                    "message": (
                        f"Client must resample to {settings.agent_input_sample_rate} Hz "
                        f"or AGENT_INPUT_SAMPLE_RATE must be changed."
                    ),
                }
            )
            await websocket.close(code=4400)
            return

        session = ConversationSession(
            stt,
            llm,
            tts,
            voice_id=voice_id,
            language=language,
            max_history_turns=settings.agent_max_history_turns,
            system_prompt=settings.agent_system_prompt or None,
        )
        vad = VadSegmenter(vad_config)
        response_task: asyncio.Task | None = None

        await send_json(
            {
                "type": "ready",
                "version": AGENT_VERSION,
                "voice_id": voice_id,
                "language": language,
                "input_sample_rate": settings.agent_input_sample_rate,
                "output_sample_rate": settings.agent_output_sample_rate,
                "tts_engine": settings.agent_tts_engine,
                "audio_format": "pcm_s16le",
                "vad": {
                    "silence_ms": settings.agent_vad_silence_ms,
                    "min_speech_ms": settings.agent_vad_min_speech_ms,
                },
            }
        )

        async def run_response(text: str | None, audio: bytes | None, sample_rate: int) -> None:
            cancelled = False
            try:
                events = (
                    session.handle_utterance(audio, sample_rate)
                    if audio is not None
                    else session.respond(text or "")
                )
                async for event in events:
                    kind = event.get("type")
                    if kind == "audio":
                        await send_audio(event["data"])
                    else:
                        await send_json(event)
            except asyncio.CancelledError:
                cancelled = True
                raise
            except Exception as error:
                LOGGER.exception("Agent turn failed")
                await send_json(
                    {"type": "error", "code": "turn_failed", "message": str(error)}
                )
            finally:
                if not cancelled:
                    await send_json({"type": "listening"})

        async def stop_response(*, notify: bool) -> bool:
            nonlocal response_task
            task, response_task = response_task, None
            if task is None or task.done():
                return False
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                LOGGER.debug("Agent response task ended with an error", exc_info=True)
            if notify:
                await send_json({"type": "interrupted"})
            return True

        async def start_response(*, text: str | None = None, audio: bytes | None = None) -> None:
            nonlocal response_task
            await stop_response(notify=False)
            response_task = asyncio.create_task(
                run_response(text, audio, settings.agent_input_sample_rate)
            )

        async def consume_vad_events(events) -> None:
            for event in events:
                if event.kind == "speech_start":
                    interrupted = await stop_response(notify=True)
                    await send_json({"type": "speech_start", "interrupted": interrupted})
                elif event.kind == "utterance":
                    await send_json(
                        {
                            "type": "vad_end",
                            "duration_seconds": round(event.duration_seconds, 3),
                        }
                    )
                    await start_response(audio=event.audio)

        try:
            if start.get("type") == "say" and start.get("text"):
                await start_response(text=str(start["text"]))
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                payload = message.get("bytes")
                if payload is not None:
                    await consume_vad_events(vad.feed_pcm16(payload))
                    continue
                raw_text = message.get("text")
                if raw_text is None:
                    continue
                try:
                    command = json.loads(raw_text)
                except ValueError:
                    await send_json({"type": "error", "code": "invalid_json"})
                    continue
                if not isinstance(command, dict):
                    await send_json({"type": "error", "code": "invalid_message"})
                    continue
                kind = command.get("type")
                if kind == "interrupt":
                    await stop_response(notify=True)
                elif kind == "end_utterance":
                    await consume_vad_events(vad.flush())
                elif kind == "say":
                    text = str(command.get("text") or "").strip()
                    if text:
                        await start_response(text=text)
                elif kind == "reset":
                    await stop_response(notify=False)
                    session.reset()
                    vad.reset()
                    await send_json({"type": "reset"})
                elif kind == "ping":
                    await send_json({"type": "pong"})
                elif kind == "stop":
                    break
                else:
                    await send_json({"type": "error", "code": "unknown_command"})
        except WebSocketDisconnect:
            pass
        except Exception:
            LOGGER.exception("Agent WebSocket loop failed")
        finally:
            await stop_response(notify=False)
            try:
                await websocket.close()
            except Exception:
                pass

    return app


def create_app_from_env() -> FastAPI:
    """Uvicorn factory: ``uvicorn agent.server:create_app_from_env --factory``."""
    return create_agent_app(Settings.from_env())
