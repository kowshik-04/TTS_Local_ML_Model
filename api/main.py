"""Production HTTP/WebSocket API for the local Telugu voice studio.

Run:
    uvicorn api.main:app --host 127.0.0.1 --port 8000

The API is deliberately thin: synthesis, text preparation, caching and voice
profiles live in ``tts.service`` so they can also be used by the Gradio UI and the
agent process.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.accounts import AccountError, AccountStore
from api.auth import AuthContext, AuthManager, require_scope
from api.metrics import METRICS, timer
from api.schemas import (
    CapabilitiesResponse,
    ErrorBody,
    ErrorResponse,
    MeResponse,
    PreviewRequest,
    PreviewResponse,
    TTSRequest,
    TTSStreamRequest,
    TransliterateRequest,
    TransliterateResponse,
    UsageResponse,
    VoiceListResponse,
    VoiceResponse,
)
from api.streaming import ThreadedStream
from tts.config import Settings
from tts.languages import canonical_language
from tts.postprocess import to_pcm16
from tts.service import SpeechRequest, TTSError, TTSService
from tts.voice_store import VoiceError


LOGGER = logging.getLogger("local-voice-studio.api")
APP_VERSION = "0.2.0"


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    payload = ErrorResponse(error=ErrorBody(code=code, message=message, request_id=request_id))
    return JSONResponse(status_code=status_code, content=payload.model_dump(), headers=headers)


def _status_for_code(code: str) -> int:
    return {
        "voice_not_found": 404,
        "text_too_long": 413,
        "queue_timeout": 503,
        "transliteration_unavailable": 503,
        "engine_unavailable": 422,
        "engine_initialization_failed": 503,
        "generation_failed": 500,
        "invalid_reference": 422,
        "cancelled": 499,
        "quota_exceeded": 402,
    }.get(code, 400)


def _code_for_status(status_code: int) -> str:
    return {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        413: "request_too_large",
        422: "validation_error",
        429: "rate_limited",
    }.get(status_code, "http_error")


def _warnings_header(warnings: tuple[str, ...] | list[str]) -> str:
    if not warnings:
        return ""
    payload = json.dumps(list(warnings), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii")


def _audio_headers(
    *,
    result,
    request_id: str,
    audio_format: str,
    payload_bytes: int,
) -> dict[str, str]:
    headers = {
        "X-Request-Id": request_id,
        "X-Audio-Format": audio_format,
        "X-Sample-Rate": str(result.sample_rate),
        "X-Channels": "1",
        "X-Audio-Duration-Ms": str(int(result.audio_seconds * 1000)),
        "X-Generation-Ms": str(int(result.generation_seconds * 1000)),
        "X-Rtf": f"{result.generation_seconds / max(result.audio_seconds, 1e-6):.3f}",
        "X-Engine": result.engine,
        "X-Voice-Id": result.voice_id,
        "X-Cache": "hit" if result.cache_hit else "miss",
        "X-Text-Chars": str(len(result.normalized_text)),
        "X-Warning-Count": str(len(result.warnings)),
        "X-Warnings-B64": _warnings_header(result.warnings),
        "Content-Length": str(payload_bytes),
        "Cache-Control": "no-store",
    }
    if result.experimental:
        headers["X-Experimental"] = "true"
    return headers


def _speech_request(payload: TTSRequest | TTSStreamRequest, auth: AuthContext) -> SpeechRequest:
    return SpeechRequest(
        text=payload.text,
        language=payload.language,
        voice_id=payload.voice_id,
        engine=payload.engine,
        profile=payload.profile,
        speed=payload.speed,
        use_cache=payload.use_cache,
        owner_key_id=auth.key_id,
    )


def _language_or_400(value: str) -> str:
    try:
        return canonical_language(value)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _voice_dict(profile, *, include_transcript: bool) -> dict:
    if hasattr(profile, "as_dict"):
        data = profile.as_dict(include_transcript=include_transcript)
    else:
        data = dict(profile)
        if not include_transcript:
            data.pop("transcript", None)
    data.pop("owner_key_id", None)
    return data


def create_app(
    settings: Settings | None = None,
    *,
    service: TTSService | None = None,
    accounts: AccountStore | None = None,
    auth: AuthManager | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    upload_dir = settings.cache_dir / "uploads"
    accounts = accounts or AccountStore(settings.api_db)
    auth = auth or AuthManager(settings, accounts)
    service = service or TTSService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.warmup:
            try:
                await asyncio.to_thread(service.router.warmup, settings.engine)  # type: ignore[attr-defined]
                LOGGER.info("Model warmup complete for configured engine=%s", settings.engine)
            except Exception as error:
                LOGGER.exception("Model warmup failed; /readyz will report not ready")
                app.state.warmup_error = f"{type(error).__name__}: {error}"
        yield

    app = FastAPI(
        title="Local Voice Studio API",
        version=APP_VERSION,
        description=(
            "Custom-reference Telugu and experimental Telugu-English speech. "
            "Local inference; no cloud TTS fallback. Voice cloning requires consent."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.settings = settings
    app.state.accounts = accounts
    app.state.auth = auth
    app.state.service = service

    if settings.api_cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.api_cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-Id"],
            expose_headers=[
                "X-Request-Id", "X-Audio-Format", "X-Sample-Rate", "X-Audio-Duration-Ms",
                "X-Generation-Ms", "X-Rtf", "X-Engine", "X-Voice-Id", "X-Cache",
                "X-Warning-Count", "X-Warnings-B64", "X-Experimental",
            ],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:16]}"
        request.state.request_id = request_id
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > settings.api_max_request_bytes:
                    return _error_response(
                        413, "request_too_large",
                        f"Request body exceeds {settings.api_max_request_bytes} bytes.",
                        request_id,
                    )
            except ValueError:
                return _error_response(400, "invalid_content_length", "Invalid Content-Length.", request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            METRICS.inc("tts_http_requests_total", labels={"status": "500"})
            raise
        elapsed = (time.perf_counter() - started) * 1000.0
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        METRICS.inc("tts_http_requests_total", labels={"status": str(response.status_code)})
        METRICS.observe("tts_http_request_duration_ms", elapsed)
        return response

    @app.exception_handler(TTSError)
    async def tts_error_handler(request: Request, exc: TTSError):
        return _error_response(_status_for_code(exc.code), exc.code, str(exc), _request_id(request))

    @app.exception_handler(VoiceError)
    async def voice_error_handler(request: Request, exc: VoiceError):
        return _error_response(400, "voice_error", str(exc), _request_id(request))

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return _error_response(
            422, "validation_error", "Request validation failed.",
            _request_id(request),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return _error_response(
            exc.status_code, _code_for_status(exc.status_code), detail, _request_id(request),
            headers=dict(exc.headers or {}),
        )

    # ---------- liveness / readiness ----------
    @app.get("/healthz", tags=["system"])
    async def healthz():
        return {"status": "ok", "version": APP_VERSION}

    @app.get("/readyz", tags=["system"])
    async def readyz(request: Request):
        error = getattr(request.app.state, "warmup_error", None)
        info = {}
        try:
            info = service.router.info()  # type: ignore[attr-defined]
        except Exception:
            info = {}
        # ``auto`` keeps IndicF5 as the validation-critical default. An operator
        # who explicitly selects Chatterbox can run (and become ready) without
        # IndicF5 weights; readiness follows the configured engine instead of
        # silently requiring the unselected one.
        readiness_engine = "indicf5" if settings.engine == "auto" else settings.engine
        selected = info.get(readiness_engine, {}) if isinstance(info, dict) else {}
        loaded = bool(selected.get("loaded"))
        ready = loaded or not settings.warmup
        payload = {
            "status": "ready" if ready and not error else "not_ready",
            "model_loaded": loaded,
            "readiness_engine": readiness_engine,
            "configured_engine": settings.engine,
            "warmup_error": error,
            "engine": info,
        }
        return JSONResponse(status_code=200 if ready and not error else 503, content=payload)

    # ---------- account ----------
    @app.get("/v1/me", response_model=MeResponse, tags=["account"])
    async def me(request: Request, auth_context: AuthContext = Depends(require_scope("tts"))):
        usage = accounts.usage(auth_context.account)
        return MeResponse(
            key_id=auth_context.key_id,
            name=auth_context.account.name,
            scopes=list(auth_context.account.scopes),
            usage=UsageResponse(**usage.as_dict()),
        )

    @app.get("/v1/capabilities", response_model=CapabilitiesResponse, tags=["system"])
    async def capabilities(auth_context: AuthContext = Depends(require_scope("tts"))):
        return service.capabilities()  # type: ignore[attr-defined]

    @app.get("/v1/health", tags=["system"])
    async def health(auth_context: AuthContext = Depends(require_scope("tts"))):
        return service.health()  # type: ignore[attr-defined]

    @app.get("/metrics", include_in_schema=False)
    async def metrics(request: Request, auth_context: AuthContext = Depends(require_scope("admin"))):
        if not settings.api_metrics_enabled:
            raise HTTPException(status_code=404, detail="Metrics are disabled.")
        return Response(content=METRICS.render(), media_type="text/plain; version=0.0.4")

    # ---------- TTS ----------
    @app.post("/v1/tts", tags=["speech"], responses={200: {"content": {"audio/wav": {}}}})
    async def synthesize(
        request: Request,
        payload: TTSRequest,
        auth_context: AuthContext = Depends(require_scope("tts")),
    ):
        request_id = _request_id(request)
        chars = len(payload.text)
        if not accounts.reserve_chars(auth_context.account, chars):
            raise TTSError("Monthly character quota exhausted.", "quota_exceeded")
        speech = _speech_request(payload, auth_context)
        try:
            with timer(METRICS, "tts_synthesis_seconds", labels={"engine": payload.engine}):
                result = await asyncio.to_thread(service.synthesize, speech)
            payload_bytes, media_type = await asyncio.to_thread(
                _encode_result, result, payload.format
            )
        except Exception:
            accounts.refund_chars(auth_context.account, chars)
            raise
        headers = _audio_headers(
            result=result,
            request_id=request_id,
            audio_format=payload.format,
            payload_bytes=len(payload_bytes),
        )
        accounts.record_audio_ms(auth_context.account, int(result.audio_seconds * 1000))
        METRICS.inc("tts_chars_total", chars, labels={"engine": result.engine})
        METRICS.inc("tts_audio_ms_total", int(result.audio_seconds * 1000), labels={"engine": result.engine})
        return Response(content=payload_bytes, media_type=media_type, headers=headers)

    @app.post(
        "/v1/tts/stream",
        tags=["speech"],
        responses={200: {"content": {"audio/L16": {}}}},
    )
    async def synthesize_stream(
        request: Request,
        payload: TTSStreamRequest,
        auth_context: AuthContext = Depends(require_scope("tts")),
    ):
        request_id = _request_id(request)
        chars = len(payload.text)
        if not accounts.reserve_chars(auth_context.account, chars):
            raise TTSError("Monthly character quota exhausted.", "quota_exceeded")
        speech = _speech_request(payload, auth_context)
        try:
            plan = await asyncio.to_thread(service.plan_stream, speech)
        except Exception:
            accounts.refund_chars(auth_context.account, chars)
            raise
        cancel_event = threading.Event()
        stream = ThreadedStream(
            lambda: service.stream(plan, cancel_event=cancel_event),
            queue_size=4,
        )
        iterator = stream.__aiter__()
        # Prime the first chunk before response headers.  A model that loads but
        # fails during generation then returns a JSON error instead of a 200
        # response with an empty body.
        try:
            first_chunk = await anext(iterator)
        except StopAsyncIteration:
            stream.cancel()
            accounts.refund_chars(auth_context.account, chars)
            raise TTSError(
                f"The {plan.engine} engine returned no audio.",
                "generation_failed",
            )
        except Exception:
            stream.cancel()
            accounts.refund_chars(auth_context.account, chars)
            raise

        headers = {
            "X-Request-Id": request_id,
            "X-Audio-Format": "pcm_s16le",
            "X-Sample-Rate": "24000",
            "X-Channels": "1",
            "X-Engine": plan.engine,
            "X-Voice-Id": plan.voice.voice_id,
            "X-Cache": "hit" if plan.cached_audio is not None else "miss",
            "X-Warning-Count": str(len(plan.warnings)),
            "X-Warnings-B64": _warnings_header(plan.warnings),
            "Cache-Control": "no-store",
        }
        if plan.experimental:
            headers["X-Experimental"] = "true"

        async def audio_iter():
            sent_samples = 0
            try:
                data = to_pcm16(first_chunk.audio)
                if data:
                    sent_samples += len(first_chunk.audio)
                    yield data
                async for chunk in iterator:
                    data = to_pcm16(chunk.audio)
                    if data:
                        sent_samples += len(chunk.audio)
                        yield data
            except asyncio.CancelledError:
                cancel_event.set()
                raise
            except TTSError as error:
                if error.code != "cancelled":
                    LOGGER.exception("Streaming synthesis failed after headers were sent")
            except Exception:
                LOGGER.exception("Streaming synthesis failed after headers were sent")
            finally:
                stream.cancel()
                if sent_samples == 0:
                    accounts.refund_chars(auth_context.account, chars)
                else:
                    accounts.record_audio_ms(
                        auth_context.account, int(sent_samples / 24_000 * 1000)
                    )
                    METRICS.inc("tts_stream_audio_ms_total", int(sent_samples / 24_000 * 1000))
                METRICS.inc("tts_stream_requests_total")

        return StreamingResponse(
            audio_iter(),
            media_type="audio/L16",
            headers=headers,
        )

    # ---------- text helpers ----------
    @app.post("/v1/transliterate", response_model=TransliterateResponse, tags=["text"])
    async def transliterate(
        payload: TransliterateRequest,
        auth_context: AuthContext = Depends(require_scope("tts")),
    ):
        telugu = await asyncio.to_thread(service.transliterate, payload.text)
        return TransliterateResponse(telugu=telugu)

    @app.post("/v1/preview", response_model=PreviewResponse, tags=["text"])
    async def preview(
        payload: PreviewRequest,
        auth_context: AuthContext = Depends(require_scope("tts")),
    ):
        _language_or_400(payload.language)
        result = await asyncio.to_thread(service.preview, payload.text, payload.language)
        return PreviewResponse(**result)

    # ---------- voice library ----------
    @app.get("/v1/voices", response_model=VoiceListResponse, tags=["voices"])
    async def list_voices(
        include_transcript: bool = Query(False),
        auth_context: AuthContext = Depends(require_scope("voices")),
    ):
        voices = service.voices.list(  # type: ignore[attr-defined]
            include_transcript=include_transcript,
            owner_key_id=auth_context.key_id,
            allow_admin=auth_context.account.has_scope("admin"),
        )
        return VoiceListResponse(
            voices=[
                VoiceResponse(**_voice_dict(voice, include_transcript=include_transcript))
                for voice in voices
            ]
        )

    @app.post("/v1/voices", response_model=VoiceResponse, tags=["voices"])
    async def create_voice(
        name: str = Form(..., min_length=1, max_length=100),
        transcript: str = Form(..., min_length=1, max_length=5000),
        consent: bool = Form(...),
        language: str = Form("Telugu"),
        engine_hint: str = Form("indicf5"),
        audio: UploadFile = File(...),
        auth_context: AuthContext = Depends(require_scope("voices")),
    ):
        if not consent:
            raise HTTPException(
                status_code=400,
                detail="Explicit consent is required to clone a voice.",
            )
        language = _language_or_400(language)
        if engine_hint not in {"indicf5", "chatterbox", "auto"}:
            raise HTTPException(status_code=400, detail="engine_hint must be indicf5, chatterbox or auto.")
        upload_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(audio.filename or "upload.wav").suffix[:10] or ".wav"
        fd, temp_name = tempfile.mkstemp(
            prefix="voice-upload-", suffix=suffix, dir=upload_dir
        )
        temp_path = Path(temp_name)
        try:
            total = 0
            with os.fdopen(fd, "wb") as handle:
                while True:
                    block = await audio.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > settings.api_max_request_bytes:
                        raise HTTPException(status_code=413, detail="Reference audio is too large.")
                    handle.write(block)
            profile = await asyncio.to_thread(
                service.voices.create,  # type: ignore[attr-defined]
                name,
                temp_path,
                transcript,
                language=language,
                consent=True,
                engine_hint=engine_hint,
                owner_key_id=auth_context.key_id,
            )
        finally:
            temp_path.unlink(missing_ok=True)
        return VoiceResponse(**_voice_dict(profile, include_transcript=True))

    @app.get("/v1/voices/{voice_id}", response_model=VoiceResponse, tags=["voices"])
    async def get_voice(
        voice_id: str,
        auth_context: AuthContext = Depends(require_scope("voices")),
    ):
        try:
            profile = service.voices.get(  # type: ignore[attr-defined]
                voice_id,
                owner_key_id=auth_context.key_id,
                allow_admin=auth_context.account.has_scope("admin"),
            )
        except VoiceError as error:
            raise TTSError(str(error), "voice_not_found") from error
        return VoiceResponse(**_voice_dict(profile, include_transcript=True))

    @app.delete("/v1/voices/{voice_id}", tags=["voices"])
    async def delete_voice(
        voice_id: str,
        auth_context: AuthContext = Depends(require_scope("voices")),
    ):
        try:
            await asyncio.to_thread(
                service.voices.delete,  # type: ignore[attr-defined]
                voice_id,
                owner_key_id=auth_context.key_id,
                allow_admin=auth_context.account.has_scope("admin"),
            )
        except VoiceError as error:
            raise TTSError(str(error), "voice_not_found") from error
        return {"deleted": voice_id}

    # ---------- WebSocket streaming ----------
    @app.websocket("/v1/tts/ws")
    async def tts_websocket(websocket: WebSocket):
        await websocket.accept()
        raw_key = None
        authorization = websocket.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            raw_key = authorization[7:].strip()
        raw_key = raw_key or websocket.headers.get("x-api-key")
        try:
            start = await asyncio.wait_for(websocket.receive_json(), timeout=20.0)
        except (asyncio.TimeoutError, WebSocketDisconnect):
            await websocket.close(code=4400, reason="Expected a start message.")
            return
        if not isinstance(start, dict) or start.get("type") != "start":
            await websocket.close(code=4400, reason="First message must be type=start.")
            return
        raw_key = raw_key or str(start.get("api_key") or "")
        auth_context = auth.authenticate(raw_key)
        if auth_context is None:
            await websocket.close(code=4401, reason="Missing or invalid API key.")
            return
        if not auth_context.account.has_scope("tts"):
            await websocket.close(code=4403, reason="API key lacks tts scope.")
            return
        allowed, retry_after = auth.rate_limiter.allow(
            auth_context.key_id, auth_context.account.rate_limit_per_min
        )
        if not allowed:
            await websocket.send_json(
                {"type": "error", "code": "rate_limited", "retry_after": retry_after}
            )
            await websocket.close(code=4429, reason="Rate limit exceeded.")
            return
        try:
            text = str(start.get("text") or "")
            voice_id = str(start.get("voice_id") or "")
            if not text or not voice_id:
                raise TTSError("text and voice_id are required.", "empty_text")
            requested_engine = start.get("engine")
            payload = TTSStreamRequest(
                text=text,
                language=str(start.get("language") or "Telugu"),
                voice_id=voice_id,
                engine=str(requested_engine) if requested_engine else None,
                profile=start.get("profile"),
                speed=start.get("speed"),
                use_cache=bool(start.get("use_cache", True)),
            )
        except Exception as error:
            await websocket.send_json({"type": "error", "code": "invalid_request", "message": str(error)})
            await websocket.close(code=4400)
            return
        chars = len(payload.text)
        if not accounts.reserve_chars(auth_context.account, chars):
            await websocket.send_json(
                {"type": "error", "code": "quota_exceeded", "message": "Monthly character quota exhausted."}
            )
            await websocket.close(code=4402)
            return
        speech = _speech_request(payload, auth_context)
        cancel_event = threading.Event()
        receiver_task = None
        sent_samples = 0
        try:
            plan = await asyncio.to_thread(service.plan_stream, speech)
            await websocket.send_json(
                {
                    "type": "ready",
                    "request_id": f"ws_{uuid.uuid4().hex[:16]}",
                    "normalized_text": plan.normalized.text,
                    "engine": plan.engine,
                    "voice_id": plan.voice.voice_id,
                    "sample_rate": 24000,
                    "audio_format": "pcm_s16le",
                    "cache_hit": plan.cached_audio is not None,
                    "warnings": list(plan.warnings),
                    "text_chars": len(plan.normalized.text),
                }
            )

            async def receive_control():
                try:
                    while True:
                        message = await websocket.receive_json()
                        if isinstance(message, dict) and message.get("type") == "cancel":
                            cancel_event.set()
                            return
                except (WebSocketDisconnect, RuntimeError):
                    cancel_event.set()

            receiver_task = asyncio.create_task(receive_control())
            stream = ThreadedStream(
                lambda: service.stream(
                    plan, cancel_event=cancel_event, collect_for_cache=True
                ),
                queue_size=4,
            )
            async for chunk in stream:
                if cancel_event.is_set():
                    break
                data = to_pcm16(chunk.audio)
                if data:
                    sent_samples += len(chunk.audio)
                    await websocket.send_bytes(data)
            await websocket.send_json(
                {
                    "type": "done",
                    "audio_ms": int(sent_samples / 24_000 * 1000),
                    "cancelled": cancel_event.is_set(),
                    "cache_hit": plan.cached_audio is not None,
                }
            )
        except (WebSocketDisconnect, RuntimeError):
            cancel_event.set()
        except TTSError as error:
            if error.code != "cancelled":
                cancel_event.set()
                try:
                    await websocket.send_json(
                        {"type": "error", "code": error.code, "message": str(error)}
                    )
                except Exception:
                    pass
            else:
                cancel_event.set()
        except Exception as error:
            LOGGER.exception("WebSocket synthesis failed")
            try:
                await websocket.send_json(
                    {"type": "error", "code": getattr(error, "code", "synthesis_failed"), "message": str(error)}
                )
            except Exception:
                pass
        finally:
            cancel_event.set()
            if receiver_task is not None:
                receiver_task.cancel()
                try:
                    await receiver_task
                except (asyncio.CancelledError, Exception):
                    pass
            if sent_samples == 0:
                accounts.refund_chars(auth_context.account, chars)
            else:
                accounts.record_audio_ms(auth_context.account, int(sent_samples / 24_000 * 1000))
            try:
                await websocket.close()
            except Exception:
                pass

    return app


def _encode_result(result, audio_format: str):
    from tts.audio_utils import encode_audio_bytes

    return encode_audio_bytes(result.audio, result.sample_rate, audio_format)


app = create_app()
