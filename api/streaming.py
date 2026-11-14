"""Bridge a blocking TTS generator to ASGI streaming responses/WebSockets."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Generic, Iterable, TypeVar


T = TypeVar("T")


class StreamCancelled(RuntimeError):
    pass


class ThreadedStream(Generic[T]):
    """Consume a blocking iterator from a worker thread without blocking the loop.

    The queue is deliberately small so a slow client applies backpressure to the
    model worker instead of buffering an unbounded amount of audio in memory.
    """

    def __init__(self, factory: Callable[[], Iterable[T]], *, queue_size: int = 8) -> None:
        self._factory = factory
        self._queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue(maxsize=queue_size)
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._started = False

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def _run(self) -> None:
        assert self._loop is not None
        try:
            for item in self._factory():
                if self._cancel.is_set():
                    break
                future = asyncio.run_coroutine_threadsafe(self._queue.put(("item", item)), self._loop)
                while True:
                    try:
                        future.result(timeout=0.5)
                        break
                    except TimeoutError:
                        if self._cancel.is_set():
                            future.cancel()
                            return
            if not self._cancel.is_set():
                asyncio.run_coroutine_threadsafe(self._queue.put(("done", None)), self._loop).result()
        except Exception as error:  # delivered to the async consumer
            try:
                asyncio.run_coroutine_threadsafe(self._queue.put(("error", error)), self._loop).result(timeout=5)
            except Exception:
                pass
        finally:
            self._cancel.set()

    def _start(self) -> None:
        if self._started:
            return
        self._started = True
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._run, name="tts-stream-worker", daemon=True)
        self._thread.start()

    async def __aiter__(self) -> AsyncIterator[T]:
        self._start()
        try:
            while True:
                kind, payload = await self._queue.get()
                if kind == "item":
                    yield payload  # type: ignore[misc]
                elif kind == "done":
                    return
                elif kind == "error":
                    if isinstance(payload, StreamCancelled):
                        return
                    raise payload  # type: ignore[misc]
        finally:
            self.cancel()
            if self._thread is not None and self._thread.is_alive():
                # Do not block the event loop waiting for a forward pass; the
                # worker observes the cancel flag before the next chunk.
                self._thread.join(timeout=0.05)


@dataclass
class StreamResult:
    """Metadata sent before/after binary audio on a WebSocket."""

    sample_rate: int
    duration_seconds: float
    engine: str
    voice_id: str
    cache_hit: bool
    normalized_text: str
    warnings: list[str]
