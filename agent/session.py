"""Conversation state and the streaming LLM -> sentence -> TTS pipeline."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import AsyncIterator

import numpy as np

from agent.llm import DEFAULT_SYSTEM_PROMPT
from agent.stt import SpeechToText
from agent.tts_client import TTSClient
from agent.vad import pcm16_to_float


_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "vs", "etc", "e.g", "i.e",
    "no", "fig", "figs", "st", "inc", "ltd", "govt", "approx", "dept", "univ",
    "p.m", "a.m", "u.s", "u.k", "డా", "శ్రీ", "శ్రీమతి", "ఉదా",
}
_END_PUNCT = re.compile(r"[.!?…।]+[\"')\]]*(?=\s|$)")
_COMMA = re.compile(r"[,;:](?=\s)")
LOGGER = logging.getLogger("local-voice-studio.agent.session")


class SentenceBuffer:
    """Incrementally split streamed LLM tokens into speakable phrases.

    The first phrase may end at a comma so the first audio starts sooner.  Later
    phrases wait for sentence-ending punctuation (or the max length), with a
    small abbreviation guard.
    """

    def __init__(self, *, min_chars: int = 8, max_chars: int = 240, first_min_chars: int = 4) -> None:
        if min_chars < 1 or max_chars < min_chars:
            raise ValueError("Invalid sentence buffer limits.")
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.first_min_chars = first_min_chars
        self._text = ""
        self._emitted = 0

    @property
    def buffered(self) -> str:
        return self._text

    def feed(self, token: str) -> list[str]:
        self._text += token
        output: list[str] = []
        while True:
            boundary = self._next_boundary()
            if boundary is None:
                break
            phrase = self._text[:boundary].strip()
            self._text = self._text[boundary:].lstrip()
            if phrase:
                output.append(phrase)
                self._emitted += 1
        return output

    def flush(self) -> list[str]:
        phrase = self._text.strip()
        self._text = ""
        if not phrase:
            return []
        self._emitted += 1
        return [phrase]

    def _next_boundary(self) -> int | None:
        # First phrase: a comma is an acceptable low-latency boundary.
        if self._emitted == 0:
            match = _COMMA.search(self._text)
            if match and len(self._text[:match.end()].strip()) >= self.first_min_chars:
                return match.end()
        for match in _END_PUNCT.finditer(self._text):
            candidate = self._text[:match.end()].strip()
            if len(candidate) < self.min_chars:
                continue
            if self._ends_in_abbreviation(candidate):
                continue
            return match.end()
        if len(self._text) >= self.max_chars:
            split = self._text.rfind(" ", 0, self.max_chars)
            if split <= 0:
                split = self.max_chars
            return split
        return None

    @staticmethod
    def _ends_in_abbreviation(text: str) -> bool:
        stripped = text.rstrip(".!?…।\"')] ").strip()
        last = stripped.rsplit(" ", 1)[-1].lower()
        return last in _ABBREVIATIONS


class ConversationSession:
    """One caller's conversation.

    ``handle_utterance`` performs STT and then streams response events.  Event
    types are JSON-friendly except ``audio``, whose ``data`` field is raw PCM16.
    """

    def __init__(
        self,
        stt: SpeechToText,
        llm,
        tts: TTSClient,
        *,
        voice_id: str,
        language: str = "Telugu",
        max_history_turns: int = 8,
        system_prompt: str | None = None,
        sentence_buffer_factory=SentenceBuffer,
    ) -> None:
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.voice_id = voice_id
        self.language = language
        self.max_history_turns = max(1, int(max_history_turns))
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._sentence_buffer_factory = sentence_buffer_factory
        self.history: list[dict] = []
        self._active_turns = 0

    @property
    def busy(self) -> bool:
        return self._active_turns > 0

    def reset(self) -> None:
        self.history.clear()

    async def handle_utterance(self, audio, sample_rate: int) -> AsyncIterator[dict]:
        # The WebSocket/VAD layer emits PCM16 bytes; accept float arrays too so
        # the session is convenient to drive from tests and other callers.
        if isinstance(audio, (bytes, bytearray, memoryview)):
            audio = pcm16_to_float(bytes(audio))
        else:
            audio = np.asarray(audio, dtype=np.float32)
        try:
            text = await self.stt.transcribe(audio, sample_rate)
        except Exception as error:
            yield {"type": "error", "code": "stt_failed", "message": str(error)}
            return
        text = (text or "").strip()
        if not text:
            yield {"type": "empty_transcript"}
            return
        yield {"type": "transcript", "text": text}
        async for event in self.respond(text):
            yield event

    async def respond(self, user_text: str) -> AsyncIterator[dict]:
        user_text = (user_text or "").strip()
        if not user_text:
            yield {"type": "empty_transcript"}
            return
        self.history.append({"role": "user", "content": user_text})
        self._trim_history()
        messages = [{"role": "system", "content": self.system_prompt}, *self.history]

        events: asyncio.Queue[dict] = asyncio.Queue(maxsize=24)
        sentences: asyncio.Queue[str | None] = asyncio.Queue(maxsize=8)
        producer = asyncio.create_task(self._produce(messages, sentences, events))
        speaker = asyncio.create_task(self._speak(sentences, events))
        self._active_turns += 1
        llm_done = False
        tts_done = False
        failed = False
        try:
            while not (llm_done and tts_done):
                event = await events.get()
                kind = event.get("type")
                if kind == "llm_done":
                    llm_done = True
                    continue
                if kind == "tts_done":
                    tts_done = True
                    continue
                if kind == "error":
                    failed = True
                    yield event
                    break
                yield event
            if not failed:
                assistant_text = await producer
                if not assistant_text:
                    failed = True
                    LOGGER.warning("LLM returned no speakable text; failing the turn explicitly")
                    yield {
                        "type": "error",
                        "code": "empty_llm_response",
                        "message": "The language model returned no speakable text.",
                    }
                else:
                    # Commit the answer only after it was fully produced.  A
                    # barge-in therefore cannot leave a half-sentence in history.
                    events_done = False
                    while not events.empty():
                        event = events.get_nowait()
                        if event.get("type") == "error":
                            yield event
                            events_done = True
                            break
                    if events_done:
                        failed = True
                    else:
                        self.history.append({"role": "assistant", "content": assistant_text})
                        self._trim_history()
                if not failed:
                    yield {"type": "turn_end", "text": assistant_text}
        finally:
            for task in (producer, speaker):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(producer, speaker, return_exceptions=True)
            self._active_turns = max(0, self._active_turns - 1)

    async def _produce(self, messages, sentences: asyncio.Queue, events: asyncio.Queue) -> str:
        buffer = self._sentence_buffer_factory()
        parts: list[str] = []
        try:
            async for delta in self.llm.stream(messages):
                if not delta:
                    continue
                parts.append(delta)
                await events.put({"type": "assistant_delta", "text": delta})
                for phrase in buffer.feed(delta):
                    await events.put({"type": "phrase", "text": phrase})
                    await sentences.put(phrase)
            for phrase in buffer.flush():
                await events.put({"type": "phrase", "text": phrase})
                await sentences.put(phrase)
            return "".join(parts).strip()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await events.put(
                {"type": "error", "code": "llm_failed", "message": str(error)}
            )
            return "".join(parts).strip()
        finally:
            await sentences.put(None)
            await events.put({"type": "llm_done"})

    async def _speak(self, sentences: asyncio.Queue, events: asyncio.Queue) -> None:
        index = 0
        try:
            while True:
                phrase = await sentences.get()
                if phrase is None:
                    break
                async for chunk in self.tts.stream_pcm(
                    phrase, voice_id=self.voice_id, language=self.language
                ):
                    if chunk:
                        await events.put({"type": "audio", "data": chunk})
                index += 1
                await events.put({"type": "audio_end", "text": phrase, "index": index})
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await events.put(
                {
                    "type": "error",
                    "code": "tts_failed",
                    "tts_code": getattr(error, "code", "tts_error"),
                    "status_code": getattr(error, "status_code", 0),
                    "message": str(error),
                }
            )
        finally:
            await events.put({"type": "tts_done"})

    def _trim_history(self) -> None:
        limit = self.max_history_turns * 2
        if len(self.history) > limit:
            self.history = self.history[-limit:]
