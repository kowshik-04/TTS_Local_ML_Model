import asyncio
import unittest

from agent.session import ConversationSession
from agent.vad import float_to_pcm16
from tests.support import FakeLLM, FakeSTT, FakeTTSClient, tone


async def collect(events):
    return [event async for event in events]


class AgentSessionTests(unittest.IsolatedAsyncioTestCase):
    def make_session(self, **overrides):
        stt = overrides.pop("stt", FakeSTT())
        llm = overrides.pop("llm", FakeLLM())
        tts = overrides.pop("tts", FakeTTSClient())
        session = ConversationSession(
            stt,
            llm,
            tts,
            voice_id="voice-a",
            language="Telugu",
            **overrides,
        )
        return session, stt, llm, tts

    async def test_respond_streams_text_phrases_and_audio(self):
        session, stt, llm, tts = self.make_session()
        events = await collect(session.respond("నమస్కారం"))

        kinds = [event["type"] for event in events]
        self.assertEqual(kinds.count("assistant_delta"), 3)
        self.assertEqual(kinds.count("phrase"), 3)
        self.assertEqual(kinds.count("audio"), 6)
        self.assertEqual(kinds.count("audio_end"), 3)
        self.assertEqual(kinds[-1], "turn_end")
        self.assertEqual(
            [event["index"] for event in events if event["type"] == "audio_end"],
            [1, 2, 3],
        )
        self.assertEqual([call["text"] for call in tts.calls], [
            "నమస్కారం,",
            "ఎలా ఉన్నారు?",
            "ధన్యవాదాలు.",
        ])
        self.assertEqual(llm.calls[0][0]["role"], "system")
        self.assertEqual(llm.calls[0][-1], {"role": "user", "content": "నమస్కారం"})
        self.assertEqual(session.history, [
            {"role": "user", "content": "నమస్కారం"},
            {"role": "assistant", "content": "నమస్కారం, ఎలా ఉన్నారు? ధన్యవాదాలు."},
        ])
        self.assertFalse(session.busy)

    async def test_handle_utterance_transcribes_pcm16_bytes(self):
        session, stt, _, _ = self.make_session(stt=FakeSTT("ధన్యవాదాలు."))
        audio = float_to_pcm16(tone(0.2, rate=16_000, amplitude=0.1))
        events = await collect(session.handle_utterance(audio, 16_000))

        self.assertEqual(events[0], {"type": "transcript", "text": "ధన్యవాదాలు."})
        self.assertEqual(stt.calls[0]["sample_rate"], 16_000)
        self.assertEqual(stt.calls[0]["length"], 3200)
        self.assertEqual(events[-1]["type"], "turn_end")

    async def test_empty_transcript_does_not_call_llm(self):
        session, _, llm, tts = self.make_session(stt=FakeSTT("   "))
        events = await collect(session.handle_utterance(b"\x00\x00" * 100, 16_000))
        self.assertEqual(events, [{"type": "empty_transcript"}])
        self.assertEqual(llm.calls, [])
        self.assertEqual(tts.calls, [])
        self.assertEqual(session.history, [])

    async def test_empty_text_does_not_call_llm(self):
        session, _, llm, _ = self.make_session()
        events = await collect(session.respond("  "))
        self.assertEqual(events, [{"type": "empty_transcript"}])
        self.assertEqual(llm.calls, [])

    async def test_stt_error_is_reported(self):
        session, *_ = self.make_session(stt=FakeSTT(error=RuntimeError("no model")))
        events = await collect(session.handle_utterance(b"\x00\x00" * 10, 16_000))
        self.assertEqual(events[0]["type"], "error")
        self.assertEqual(events[0]["code"], "stt_failed")
        self.assertIn("no model", events[0]["message"])
        self.assertEqual(session.history, [])

    async def test_llm_error_is_reported_without_committing_assistant(self):
        session, _, _, _ = self.make_session(llm=FakeLLM(error=RuntimeError("offline")))
        events = await collect(session.respond("ప్రశ్న"))
        self.assertEqual(events[0]["type"], "error")
        self.assertEqual(events[0]["code"], "llm_failed")
        self.assertEqual([event["type"] for event in events], ["error"])
        self.assertEqual(session.history, [{"role": "user", "content": "ప్రశ్న"}])

    async def test_empty_llm_response_is_reported_without_committing_assistant(self):
        class EmptyLLM:
            async def stream(self, messages):
                if False:
                    yield ""

        session, _, _, tts = self.make_session(llm=EmptyLLM())
        events = await collect(session.respond("ప్రశ్న"))
        self.assertEqual(events[0]["type"], "error")
        self.assertEqual(events[0]["code"], "empty_llm_response")
        self.assertNotIn("turn_end", [event["type"] for event in events])
        self.assertEqual(tts.calls, [])
        self.assertEqual(session.history, [{"role": "user", "content": "ప్రశ్న"}])

    async def test_partial_llm_failure_does_not_commit_assistant(self):
        class PartialFailureLLM:
            async def stream(self, messages):
                yield "Hello, "
                raise RuntimeError("connection dropped")

        session, _, _, tts = self.make_session(llm=PartialFailureLLM())
        events = await collect(session.respond("hi"))
        kinds = [event["type"] for event in events]
        self.assertIn("assistant_delta", kinds)
        self.assertIn("error", kinds)
        self.assertNotIn("turn_end", kinds)
        self.assertEqual(session.history, [{"role": "user", "content": "hi"}])
        self.assertGreaterEqual(len(tts.calls), 1)

    async def test_tts_error_is_reported(self):
        session, _, _, _ = self.make_session(
            tts=FakeTTSClient(error=RuntimeError("synthesis failed"))
        )
        events = await collect(session.respond("ప్రశ్న"))
        errors = [event for event in events if event["type"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["code"], "tts_failed")
        self.assertEqual(session.history, [{"role": "user", "content": "ప్రశ్న"}])

    async def test_cancellation_does_not_commit_assistant(self):
        tts = FakeTTSClient(block_after_first=True, block_seconds=30.0)
        session, _, _, _ = self.make_session(tts=tts)
        audio_seen = asyncio.Event()

        async def consume():
            async for event in session.respond("ప్రశ్న"):
                if event["type"] == "audio":
                    audio_seen.set()

        task = asyncio.create_task(consume())
        await asyncio.wait_for(audio_seen.wait(), timeout=2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)

        self.assertEqual(session.history, [{"role": "user", "content": "ప్రశ్న"}])
        self.assertFalse(session.busy)

    async def test_history_is_trimmed(self):
        session, _, _, _ = self.make_session(
            llm=FakeLLM(["One."]),
            tts=FakeTTSClient([b"\x00\x00"]),
            max_history_turns=1,
        )
        await collect(session.respond("First"))
        await collect(session.respond("Second"))
        self.assertEqual(session.history, [
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "One."},
        ])

    async def test_reset_clears_history(self):
        session, _, _, _ = self.make_session(
            llm=FakeLLM(["One."]),
            tts=FakeTTSClient([b"\x00\x00"]),
        )
        await collect(session.respond("First"))
        self.assertEqual(len(session.history), 2)
        session.reset()
        self.assertEqual(session.history, [])

    async def test_custom_system_prompt(self):
        session, _, llm, _ = self.make_session(
            llm=FakeLLM(["One."]),
            tts=FakeTTSClient([b"\x00\x00"]),
            system_prompt="Be brief.",
        )
        await collect(session.respond("hello"))
        self.assertEqual(llm.calls[0][0], {"role": "system", "content": "Be brief."})

    async def test_audio_chunks_are_forwarded_unchanged(self):
        chunks = [b"abc", b"def"]
        session, _, _, _ = self.make_session(
            llm=FakeLLM(["Hello."]),
            tts=FakeTTSClient(chunks),
        )
        events = await collect(session.respond("hello"))
        audio = [event["data"] for event in events if event["type"] == "audio"]
        self.assertEqual(audio, chunks)


if __name__ == "__main__":
    unittest.main()
