from __future__ import annotations

import contextlib
import asyncio
import io
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    LLMContextFrame,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext

from memory.caller_identity import DEFAULT_TEST_CALLER_ID, resolve_caller_id
from memory.integration import (
    VoiceMemContextProcessor,
    VoiceMemIngestProcessor,
    is_caller_history_query,
    is_useful_memory,
    needs_memory,
    temporary_memory_instruction,
)
from memory.voicemem_adapter import VoiceMemAdapter


def client_for(handler):
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:8765")


class VoiceMemAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_caller_identity_is_stable(self):
        with patch.dict(os.environ, {"TEST_CALLER_ID": "terminal_ali_001"}):
            self.assertEqual(resolve_caller_id(), "terminal_ali_001")
            self.assertEqual(resolve_caller_id(), "terminal_ali_001")
        with patch.dict(os.environ, {"TEST_CALLER_ID": ""}):
            self.assertEqual(resolve_caller_id(), DEFAULT_TEST_CALLER_ID)

    async def test_success_and_caller_id_are_sent_to_sidecar(self):
        seen = []

        def handler(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={
                    "alive": True, "voicemem_available": True, "error_type": None})
            seen.append(request.read().decode())
            return httpx.Response(200, json={
                "ok": True, "memories": ["Runs a dental clinic"], "count": 1,
                "latency_ms": 4.0})

        client = client_for(handler)
        adapter = VoiceMemAdapter(client=client)
        self.assertTrue(await adapter.initialize())
        result = await adapter.retrieve("caller_a", "What business do I run?")
        self.assertEqual(result["memories"], ["Runs a dental clinic"])
        self.assertIn('"caller_id":"caller_a"', seen[0])
        await client.aclose()

    async def test_unavailable_sidecar_is_fail_open(self):
        async def fail(_request):
            raise httpx.ConnectError("unavailable")

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(fail), base_url="http://127.0.0.1:8765")
        adapter = VoiceMemAdapter(client=client)
        self.assertFalse(await adapter.initialize())
        self.assertEqual(
            await adapter.retrieve("caller", "remember me"),
            {"memories": [], "latency_ms": None},
        )
        await client.aclose()

    async def test_timeout_is_fail_open(self):
        async def timeout(_request):
            raise httpx.ReadTimeout("timed out")

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(timeout), base_url="http://127.0.0.1:8765")
        adapter = VoiceMemAdapter(client=client)
        self.assertFalse(await adapter.initialize())
        self.assertIn("ReadTimeout", adapter.health()["error"])
        await client.aclose()

    async def test_invalid_search_response_is_fail_open(self):
        def handler(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={
                    "alive": True, "voicemem_available": True})
            return httpx.Response(200, json={"ok": True, "memories": "not-a-list", "count": 1})

        client = client_for(handler)
        adapter = VoiceMemAdapter(client=client)
        await adapter.initialize()
        result = await adapter.retrieve("caller", "What did I say?")
        self.assertEqual(result, {"memories": [], "latency_ms": None})
        self.assertFalse(adapter.health()["enabled"])
        await client.aclose()

    async def test_ingest_failures_are_safe(self):
        cases = [
            httpx.ConnectError("unavailable"),
            httpx.ReadTimeout("timed out"),
            None,
        ]
        for failure in cases:
            with self.subTest(failure=type(failure).__name__ if failure else "invalid"):
                def handler(request, failure=failure):
                    if request.url.path == "/health":
                        return httpx.Response(200, json={
                            "alive": True, "voicemem_available": True})
                    if failure:
                        raise failure
                    return httpx.Response(200, json={"ok": False})

                client = client_for(handler)
                adapter = VoiceMemAdapter(client=client, write_timeout_seconds=0.01)
                self.assertTrue(await adapter.initialize())
                self.assertFalse(await adapter.ingest("caller", "I run a clinic."))
                await client.aclose()

    async def test_zero_new_memory_is_a_safe_noop(self):
        def handler(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={
                    "alive": True, "voicemem_available": True})
            return httpx.Response(200, json={
                "ok": True, "persisted": False, "memory_count": 0})

        client = client_for(handler)
        adapter = VoiceMemAdapter(client=client)
        self.assertTrue(await adapter.initialize())
        self.assertIsNone(await adapter.ingest("caller", "I run a clinic."))
        self.assertTrue(adapter.health()["enabled"])
        await client.aclose()

    async def test_legacy_sidecar_write_confirmation_supports_new_callers(self):
        memories = {"existing": ["Runs a bakery"]}

        def handler(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={
                    "alive": True, "voicemem_available": True})
            payload = json.loads(request.read())
            if request.url.path == "/memory/ingest":
                memories.setdefault(payload["caller_id"], []).append(payload["content"])
                return httpx.Response(200, json={
                    "ok": True, "memory_count": 1})
            found = memories.get(payload["caller_id"], [])
            return httpx.Response(200, json={
                "ok": True, "memories": found, "count": len(found),
                "latency_ms": 1.0})

        client = client_for(handler)
        adapter = VoiceMemAdapter(client=client)
        self.assertTrue(await adapter.initialize())
        self.assertTrue(await adapter.ingest("existing", "Prefers mornings"))
        self.assertTrue(await adapter.ingest("brand_new", "My name is Ali."))
        self.assertTrue(await adapter.ingest("brand_new", "Runs a dental clinic"))

        # Switch A -> B -> A without restarting the sidecar.
        existing = await adapter.retrieve("existing", "business")
        brand_new = await adapter.retrieve("brand_new", "identity")
        existing_again = await adapter.retrieve("existing", "preference")
        self.assertEqual(existing["memories"], ["Runs a bakery", "Prefers mornings"])
        self.assertEqual(
            brand_new["memories"], ["My name is Ali.", "Runs a dental clinic"])
        self.assertEqual(existing_again["memories"], existing["memories"])

        # A new main client sees the same sidecar-backed caller namespaces.
        restarted_client = client_for(handler)
        restarted_adapter = VoiceMemAdapter(client=restarted_client)
        self.assertTrue(await restarted_adapter.initialize())
        persisted = await restarted_adapter.retrieve("brand_new", "What do I run?")
        self.assertEqual(persisted["memories"], brand_new["memories"])
        self.assertNotIn("Runs a bakery", persisted["memories"])
        self.assertNotIn("Runs a dental clinic", existing["memories"])
        await restarted_client.aclose()
        await client.aclose()

    async def test_one_caller_ingest_failure_does_not_disable_other_callers(self):
        def handler(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={
                    "alive": True, "voicemem_available": True})
            payload = json.loads(request.read())
            if payload["caller_id"] == "broken":
                return httpx.Response(200, json={
                    "ok": False, "persisted": False, "memory_count": 0,
                    "error_type": "ValueError"})
            return httpx.Response(200, json={
                "ok": True, "persisted": True, "memory_count": 1})

        client = client_for(handler)
        adapter = VoiceMemAdapter(client=client)
        self.assertTrue(await adapter.initialize())
        self.assertFalse(await adapter.ingest("broken", "My name is Broken."))
        self.assertTrue(adapter.health()["enabled"])
        self.assertTrue(await adapter.ingest("healthy", "My name is Healthy."))
        self.assertTrue(adapter.health()["enabled"])
        await client.aclose()

    async def test_live_processor_never_ingests_and_context_is_temporary(self):
        adapter = AsyncMock(spec=VoiceMemAdapter)
        adapter.initialize.return_value = True
        adapter.retrieve.return_value = {
            "memories": ["The caller runs a dental clinic."], "latency_ms": 3.2}
        processor = VoiceMemContextProcessor(adapter)
        processor.push_frame = AsyncMock()
        original = LLMContext(messages=[
            {"role": "user", "content": "What business did I tell you I run?"}])
        frame = LLMContextFrame(context=original)

        with contextlib.redirect_stdout(io.StringIO()):
            await processor.process_frame(frame, None)
            await processor.process_frame(frame, None)

        adapter.retrieve.assert_awaited_once()
        adapter.ingest.assert_not_called()
        self.assertEqual(len(original.messages), 1)
        pushed = processor.push_frame.await_args_list[0].args[0]
        self.assertIsNot(pushed.context, original)
        self.assertIn("[CALLER MEMORY]", pushed.context.messages[0]["content"])

    async def test_routing_skips_greeting_and_retrieves_explicit_memory_question(self):
        positive_queries = [
            "What is my name?",
            "What's my name?",
            "Do you remember my name?",
            "Who am I?",
            "What business do I run?",
            "Which company did I say I own?",
            "What job did I mention before?",
            "What did I tell you about my business?",
            "What do you know about me?",
            "Which meeting time did I prefer?",
            "What time do I prefer meetings?",
            "Do you remember my meeting preference?",
            "What do I prefer?",
            "Do you remember what I prefer?",
            "What did I tell you last time?",
            "Which problem did I mention?",
            "What option did I choose?",
            "And what about my name?",
            "And what about my preferred time?",
        ]
        negative_queries = [
            "What is your name?",
            "What is Agentix Labs AI?",
            "What time is the meeting?",
            "What services does Agentix offer?",
            "Tell me about PongVerse.",
            "Hello",
            "Thanks",
        ]
        for query in positive_queries:
            with self.subTest(query=query):
                self.assertTrue(needs_memory(query))
        for query in negative_queries:
            with self.subTest(query=query):
                self.assertFalse(needs_memory(query))

    async def test_current_workflow_history_routes_by_intent_category(self):
        variants = [
            "Tell me how do we manage our leads.",
            "How do I currently manage leads?",
            "What system do we use for leads?",
            "How are we handling follow-up?",
            "What did I tell you about our current process?",
            "How does my business currently handle bookings?",
            "What problem did I say we have?",
            "How many leads did I say we receive?",
            "What do we currently do manually?",
        ]
        for query in variants:
            with self.subTest(query=query):
                self.assertTrue(is_caller_history_query(query))
                self.assertTrue(needs_memory(query))

    async def test_general_lead_advice_is_not_forced_into_memory_recall(self):
        advice_queries = [
            "How should we manage our leads?",
            "How can I improve lead management?",
            "Give me general advice for managing leads.",
        ]
        for query in advice_queries:
            with self.subTest(query=query):
                self.assertFalse(is_caller_history_query(query))
                self.assertFalse(needs_memory(query))

    async def test_recalled_workflow_volume_and_problem_are_strictly_grounded(self):
        cases = [
            ("How do we manage our leads?", "We manage our leads manually."),
            ("How many leads did I say we receive?", "The caller receives 200 leads weekly."),
            ("What problem did I say we have?", "The caller misses calls after hours."),
        ]
        for query, memory in cases:
            with self.subTest(query=query):
                adapter = AsyncMock(spec=VoiceMemAdapter)
                adapter.initialize.return_value = True
                adapter.retrieve.return_value = {
                    "memories": [memory], "latency_ms": 2.0}
                adapter.health.return_value = {"enabled": True, "error": None}
                processor = VoiceMemContextProcessor(adapter)
                result = await processor._retrieve_turn(query)
                self.assertIn(memory, result["instruction"])
                self.assertIn("current conversation facts first", result["instruction"])
                self.assertIn("Do not infer missing caller facts", result["instruction"])
                self.assertIn("Answer in one short sentence", result["instruction"])

    async def test_no_memory_history_question_gets_short_closed_world_fallback(self):
        adapter = AsyncMock(spec=VoiceMemAdapter)
        adapter.initialize.return_value = True
        adapter.retrieve.return_value = {"memories": [], "latency_ms": 2.0}
        adapter.health.return_value = {"enabled": True, "error": None}
        processor = VoiceMemContextProcessor(adapter)
        result = await processor._retrieve_turn("How do we manage our leads?")
        self.assertIn("No relevant stored caller fact was found", result["instruction"])
        self.assertIn(
            'respond exactly: "I don\'t have that detail yet."',
            result["instruction"],
        )

    async def test_memory_instruction_has_no_internal_metadata(self):
        instruction = temporary_memory_instruction(["Prefers evening meetings"])
        self.assertIn("Prefers evening meetings", instruction)
        self.assertNotIn("caller_id", instruction)
        self.assertNotIn("score", instruction)
        self.assertNotIn("storage", instruction)

    async def test_usefulness_filter_allows_facts_and_skips_noise_or_contact_pii(self):
        useful = [
            "My name is Hamza.",
            "I run a real estate business.",
            "I have a dental clinic.",
            "We miss customer calls after business hours.",
            "Yes, I prefer evening meetings.",
            "I want to automate our lead booking process.",
        ]
        skipped = [
            "Hello.",
            "Okay",
            "Sure.",
            "Thanks",
            "My email is hamza@example.com.",
            "My phone number is 03001234567.",
            "Tomorrow at 5 PM.",
            "I want to book a meeting tomorrow at 5 PM.",
            "I am fine.",
            "What is my name?",
            "What's my name?",
            "Who am I?",
            "What business do I run?",
            "What did I tell you about my business?",
            "Which meeting time do I prefer?",
            "Do you remember my preference?",
            "What do you know about me?",
        ]
        for text in useful:
            with self.subTest(text=text):
                self.assertTrue(is_useful_memory(text))
        for text in skipped:
            with self.subTest(text=text):
                self.assertFalse(is_useful_memory(text))

    async def test_assertions_queue_but_recall_questions_do_not(self):
        adapter = AsyncMock(spec=VoiceMemAdapter)
        adapter.initialize.return_value = True
        adapter.ingest.return_value = True
        processor = VoiceMemIngestProcessor(adapter)
        processor.push_frame = AsyncMock()
        processor.create_task = lambda coro, name=None: asyncio.create_task(coro)
        assertions = [
            "My name is Hamza.",
            "I run a real estate business.",
            "I prefer morning meetings.",
            "We struggle with missed calls.",
            "I want to automate our lead booking process.",
            "Uh, I run a dental clinic.",
            "Yeah, I actually prefer evenings.",
        ]
        recall_questions = [
            "What is my name?",
            "Who am I?",
            "What business do I run?",
            "What did I tell you about my business?",
            "Which meeting time do I prefer?",
            "Do you remember my preference?",
            "What do you know about me?",
        ]
        frames = [
            TranscriptionFrame(text, "user", str(index), finalized=True)
            for index, text in enumerate(assertions + recall_questions)
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            for frame in frames:
                await processor.process_frame(frame, None)
            await asyncio.sleep(0)
        self.assertEqual(adapter.ingest.await_count, len(assertions))
        self.assertEqual(
            [call.args[1] for call in adapter.ingest.await_args_list],
            assertions,
        )

    async def test_only_finalized_user_turns_are_queued_and_duplicates_are_skipped(self):
        adapter = AsyncMock(spec=VoiceMemAdapter)
        adapter.initialize.return_value = True
        adapter.ingest.return_value = True
        processor = VoiceMemIngestProcessor(adapter)
        processor.push_frame = AsyncMock()
        processor.create_task = lambda coro, name=None: asyncio.create_task(coro)

        interim = InterimTranscriptionFrame(
            "I run a clinic", "user", "now")
        partial = TranscriptionFrame(
            "I run a clinic", "user", "now", finalized=False)
        final = TranscriptionFrame(
            "I run a dental clinic.", "user", "now", finalized=True)
        with contextlib.redirect_stdout(io.StringIO()):
            await processor.process_frame(interim, None)
            await processor.process_frame(partial, None)
            await processor.process_frame(final, None)
            await processor.process_frame(final, None)
            await asyncio.sleep(0)

        adapter.ingest.assert_awaited_once_with(
            DEFAULT_TEST_CALLER_ID, "I run a dental clinic.")

    async def test_different_useful_turns_use_resolved_caller_namespace(self):
        adapter = AsyncMock(spec=VoiceMemAdapter)
        adapter.initialize.return_value = True
        adapter.ingest.return_value = True
        processor = VoiceMemIngestProcessor(adapter)
        processor.push_frame = AsyncMock()
        processor.create_task = lambda coro, name=None: asyncio.create_task(coro)
        turns = [
            TranscriptionFrame("My name is Ali.", "user", "one", finalized=True),
            TranscriptionFrame(
                "I prefer morning meetings.", "user", "two", finalized=True),
        ]
        with patch.dict(os.environ, {"TEST_CALLER_ID": "caller_a"}):
            with contextlib.redirect_stdout(io.StringIO()):
                for frame in turns:
                    await processor.process_frame(frame, None)
                await asyncio.sleep(0)
        self.assertEqual(adapter.ingest.await_count, 2)
        self.assertEqual(
            [call.args[0] for call in adapter.ingest.await_args_list],
            ["caller_a", "caller_a"],
        )

    async def test_ingest_is_queued_without_blocking_frame_path(self):
        gate = asyncio.Event()
        adapter = AsyncMock(spec=VoiceMemAdapter)
        adapter.initialize.return_value = True
        async def blocked_ingest(*_args):
            await gate.wait()
            return True
        adapter.ingest.side_effect = blocked_ingest
        processor = VoiceMemIngestProcessor(adapter)
        processor.push_frame = AsyncMock()
        processor.create_task = lambda coro, name=None: asyncio.create_task(coro)
        frame = TranscriptionFrame(
            "I run a dental clinic.", "user", "now", finalized=True)

        with contextlib.redirect_stdout(io.StringIO()):
            await asyncio.wait_for(processor.process_frame(frame, None), timeout=0.05)
            await asyncio.sleep(0)
        processor.push_frame.assert_awaited_once_with(frame, None)
        self.assertTrue(processor._pending)
        gate.set()
        await asyncio.gather(*processor._pending)

    async def test_non_transcript_frames_never_write_internal_context(self):
        adapter = AsyncMock(spec=VoiceMemAdapter)
        processor = VoiceMemIngestProcessor(adapter)
        processor.push_frame = AsyncMock()
        internal = LLMContextFrame(context=LLMContext(messages=[
            {"role": "system", "content": "secret prompt"},
            {"role": "developer", "content": "RAG and tool metadata"},
        ]))
        await processor.process_frame(internal, None)
        adapter.ingest.assert_not_called()


if __name__ == "__main__":
    unittest.main()
