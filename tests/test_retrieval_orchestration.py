from __future__ import annotations

import asyncio
import contextlib
import io
import time
import unittest
from unittest.mock import AsyncMock

from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext

from memory.integration import temporary_memory_instruction
from memory.orchestration import RetrievalPrefetchProcessor
from rag.integration import RAGContextProcessor, temporary_instruction
from rag.retrieve import RetrievalResult


class FakeSource:
    def __init__(self, delay=0.0, error=False):
        self.delay = delay
        self.error = error
        self.calls = 0

    _latest_user = staticmethod(RAGContextProcessor._latest_user)

    def prefetch(self, _frame):
        self.calls += 1

        async def work():
            await asyncio.sleep(self.delay)
            if self.error:
                raise ConnectionError("unavailable")
            return {"ok": True}

        return asyncio.create_task(work())


class RetrievalOrchestrationTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def frame(query):
        return LLMContextFrame(
            context=LLMContext(messages=[{"role": "user", "content": query}]))

    async def run_route(self, query, rag=None, memory=None):
        rag = rag or FakeSource()
        memory = memory or FakeSource()
        processor = RetrievalPrefetchProcessor(rag, memory)
        processor.push_frame = AsyncMock()
        processor.create_task = lambda coro, name=None: asyncio.create_task(coro)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            await processor.process_frame(self.frame(query), None)
            if processor._monitors:
                await asyncio.gather(*tuple(processor._monitors))
        return rag, memory, output.getvalue()

    async def test_four_routes(self):
        cases = [
            ("Hello", 0, 0, "skipped"),
            ("What services does Agentix offer?", 1, 0, "RAG"),
            ("What business do I run?", 0, 1, "VoiceMem"),
            ("Which Agentix service would fit my dental clinic?", 1, 1, "RAG+VoiceMem"),
        ]
        for query, rag_calls, memory_calls, route in cases:
            with self.subTest(query=query):
                rag, memory, output = await self.run_route(query)
                self.assertEqual(rag.calls, rag_calls)
                self.assertEqual(memory.calls, memory_calls)
                self.assertIn(f"Retrieval route: {route}", output)

    async def test_both_retrievals_are_concurrent(self):
        rag = FakeSource(delay=0.15)
        memory = FakeSource(delay=0.15)
        started = time.perf_counter()
        _, _, output = await self.run_route(
            "Which Agentix service would fit my dental clinic?", rag, memory)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 0.25)
        self.assertGreaterEqual(elapsed, 0.14)
        self.assertIn("Combined retrieval:", output)

    async def test_duplicate_context_frame_does_not_repeat_retrieval(self):
        rag = FakeSource()
        memory = FakeSource()
        processor = RetrievalPrefetchProcessor(rag, memory)
        processor.push_frame = AsyncMock()
        processor.create_task = lambda coro, name=None: asyncio.create_task(coro)
        frame = self.frame("Which Agentix service would fit my dental clinic?")
        with contextlib.redirect_stdout(io.StringIO()):
            await processor.process_frame(frame, None)
            await processor.process_frame(frame, None)
            await asyncio.gather(*tuple(processor._monitors))
        self.assertEqual(rag.calls, 1)
        self.assertEqual(memory.calls, 1)

    async def test_failure_matrix_always_forwards_the_turn(self):
        for rag_error, memory_error in [
            (False, False), (False, True), (True, False), (True, True)
        ]:
            with self.subTest(rag_error=rag_error, memory_error=memory_error):
                rag = FakeSource(error=rag_error)
                memory = FakeSource(error=memory_error)
                _, _, output = await self.run_route(
                    "Which Agentix service would fit my dental clinic?", rag, memory)
                self.assertIn("Combined retrieval:", output)

    async def test_context_authority_and_separation(self):
        rag = temporary_instruction([
            RetrievalResult(
                text="Agentix offers AI voice agents.",
                heading="AI Voice Agents",
                score=0.9,
                rank=1,
            )
        ], query="Which service fits my clinic?")
        memory = temporary_memory_instruction([
            "The caller runs a dental clinic.",
            "The caller needs an Android app.",
        ])
        self.assertIn("Temporary verified company knowledge", rag)
        self.assertIn("CLOSED-WORLD", rag)
        self.assertIn("[CALLER MEMORY]", memory)
        self.assertIn("not authoritative company knowledge", memory)
        self.assertIn("Current explicit caller statements override stale memory", memory)
        self.assertNotIn("[CALLER MEMORY]", rag)
        self.assertNotIn("Temporary verified company knowledge", memory)


if __name__ == "__main__":
    unittest.main()
