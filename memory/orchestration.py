"""Start existing RAG and VoiceMem retrievals concurrently when both apply."""

from __future__ import annotations

import asyncio
from time import perf_counter

from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.frame_processor import FrameProcessor

from memory.integration import VoiceMemContextProcessor, needs_memory
from rag.integration import RAGContextProcessor, needs_rag
from trace_recorder import trace_recorder


class RetrievalPrefetchProcessor(FrameProcessor):
    def __init__(self, rag: RAGContextProcessor, memory: VoiceMemContextProcessor):
        super().__init__()
        self._rag = rag
        self._memory = memory
        self._monitors: set[asyncio.Task] = set()
        self._turn_key = None

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame) or frame.speculation:
            await self.push_frame(frame, direction)
            return

        messages = frame.context.get_messages()
        user_index, query = self._rag._latest_user(messages)
        turn_key = (user_index, query)
        if turn_key == self._turn_key:
            await self.push_frame(frame, direction)
            return
        self._turn_key = turn_key
        recent = [
            message.get("content", "")
            for message in messages[:user_index or 0]
            if isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ]
        rag_used = needs_rag(query, recent)
        memory_used = needs_memory(query)
        route = (
            "RAG+VoiceMem" if rag_used and memory_used
            else "RAG" if rag_used
            else "VoiceMem" if memory_used
            else "skipped"
        )
        print(f"Retrieval route: {route}")
        trace_recorder.record(
            "retrieval_route",
            rag_routed=rag_used,
            memory_routed=memory_used,
            mode=route,
        )

        rag_task = self._rag.prefetch(frame) if rag_used else None
        memory_task = self._memory.prefetch(frame) if memory_used else None
        if rag_task is not None and memory_task is not None:
            started = perf_counter()
            monitor = self.create_task(
                self._log_combined(rag_task, memory_task, started),
                name="combined-retrieval-monitor",
            )
            self._monitors.add(monitor)
            monitor.add_done_callback(self._monitors.discard)
        await self.push_frame(frame, direction)

    @staticmethod
    async def _log_combined(rag_task, memory_task, started):
        await asyncio.gather(rag_task, memory_task, return_exceptions=True)
        latency_ms = (perf_counter() - started) * 1000
        print(f"Combined retrieval: {latency_ms:.2f} ms")
        trace_recorder.record(
            "combined_retrieval",
            mode="RAG+VoiceMem",
            latency_ms=round(latency_ms, 3),
        )

    async def cleanup(self):
        if self._monitors:
            await asyncio.gather(*tuple(self._monitors), return_exceptions=True)
        await super().cleanup()
