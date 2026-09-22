"""Credit-free checks for temporary RAG context in the voice pipeline."""

import contextlib
import io
import unittest
from unittest.mock import AsyncMock

from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext

from rag.integration import (
    RAGContextProcessor,
    company_service_verdict,
    entity_hint,
    needs_rag,
    retrieval_query,
    select_results,
    temporary_instruction,
)
from rag.retrieve import RetrievalResult


class FakeRetriever:
    def __init__(self, results=None):
        self.calls = 0
        self.queries = []
        self.results = results

    def search(self, query):
        self.calls += 1
        self.queries.append(query)
        return (self.results or [
            RetrievalResult(1, 0.82, "AI Voice Agents", "Voice agents handle customer calls."),
            RetrievalResult(2, 0.60, "Suitable problems", "Businesses may automate calls."),
            RetrievalResult(3, 0.30, "Unrelated", "Other content."),
        ], 12.5)


class RAGIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def test_routing_decision(self):
        needed = (
            "What does Agentix Labs AI do?",
            "What does Agentix Labs AI do",
            "Can you build something to handle customer calls and bookings?",
            "What is PongVerse?",
            "How much does a voice agent cost?",
            "Do you have offices in Dubai?",
            "Do you use DeepSORT?",
            "Do you build workflows with LangGraph?",
            "What is pong verse?",
            "What is pong versus?",
            "What does the GenTech Labs AI do?",
            "What do you know about Gongverse?",
            "What is bone verse?",
            "I want to book a meeting; what services do you provide?",
        )
        skipped = (
            "Hello",
            "Thanks",
            "I would like to book a meeting",
            "Tomorrow",
            "5 PM",
            "test@example.com",
            "My name is Test Caller",
            "Tell me about Converse shoes",
            "What is the weather like?",
        )
        for query in needed:
            with self.subTest(query=query):
                self.assertTrue(needs_rag(query))
        for query in skipped:
            with self.subTest(query=query):
                self.assertFalse(needs_rag(query))

    def test_voice_entity_hints_are_conservative(self):
        for query, expected in (
            ("What is PongVerse?", "PongVerse"),
            ("What is pong verse?", "PongVerse"),
            ("What is pong versus?", "PongVerse"),
            ("What does the GenTech Labs AI do?", "Agentix Labs AI"),
            ("Do you use lang graph?", "LangGraph"),
            ("Do you use deep sort?", "DeepSORT"),
            ("What do you know about Gongverse?", "PongVerse"),
            ("What is bone verse?", "PongVerse"),
        ):
            with self.subTest(query=query):
                self.assertEqual(entity_hint(query)[0], expected)
        self.assertEqual(entity_hint("Tell me about Converse shoes"), (None, False))

    def test_correction_uses_recent_user_context(self):
        recent = ["What is converse?"]
        self.assertEqual(
            entity_hint("No, Pong, ping-pong wala Pong.", recent),
            ("PongVerse", False),
        )
        self.assertTrue(needs_rag("No, Pong, ping-pong wala Pong.", recent))
        self.assertEqual(
            entity_hint("What do you know about Tongva's?", ["What is Gongverse?"]),
            ("PongVerse", False),
        )

    def test_only_broad_service_queries_are_expanded(self):
        expanded = retrieval_query("While we book, what services do you offer?")
        self.assertIn("voice agents", expanded)
        self.assertIn("computer vision", expanded)
        exact = "What is PongVerse?"
        self.assertEqual(retrieval_query(exact), exact)

    def test_knowledge_boundaries_take_priority(self):
        results = [
            RetrievalResult(1, 0.72, "Company Identity", "General company information."),
            RetrievalResult(2, 0.65, "22. Knowledge Boundaries", "Do not claim an office address."),
            RetrievalResult(3, 0.50, "Other", "Other information."),
        ]
        selected = select_results("Do you have offices in Dubai?", results)
        self.assertEqual([item.heading for item in selected], ["22. Knowledge Boundaries"])

    def test_one_strong_chunk_is_preferred_for_broad_service_question(self):
        results = [
            RetrievalResult(1, 0.70, "4. Service Hierarchy", "The complete service list."),
            RetrievalResult(2, 0.66, "Company Description", "A shorter overlapping list."),
            RetrievalResult(3, 0.64, "Modern AI Stack", "Technology names."),
        ]
        selected = select_results("What solutions can your team provide?", results)
        self.assertEqual([item.heading for item in selected], ["4. Service Hierarchy"])

    def test_company_service_questions_use_closed_world_verdicts(self):
        cases = (
            (
                "Do you build Android and iOS mobile apps?",
                "Full-Stack AI Applications",
                "We build full-stack AI applications, web portals, dashboards, and APIs.",
                False,
            ),
            (
                "Do you work on surveillance systems?",
                "Computer Vision",
                "Capabilities include tracking, multi-camera systems, and real-time monitoring.",
                False,
            ),
            (
                "Do you provide AI Voice Agents?",
                "AI Voice Agents",
                "AI Voice Agents handle reception, qualification, and booking.",
                True,
            ),
            (
                "Do you build RAG systems?",
                "LLM and RAG Systems",
                "Agentix Labs AI builds RAG systems using embeddings and vector retrieval.",
                True,
            ),
            (
                "Do you work on multi-camera 3D reconstruction?",
                "Computer Vision",
                "Capabilities include multi-camera systems and 3D reconstruction.",
                True,
            ),
            (
                "Do you build full-stack AI applications?",
                "Full-Stack AI Applications",
                "Agentix Labs AI builds full-stack AI applications.",
                True,
            ),
        )
        for query, heading, text, expected in cases:
            with self.subTest(query=query):
                results = [RetrievalResult(1, 0.8, heading, text)]
                self.assertIs(company_service_verdict(query, results), expected)

    def test_unlisted_service_instruction_requires_direct_no(self):
        results = [RetrievalResult(
            1,
            0.8,
            "Computer Vision",
            "Capabilities include tracking, multi-camera systems, and real-time monitoring.",
        )]
        instruction = temporary_instruction(
            results,
            query="Do you build surveillance systems?",
        )
        self.assertIn("CLOSED-WORLD SERVICE VERDICT: NOT LISTED", instruction)
        self.assertIn("No, that is not one of our listed services.", instruction)
        self.assertIn("Do not suggest checking with the team", instruction)

    async def test_context_is_temporary_during_booking_and_retrieval_runs_once_per_turn(self):
        fake = FakeRetriever()
        processor = RAGContextProcessor(retriever_factory=lambda: fake)
        processor.push_frame = AsyncMock()
        original = LLMContext(
            messages=[
                {"role": "user", "content": "I want to book a meeting."},
                {"role": "assistant", "content": "Which date works for you?"},
                {"role": "user", "content": "Before we continue, what voice-agent services do you offer?"},
            ],
            tools=[],
        )
        frame = LLMContextFrame(context=original)

        with contextlib.redirect_stdout(io.StringIO()) as output:
            await processor.process_frame(frame, None)
            await processor.process_frame(frame, None)

        self.assertEqual(fake.calls, 1)
        self.assertEqual(len(original.messages), 3)
        self.assertNotIn("temporary", str(original.messages).lower())
        pushed = processor.push_frame.await_args_list[0].args[0]
        self.assertIsNot(pushed.context, original)
        self.assertEqual(len(pushed.context.messages), 4)
        self.assertIn("Temporary verified company knowledge", pushed.context.messages[2]["content"])
        self.assertEqual(output.getvalue().count("RAG:"), 1)

    async def test_skipped_turn_does_not_initialize_retriever(self):
        factory = unittest.mock.Mock()
        processor = RAGContextProcessor(retriever_factory=factory)
        processor.push_frame = AsyncMock()
        frame = LLMContextFrame(context=LLMContext(messages=[{"role": "user", "content": "Hello"}]))

        with contextlib.redirect_stdout(io.StringIO()) as output:
            await processor.process_frame(frame, None)

        factory.assert_not_called()
        self.assertEqual(output.getvalue().strip(), "RAG: skipped")
        self.assertIs(processor.push_frame.await_args.args[0], frame)

    async def test_qdrant_failure_keeps_turn_usable_with_safe_temporary_context(self):
        factory = unittest.mock.Mock(side_effect=ConnectionError("Qdrant unavailable"))
        processor = RAGContextProcessor(retriever_factory=factory)
        processor.push_frame = AsyncMock()
        original = LLMContext(messages=[
            {"role": "user", "content": "Do you have offices in Dubai?"}
        ])

        with contextlib.redirect_stdout(io.StringIO()) as output:
            await processor.process_frame(LLMContextFrame(context=original), None)

        pushed = processor.push_frame.await_args.args[0]
        instruction = pushed.context.messages[0]["content"]
        self.assertIn("No useful verified knowledge", instruction)
        self.assertIn("answer yes only when the retrieved text explicitly names", instruction)
        self.assertIn("Never expand the company's scope", instruction)
        self.assertEqual(len(original.messages), 1)
        self.assertRegex(output.getvalue().strip(), r"^RAG: 0 chunks \| \d+\.\d{2} ms$")

    async def test_ambiguous_converse_asks_for_clarification_when_not_verified(self):
        fake = FakeRetriever(results=[
            RetrievalResult(1, 0.75, "AI Receptionist", "Answers company calls."),
        ])
        processor = RAGContextProcessor(retriever_factory=lambda: fake)
        processor.push_frame = AsyncMock()
        frame = LLMContextFrame(context=LLMContext(messages=[
            {"role": "user", "content": "What is converse?"}
        ]))

        await processor.process_frame(frame, None)

        pushed = processor.push_frame.await_args.args[0]
        instruction = pushed.context.messages[0]["content"]
        self.assertEqual(fake.calls, 1)
        self.assertIn("Ask one short clarification question", instruction)
        self.assertIn("do not answer from general knowledge", instruction)

    async def test_correction_turn_recovers_pongverse_with_one_retrieval(self):
        fake = FakeRetriever(results=[
            RetrievalResult(1, 0.83, "PongVerse", "PongVerse is an Agentix Labs AI project."),
        ])
        processor = RAGContextProcessor(retriever_factory=lambda: fake)
        processor.push_frame = AsyncMock()
        frame = LLMContextFrame(context=LLMContext(messages=[
            {"role": "user", "content": "What is converse?"},
            {"role": "assistant", "content": "Could you clarify the project name?"},
            {"role": "user", "content": "No, Pong, ping-pong wala Pong."},
        ]))

        await processor.process_frame(frame, None)

        self.assertEqual(fake.calls, 1)
        self.assertIn("PongVerse", fake.queries[0])
        pushed = processor.push_frame.await_args.args[0]
        self.assertIn("PongVerse is an Agentix", pushed.context.messages[2]["content"])


if __name__ == "__main__":
    unittest.main()
