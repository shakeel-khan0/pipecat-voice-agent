"""Security and schema checks for the one-shot representative trace."""

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from rag.retrieve import RetrievalResult
from trace_recorder import TraceRecorder


class TraceRecorderTests(unittest.TestCase):
    def test_trace_file_redacts_secrets_ids_and_customer_pii(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "representative_trace.json"
            recorder = TraceRecorder(path)
            recorder.user_turn(
                "My name is Real Person, and my email is real.person@example.com. "
                "Call +92 300 1234567."
            )
            recorder.record(
                "tool_call",
                name="Real Person",
                email="real.person@example.com",
                phone="+92 300 1234567",
                api_key="gsk_this_is_a_fake_but_sensitive_api_key",
                oauth_token="fake-oauth-token",
                spreadsheet_id="fake-spreadsheet-id",
                event_id="fake-calendar-event-id",
            )
            recorder.record("error", error_type="RuntimeError", detail="Bearer fake-token-value")
            recorder.write()

            raw = path.read_text(encoding="utf-8")
            for forbidden in (
                "Real Person",
                "real.person@example.com",
                "+92 300 1234567",
                "gsk_this_is_a_fake_but_sensitive_api_key",
                "fake-oauth-token",
                "fake-spreadsheet-id",
                "fake-calendar-event-id",
                "fake-token-value",
            ):
                self.assertNotIn(forbidden, raw)
            document = json.loads(raw)
            self.assertTrue(document["trace_id"].startswith("trace-"))
            self.assertEqual(
                [event["sequence"] for event in document["events"]],
                list(range(1, len(document["events"]) + 1)),
            )

    def test_only_repository_knowledge_can_be_written_as_rag_excerpt(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = TraceRecorder(Path(directory) / "trace.json")
            public_text = "Agentix Labs AI builds custom AI systems and automation around real business workflows, including voice agents, agentic workflows, RAG systems, computer vision, AI integrations, and full-stack AI applications."
            recorder.rag_event(
                used=True,
                latency_ms=12.5,
                results=[
                    RetrievalResult(1, 0.8, "What does Agentix Labs AI do?", public_text),
                    RetrievalResult(2, 0.7, "External", "Untrusted external text"),
                ],
            )
            event = recorder._events[-1]
            self.assertEqual(event["data"]["results"][0]["text_excerpt"], public_text)
            self.assertIsNone(event["data"]["results"][1]["text_excerpt"])

    def test_configured_secret_is_removed_before_disk_write(self):
        configured_secret = "secret-value-that-must-never-be-written"
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"TEST_API_KEY": configured_secret}
        ):
            path = Path(directory) / "trace.json"
            recorder = TraceRecorder(path)
            recorder.record("error", detail=configured_secret)
            recorder.write()
            self.assertNotIn(configured_secret, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
