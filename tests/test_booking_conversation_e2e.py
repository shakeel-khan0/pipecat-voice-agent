"""Credit-free regression tests for the complete conversational booking flow."""

import contextlib
import importlib
import io
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import main
from agent.booking_session import BookingSession
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame


workflow = importlib.import_module("agent.graph")
DATE = (datetime.now(ZoneInfo("Asia/Karachi")) + timedelta(days=1)).strftime("%Y-%m-%d")


class FakeLLM:
    """Records normal downstream LLM frames without calling a provider."""

    def __init__(self):
        self.frames = []

    async def push_frame(self, frame):
        self.frames.append(frame)


class BookingConversationE2ETests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.slots = self.stack.enter_context(
            patch.object(workflow, "get_available_slots", return_value=["04:00 PM", "05:00 PM"])
        )
        self.available = self.stack.enter_context(
            patch.object(workflow, "is_slot_available", return_value=True)
        )
        self.create = self.stack.enter_context(
            patch.object(workflow, "create_calendar_appointment", return_value={"success": True})
        )
        self.save = self.stack.enter_context(
            patch.object(workflow, "save_lead_to_sheet", return_value={"success": True})
        )

    def session(self):
        return BookingSession()

    async def finish(self, session, time="05:00 PM"):
        first = await session.advance(meeting_date=DATE, meeting_time=time)
        self.assertIn("name and email", first["spoken_response"])
        return await session.advance(name="Test Caller", email="test@example.com")

    async def test_a_requested_five_pm_available(self):
        session = self.session()
        result = await session.advance(
            meeting_date=DATE,
            meeting_time="05:00 PM",
            time_preference="morning around four",
        )
        self.assertEqual(session._details["meeting_time"], "05:00 PM")
        self.assertIn("name and email", result["spoken_response"])
        self.assertNotIn("morning", result["spoken_response"].lower())

        completed = await session.advance(name="Test Caller", email="test@example.com")
        self.assertEqual(completed["status"], "completed")
        self.assertIn("05:00 PM", completed["spoken_response"])

    async def test_b_requested_five_pm_unavailable_offers_real_nearby_slot(self):
        self.slots.return_value = ["01:00 PM", "02:00 PM", "03:00 PM", "04:00 PM"]
        session = self.session()
        result = await session.advance(meeting_date=DATE, meeting_time="05:00 PM")

        self.assertEqual(result["status"], "needs_input")
        self.assertIn("05:00 PM isn't available", result["spoken_response"])
        self.assertIn("04:00 PM", result["offered_slots"])
        self.assertNotIn("confirm", result["spoken_response"].lower())
        self.assertNotIn("morning, afternoon, or evening", result["spoken_response"])
        self.create.assert_not_called()
        self.save.assert_not_called()

    async def test_unavailable_time_does_not_drop_identity_from_same_turn(self):
        self.slots.return_value = ["04:00 PM"]
        session = self.session()
        await session.advance(meeting_date=DATE, meeting_time="04:00 PM")
        result = await session.advance(
            meeting_time="05:00 PM",
            name="Test Caller",
            email="test@example.com",
        )
        self.assertIn("04:00 PM", result["offered_slots"])
        self.assertEqual(session._details["name"], "Test Caller")
        self.assertEqual(session._details["email"], "test@example.com")

        completed = await session.advance(meeting_time="04:00 PM")
        self.assertEqual(completed["status"], "completed")

    async def test_date_change_validates_time_against_new_date(self):
        session = self.session()
        await session.advance(meeting_date=DATE, meeting_time="05:00 PM")
        new_date = (
            datetime.now(ZoneInfo("Asia/Karachi")) + timedelta(days=2)
        ).strftime("%Y-%m-%d")
        self.slots.return_value = ["05:00 PM"]
        result = await session.advance(meeting_date=new_date, meeting_time="05:00 PM")
        self.assertIn("name and email", result["spoken_response"])
        self.assertEqual(session._details["meeting_date"], new_date)
        self.assertEqual(session._details["meeting_time"], "05:00 PM")

    async def test_c_user_changes_selected_time(self):
        session = self.session()
        await session.advance(meeting_date=DATE, meeting_time="05:00 PM")
        changed = await session.advance(meeting_time="04:00 PM")
        self.assertEqual(session._details["meeting_time"], "04:00 PM")
        self.assertIn("name and email", changed["spoken_response"])

        completed = await session.advance(name="Test Caller", email="test@example.com")
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(self.create.call_args.kwargs["time"], "04:00 PM")

    async def test_d_time_is_resolved_before_identity_is_requested(self):
        session = self.session()
        result = await session.advance(meeting_date=DATE)
        request = session._result["__interrupt__"][0].value

        self.assertEqual(request["fields"], ["meeting_time"])
        self.assertNotIn("name", result["spoken_response"].lower())
        self.assertNotIn("email", result["spoken_response"].lower())

    async def test_e_identity_collection_preserves_selected_date_and_time(self):
        session = self.session()
        await session.advance(meeting_date=DATE)
        identity = await session.advance(meeting_time="05:00 PM")
        self.assertIn("name and email", identity["spoken_response"])

        completed = await session.advance(name="Test Caller", email="test@example.com")
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(self.create.call_args.kwargs["date"], DATE)
        self.assertEqual(self.create.call_args.kwargs["time"], "05:00 PM")

    async def test_f_duplicate_utterance_does_not_duplicate_booking(self):
        session = self.session()
        await self.finish(session)
        repeated = await session.advance(
            meeting_date=DATE,
            meeting_time="05:00 PM",
            name="Test Caller",
            email="test@example.com",
        )
        self.assertEqual(repeated["status"], "completed")
        self.create.assert_called_once()
        self.save.assert_called_once()

    async def test_g_calendar_and_sheets_are_written_exactly_once(self):
        session = self.session()
        completed = await self.finish(session)
        self.assertTrue(completed["booking_confirmed"])
        self.assertTrue(completed["lead_saved"])
        self.create.assert_called_once()
        self.save.assert_called_once()

    async def test_h_one_exact_public_response_uses_no_provider_follow_up(self):
        session = self.session()
        fake_llm = FakeLLM()
        callback = AsyncMock()
        params = SimpleNamespace(result_callback=callback, llm=fake_llm)

        with patch.object(main, "booking_session", session):
            await main.booking_workflow(params, meeting_date=DATE)

        callback.assert_awaited_once()
        result = callback.await_args.args[0]
        properties = callback.await_args.kwargs["properties"]
        self.assertFalse(properties.run_llm)
        self.assertIsNotNone(properties.on_context_updated)

        await properties.on_context_updated()
        self.assertEqual(len(fake_llm.frames), 3)
        self.assertIsInstance(fake_llm.frames[0], LLMFullResponseStartFrame)
        self.assertIsInstance(fake_llm.frames[1], LLMTextFrame)
        self.assertEqual(fake_llm.frames[1].text, result["spoken_response"])
        self.assertIsInstance(fake_llm.frames[2], LLMFullResponseEndFrame)
        for forbidden in ("we need to", "workflow", "interrupt", "internal state"):
            self.assertNotIn(forbidden, fake_llm.frames[1].text.lower())

    async def test_i_natural_time_phrases_resolve_to_five_pm(self):
        cases = {
            "around five": {"time_preference": "around five"},
            "five PM": {"meeting_time": "05:00 PM"},
            "tomorrow evening at five": {"time_preference": "tomorrow evening at five"},
            "is 5 available tomorrow?": {"time_preference": "5"},
        }
        for utterance, extracted in cases.items():
            with self.subTest(utterance=utterance):
                session = self.session()
                result = await session.advance(meeting_date=DATE, **extracted)
                self.assertEqual(session._details.get("meeting_time"), "05:00 PM")
                self.assertIn("name and email", result["spoken_response"])
                self.assertNotIn("Which time", result["spoken_response"])


if __name__ == "__main__":
    unittest.main()
