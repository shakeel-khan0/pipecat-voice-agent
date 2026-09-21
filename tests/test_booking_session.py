"""Offline integration checks: real graph/checkpoints, mocked Google actions."""

import asyncio
import contextlib
import importlib
import io
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

from agent.booking_session import BookingSession

workflow = importlib.import_module("agent.graph")
DATE = (datetime.now(ZoneInfo("Asia/Karachi")) + timedelta(days=2)).strftime("%Y-%m-%d")
OTHER_DATE = (datetime.now(ZoneInfo("Asia/Karachi")) + timedelta(days=3)).strftime("%Y-%m-%d")


class BookingSessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.slots = self.stack.enter_context(patch.object(workflow, "get_available_slots", return_value=["10:00 AM", "11:00 AM"]))
        self.available = self.stack.enter_context(patch.object(workflow, "is_slot_available", return_value=True))
        self.create = self.stack.enter_context(patch.object(workflow, "create_calendar_appointment", return_value={"success": True}))
        self.save = self.stack.enter_context(patch.object(workflow, "save_lead_to_sheet", return_value={"success": True}))
        self.session = BookingSession()

    async def details(self):
        return await self.session.advance(meeting_date=DATE, name="Test Caller", email="test@example.com")

    async def test_partial_turns_resume_same_thread(self):
        config = self.session.config.copy()
        self.assertEqual((await self.session.advance())["spoken_response"], "Sure. Which date would you like to meet?")
        availability = await self.session.advance(meeting_date=DATE)
        self.assertEqual(availability["offered_slots"], ["10:00 AM", "11:00 AM"])
        self.assertIn("Which works best?", availability["spoken_response"])
        self.assertIn("name and email", (await self.session.advance(meeting_time="10:00 AM"))["spoken_response"])
        self.assertIn("email", (await self.session.advance(name="Test Caller"))["spoken_response"])
        result = await self.session.advance(email="test@example.com")
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["lead_saved"])
        self.assertEqual(config, self.session.config)
        self.create.assert_called_once()
        self.save.assert_called_once()

    async def test_duplicate_concurrent_calls_do_not_duplicate_writes(self):
        await self.details()
        await asyncio.gather(*(self.session.advance(meeting_time="10:00 AM") for _ in range(3)))
        self.create.assert_called_once()
        self.save.assert_called_once()

    async def test_stale_slot_rechecks_and_requests_new_time(self):
        await self.details()
        self.available.return_value = False
        self.slots.return_value = ["11:00 AM"]
        result = await self.session.advance(meeting_time="10:00 AM")
        self.assertEqual(result["offered_slots"], ["11:00 AM"])
        self.assertIn("just taken", result["spoken_response"])
        self.create.assert_not_called()
        self.available.return_value = True
        self.assertTrue((await self.session.advance(meeting_time="11:00 AM"))["booking_confirmed"])

    async def test_invalid_time_and_email_do_not_write(self):
        self.assertEqual((await self.session.advance(email="invalid"))["status"], "needs_input")
        await self.details()
        result = await self.session.advance(meeting_time="09:00 PM")
        self.assertCountEqual(result["offered_slots"], ["10:00 AM", "11:00 AM"])
        self.create.assert_not_called()

    async def test_new_date_after_no_availability(self):
        self.slots.return_value = []
        self.assertEqual((await self.session.advance(meeting_date=DATE))["status"], "unavailable")
        self.slots.return_value = ["11:00 AM"]
        result = await self.session.advance(meeting_date=OTHER_DATE)
        self.assertEqual(result["status"], "needs_input")
        self.assertEqual(result["offered_slots"], ["11:00 AM"])

    async def test_date_change_during_interrupt(self):
        await self.details()
        result = await self.session.advance(meeting_date=OTHER_DATE)
        self.assertEqual(result["offered_slots"], ["10:00 AM", "11:00 AM"])
        await self.session.advance(meeting_time="11:00 AM")
        self.assertEqual(self.create.call_args.kwargs["date"], OTHER_DATE)

    async def test_sheets_failure_preserves_booking_without_retries(self):
        self.save.return_value = {"success": False}
        await self.details()
        result = await self.session.advance(meeting_time="10:00 AM")
        self.assertTrue(result["booking_confirmed"])
        self.assertFalse(result["lead_saved"])
        self.assertEqual(result["status"], "error")
        await self.session.advance()
        self.create.assert_called_once()
        self.save.assert_called_once()

    async def test_uncertain_calendar_write_not_retried(self):
        self.create.side_effect = TimeoutError("unknown remote outcome")
        await self.details()
        self.assertEqual((await self.session.advance(meeting_time="10:00 AM"))["status"], "error")
        await self.session.advance(meeting_time="10:00 AM")
        self.create.assert_called_once()
        self.save.assert_not_called()

    async def test_calendar_final_recheck_conflict_requests_another_time(self):
        await self.details()
        self.create.return_value = {"success": False, "error": "Selected slot is no longer available."}
        self.slots.return_value = ["11:00 AM"]
        result = await self.session.advance(meeting_time="10:00 AM")
        self.assertEqual(result["status"], "needs_input")
        self.assertEqual(result["offered_slots"], ["11:00 AM"])
        self.assertIn("just taken", result["spoken_response"])
        self.save.assert_not_called()

    async def test_sheets_exception_preserves_confirmed_booking(self):
        await self.details()
        self.save.side_effect = TimeoutError("unknown remote outcome")
        result = await self.session.advance(meeting_time="10:00 AM")
        self.assertTrue(result["booking_confirmed"])
        self.assertEqual(result["status"], "error")
        await self.session.advance()
        self.save.assert_called_once()

    async def test_normal_conversation_graph_path_does_not_call_google(self):
        result = await workflow.graph.ainvoke(workflow.initial_state(message="Hello"), config=self.session.config)
        self.assertEqual(result["intent"], "conversation")
        self.slots.assert_not_called()
        self.create.assert_not_called()
        self.save.assert_not_called()

    async def test_independent_callers_do_not_share_state(self):
        await self.details()
        other = BookingSession()
        self.assertNotEqual(other.config, self.session.config)
        self.assertIn("Which date", (await other.advance())["spoken_response"])

    async def test_voice_result_contains_no_internal_status_language(self):
        result = await self.session.advance()
        spoken = result["spoken_response"].lower()
        for forbidden in ("(waiting", "interrupt", "resume", "workflow", "langgraph", "tool result"):
            self.assertNotIn(forbidden, spoken)
        self.assertNotIn("question", result)
        self.assertNotIn("fields", result)
        self.assertNotIn("available_slots", result)

    async def test_large_slot_list_asks_day_part_without_dumping_slots(self):
        all_slots = ["10:00 AM", "10:30 AM", "11:00 AM", "12:00 PM", "02:00 PM", "03:00 PM", "05:00 PM"]
        self.slots.return_value = all_slots
        result = await self.session.advance(meeting_date=DATE)
        self.assertEqual(result["spoken_response"], "What works better for you: morning, afternoon, or evening?")
        self.assertEqual(result["offered_slots"], [])
        self.assertTrue(all(slot not in result["spoken_response"] for slot in all_slots))

    async def test_day_part_filters_real_slots_and_offers_three(self):
        self.slots.return_value = ["10:00 AM", "11:30 AM", "12:00 PM", "02:00 PM", "03:00 PM", "04:30 PM", "05:00 PM"]
        await self.session.advance(meeting_date=DATE)
        result = await self.session.advance(time_preference="afternoon")
        self.assertEqual(result["offered_slots"], ["12:00 PM", "02:00 PM", "03:00 PM"])
        self.assertNotIn("10:00 AM", result["spoken_response"])
        self.assertNotIn("05:00 PM", result["spoken_response"])

    async def test_specific_spoken_time_is_selected_and_booking_continues(self):
        self.slots.return_value = ["02:00 PM", "03:00 PM", "04:00 PM", "05:00 PM", "05:30 PM"]
        first = await self.session.advance(meeting_date=DATE)
        self.assertIn("morning, afternoon, or evening", first["spoken_response"])

        result = await self.session.advance(time_preference="evening around five PM")
        self.assertEqual(
            result["spoken_response"],
            "Great. Could I get your name and email to finalize the booking?",
        )
        self.assertEqual(self.session._details["meeting_time"], "05:00 PM")

        completed = await self.session.advance(name="Test Caller", email="test@example.com")
        self.assertEqual(completed["status"], "completed")
        self.assertIn("05:00 PM", completed["spoken_response"])
        self.assertEqual(self.create.call_args.kwargs["time"], "05:00 PM")
        self.create.assert_called_once()
        self.save.assert_called_once()

    async def test_completion_returns_immediate_natural_confirmation(self):
        await self.details()
        result = await self.session.advance(meeting_time="10:00 AM")
        self.assertEqual(result["status"], "completed")
        self.assertIn("You're all set", result["spoken_response"])
        self.assertIn("10:00 AM", result["spoken_response"])

    async def test_cancelled_response_does_not_cancel_booking(self):
        started, release = asyncio.Event(), asyncio.Event()
        original = self.session.graph

        class DelayedGraph:
            async def ainvoke(inner_self, value, config):
                started.set()
                await release.wait()
                return await original.ainvoke(value, config=config)

        self.session.graph = DelayedGraph()
        task = asyncio.create_task(self.session.advance(meeting_date=DATE, meeting_time="10:00 AM", name="Test Caller", email="test@example.com"))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        release.set()
        await asyncio.gather(*self.session._tasks)
        self.create.assert_called_once()
        self.save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
