"""Pipecat callback contract for a completed booking."""

import unittest
import importlib
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import main
from agent.booking_session import BookingSession
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.turns.user_mute import FunctionCallUserMuteStrategy

workflow = importlib.import_module("agent.graph")


class BookingWorkflowCallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_booking_emits_one_exact_llm_tts_response(self):
        date = (datetime.now(ZoneInfo("Asia/Karachi")) + timedelta(days=1)).strftime("%Y-%m-%d")
        callback = AsyncMock()
        llm = SimpleNamespace(push_frame=AsyncMock())
        params = SimpleNamespace(result_callback=callback, llm=llm)
        session = BookingSession()

        with (
            patch.object(workflow, "get_available_slots", return_value=["04:00 PM"]),
            patch.object(workflow, "is_slot_available", return_value=True),
            patch.object(workflow, "create_calendar_appointment", return_value={"success": True}) as create,
            patch.object(workflow, "save_lead_to_sheet", return_value={"success": True}) as save,
            patch.object(main, "booking_session", session),
        ):
            await main.booking_workflow(
                params,
                meeting_date=date,
                meeting_time="04:00 PM",
                name="Test Caller",
                email="test@example.com",
            )

        create.assert_called_once()
        save.assert_called_once()
        callback.assert_awaited_once()
        args, kwargs = callback.await_args
        self.assertTrue(args[0]["booking_confirmed"])
        self.assertTrue(args[0]["lead_saved"])
        self.assertEqual(args[0]["status"], "completed")
        self.assertFalse(kwargs["properties"].run_llm)
        self.assertIn("You're all set", args[0]["spoken_response"])
        self.assertIn("tomorrow at 04:00 PM", args[0]["spoken_response"])
        await kwargs["properties"].on_context_updated()
        frames = [call.args[0] for call in llm.push_frame.await_args_list]
        self.assertEqual(len(frames), 3)
        self.assertIsInstance(frames[0], LLMFullResponseStartFrame)
        self.assertIsInstance(frames[1], LLMTextFrame)
        self.assertEqual(frames[1].text, args[0]["spoken_response"])
        self.assertIsInstance(frames[2], LLMFullResponseEndFrame)

    def test_function_call_mute_ends_before_tts_barge_in(self):
        strategies = main.user_aggregator._params.user_mute_strategies
        self.assertTrue(any(isinstance(item, FunctionCallUserMuteStrategy) for item in strategies))
        self.assertTrue(main.booking_workflow._pipecat_cancel_on_interruption)
        self.assertTrue(main.stt._should_interrupt)

    def test_reasoning_is_hidden_from_spoken_output(self):
        self.assertEqual(
            main.llm._settings.extra["extra_body"]["reasoning_format"],
            "hidden",
        )


if __name__ == "__main__":
    unittest.main()
