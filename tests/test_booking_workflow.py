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
    async def asyncSetUp(self):
        main._pending_email = ""
        main._pending_email_turn = 0
        main.transcription_printer.latest_text = ""
        main.transcription_printer.turn_id = 0

    async def test_model_cannot_supply_an_unspoken_day_part(self):
        callback = AsyncMock()
        params = SimpleNamespace(result_callback=callback, llm=SimpleNamespace(push_frame=AsyncMock()))
        session = SimpleNamespace(advance=AsyncMock(return_value={
            "status": "needs_input",
            "spoken_response": "What works better for you: morning, afternoon, or evening?",
        }))

        with (
            patch.object(main, "booking_session", session),
            patch.object(main.transcription_printer, "latest_text", "What about tomorrow?"),
        ):
            await main.booking_workflow(
                params,
                meeting_date="2026-09-23",
                time_preference="afternoon",
            )

        self.assertEqual(session.advance.await_args.kwargs["time_preference"], "")

    async def test_ambiguous_spoken_email_is_confirmed_before_booking(self):
        callback = AsyncMock()
        params = SimpleNamespace(result_callback=callback, llm=SimpleNamespace(push_frame=AsyncMock()))
        session = SimpleNamespace(advance=AsyncMock(return_value={
            "status": "needs_input",
            "spoken_response": "Thanks. What's the best email address for the booking?",
        }))

        with (
            patch.object(main, "booking_session", session),
            patch.object(
                main.transcription_printer,
                "latest_text",
                "My name is Ali Juan, and the email is ali con at g mail dot com.",
            ),
            patch.object(main.transcription_printer, "turn_id", 4),
        ):
            await main.booking_workflow(params, name="Ali Juan", email="ali@gmail.com")

        self.assertEqual(session.advance.await_count, 1)
        self.assertEqual(session.advance.await_args.kwargs["email"], "")
        result = callback.await_args.args[0]
        self.assertEqual(result["spoken_response"], "I heard ali at gmail dot com. Is that correct?")

    async def test_email_cannot_be_self_confirmed_in_the_same_caller_turn(self):
        session = SimpleNamespace(advance=AsyncMock(return_value={
            "status": "needs_input",
            "spoken_response": "Thanks. What's the best email address for the booking?",
        }))

        async def invoke():
            params = SimpleNamespace(result_callback=AsyncMock(), llm=SimpleNamespace(push_frame=AsyncMock()))
            await main.booking_workflow(params, name="Test Caller", email="tester@gmail.com")

        with patch.object(main, "booking_session", session):
            main.transcription_printer.latest_text = "My email is tester at gmail dot com."
            main.transcription_printer.turn_id = 7
            await invoke()
            await invoke()
            self.assertEqual(session.advance.await_args_list[0].kwargs["email"], "")
            self.assertEqual(session.advance.await_args_list[1].kwargs["email"], "")

            main.transcription_printer.latest_text = "Yes, that is correct."
            main.transcription_printer.turn_id = 8
            await invoke()
            self.assertEqual(session.advance.await_args_list[2].kwargs["email"], "tester@gmail.com")

    async def test_tts_filter_removes_visual_quotes_and_markdown(self):
        filtered = await main.SpokenTextFilter().filter('She said "**Agentix Labs AI**".')
        self.assertEqual(filtered, "She said Agentix Labs AI.")

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

    def test_flux_boosts_company_specific_spoken_terms(self):
        self.assertEqual(
            main.stt._settings.keyterm,
            ["Agentix Labs AI", "PongVerse", "LangGraph", "DeepSORT"],
        )

    def test_representative_trace_metrics_are_enabled(self):
        self.assertTrue(main.worker._params.enable_metrics)
        self.assertTrue(main.worker._params.enable_usage_metrics)


if __name__ == "__main__":
    unittest.main()
