import asyncio
import os
import re
from time import perf_counter
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from loguru import logger

from dotenv import load_dotenv

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TranscriptionFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    FunctionCallResultProperties,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.workers.runner import WorkerRunner

from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.turns.user_mute import FunctionCallUserMuteStrategy
from pipecat.utils.text.base_text_filter import BaseTextFilter

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from agent.booking_session import BookingSession
from pipecat.adapters.schemas.direct_function import tool_options
from rag.integration import RAGContextProcessor
from trace_recorder import TraceMetricsObserver, trace_recorder


logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    level="WARNING"
)

load_dotenv()

stt = DeepgramFluxSTTService(
    api_key=os.getenv("DEEPGRAM_API_KEY"),
    settings=DeepgramFluxSTTService.Settings(
        model="flux-general-en",
        min_confidence=0.3,
        eot_threshold=0.8,
        keyterm=["Agentix Labs AI", "PongVerse", "LangGraph", "DeepSORT"],
    ),
)

# One process / local microphone conversation is one caller session.
booking_session = BookingSession()
_pending_email = ""
_pending_email_turn = 0

_DAY_PARTS = ("morning", "afternoon", "evening")


def _latest_caller_text():
    printer = globals().get("transcription_printer")
    return getattr(printer, "latest_text", "")


def _latest_caller_turn():
    printer = globals().get("transcription_printer")
    return getattr(printer, "turn_id", 0)


def _caller_supported_time_preference(preference: str, caller_text: str) -> str:
    """Reject a model-invented day-part while preserving caller-supplied preferences."""
    preference_lower = preference.lower()
    caller_lower = caller_text.lower()
    stated_parts = [part for part in _DAY_PARTS if part in preference_lower]
    if stated_parts and not any(part in caller_lower for part in stated_parts):
        return ""
    return preference


def _say_email(email: str) -> str:
    return email.replace("@", " at ").replace(".", " dot ")


def _email_requires_confirmation(email: str, caller_text: str, caller_turn: int) -> bool:
    """Require approval from a later caller turn before accepting an email."""
    global _pending_email, _pending_email_turn

    normalized_email = email.strip().lower()
    affirmative = bool(re.search(r"\b(?:yes|correct|right|confirmed|that's right|that is right)\b", caller_text, re.IGNORECASE))
    if (
        _pending_email == normalized_email
        and caller_turn > _pending_email_turn
        and affirmative
    ):
        _pending_email = ""
        _pending_email_turn = 0
        return False

    _pending_email = normalized_email
    _pending_email_turn = caller_turn
    return True


@tool_options(cancel_on_interruption=True, timeout_secs=120)
async def booking_workflow(
    params: FunctionCallParams,
    meeting_date: str = "",
    meeting_time: str = "",
    name: str = "",
    email: str = "",
    time_preference: str = "",
):
    """Start or continue the caller's booking workflow; also get its current status.

    Args:
        meeting_date: Caller-provided date in YYYY-MM-DD, otherwise empty.
        meeting_time: Caller-selected time in HH:MM AM/PM, otherwise empty.
        name: Caller-provided name, otherwise empty.
        email: Caller-provided email address, otherwise empty. On the first email
            turn, call this tool immediately so it can ask for confirmation. After
            the caller confirms on their next turn, pass the same email again.
        time_preference: Caller preference such as morning, afternoon, evening,
            after 3 PM, before lunch, or around 5 PM; otherwise empty.
    """
    started = perf_counter()
    trace_recorder.record(
        "tool_call",
        tool="booking_workflow",
        arguments={
            "meeting_date": meeting_date,
            "meeting_time": meeting_time,
            "name": name,
            "email": email,
            "time_preference": time_preference,
        },
        timeout_seconds=120,
        application_retries=0,
    )
    caller_text = _latest_caller_text()
    caller_turn = _latest_caller_turn()
    time_preference = _caller_supported_time_preference(time_preference, caller_text)
    confirm_email = bool(
        email
        and caller_turn > 0
        and _email_requires_confirmation(email, caller_text, caller_turn)
    )
    result = await booking_session.advance(
        meeting_date=meeting_date,
        meeting_time=meeting_time,
        name=name,
        email="" if confirm_email else email,
        time_preference=time_preference,
    )
    if confirm_email:
        # Preserve all other booking progress, but do not write an uncertain address.
        result = dict(result)
        result["spoken_response"] = f"I heard {_say_email(email)}. Is that correct?"
    trace_recorder.record(
        "tool_result",
        tool="booking_workflow",
        duration_ms=round((perf_counter() - started) * 1000, 3),
        result={
            "status": result.get("status"),
            "booking_confirmed": result.get("booking_confirmed"),
            "lead_saved": result.get("lead_saved"),
            "spoken_response": result.get("spoken_response"),
            "offered_slots": result.get("offered_slots"),
        },
        error_type=None if result.get("status") != "error" else "BookingWorkflowError",
    )
    if result.get("status") == "completed":
        trace_recorder.finalize_after_speech()
    async def emit_spoken_response():
        # The workflow has already produced the exact public response. Emit it
        # through the existing LLM-frame -> TTS path without another provider
        # generation that could add reasoning, paraphrases, or duplicates.
        await params.llm.push_frame(LLMFullResponseStartFrame())
        await params.llm.push_frame(LLMTextFrame(result["spoken_response"]))
        await params.llm.push_frame(LLMFullResponseEndFrame())

    await params.result_callback(
        result,
        properties=FunctionCallResultProperties(
            run_llm=False,
            on_context_updated=emit_spoken_response,
        ),
    )


def load_system_prompt():
    with open(
        Path(__file__).parent / "prompts" / "receptionist.md",
        "r",
        encoding="utf-8"
    ) as file:
        return file.read()


SYSTEM_PROMPT = load_system_prompt() + (
    "\nCurrent local date: "
    + datetime.now(ZoneInfo("Asia/Karachi")).strftime("%Y-%m-%d")
    + ". Resolve relative dates in Asia/Karachi timezone.\n"
)

llm = GroqLLMService(
    api_key=os.getenv("GROQ_API_KEY"),
    settings=GroqLLMService.Settings(
        model="openai/gpt-oss-120b",
        temperature=0.3,
        system_instruction=SYSTEM_PROMPT,
        extra={"extra_body": {"reasoning_format": "hidden"}},
    ),
)

class SpokenTextFilter(BaseTextFilter):
    async def filter(self, text: str) -> str:
        # Quotes and Markdown emphasis are visual punctuation and should not be spoken.
        return text.translate(str.maketrans("", "", '"“”*`'))


tts = DeepgramTTSService(
    api_key=os.getenv("DEEPGRAM_API_KEY"),
    settings=DeepgramTTSService.Settings(
        voice="aura-2-thalia-en",
    ),
    text_filters=[SpokenTextFilter()],
)

transport = LocalAudioTransport(
    LocalAudioTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    )
)

context = LLMContext(
    tools=[
        booking_workflow,
    ]
)

user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(),
        # Prevent a false interruption from cancelling the callback after the
        # shielded Calendar/Sheets work has already completed. This mute ends
        # with the function result, before TTS starts, so TTS barge-in remains.
        user_mute_strategies=[FunctionCallUserMuteStrategy()],
    ),
)

rag_context = RAGContextProcessor()

class LLMPrintProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            trace_recorder.start_assistant_response()

        if isinstance(frame, TextFrame):
            trace_recorder.assistant_text(frame.text)
            print(frame.text, end="", flush=True)

        if isinstance(frame, LLMFullResponseEndFrame):
            trace_recorder.end_assistant_response()
            print()

        await self.push_frame(frame, direction)

class TranscriptionPrinter(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.latest_text = ""
        self.turn_id = 0

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            self.latest_text = frame.text
            self.turn_id += 1
            trace_recorder.user_turn(frame.text)
            print(f"You: {frame.text}")

        await self.push_frame(frame, direction)


transcription_printer = TranscriptionPrinter()

pipeline = Pipeline([
    transport.input(),
    stt,
    transcription_printer,
    user_aggregator,
    rag_context,
    llm,
    LLMPrintProcessor(),
    tts,
    transport.output(),
    assistant_aggregator,
])

worker = PipelineWorker(
    pipeline,
    params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    observers=[TraceMetricsObserver(trace_recorder)],
)


async def main():
    runner = WorkerRunner()

    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
