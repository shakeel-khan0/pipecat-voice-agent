import asyncio
import os
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
from pipecat.pipeline.worker import PipelineWorker
from pipecat.workers.runner import WorkerRunner

from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.turns.user_mute import FunctionCallUserMuteStrategy

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from agent.booking_session import BookingSession
from pipecat.adapters.schemas.direct_function import tool_options


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
    ),
)

# One process / local microphone conversation is one caller session.
booking_session = BookingSession()


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
        email: Caller-provided email address, otherwise empty.
        time_preference: Caller preference such as morning, afternoon, evening,
            after 3 PM, before lunch, or around 5 PM; otherwise empty.
    """
    result = await booking_session.advance(
        meeting_date=meeting_date, meeting_time=meeting_time, name=name, email=email,
        time_preference=time_preference,
    )
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

tts = DeepgramTTSService(
    api_key=os.getenv("DEEPGRAM_API_KEY"),
    settings=DeepgramTTSService.Settings(
        voice="aura-2-thalia-en",
    ),
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

class LLMPrintProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            print(frame.text, end="", flush=True)

        if isinstance(frame, LLMFullResponseEndFrame):
            print()

        await self.push_frame(frame, direction)

class TranscriptionPrinter(FrameProcessor):
    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            print(f"You: {frame.text}")

        await self.push_frame(frame, direction)


pipeline = Pipeline([
    transport.input(),
    stt,
    TranscriptionPrinter(),
    user_aggregator,
    llm,
    LLMPrintProcessor(),
    tts,
    transport.output(),
    assistant_aggregator,
])

worker = PipelineWorker(pipeline)


async def main():
    runner = WorkerRunner()

    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
