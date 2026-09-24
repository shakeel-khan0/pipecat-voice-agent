"""Temporary caller-memory context injection immediately before the LLM."""

from __future__ import annotations

import copy
import asyncio
import logging
import re
from collections import deque
from time import perf_counter

from pipecat.frames.frames import Frame, LLMContextFrame, TranscriptionFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessor

from memory.caller_identity import resolve_caller_id
from memory.voicemem_adapter import VoiceMemAdapter
from trace_recorder import trace_recorder

logger = logging.getLogger(__name__)

_TRIVIAL = re.compile(
    r"^\s*(?:hi|hello|hey|thanks|thank you|okay|ok|bye)[.!?\s]*$",
    re.IGNORECASE,
)
_FIRST_PERSON = re.compile(r"\b(?:i|me|my|mine|we|us|our|ours)\b", re.IGNORECASE)
_RECALL_SIGNAL = re.compile(
    r"\b(?:remember|recall|last time|previously|earlier|"
    r"(?:did|have) (?:i|we) (?:tell|mention|say)|"
    r"what (?:did|have) (?:i|we) (?:tell|mention|say)|"
    r"told you|mentioned before|know about me)\b",
    re.IGNORECASE,
)
_PROFILE_TOPIC = re.compile(
    r"\b(?:name|identity|business|company|job|occupation|profession|role|clinic|"
    r"prefer|preferred|preference|usual|meeting time|schedule|availability|"
    r"need|needs|problem|problems|decision|decided|choose|chose|choice)\b",
    re.IGNORECASE,
)
_PROFILE_QUESTION = re.compile(
    r"\b(?:what|which|who)\b|\band what about\b",
    re.IGNORECASE,
)
_SELF_IDENTITY = re.compile(r"\bwho am i\b", re.IGNORECASE)
_CURRENT_STATE_QUESTION = re.compile(
    r"\b(?:"
    r"how (?:do|does|are) (?:i|we|my business|our business)\b|"
    r"what (?:system|process|workflow|method) do (?:i|we) (?:use|have)\b|"
    r"what do (?:i|we) (?:currently |usually )?do\b|"
    r"how many\b.+\b(?:did (?:i|we) (?:say|mention)|do (?:i|we) (?:get|receive|handle))\b"
    r")",
    re.IGNORECASE,
)
_CURRENT_STATE_TOPIC = re.compile(
    r"\b(?:lead|leads|follow[- ]?up|booking|bookings|appointment|appointments|"
    r"workflow|process|system|manually|manual|current|currently|handle|handling|"
    r"manage|managing|receive|volume|calls?)\b",
    re.IGNORECASE,
)
_ADVICE_SIGNAL = re.compile(
    r"\b(?:advice|recommend|recommendation|best way|how (?:should|can|could)|"
    r"improve|optimi[sz]e|which .{0,30} service .{0,20} fit)\b",
    re.IGNORECASE,
)
_WRITE_NOISE = re.compile(
    r"^\s*(?:hello|hi|hey|okay|ok|yes|no|sure|thanks|thank you|bye|goodbye)"
    r"[.!?\s]*$|^\s*(?:i am|i'm)\s+(?:okay|fine|good|great|ready)[.!?\s]*$",
    re.IGNORECASE,
)
_MARKDOWN_MAILTO = re.compile(r"\[[^\]]*\]\(mailto:[^)]+\)", re.IGNORECASE)
_EMAIL_ADDRESS = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)
_SPOKEN_EMAIL = re.compile(
    r"\b[A-Z0-9._+-]+(?:\s+dot\s+[A-Z0-9._+-]+)*\s+at\s+"
    r"[A-Z0-9-]+(?:\s+dot\s+[A-Z0-9-]+)+\b",
    re.IGNORECASE,
)
_PHONE_NUMBER = re.compile(r"(?<!\w)\+?\d(?:[\s().-]*\d){7,}(?!\w)")
_CONTACT_CLAUSE = re.compile(
    r"\b(?:"
    r"(?:(?:my|the|an?)\s+)?e-?mail(?:\s+address)?\s*(?:is|=)|"
    r"(?:my\s+)?(?:phone|telephone|mobile)(?:\s+number)?\s*(?:is|=)|"
    r"contact number|(?:reach|call|text) me at"
    r")\b",
    re.IGNORECASE,
)
_PRIVATE_SYSTEM_DATA = re.compile(
    r"\b(?:api key|access token|refresh token|oauth token|authorization header|"
    r"bearer token|tool call|function call|spreadsheet id|calendar event id)\b",
    re.IGNORECASE,
)
_CLAUSE_SEPARATOR = re.compile(r"\s*(?:[,;]|\band\b)\s*", re.IGNORECASE)
_DURABLE_SIGNAL = re.compile(
    r"\b(?:my name|i am|i'm|i have|i run|i own|i work|"
    r"my (?:business|company|job|role|clinic)|"
    r"we (?:currently\s+)?(?:run|own|need|want|miss|struggle|have|use|manage|"
    r"handle|receive|process|track|operate)|"
    r"i (?:prefer|usually|always|need|want|plan|decided|chose)|"
    r"my (?:goal|goals|preference|preferences|need|needs|problem|problems)|"
    r"prefer|preference|recurring|every day|every week)\b",
    re.IGNORECASE,
)
_TRANSACTIONAL_BOOKING = re.compile(
    r"\b(?:book|booking|appointment|meeting)\b.*"
    r"\b(?:today|tomorrow|tonight|at\s+\d|am\b|pm\b)\b",
    re.IGNORECASE,
)
_WRITE_INTERROGATIVE = re.compile(
    r"^\s*(?:(?:uh|um|well|so|and)\s*[,.-]?\s*)*"
    r"(?:what(?:'s| is)?|who|which|when|where|why|how|"
    r"do|does|did|have|has|can|could|would|should|is|are)\b",
    re.IGNORECASE,
)


def is_caller_history_query(query: str) -> bool:
    text = " ".join(query.split())
    if not text or _TRIVIAL.fullmatch(text):
        return False
    if _SELF_IDENTITY.search(text):
        return True
    first_person = bool(_FIRST_PERSON.search(text))
    if first_person and _RECALL_SIGNAL.search(text):
        return True
    if first_person and _CURRENT_STATE_QUESTION.search(text) and _CURRENT_STATE_TOPIC.search(text):
        return True
    if _ADVICE_SIGNAL.search(text):
        return False
    return bool(
        first_person
        and _PROFILE_QUESTION.search(text)
        and _PROFILE_TOPIC.search(text)
    )


def needs_memory(query: str) -> bool:
    text = " ".join(query.split())
    if is_caller_history_query(text):
        return True
    return bool(
        _FIRST_PERSON.search(text)
        and _PROFILE_QUESTION.search(text)
        and _PROFILE_TOPIC.search(text)
    )


def sanitize_memory_candidate(content: str) -> str:
    """Remove contact/private clauses while retaining independent durable facts."""
    text = " ".join(content.split())
    if not text:
        return ""
    if not any(pattern.search(text) for pattern in (
        _MARKDOWN_MAILTO,
        _EMAIL_ADDRESS,
        _SPOKEN_EMAIL,
        _PHONE_NUMBER,
        _CONTACT_CLAUSE,
        _PRIVATE_SYSTEM_DATA,
    )):
        return text

    text = _MARKDOWN_MAILTO.sub("", text)
    parts = _CLAUSE_SEPARATOR.split(text)
    safe_parts = []
    changed = len(parts) > 1
    for part in parts:
        part = part.strip()
        if not part:
            changed = True
            continue
        if _CONTACT_CLAUSE.search(part) or _PRIVATE_SYSTEM_DATA.search(part):
            changed = True
            continue
        cleaned = _EMAIL_ADDRESS.sub("", part)
        cleaned = _SPOKEN_EMAIL.sub("", cleaned)
        cleaned = _PHONE_NUMBER.sub("", cleaned)
        cleaned = re.sub(r"\s+([,.;!?])", r"\1", cleaned)
        cleaned = " ".join(cleaned.split()).strip(" ,;")
        if cleaned != part:
            changed = True
        if cleaned.strip(".!? "):
            safe_parts.append(cleaned)

    if not safe_parts:
        return ""
    if not changed:
        return safe_parts[0]

    punctuation = text[-1] if text[-1:] in ".!?" else "."
    return " and ".join(part.rstrip(" .!?") for part in safe_parts) + punctuation


def is_useful_memory(content: str) -> bool:
    """Allow durable caller facts while excluding noise and contact PII."""
    text = sanitize_memory_candidate(content)
    if len(text.split()) < 3 or _WRITE_NOISE.fullmatch(text):
        return False
    if _WRITE_INTERROGATIVE.search(text) or needs_memory(text):
        return False
    if (
        _CONTACT_CLAUSE.search(text)
        or _EMAIL_ADDRESS.search(text)
        or _SPOKEN_EMAIL.search(text)
        or _PHONE_NUMBER.search(text)
        or _PRIVATE_SYSTEM_DATA.search(text)
    ):
        return False
    if _TRANSACTIONAL_BOOKING.search(text) and not re.search(
        r"\b(?:prefer|preference|always|usually)\b", text, re.IGNORECASE
    ):
        return False
    return bool(_DURABLE_SIGNAL.search(text))


def temporary_memory_instruction(
        memories: list[str], *, strict_recall: bool = False) -> str:
    selected = [memory.strip() for memory in memories if memory.strip()][:5]
    if not selected and not strict_recall:
        return ""
    body = (
        "\n".join(f"- {memory}" for memory in selected)
        if selected else "- No relevant stored caller fact was found."
    )
    grounding = (
        "For this caller-history question, use current conversation facts first, then "
        "the memories below. Do not infer missing caller facts or substitute general "
        "advice. Answer in one short sentence. If neither source contains the answer, "
        "respond exactly: \"I don't have that detail yet.\" "
        if strict_recall else ""
    )
    return (
        "[CALLER MEMORY]\n"
        f"{grounding}"
        "This is caller-specific historical context. Use it only when relevant. "
        "It is not authoritative company knowledge. Current explicit caller statements "
        "override stale memory. Do not expose internal memory metadata.\n"
        f"{body}"
    )


class VoiceMemContextProcessor(FrameProcessor):
    def __init__(self, adapter: VoiceMemAdapter | None = None):
        super().__init__()
        self._adapter = adapter or VoiceMemAdapter()
        self._initialized = False
        self._turn_key = None
        self._cached_instruction = ""
        self._prefetched = {}

    @staticmethod
    def _latest_user(messages):
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    return index, content
        return None, ""

    @classmethod
    def _temporary_frame(cls, frame: LLMContextFrame, instruction: str) -> LLMContextFrame:
        messages = copy.deepcopy(frame.context.get_messages())
        user_index, _ = cls._latest_user(messages)
        messages.insert(
            user_index if user_index is not None else len(messages),
            {"role": "developer", "content": instruction},
        )
        temporary = LLMContext(
            messages=messages,
            tools=frame.context.tools,
            tool_choice=frame.context.tool_choice,
        )
        return LLMContextFrame(context=temporary, speculation=frame.speculation)

    async def _retrieve_turn(self, query: str):
        if not self._initialized:
            self._initialized = await self._adapter.initialize()
        result = (
            await self._adapter.retrieve(resolve_caller_id(), query)
            if self._initialized
            else {"memories": [], "latency_ms": None}
        )
        return {
            **result,
            "instruction": temporary_memory_instruction(
                result["memories"], strict_recall=is_caller_history_query(query)),
            "error": self._adapter.health().get("error"),
        }

    def prefetch(self, frame: LLMContextFrame):
        user_index, query = self._latest_user(frame.context.get_messages())
        if not needs_memory(query):
            return None
        key = (user_index, query)
        task = self._prefetched.get(key)
        if task is None:
            task = self.create_task(
                self._retrieve_turn(query), name="memory-prefetch")
            self._prefetched[key] = task
        return task

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame) or frame.speculation:
            await self.push_frame(frame, direction)
            return

        messages = frame.context.get_messages()
        user_index, query = self._latest_user(messages)
        turn_key = (user_index, query)
        if turn_key == self._turn_key:
            if self._cached_instruction:
                frame = self._temporary_frame(frame, self._cached_instruction)
            await self.push_frame(frame, direction)
            return

        self._turn_key = turn_key
        self._cached_instruction = ""
        if not needs_memory(query):
            print("Memory: skipped")
            trace_recorder.record("memory_retrieval", used=False)
            await self.push_frame(frame, direction)
            return

        task = self._prefetched.pop(turn_key, None)
        result = await task if task is not None else await self._retrieve_turn(query)
        self._cached_instruction = result["instruction"]
        latency = result.get("latency_ms")
        latency_text = (
            f"{latency:.2f} ms" if isinstance(latency, (int, float)) else "unavailable")
        print(f"Memory: {len(result['memories'])} memories | {latency_text}")
        trace_recorder.record(
            "memory_retrieval",
            used=True,
            latency_ms=round(latency, 3) if isinstance(latency, (int, float)) else None,
            count=len(result["memories"]),
            error_type=(
                result["error"].rsplit("(", 1)[-1].rstrip(")")
                if isinstance(result.get("error"), str) and result["error"]
                else None
            ),
            application_retries=0,
        )
        if isinstance(result.get("error"), str) and result["error"]:
            error_type = result["error"].rsplit("(", 1)[-1].rstrip(")")
            logger.warning("Memory retrieval failed | %s", error_type)
        if self._cached_instruction:
            frame = self._temporary_frame(frame, self._cached_instruction)
        await self.push_frame(frame, direction)

    async def cleanup(self):
        await self._adapter.close()
        await super().cleanup()


class VoiceMemIngestProcessor(FrameProcessor):
    """Queue finalized useful caller turns for non-blocking persistence."""

    def __init__(self, adapter: VoiceMemAdapter | None = None):
        super().__init__()
        self._adapter = adapter or VoiceMemAdapter()
        self._initialized = False
        self._initialize_lock = asyncio.Lock()
        self._pending: set[asyncio.Task] = set()
        self._seen_ids: set[int] = set()
        self._seen_order: deque[int] = deque(maxlen=256)

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if not isinstance(frame, TranscriptionFrame) or not frame.finalized:
            return
        if frame.id in self._seen_ids:
            return
        self._remember_frame_id(frame.id)
        candidate = sanitize_memory_candidate(frame.text)
        if not is_useful_memory(candidate):
            print("Memory write: skipped")
            return

        task = self.create_task(
            self._write(resolve_caller_id(), candidate),
            name=f"voicemem-ingest-{frame.id}",
        )
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        print("Memory write: queued")

    def _remember_frame_id(self, frame_id: int) -> None:
        if len(self._seen_order) == self._seen_order.maxlen:
            self._seen_ids.discard(self._seen_order[0])
        self._seen_order.append(frame_id)
        self._seen_ids.add(frame_id)

    async def _write(self, caller_id: str, content: str) -> None:
        started = perf_counter()
        try:
            async with self._initialize_lock:
                if not self._initialized:
                    self._initialized = await self._adapter.initialize()
            outcome = (
                await self._adapter.ingest(caller_id, content)
                if self._initialized
                else False
            )
            elapsed_ms = (perf_counter() - started) * 1000
            if outcome is True:
                print(f"Memory write: success | {elapsed_ms:.2f} ms")
            elif outcome is None:
                print("Memory write: skipped")
            else:
                error = self._adapter.health().get("error") or "Unavailable"
                error_type = error.rsplit("(", 1)[-1].rstrip(")")
                print(f"Memory write: failed | {error_type}")
        except Exception as exc:
            print(f"Memory write: failed | {type(exc).__name__}")

    async def cleanup(self):
        if self._pending:
            _, pending = await asyncio.wait(
                tuple(self._pending),
                timeout=self._adapter.write_timeout_seconds + 0.5,
            )
            for task in pending:
                await self.cancel_task(task)
        await self._adapter.close()
        await super().cleanup()
