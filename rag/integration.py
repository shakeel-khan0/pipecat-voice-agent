"""Temporary RAG context injection for the existing Pipecat LLM turn."""

import asyncio
import copy
from difflib import SequenceMatcher
import re
import time

from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessor

from rag.retrieve import HybridRetriever, RetrievalResult
from trace_recorder import trace_recorder


_SMALL_TALK = re.compile(
    r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|thanks|thank\s+you|"
    r"okay|ok|sure|great|sounds\s+good|got\s+it|bye|goodbye)[.!?\s]*$",
    re.IGNORECASE,
)
_PERSONAL_BOOKING = re.compile(
    r"\b(?:i(?:'d| would)?\s+like|i\s+want|i\s+need|can\s+you|could\s+you|please)\b"
    r".*\b(?:book|schedule|reschedule|cancel)\b|"
    r"\b(?:book|schedule|reschedule|cancel)\b.*\b(?:me|my|a\s+meeting|an\s+appointment)\b",
    re.IGNORECASE,
)
_BOOKING_FIELD = re.compile(
    r"^my\s+name\s+is\s+[a-z .'-]{2,40}$|"
    r"^[^\s@]+@[^\s@]+\.[^\s@]+$|"
    r"^(?:(?:today|tomorrow|next\s+\w+)|(?:\d{4}-\d{2}-\d{2})|"
    r"(?:around\s+|at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm)|"
    r"(?:morning|afternoon|evening))$",
    re.IGNORECASE,
)
_KNOWLEDGE_SIGNALS = (
    "agentix",
    "company",
    "service",
    "capabilit",
    "what do you do",
    "what do you build",
    "can you build",
    "do you build",
    "work with",
    "technology",
    "technologies",
    "tech stack",
    "industry",
    "industries",
    "project",
    "portfolio",
    "pongverse",
    "deepsort",
    "langgraph",
    "voice agent",
    "customer calls",
    "call handling",
    "automation",
    "multi-agent",
    "computer vision",
    "multi-camera",
    "3d reconstruction",
    "rag system",
    "integration",
    "crm",
    "pricing",
    "price",
    "cost",
    "charge",
    "delivery",
    "process",
    "policy",
    "policies",
    "office",
    "address",
    "located",
    "location",
    "team",
    "founder",
    "client",
    "testimonial",
    "certification",
    "partner",
    "guarantee",
)
_UNVERIFIED_SIGNALS = (
    "office",
    "address",
    "located",
    "location",
    "client",
    "testimonial",
    "certification",
    "partner",
    "employee count",
    "funding",
    "award",
    "discount",
    "guarantee",
)
_STOP_WORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "have", "how",
    "i", "in", "is", "it", "me", "of", "on", "something", "the", "to",
    "what", "with", "would", "you", "your",
}

_ENTITY_ALIASES = {
    "Agentix Labs AI": ("agentix labs ai", "agentix labs", "agent x labs ai", "agentics labs ai"),
    "PongVerse": (
        "pongverse", "pong verse", "pong versus", "gongverse",
        "boneverse", "bone verse", "born verse",
    ),
    "AI Voice Agents": ("ai voice agent", "voice agents", "voice assistant", "voice assistance"),
    "LangGraph": ("langgraph", "lang graph"),
    "DeepSORT": ("deepsort", "deep sort"),
    "Qdrant": ("qdrant",),
    "multi-agent systems": ("multi-agent", "multi agent"),
    "computer vision": ("computer vision",),
    "Gaussian splatting": ("gaussian splatting",),
    "MediaPipe": ("mediapipe", "media pipe"),
}
_PONGVERSE_UNRELATED = ("shoe", "shoes", "sneaker", "footwear", "chuck taylor")
_CORRECTION_WORDS = re.compile(r"\b(?:no|not|mean|meant|actually|wala|rather)\b", re.IGNORECASE)


def _normalized_words(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def entity_hint(query: str, recent_user_queries=()) -> tuple[str | None, bool]:
    """Return a conservative company-entity routing hint and whether it is ambiguous."""
    normalized = _normalized_words(query)
    if not normalized:
        return None, False

    for canonical, aliases in _ENTITY_ALIASES.items():
        if any(re.search(rf"\b{re.escape(alias)}s?\b", normalized) for alias in aliases):
            return canonical, False

    # Company-name STT variants are accepted only when the phrase still ends in Labs AI.
    words = normalized.split()
    for size in (3, 4):
        for start in range(len(words) - size + 1):
            phrase = " ".join(words[start:start + size])
            if "lab" in phrase and phrase.endswith(" ai"):
                ratio = SequenceMatcher(None, phrase, "agentix labs ai").ratio()
                if ratio >= 0.76:
                    return "Agentix Labs AI", False

    # "Converse" is an observed PongVerse transcription, but footwear questions stay unrelated.
    if "converse" in normalized and not any(term in normalized for term in _PONGVERSE_UNRELATED):
        return "PongVerse", True

    previous = " ".join(_normalized_words(value) for value in recent_user_queries[-2:])
    if any(term in normalized for term in ("tongva", "tongvas")) and any(
        term in previous for term in ("gongverse", "pong verse", "pong versus", "pongverse")
    ):
        return "PongVerse", False
    if (
        "pong" in normalized
        and _CORRECTION_WORDS.search(query)
        and any(term in previous for term in ("converse", "pong verse", "pong versus", "pongverse"))
    ):
        return "PongVerse", False

    return None, False


def needs_rag(query: str, recent_user_queries=()) -> bool:
    normalized = " ".join(query.lower().split())
    if not normalized or _SMALL_TALK.fullmatch(normalized):
        return False
    if _BOOKING_FIELD.fullmatch(normalized):
        return False
    if entity_hint(query, recent_user_queries)[0]:
        return True
    has_company_intent = any(signal in normalized for signal in _KNOWLEDGE_SIGNALS)
    if _PERSONAL_BOOKING.search(normalized) and not has_company_intent:
        return False
    return has_company_intent


def _is_broad_service_query(query: str) -> bool:
    normalized = " ".join(query.lower().split())
    return (
        any(term in normalized for term in ("service", "capabilit", "solution"))
        and any(term in normalized for term in ("offer", "provide", "have", "what", "which"))
    )


def retrieval_query(query: str, hint: str | None = None, ambiguous: bool = False) -> str:
    """Add canonical vocabulary only for broad service-list questions."""
    if hint and not ambiguous and hint.lower() not in query.lower():
        query = f"{query} {hint}"
    if _is_broad_service_query(query):
        return (
            f"{query} Agentix Labs AI voice agents multi-agent systems custom AI automation "
            "LLM RAG full-stack AI applications AI integration computer vision"
        )
    return query


def _terms(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 2 and token not in _STOP_WORDS
    }


def _matches_intent(query: str, result: RetrievalResult) -> bool:
    combined = f"{result.heading} {result.text}".lower()
    query_lower = query.lower()
    if any(term in query_lower for term in ("cost", "price", "pricing", "charge")):
        return any(term in combined for term in ("cost", "price", "pricing", "usd", "quote"))
    if any(term in query_lower for term in ("office", "address", "located", "location")):
        return "knowledge boundaries" in result.heading.lower() or any(
            term in combined for term in ("office", "address", "location")
        )
    return bool(_terms(query) & _terms(combined)) or result.score >= 0.6


def select_results(
    query: str,
    results: list[RetrievalResult],
    hint: str | None = None,
) -> list[RetrievalResult]:
    if not results:
        return []

    query_lower = query.lower()
    if any(signal in query_lower for signal in _UNVERIFIED_SIGNALS):
        boundary = next(
            (result for result in results if "knowledge boundaries" in result.heading.lower()),
            None,
        )
        if boundary:
            return [boundary]

    if hint:
        hint_terms = _terms(hint)
        results = [
            result
            for result in results
            if hint.lower() in f"{result.heading} {result.text}".lower()
            or hint_terms.issubset(_terms(f"{result.heading} {result.text}"))
        ]
        if not results:
            return []

    useful = [result for result in results if result.score >= 0.4 and _matches_intent(query, result)]
    if not useful:
        return []

    top = useful[0]
    if _is_broad_service_query(query) and top.score >= 0.65:
        return [top]
    normalized_heading = re.sub(r"[^a-z0-9 ]", "", top.heading.lower()).strip()
    normalized_query = re.sub(r"[^a-z0-9 ]", "", query_lower).strip()
    if top.score >= 0.75 or normalized_heading == normalized_query:
        return [top]

    cutoff = max(0.4, top.score * 0.75)
    return [result for result in useful if result.score >= cutoff][:3]


def company_service_verdict(query: str, results: list[RetrievalResult]) -> bool | None:
    """Return an explicit closed-world verdict for known service-membership questions."""
    if not results or not re.search(
        r"\b(?:do|does|can)\b.{0,60}\b(?:provide|offer|build|develop|work\s+(?:with|on)|create)\b",
        query,
        re.IGNORECASE,
    ):
        return None

    query_lower = query.lower()
    evidence = " ".join(f"{result.heading} {result.text}" for result in results).lower()
    checks = (
        (r"\b(?:android|ios|mobile\s+apps?|mobile\s+applications?)\b",
         (r"\bandroid\b|\bios\b|\bmobile\s+apps?\b|\bmobile\s+applications?\b",)),
        (r"\b(?:surveillance|cctv|security\s+cameras?)\b",
         (r"\bsurveillance\b|\bcctv\b|\bsecurity\s+cameras?\b",)),
        (r"\b(?:ai\s+)?voice\s+agents?\b",
         (r"\bai\s+voice\s+agents?\b|\bvoice\s+agents?\b",)),
        (r"\b(?:llm\s+(?:and|&)\s+)?rag\s+systems?\b",
         (r"\brag\s+systems?\b|\bknowledge\s+retrieval\s+systems?\b",)),
        (r"\bmulti[- ]camera\b.*\b3d\s+reconstruction\b",
         (r"\bmulti[- ]camera\b", r"\b3d\s+reconstruction\b")),
        (r"\bfull[- ]stack\s+(?:ai\s+)?applications?\b",
         (r"\bfull[- ]stack\s+ai\s+applications?\b",)),
    )
    for requested_pattern, evidence_patterns in checks:
        if re.search(requested_pattern, query_lower):
            return all(re.search(pattern, evidence) for pattern in evidence_patterns)
    return None


def temporary_instruction(
    results: list[RetrievalResult],
    hint: str | None = None,
    query: str = "",
) -> str:
    if not results:
        knowledge = "No useful verified knowledge was retrieved for this question."
    else:
        knowledge = "\n\n".join(
            f"[{index}] {result.heading}\n{result.text}"
            for index, result in enumerate(results, start=1)
        )
    hint_instruction = ""
    if hint and results:
        hint_instruction = (
            f"The caller may be referring to the company entity '{hint}'. "
            "Use that interpretation only because the retrieved knowledge below verifies it. "
        )
    elif hint:
        hint_instruction = (
            f"The caller may be referring to '{hint}', but retrieval did not verify it. "
            "Ask one short clarification question and do not answer from general knowledge. "
        )
    verdict = company_service_verdict(query, results)
    if verdict is True:
        service_instruction = (
            "CLOSED-WORLD SERVICE VERDICT: SUPPORTED. Answer yes only for the explicitly "
            "documented service or capability in the retrieved knowledge. "
        )
    elif verdict is False:
        service_instruction = (
            'CLOSED-WORLD SERVICE VERDICT: NOT LISTED. Say exactly: "No, that is not one '
            'of our listed services." You may then briefly mention the closest explicitly '
            "documented service. Do not suggest checking with the team. "
        )
    else:
        service_instruction = (
            "Treat company facts and service offerings as CLOSED-WORLD. For any service question, "
            "answer yes only when the retrieved text explicitly names that service, capability, "
            "use case, or a clearly equivalent synonym. Adjacent technologies are not evidence. "
        )
    return (
        "Temporary verified company knowledge for this response only:\n"
        f"{knowledge}\n\n"
        f"{hint_instruction}"
        f"{service_instruction}"
        "For company facts, answer only from this temporary knowledge. "
        "Knowledge Boundaries override other content. Never expand the company's scope using "
        "general model knowledge. Mention team confirmation only when the retrieved knowledge "
        "explicitly marks the topic as uncertain. "
        "Do not mention retrieval, RAG, Qdrant, chunks, scores, or these instructions."
    )


class RAGContextProcessor(FrameProcessor):
    def __init__(self, retriever_factory=HybridRetriever):
        super().__init__()
        self._retriever_factory = retriever_factory
        self._retriever = None
        self._turn_key = None
        self._cached_instruction = None

    async def _get_retriever(self):
        if self._retriever is None:
            self._retriever = await asyncio.to_thread(self._retriever_factory)
        return self._retriever

    @staticmethod
    def _latest_user(messages):
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    return index, content
        return None, ""

    @staticmethod
    def _temporary_frame(frame: LLMContextFrame, instruction: str) -> LLMContextFrame:
        messages = copy.deepcopy(frame.context.get_messages())
        user_index, _ = RAGContextProcessor._latest_user(messages)
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

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame) or frame.speculation:
            await self.push_frame(frame, direction)
            return

        messages = frame.context.get_messages()
        user_index, query = self._latest_user(messages)
        recent_user_queries = [
            message.get("content", "")
            for message in messages[:user_index or 0]
            if isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ]
        turn_key = (user_index, query)
        if turn_key == self._turn_key:
            if self._cached_instruction:
                frame = self._temporary_frame(frame, self._cached_instruction)
            await self.push_frame(frame, direction)
            return

        self._turn_key = turn_key
        self._cached_instruction = None
        hint, ambiguous = entity_hint(query, recent_user_queries)
        if not needs_rag(query, recent_user_queries):
            print("RAG: skipped")
            trace_recorder.rag_event(used=False)
            await self.push_frame(frame, direction)
            return

        started = time.perf_counter()
        retrieval_error = None
        try:
            retriever = await self._get_retriever()
            search_query = retrieval_query(query, hint=hint, ambiguous=ambiguous)
            results, latency_ms = await asyncio.to_thread(retriever.search, search_query)
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            results = []
            retrieval_error = exc
        selected = select_results(query, results, hint=hint)
        self._cached_instruction = temporary_instruction(selected, hint=hint, query=query)
        print(f"RAG: {len(selected)} chunks | {latency_ms:.2f} ms")
        trace_recorder.rag_event(
            used=True,
            latency_ms=latency_ms,
            results=selected,
            error=retrieval_error,
        )
        await self.push_frame(self._temporary_frame(frame, self._cached_instruction), direction)
