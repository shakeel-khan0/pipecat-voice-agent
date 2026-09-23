"""Small, fail-safe recorder for one sanitized AIReview execution trace."""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from pipecat.frames.frames import BotStoppedSpeakingFrame, LLMContextFrame, MetricsFrame
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    STTUsageMetricsData,
    TTFAMetricsData,
    TTFATMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed


PROJECT_ROOT = Path(__file__).resolve().parent
TRACE_PATH = PROJECT_ROOT / "traces" / "representative_trace.json"
KNOWLEDGE_PATH = PROJECT_ROOT / "knowledge" / "agentix_rag_knowledge_base.md"

_SENSITIVE_KEYS = {
    "api_key", "authorization", "credentials", "email", "event_id", "headers",
    "name", "oauth_token", "phone", "refresh_token", "spreadsheet_id", "thread_id",
    "token",
}
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
_SECRET = re.compile(
    r"\b(?:sk|gsk|AIza)[-_A-Za-z0-9]{16,}\b|\bBearer\s+\S+",
    re.IGNORECASE,
)
_NAME_PHRASE = re.compile(
    r"\b(my name is|name is|i am|i'm)\s+[A-Za-z][A-Za-z .'-]{1,60}"
    r"(?=\s+(?:and|email|my email)\b|[,.;!?]|$)",
    re.IGNORECASE,
)
_SPOKEN_EMAIL = re.compile(
    r"\b(?:my\s+)?email(?:\s+address)?\s+is\s+.+?"
    r"(?=\s+(?:and\s+my|and\s+the|for\s+the)\b|[.;!?]|$)",
    re.IGNORECASE,
)


class TraceRecorder:
    def __init__(self, output_path: Path = TRACE_PATH):
        self.output_path = output_path
        self._lock = threading.RLock()
        self._started = perf_counter()
        self._sequence = 0
        self._turn = 0
        self._events: list[dict[str, Any]] = []
        self._assistant_parts: list[str] = []
        self._final_response: str | None = None
        self._pending_finalize = False
        self._written = False
        self.trace_id = f"trace-{uuid4()}"
        try:
            self._knowledge = KNOWLEDGE_PATH.read_text(encoding="utf-8")
        except OSError:
            self._knowledge = ""
        self._summary = {
            "llm": {
                "provider": "Groq",
                "model": "openai/gpt-oss-120b",
                "calls": 0,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
            "stt": {"provider": "Deepgram", "model": "flux-general-en", "audio_seconds": None},
            "tts": {"provider": "Deepgram", "model": "aura-2-thalia-en", "characters": None},
        }

    @staticmethod
    def sanitize_text(value: str) -> str:
        text = _SECRET.sub("<redacted-secret>", str(value))
        text = _EMAIL.sub("<redacted-email>", text)
        text = _PHONE.sub("<redacted-phone>", text)
        text = _SPOKEN_EMAIL.sub("email is <redacted-email>", text)
        text = _NAME_PHRASE.sub(lambda match: f"{match.group(1)} <redacted-name>", text)
        return text[:2000]

    @classmethod
    def sanitize(cls, value: Any, key: str | None = None) -> Any:
        if key and key.lower() in _SENSITIVE_KEYS:
            return f"<redacted-{key.lower()}>"
        if isinstance(value, str):
            return cls.sanitize_text(value)
        if isinstance(value, dict):
            return {str(k): cls.sanitize(v, str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls.sanitize(item) for item in value]
        if isinstance(value, (bool, int, float)) or value is None:
            return value
        return f"<{type(value).__name__}>"

    def record(self, event_type: str, **data: Any) -> None:
        with self._lock:
            if self._written:
                return
            self._sequence += 1
            self._events.append({
                "sequence": self._sequence,
                "timestamp": datetime.now(ZoneInfo("Asia/Karachi")).isoformat(),
                "elapsed_ms": round((perf_counter() - self._started) * 1000, 3),
                "turn_id": self._turn or None,
                "type": event_type,
                "data": self.sanitize(data),
            })

    def user_turn(self, transcript: str) -> None:
        with self._lock:
            self._turn += 1
        self.record("user_transcript", transcript=transcript)

    def rag_event(self, *, used: bool, latency_ms: float | None = None, results=(), error=None):
        safe_results = []
        for result in results:
            text = str(result.text)
            # Only exact source text from the repository-owned public KB may enter the trace.
            excerpt = text[:500] if text and text in self._knowledge else None
            heading = str(result.heading)
            safe_results.append({
                "rank": int(result.rank),
                "score": float(result.score),
                "heading": heading if heading in self._knowledge else "<unverified-heading>",
                "text_excerpt": excerpt,
            })
        self.record(
            "rag_retrieval",
            used=used,
            latency_ms=round(latency_ms, 3) if latency_ms is not None else None,
            results=safe_results,
            error_type=type(error).__name__ if error else None,
            application_retries=0,
        )

    def start_assistant_response(self):
        with self._lock:
            self._assistant_parts = []

    def assistant_text(self, text: str):
        with self._lock:
            self._assistant_parts.append(text)

    def end_assistant_response(self):
        with self._lock:
            response = self.sanitize_text("".join(self._assistant_parts).strip())
            self._final_response = response
        self.record("assistant_response", text=response)

    def finalize_after_speech(self):
        with self._lock:
            self._pending_finalize = True

    def add_metric(self, metric: Any):
        payload: dict[str, Any] = {"processor": metric.processor, "model": metric.model}
        if isinstance(metric, LLMUsageMetricsData):
            usage = metric.value
            payload.update(
                metric="llm_usage",
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                reasoning_tokens=usage.reasoning_tokens,
            )
            self._summary["llm"].update(
                prompt_tokens=(self._summary["llm"]["prompt_tokens"] or 0) + usage.prompt_tokens,
                completion_tokens=(self._summary["llm"]["completion_tokens"] or 0) + usage.completion_tokens,
                total_tokens=(self._summary["llm"]["total_tokens"] or 0) + usage.total_tokens,
            )
        elif isinstance(metric, STTUsageMetricsData):
            payload.update(metric="stt_usage", audio_seconds=metric.value.audio_seconds)
            self._summary["stt"]["audio_seconds"] = (
                (self._summary["stt"]["audio_seconds"] or 0) + metric.value.audio_seconds
            )
        elif isinstance(metric, TTSUsageMetricsData):
            payload.update(metric="tts_usage", characters=metric.value)
            self._summary["tts"]["characters"] = (
                (self._summary["tts"]["characters"] or 0) + metric.value
            )
        elif isinstance(metric, TTFBMetricsData):
            payload.update(metric="ttfb", seconds=metric.value)
        elif isinstance(metric, TTFAMetricsData):
            payload.update(metric="ttfa", seconds=metric.ttfa)
        elif isinstance(metric, TTFATMetricsData):
            payload.update(metric="ttfat", seconds=metric.ttfat)
        elif isinstance(metric, ProcessingMetricsData):
            payload.update(metric="processing", seconds=metric.value)
        else:
            return
        self.record("pipecat_metric", **payload)

    def note_llm_call(self):
        self._summary["llm"]["calls"] += 1
        self.record("llm_call", provider="Groq", model="openai/gpt-oss-120b")

    def write(self):
        with self._lock:
            if self._written:
                return
            document = {
                "schema_version": "1.0",
                "trace_id": self.trace_id,
                "started_at": self._events[0]["timestamp"] if self._events else None,
                "completed_at": datetime.now(ZoneInfo("Asia/Karachi")).isoformat(),
                "summary": self.sanitize(self._summary),
                "events": list(self._events),
                "final_assistant_response": self._final_response,
            }
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.output_path.with_suffix(".tmp")
            serialized = json.dumps(document, indent=2, ensure_ascii=False)
            # Last-line defense: redact configured secrets without logging or exposing them.
            for env_name, env_value in os.environ.items():
                if (
                    len(env_value) >= 8
                    and any(term in env_name.upper() for term in ("KEY", "TOKEN", "SECRET", "CREDENTIAL", "GOOGLE_SHEET_ID"))
                ):
                    serialized = serialized.replace(env_value, "<redacted-configured-secret>")
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(self.output_path)
            self._written = True


class TraceMetricsObserver(BaseObserver):
    def __init__(self, recorder: TraceRecorder):
        super().__init__()
        self.recorder = recorder
        self._metric_frames = set()
        self._llm_frames = set()
        self._bot_stopped_frames = set()

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if isinstance(frame, MetricsFrame) and frame.id not in self._metric_frames:
            self._metric_frames.add(frame.id)
            for metric in frame.data:
                self.recorder.add_metric(metric)
        elif isinstance(frame, LLMContextFrame) and frame.id not in self._llm_frames:
            destination_name = data.destination.__class__.__name__
            if destination_name == "GroqLLMService":
                self._llm_frames.add(frame.id)
                self.recorder.note_llm_call()
        elif isinstance(frame, BotStoppedSpeakingFrame) and frame.id not in self._bot_stopped_frames:
            self._bot_stopped_frames.add(frame.id)
            if self.recorder._pending_finalize:
                self.recorder.write()


trace_recorder = TraceRecorder()
