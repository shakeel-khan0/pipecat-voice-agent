"""Translate conversational tool arguments into one caller's graph execution."""

import asyncio
import re
from datetime import datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

from langgraph.types import Command

from agent.graph import graph, initial_state
from trace_recorder import trace_recorder


_TIME_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


class BookingSession:
    def __init__(self, workflow=graph, thread_id=None):
        self.graph = workflow
        self.config = {"configurable": {"thread_id": thread_id or str(uuid4())}}
        self._lock = asyncio.Lock()
        self._details = {}
        self._result = None
        self._failure = None
        self._tasks = set()
        self._time_preference = ""
        self._requested_time = ""
        self._slot_was_taken = False
        self._last_offered_slots = []

    async def advance(self, **details):
        # Keep an in-flight write alive even if its voice/tool response is interrupted.
        task = asyncio.create_task(self._advance(details))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return await asyncio.shield(task)

    async def _advance(self, details):
        async with self._lock:
            if self._failure:
                return self._failure
            if self._result is not None and (
                self._result.get("booking_confirmed") or self._result.get("error")
            ):
                return self._response()

            preference = str(details.pop("time_preference", "") or "").strip()
            request_alternatives = bool(details.pop("request_alternatives", False))
            if preference:
                if preference.lower() != self._time_preference.lower():
                    self._last_offered_slots = []
                self._time_preference = preference
            supplied = {
                key: str(value).strip()
                for key, value in details.items()
                if value is not None and str(value).strip()
            }
            explicit_time = "meeting_time" in supplied
            date = supplied.get("meeting_date")
            if date:
                try:
                    parsed = datetime.strptime(date, "%Y-%m-%d").date()
                    if parsed < datetime.now(ZoneInfo("Asia/Karachi")).date():
                        raise ValueError("past date")
                except ValueError:
                    return self._public("Please give me a valid date today or later.")
            if "email" in supplied and not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", supplied["email"]):
                return self._public("Could you spell your email address for me?")
            if "meeting_time" in supplied:
                try:
                    supplied["meeting_time"] = datetime.strptime(
                        supplied["meeting_time"].upper(), "%I:%M %p"
                    ).strftime("%I:%M %p")
                    self._requested_time = supplied["meeting_time"]
                except ValueError:
                    return self._public("What time would you like, including AM or PM?")
            elif preference:
                self._requested_time = self._specific_time_from_preference(preference) or ""

            date_changed = bool(
                date and self._details.get("meeting_date") not in (None, date)
            )
            known_slots = [] if date_changed else (self._result or {}).get("available_slots", [])
            if preference and not explicit_time:
                selected_from_preference = self._slot_from_preference(known_slots)
                if selected_from_preference:
                    supplied["meeting_time"] = selected_from_preference
            requested_unavailable = False
            if supplied.get("meeting_time") and known_slots:
                if supplied["meeting_time"] not in known_slots:
                    self._details.pop("meeting_time", None)
                    supplied.pop("meeting_time")
                    requested_unavailable = True

            try:
                previously_selected = bool(self._details.get("meeting_time"))
                if date and self._details.get("meeting_date") not in (None, date):
                    # Abandon only an unbooked attempt, keeping the caller's thread.
                    if self._result is not None:
                        await self.graph.aupdate_state(self.config, {}, as_node="complete")
                    self._result = None
                    self._details.pop("meeting_time", None)
                    self._last_offered_slots = []
                    self._time_preference = preference
                    self._requested_time = (
                        supplied.get("meeting_time")
                        or self._specific_time_from_preference(preference)
                        or ""
                    )
                    previously_selected = False
                self._details.update(supplied)
                if requested_unavailable:
                    return self._response(["meeting_time"])
                if self._result is None:
                    value = initial_state(message="Book an appointment", intent="booking", **self._details)
                elif not self._result.get("__interrupt__"):
                    return self._response()
                else:
                    request = self._result["__interrupt__"][0].value
                    fields = request["fields"]
                    missing = [key for key in fields if not self._details.get(key)]
                    if missing:
                        return self._response(
                            missing,
                            request_alternatives=request_alternatives,
                        )
                    if fields == ["meeting_time"]:
                        selected = self._details["meeting_time"]
                        if selected not in request["available_slots"]:
                            self._details.pop("meeting_time", None)
                            self._slot_was_taken = True
                            return self._response(["meeting_time"])
                        value = Command(resume=selected, update=self._details)
                    elif fields == ["meeting_date"]:
                        value = Command(resume=self._details["meeting_date"], update=self._details)
                    else:
                        value = Command(resume={key: self._details[key] for key in ("name", "email")}, update=self._details)

                self._result = await self.graph.ainvoke(value, config=self.config)
                selected_from_preference = (
                    self._slot_from_preference(self._result.get("available_slots", []))
                    if preference and not explicit_time
                    else None
                )
                if selected_from_preference and not self._details.get("meeting_time"):
                    self._details["meeting_time"] = selected_from_preference
                    interrupts = self._result.get("__interrupt__", [])
                    if interrupts and interrupts[0].value["fields"] == ["meeting_time"]:
                        self._result = await self.graph.ainvoke(
                            Command(
                                resume=selected_from_preference,
                                update=self._details,
                            ),
                            config=self.config,
                        )
                # A stale slot is removed by the graph; never automatically reuse it.
                if not self._result.get("meeting_time"):
                    if previously_selected or self._result.get("slot_conflict"):
                        self._slot_was_taken = True
                    self._details.pop("meeting_time", None)
                return self._response(request_alternatives=request_alternatives)
            except Exception as exc:
                trace_recorder.record(
                    "error",
                    component="booking_session",
                    error_type=type(exc).__name__,
                    application_retries=0,
                )
                # Do not replay a write whose remote outcome might be unknown.
                snapshot = await self.graph.aget_state(self.config)
                self._failure = {
                    "status": "error",
                    "booking_confirmed": snapshot.values.get("booking_confirmed", False),
                    "lead_saved": snapshot.values.get("lead_saved", False),
                    "spoken_response": "Sorry, I couldn't finish that booking. Please check the appointment details before trying again.",
                }
                return self._failure

    def _response(self, missing=None, request_alternatives=False):
        state = self._result
        response = {key: state.get(key) for key in ("booking_confirmed", "lead_saved")}
        if state.get("__interrupt__"):
            request = state["__interrupt__"][0].value
            fields = missing or [key for key in request["fields"] if not self._details.get(key)] or request["fields"]
            slots = request.get("available_slots") or state.get("available_slots", [])
            if slots and not self._details.get("meeting_time"):
                candidate_slots = (
                    [slot for slot in slots if slot not in self._last_offered_slots]
                    if request_alternatives else slots
                )
                offered = self._choose_slots(candidate_slots)
                if not self._time_preference and not self._requested_time and len(slots) > 3:
                    spoken = "What works better for you: morning, afternoon, or evening?"
                    offered = []
                elif offered:
                    if self._slot_was_taken:
                        prefix = "That slot was just taken. "
                    elif self._requested_time:
                        prefix = f"{self._requested_time} isn't available. "
                    else:
                        prefix = ""
                    spoken = prefix + f"I can offer {self._say_slots(offered)}. Which works best?"
                    self._slot_was_taken = False
                else:
                    alternatives = self._nearest_slots(candidate_slots)
                    spoken = (
                        "I don't have any other availability in that period."
                        if request_alternatives
                        else "I don't have availability in that period."
                    )
                    if alternatives:
                        spoken += f" I can offer {self._say_slots(alternatives)}. Which works for you?"
                    else:
                        spoken += " Would you like to try another time or date?"
                    offered = alternatives
                if offered:
                    self._last_offered_slots = offered
                response.update(status="needs_input", spoken_response=spoken,
                                offered_slots=offered)
            elif "meeting_date" in fields:
                response.update(status="needs_input", spoken_response="Sure. Which date would you like to meet?")
            elif "name" in fields and "email" in fields:
                response.update(status="needs_input", spoken_response="Great. Could I get your name and email to finalize the booking?")
            elif "name" in fields:
                response.update(status="needs_input", spoken_response="Great. What name should I use for the booking?")
            elif "email" in fields:
                response.update(status="needs_input", spoken_response="Thanks. What's the best email address for the booking?")
            else:
                response.update(status="needs_input", spoken_response="What time would work best for you?")
        elif state.get("error"):
            if state.get("booking_confirmed"):
                spoken = "Your meeting is confirmed, but I couldn't finish saving the contact details."
            else:
                spoken = "Sorry, I couldn't complete that booking. Please try again in a moment."
            response.update(status="error", spoken_response=spoken)
        elif state.get("booking_confirmed"):
            date_text = self._say_date(state["meeting_date"])
            response.update(
                status="completed",
                spoken_response=f"You're all set. Your meeting is confirmed for {date_text} at {state['meeting_time']}.",
            )
        else:
            response.update(status="unavailable", spoken_response="I don't have availability on that date. Would you like to try another date?")
        return response

    @staticmethod
    def _public(spoken_response):
        return {"status": "needs_input", "spoken_response": spoken_response}

    def _choose_slots(self, slots):
        if self._requested_time:
            target = self._minutes(self._requested_time)
            return [item[0] for item in sorted(
                ((slot, abs(self._minutes(slot) - target)) for slot in slots),
                key=lambda item: item[1],
            )[:3]]
        if not self._time_preference:
            return slots[:3]
        parsed = [(slot, datetime.strptime(slot, "%I:%M %p")) for slot in slots]
        preference = self._time_preference.lower()
        expression = self._time_expression(preference)
        if expression:
            kind, target = expression
            values = [(slot, value.hour * 60 + value.minute) for slot, value in parsed]
            if kind == "after":
                chosen = [slot for slot, value in values if value >= target]
            elif kind == "before":
                chosen = [slot for slot, value in values if value < target]
            else:
                chosen = [slot for slot, _ in sorted(values, key=lambda item: abs(item[1] - target))]
        elif "morning" in preference or "before lunch" in preference:
            chosen = [slot for slot, value in parsed if 10 <= value.hour < 12]
        elif "afternoon" in preference:
            chosen = [slot for slot, value in parsed if 12 <= value.hour < 17]
        elif "evening" in preference:
            chosen = [slot for slot, value in parsed if value.hour >= 17]
        else:
            return slots[:3]
        return chosen[:3]

    def _nearest_slots(self, slots):
        preference = self._time_preference.lower()
        anchors = {"morning": 11 * 60, "afternoon": 14 * 60, "evening": 17 * 60,
                   "before lunch": 11 * 60}
        target = next((value for key, value in anchors.items() if key in preference), None)
        expression = self._time_expression(preference)
        if expression:
            _, target = expression
        if target is None:
            return slots[:3]
        return [item[0] for item in sorted(
            ((slot, abs(datetime.strptime(slot, "%I:%M %p").hour * 60
                        + datetime.strptime(slot, "%I:%M %p").minute - target)) for slot in slots),
            key=lambda item: item[1],
        )[:3]]

    def _slot_from_preference(self, slots):
        if not self._time_preference or not slots:
            return None
        expression = self._time_expression(self._time_preference)
        if not expression or expression[0] in ("after", "before"):
            return None
        target = expression[1]
        candidate = datetime.strptime("00:00", "%H:%M").replace(
            hour=target // 60,
            minute=target % 60,
        ).strftime("%I:%M %p")
        return candidate if candidate in slots else None

    @classmethod
    def _specific_time_from_preference(cls, preference):
        expression = cls._time_expression(preference)
        if not expression or expression[0] in ("after", "before"):
            return None
        target = expression[1]
        return datetime.strptime("00:00", "%H:%M").replace(
            hour=target // 60,
            minute=target % 60,
        ).strftime("%I:%M %p")

    @staticmethod
    def _minutes(value):
        parsed = datetime.strptime(value, "%I:%M %p")
        return parsed.hour * 60 + parsed.minute

    @staticmethod
    def _time_expression(preference):
        normalized = preference.lower()
        for word, number in _TIME_WORDS.items():
            normalized = re.sub(rf"\b{word}\b", str(number), normalized)
        match = re.search(
            r"\b(?:(after|before|around|about|near|at)\s+)?"
            r"(\d{1,2})(?::(\d{2}))?\s*(a\.?\s*m\.?|p\.?\s*m\.?)?\b",
            normalized,
        )
        if not match:
            return None
        kind = match.group(1)
        hour, minute = int(match.group(2)), int(match.group(3) or 0)
        meridiem = (match.group(4) or "").replace(".", "").replace(" ", "")
        if not 1 <= hour <= 12 or not 0 <= minute <= 59:
            return None
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        elif not meridiem:
            if "afternoon" in normalized or "evening" in normalized or hour < 8:
                if hour < 12:
                    hour += 12
        return kind, hour * 60 + minute

    @staticmethod
    def _say_slots(slots):
        if len(slots) == 1:
            return slots[0]
        return ", ".join(slots[:-1]) + ", or " + slots[-1]

    @staticmethod
    def _say_date(value):
        date = datetime.strptime(value, "%Y-%m-%d").date()
        today = datetime.now(ZoneInfo("Asia/Karachi")).date()
        if date == today:
            return "today"
        if date == today + timedelta(days=1):
            return "tomorrow"
        return date.strftime("%A, %B %d").replace(" 0", " ")
