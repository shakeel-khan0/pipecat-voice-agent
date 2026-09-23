from datetime import datetime, timedelta
from time import perf_counter
from zoneinfo import ZoneInfo

from services.google_auth import get_calendar_service
from trace_recorder import trace_recorder


TIMEZONE = "Asia/Karachi"

BUSINESS_START_HOUR = 10
BUSINESS_END_HOUR = 18

APPOINTMENT_DURATION_MINUTES = 60
SLOT_STEP_MINUTES = 30


def _parse_start(date: str, time: str):
    tz = ZoneInfo(TIMEZONE)

    return datetime.strptime(
        f"{date} {time}",
        "%Y-%m-%d %I:%M %p"
    ).replace(tzinfo=tz)


def _get_busy_times(date: str):
    started = perf_counter()
    try:
        service = get_calendar_service()
    except Exception as exc:
        trace_recorder.record(
            "external_operation",
            service="google_calendar",
            operation="events.list",
            duration_ms=round((perf_counter() - started) * 1000, 3),
            success=False,
            error_type=type(exc).__name__,
            application_retries=0,
        )
        raise
    tz = ZoneInfo(TIMEZONE)

    target_date = datetime.strptime(
        date,
        "%Y-%m-%d"
    ).date()

    day_start = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        BUSINESS_START_HOUR,
        0,
        tzinfo=tz,
    )

    day_end = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        BUSINESS_END_HOUR,
        0,
        tzinfo=tz,
    )

    try:
        events = service.events().list(
            calendarId="primary",
            timeMin=day_start.isoformat(),
            timeMax=day_end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        ).execute()
    except Exception as exc:
        trace_recorder.record(
            "external_operation",
            service="google_calendar",
            operation="events.list",
            duration_ms=round((perf_counter() - started) * 1000, 3),
            success=False,
            error_type=type(exc).__name__,
            application_retries=0,
        )
        raise

    busy_times = []

    for event in events.get("items", []):
        start = event["start"].get("dateTime")
        end = event["end"].get("dateTime")

        if start and end:
            busy_times.append(
                (
                    datetime.fromisoformat(start),
                    datetime.fromisoformat(end),
                )
            )

    trace_recorder.record(
        "external_operation",
        service="google_calendar",
        operation="events.list",
        duration_ms=round((perf_counter() - started) * 1000, 3),
        success=True,
        returned_event_count=len(events.get("items", [])),
        busy_interval_count=len(busy_times),
        application_retries=0,
    )
    return busy_times


def is_slot_available(
    date: str,
    time: str,
):
    started = perf_counter()
    tz = ZoneInfo(TIMEZONE)

    start_time = _parse_start(date, time)

    end_time = start_time + timedelta(
        minutes=APPOINTMENT_DURATION_MINUTES
    )

    day_start = start_time.replace(
        hour=BUSINESS_START_HOUR,
        minute=0,
        second=0,
        microsecond=0,
    )

    day_end = start_time.replace(
        hour=BUSINESS_END_HOUR,
        minute=0,
        second=0,
        microsecond=0,
    )

    # Appointment must stay inside business hours
    if start_time < day_start or end_time > day_end:
        trace_recorder.record(
            "calendar_slot_check",
            duration_ms=round((perf_counter() - started) * 1000, 3),
            available=False,
            reason="outside_business_hours",
        )
        return False

    busy_times = _get_busy_times(date)

    is_busy = any(
        start_time < busy_end
        and end_time > busy_start
        for busy_start, busy_end in busy_times
    )

    available = not is_busy
    trace_recorder.record(
        "calendar_slot_check",
        duration_ms=round((perf_counter() - started) * 1000, 3),
        available=available,
        reason=None if available else "busy",
    )
    return available


def get_available_slots(date: str):
    started = perf_counter()
    tz = ZoneInfo(TIMEZONE)

    target_date = datetime.strptime(
        date,
        "%Y-%m-%d"
    ).date()

    day_start = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        BUSINESS_START_HOUR,
        0,
        tzinfo=tz,
    )

    day_end = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        BUSINESS_END_HOUR,
        0,
        tzinfo=tz,
    )

    busy_times = _get_busy_times(date)

    available_slots = []

    slot = day_start

    while (
        slot + timedelta(
            minutes=APPOINTMENT_DURATION_MINUTES
        )
        <= day_end
    ):
        slot_end = slot + timedelta(
            minutes=APPOINTMENT_DURATION_MINUTES
        )

        is_busy = any(
            slot < busy_end
            and slot_end > busy_start
            for busy_start, busy_end in busy_times
        )

        if not is_busy:
            available_slots.append(
                slot.strftime("%I:%M %p")
            )

        slot += timedelta(
            minutes=SLOT_STEP_MINUTES
        )

    trace_recorder.record(
        "calendar_availability",
        duration_ms=round((perf_counter() - started) * 1000, 3),
        available_slot_count=len(available_slots),
        success=True,
    )
    return available_slots


def create_calendar_appointment(
    date: str,
    time: str,
    name: str,
    email: str,
):
    started = perf_counter()
    try:
        service = get_calendar_service()
    except Exception as exc:
        trace_recorder.record(
            "external_operation",
            service="google_calendar",
            operation="events.insert",
            duration_ms=round((perf_counter() - started) * 1000, 3),
            success=False,
            error_type=type(exc).__name__,
            application_retries=0,
        )
        raise

    # Re-check immediately before booking
    if not is_slot_available(date, time):
        trace_recorder.record(
            "external_operation",
            service="google_calendar",
            operation="events.insert",
            duration_ms=round((perf_counter() - started) * 1000, 3),
            success=False,
            error_type="SlotUnavailable",
            application_retries=0,
        )
        return {
            "success": False,
            "error": "Selected slot is no longer available.",
        }

    start_time = _parse_start(date, time)

    end_time = start_time + timedelta(
        minutes=APPOINTMENT_DURATION_MINUTES
    )

    event = {
        "summary": f"Agentix Labs AI Meeting - {name}",
        "description": (
            "Appointment booked by Agentix Labs AI voice agent.\n"
            f"Name: {name}\n"
            f"Email: {email}"
        ),
        "start": {
            "dateTime": start_time.isoformat(),
            "timeZone": TIMEZONE,
        },
        "end": {
            "dateTime": end_time.isoformat(),
            "timeZone": TIMEZONE,
        },
    }

    try:
        created_event = service.events().insert(
            calendarId="primary",
            body=event,
        ).execute()
    except Exception as exc:
        trace_recorder.record(
            "external_operation",
            service="google_calendar",
            operation="events.insert",
            duration_ms=round((perf_counter() - started) * 1000, 3),
            success=False,
            error_type=type(exc).__name__,
            application_retries=0,
        )
        raise

    trace_recorder.record(
        "external_operation",
        service="google_calendar",
        operation="events.insert",
        duration_ms=round((perf_counter() - started) * 1000, 3),
        success=True,
        event_id="<redacted-event_id>",
        application_retries=0,
    )

    return {
        "success": True,
        "event_id": created_event["id"],
        "date": date,
        "time": time,
        "name": name,
        "email": email,
    }
