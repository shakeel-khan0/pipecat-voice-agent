from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

from services.google_calendar import (
    get_available_slots,
    create_calendar_appointment,
    is_slot_available,
)

from services.google_sheets import (
    save_lead as save_lead_to_sheet,
)


# =========================================================
# STATE
# =========================================================

class AgentState(TypedDict):
    message: str
    intent: str

    name: str
    email: str

    meeting_date: str
    meeting_time: str

    availability_checked: bool
    booking_confirmed: bool
    lead_saved: bool

    available_slots: list[str]
    slot_conflict: bool
    error: str


# =========================================================
# NODES
# =========================================================

def initial_state(**values) -> AgentState:
    """Fresh booking state; each caller keeps its own checkpoint thread."""
    state = AgentState(
        message="", intent="", name="", email="", meeting_date="",
        meeting_time="", availability_checked=False, booking_confirmed=False,
        lead_saved=False, available_slots=[], slot_conflict=False, error="",
    )
    state.update(values)
    return state


def collect_date_node(state: AgentState):
    date = interrupt({"question": "What date would you like to meet?",
                      "fields": ["meeting_date"]})
    return {"meeting_date": date}


def detect_intent(state: AgentState):
    message = state["message"].lower()

    if (
        state.get("intent") == "booking"
        or "appointment" in message
        or "meeting" in message
    ):
        return {
            "intent": "booking"
        }

    return {
        "intent": "conversation"
    }


def booking_node(state: AgentState):
    print("\n--- BOOKING FLOW ---")
    return {}


def check_availability_node(state: AgentState):
    date = state["meeting_date"]

    print(
        f"\nChecking real calendar for: {date}"
    )

    slots = get_available_slots(date)

    print(
        f"Available slots: {slots}"
    )

    return {
        "available_slots": slots,
        "availability_checked": True,
        "meeting_time": state["meeting_time"] if state["meeting_time"] in slots else "",
    }


def collect_details_node(state: AgentState):

    details = interrupt({
        "question":
            "Please provide your name and email.",
        "fields": [key for key in ("name", "email") if not state[key]],
    })

    return {
        "name": details["name"],
        "email": details["email"],
    }


def collect_time_node(state: AgentState):

    selected_time = interrupt({
        "question":
            "Which available time would you like?",
        "fields": ["meeting_time"],
        "available_slots":
            state["available_slots"],
    })

    return {
        "meeting_time": selected_time,
        "slot_conflict": False,
    }


def book_appointment_node(state: AgentState):

    print(
        "\nRe-checking slot before booking:"
        f" {state['meeting_date']}"
        f" {state['meeting_time']}"
    )

    # Safety re-check
    available = is_slot_available(
        state["meeting_date"],
        state["meeting_time"],
    )

    if not available:
        print(
            "Selected slot is no longer available."
        )

        # Reset booking flow
        return {
            "meeting_time": "",
            "availability_checked": False,
            "available_slots": [],
            "slot_conflict": True,
        }

    print("\nCreating real Calendar event...")

    result = create_calendar_appointment(
        date=state["meeting_date"],
        time=state["meeting_time"],
        name=state["name"],
        email=state["email"],
    )

    if result.get("success"):
        print("Appointment created successfully.")

        return {
            "booking_confirmed": True
        }

    print(
        "Booking failed:",
        result.get("error")
    )

    return {
        "booking_confirmed": False,
        "meeting_time": "",
        "availability_checked": False,
        "available_slots": [],
        "slot_conflict": result.get("error") == "Selected slot is no longer available.",
        "error": (
            "" if result.get("error") == "Selected slot is no longer available."
            else result.get("error") or "Booking failed."
        ),
    }


def save_lead_node(state: AgentState):

    print("\nSaving lead to Google Sheets...")

    result = save_lead_to_sheet(
        name=state["name"],
        email=state["email"],

        business="",
        industry="",
        problem="",
        volume="",
        current_system="",
        interested_service="",

        meeting_date=state["meeting_date"],
        meeting_time=state["meeting_time"],
    )

    if result.get("success"):
        print("Lead saved successfully.")

        return {
            "lead_saved": True
        }

    print("Lead save failed.")

    return {
        "lead_saved": False,
        "error": "The appointment is booked, but saving the lead failed.",
    }


def complete_node(state: AgentState):

    print("\n============================")
    print("BOOKING WORKFLOW COMPLETE")
    print("============================")

    print(
        f"Name: {state['name']}"
    )

    print(
        f"Email: {state['email']}"
    )

    print(
        f"Meeting: "
        f"{state['meeting_date']} "
        f"{state['meeting_time']}"
    )

    return {}


def conversation_node(state: AgentState):
    print("\n--- NORMAL CONVERSATION ---")
    return {}


# =========================================================
# ROUTERS
# =========================================================

def route_intent(state: AgentState):
    return state["intent"]


def booking_router(state: AgentState):

    if state.get("error"):
        return "complete"

    if not state["meeting_date"]:
        return "collect_date"

    if not state["availability_checked"]:
        return "check_availability"

    if not state["available_slots"]:
        return "complete"

    if not state["meeting_time"] or state["meeting_time"] not in state["available_slots"]:
        return "collect_time"

    if not state["name"] or not state["email"]:
        return "collect_details"

    if not state["booking_confirmed"]:
        return "book"

    if not state["lead_saved"]:
        return "save_lead"

    return "complete"


# =========================================================
# GRAPH
# =========================================================

builder = StateGraph(AgentState)
builder.add_node("collect_date", collect_date_node)
builder.add_edge("collect_date", "booking")


# Nodes

builder.add_node(
    "detect_intent",
    detect_intent
)

builder.add_node(
    "booking",
    booking_node
)

builder.add_node(
    "check_availability",
    check_availability_node
)

builder.add_node(
    "collect_details",
    collect_details_node
)

builder.add_node(
    "collect_time",
    collect_time_node
)

builder.add_node(
    "book",
    book_appointment_node
)

builder.add_node(
    "save_lead",
    save_lead_node
)

builder.add_node(
    "complete",
    complete_node
)

builder.add_node(
    "conversation",
    conversation_node
)


# =========================================================
# EDGES
# =========================================================

builder.add_edge(
    START,
    "detect_intent"
)


builder.add_conditional_edges(
    "detect_intent",
    route_intent,
    {
        "booking": "booking",
        "conversation": "conversation",
    }
)


builder.add_conditional_edges(
    "booking",
    booking_router,
    {
        "collect_date":
            "collect_date",
            
        "check_availability":
            "check_availability",

        "collect_details":
            "collect_details",

        "collect_time":
            "collect_time",

        "book":
            "book",

        "save_lead":
            "save_lead",

        "complete":
            "complete",
    }
)


# After each booking action,
# return to booking router.

builder.add_edge(
    "check_availability",
    "booking"
)

builder.add_edge(
    "collect_details",
    "booking"
)

builder.add_edge(
    "collect_time",
    "booking"
)

builder.add_edge(
    "book",
    "booking"
)

builder.add_edge(
    "save_lead",
    "booking"
)


builder.add_edge(
    "complete",
    END
)

builder.add_edge(
    "conversation",
    END
)


# =========================================================
# MEMORY
# =========================================================

memory = MemorySaver()

graph = builder.compile(
    checkpointer=memory
)


