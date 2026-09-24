"""Stable caller identity for the current local-terminal transport."""

from __future__ import annotations

import os


DEFAULT_TEST_CALLER_ID = "terminal_test_001"


def resolve_caller_id() -> str:
    """Return the configured terminal caller ID, or a stable safe default."""
    caller_id = os.environ.get("TEST_CALLER_ID", DEFAULT_TEST_CALLER_ID).strip()
    return caller_id or DEFAULT_TEST_CALLER_ID
