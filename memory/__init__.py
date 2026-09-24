"""Optional long-term memory infrastructure.

Phase 1 only: importing this package does not initialize VoiceMem or connect it
to the live voice pipeline.
"""

from .caller_identity import resolve_caller_id
from .voicemem_adapter import VoiceMemAdapter

__all__ = ["VoiceMemAdapter", "resolve_caller_id"]
