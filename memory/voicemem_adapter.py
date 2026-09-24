"""Async fail-open client for the localhost VoiceMem sidecar."""

from __future__ import annotations

import logging
import os
from time import perf_counter
from typing import Any

import httpx

logger = logging.getLogger(__name__)
DEFAULT_SIDECAR_URL = "http://127.0.0.1:8765"
DEFAULT_TIMEOUT_SECONDS = 0.75
DEFAULT_WRITE_TIMEOUT_SECONDS = 15.0


class VoiceMemAdapter:
    """Reusable localhost HTTP boundary; never imports VoiceMem."""

    def __init__(self, *, base_url: str | None = None,
                 timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
                 write_timeout_seconds: float = DEFAULT_WRITE_TIMEOUT_SECONDS,
                 client: httpx.AsyncClient | None = None) -> None:
        self.base_url = (base_url or os.environ.get(
            "VOICEMEM_SIDECAR_URL", DEFAULT_SIDECAR_URL)).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.write_timeout_seconds = write_timeout_seconds
        self._client = client
        self._owns_client = client is None
        self._enabled = False
        self._last_error: str | None = None

    async def initialize(self) -> bool:
        try:
            response = await self._get_client().get("/health")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or payload.get("alive") is not True:
                raise ValueError("invalid health response")
            if payload.get("voicemem_available") is not True:
                raise RuntimeError("VoiceMem unavailable")
            self._enabled = True
            self._last_error = None
            return True
        except Exception as exc:
            self._disable("initialization", exc)
            return False

    async def retrieve(self, caller_id: str, query: str, *, top_k: int = 5) -> dict[str, Any]:
        empty = {"memories": [], "latency_ms": None}
        if not self._enabled or not caller_id.strip() or not query.strip():
            return empty
        started = perf_counter()
        try:
            response = await self._get_client().post(
                "/memory/search",
                json={"caller_id": caller_id, "query": query, "top_k": top_k},
            )
            response.raise_for_status()
            payload = response.json()
            memories = payload.get("memories") if isinstance(payload, dict) else None
            count = payload.get("count") if isinstance(payload, dict) else None
            if (not isinstance(payload, dict) or payload.get("ok") is not True
                    or not isinstance(memories, list)
                    or not all(isinstance(item, str) for item in memories)
                    or count != len(memories)):
                raise ValueError("invalid search response")
            return {
                "memories": memories,
                "latency_ms": round((perf_counter() - started) * 1000, 3),
            }
        except Exception as exc:
            self._disable("retrieval", exc)
            return empty

    async def ingest(self, caller_id: str, content: str) -> bool | None:
        """Manual API only; the Phase 2 live pipeline never calls this."""
        if not self._enabled or not caller_id.strip() or not content.strip():
            return False
        try:
            response = await self._get_client().post(
                "/memory/ingest",
                json={"caller_id": caller_id, "content": content},
                timeout=httpx.Timeout(
                    self.write_timeout_seconds,
                    connect=min(0.2, self.write_timeout_seconds),
                ),
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("invalid ingest response")
            if payload.get("ok") is not True:
                self._record_operation_failure(
                    "ingestion",
                    str(payload.get("error_type") or "VoiceMemIngestError"),
                )
                return False
            persisted = payload.get("persisted")
            if isinstance(persisted, bool):
                return True if persisted else None

            # Sidecars started before the Phase 3 response extension confirm a
            # successful write with memory_count only. Accept that response so
            # a newly seen caller does not require a sidecar restart.
            memory_count = payload.get("memory_count")
            if (isinstance(memory_count, int) and not isinstance(memory_count, bool)
                    and memory_count >= 0):
                return True if memory_count > 0 else None
            raise ValueError("invalid ingest response")
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            self._disable("ingestion", exc)
            return False
        except Exception as exc:
            # A malformed or caller-specific ingest result must not take memory
            # away from every other caller while the sidecar remains reachable.
            self._record_operation_failure("ingestion", type(exc).__name__)
            return False

    def health(self) -> dict[str, Any]:
        return {"enabled": self._enabled, "error": self._last_error}

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None
        self._enabled = False

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = httpx.Timeout(
                self.timeout_seconds, connect=min(0.2, self.timeout_seconds))
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=timeout,
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
                trust_env=False,
            )
        return self._client

    def _disable(self, operation: str, exc: Exception) -> None:
        self._enabled = False
        self._last_error = f"{operation} failed ({type(exc).__name__})"
        logger.warning(
            "VoiceMem %s; memory disabled: %s", operation, type(exc).__name__)

    def _record_operation_failure(self, operation: str, error_type: str) -> None:
        self._last_error = f"{operation} failed ({error_type})"
        logger.warning("VoiceMem %s failed: %s", operation, error_type)
