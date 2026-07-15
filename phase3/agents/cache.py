"""In-process TTL cache for MCP / tool results."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

_REFRESH_RE = re.compile(
    r"\b(refresh|live|force\s+update|force\s+lookup|bypass\s+cache)\b",
    re.IGNORECASE,
)


class ToolCache:
    """Simple TTL cache keyed by tool name + JSON-stable kwargs."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl_seconds = max(0, int(ttl_seconds))
        self._store: dict[str, tuple[float, str]] = {}

    def _key(self, tool_name: str, kwargs: dict[str, Any]) -> str:
        payload = json.dumps(
            {"tool": tool_name, "kwargs": kwargs},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, tool_name: str, kwargs: dict[str, Any] | None = None) -> str | None:
        if self.ttl_seconds <= 0:
            return None
        key = self._key(tool_name, kwargs or {})
        item = self._store.get(key)
        if not item:
            return None
        expires_at, value = item
        if time.monotonic() > expires_at:
            self._store.pop(key, None)
            return None
        logger.info("Tool cache HIT %s", tool_name)
        return value

    def set(
        self,
        tool_name: str,
        value: str,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        if self.ttl_seconds <= 0:
            return
        key = self._key(tool_name, kwargs or {})
        self._store[key] = (time.monotonic() + self.ttl_seconds, value)
        logger.info("Tool cache SET %s (ttl=%ss)", tool_name, self.ttl_seconds)

    def clear(self) -> None:
        self._store.clear()
        logger.info("Tool cache cleared")


def user_requests_refresh(text: str) -> bool:
    return bool(_REFRESH_RE.search(text or ""))


# Process-wide default cache (Streamlit / CLI share within process)
_default_cache: ToolCache | None = None


def get_tool_cache(ttl_seconds: int | None = None) -> ToolCache:
    global _default_cache
    if _default_cache is None:
        from phase3.config import get_tool_cache_ttl

        _default_cache = ToolCache(ttl_seconds=ttl_seconds or get_tool_cache_ttl())
    elif ttl_seconds is not None and ttl_seconds != _default_cache.ttl_seconds:
        _default_cache.ttl_seconds = max(0, int(ttl_seconds))
    return _default_cache
