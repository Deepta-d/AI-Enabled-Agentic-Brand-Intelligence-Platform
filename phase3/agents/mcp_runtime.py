"""Persistent Unified MCP session on a dedicated event-loop thread.

Avoids spawning a new stdio MCP process for every tool call / Streamlit rerun.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import threading
from typing import Any

from langchain_core.tools import BaseTool

from phase3.agents.mcp_client import SERVER_NAME, create_mcp_client, partition_tools

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_RUNTIME: "McpRuntime | None" = None


class McpRuntime:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="mcp-runtime", daemon=True
        )
        self._ready = threading.Event()
        self._shutdown = asyncio.Event()  # created on the loop thread
        self._client = create_mcp_client()
        self.tools: list[BaseTool] = []
        self.buckets: dict[str, list[BaseTool]] = {}
        self._error: BaseException | None = None
        self._thread.start()
        if not self._ready.wait(timeout=120):
            raise TimeoutError("Unified MCP runtime failed to start within 120s")
        if self._error is not None:
            raise RuntimeError(f"Unified MCP runtime failed: {self._error}") from self._error
        if not self.tools:
            raise RuntimeError("Unified MCP runtime started with zero tools")

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._shutdown = asyncio.Event()
        self._loop.create_task(self._session_lifetime())
        try:
            self._loop.run_forever()
        except Exception as exc:  # noqa: BLE001
            self._error = exc
            logger.exception("MCP runtime loop crashed")
            self._ready.set()

    async def _session_lifetime(self) -> None:
        from langchain_mcp_adapters.tools import load_mcp_tools

        try:
            async with self._client.session(SERVER_NAME) as session:
                self.tools = await load_mcp_tools(session, server_name=SERVER_NAME)
                self.buckets = partition_tools(self.tools)
                names = sorted(t.name for t in self.tools)
                logger.info(
                    "Persistent MCP session ready (%s tools): %s", len(names), names
                )
                self._ready.set()
                await self._shutdown.wait()
        except Exception as exc:  # noqa: BLE001
            self._error = exc
            logger.exception("MCP session lifetime failed")
            self._ready.set()
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def run(self, coro, *, timeout: float = 180.0):
        if not self._thread.is_alive():
            raise RuntimeError("MCP runtime thread is not alive")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    def close(self) -> None:
        if not self._loop.is_running():
            return
        self._loop.call_soon_threadsafe(self._shutdown.set)
        self._thread.join(timeout=15)


def get_mcp_runtime() -> McpRuntime:
    global _RUNTIME
    with _LOCK:
        if _RUNTIME is None:
            _RUNTIME = McpRuntime()
            atexit.register(_shutdown_runtime)
        return _RUNTIME


def _shutdown_runtime() -> None:
    global _RUNTIME
    rt = _RUNTIME
    _RUNTIME = None
    if rt is not None:
        try:
            rt.close()
        except Exception:  # noqa: BLE001
            logger.debug("MCP runtime atexit close ignored", exc_info=True)
