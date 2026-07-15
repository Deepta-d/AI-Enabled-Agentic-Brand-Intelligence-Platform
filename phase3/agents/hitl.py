"""HITL helpers: start/resume LangGraph runs for Streamlit."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import uuid
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from phase3.agents.formatters import (
    STATUS_CANNOT_SEND_WITHOUT_APPROVE,
    STATUS_DRAFT_READY,
    assemble_final_answer,
    format_email_draft,
    format_whatsapp_draft,
)
from phase3.agents.graph import (
    _format_answer_from_state,
    _initial_state,
    build_graph,
    wants_immediate_send,
)
from phase3.agents.llm import LLMOutageError, outage_plain_text

logger = logging.getLogger(__name__)

# Process-wide checkpointer so Streamlit can resume by thread_id
_CHECKPOINTER = MemorySaver()
_APP_CACHE: dict[str, Any] = {}


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 24}


def _run_async(coro, *, timeout: float = 180.0):
    """Run a coroutine on the persistent MCP loop when available, else a worker thread."""
    # Prefer the long-lived MCP runtime loop so stdio sessions stay valid.
    try:
        from phase3.agents.mcp_runtime import get_mcp_runtime

        runtime = get_mcp_runtime()
        return runtime.run(coro, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MCP runtime unavailable (%s); falling back to thread loop", exc)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result(timeout=timeout)


async def _ensure_app(*, use_mcp: bool = True):
    key = "mcp" if use_mcp else "local"
    if key in _APP_CACHE:
        return _APP_CACHE[key]
    if use_mcp:
        from phase3.agents.mcp_runtime import get_mcp_runtime

        runtime = get_mcp_runtime()
        # Tools must stay bound to the runtime session/loop.
        buckets = runtime.buckets
        app = build_graph(tool_buckets=buckets, hitl=True, checkpointer=_CHECKPOINTER)
    else:
        app = build_graph(tool_buckets=None, hitl=True, checkpointer=_CHECKPOINTER)
    _APP_CACHE[key] = app
    return app


def _interrupt_payload(app, config: dict[str, Any]) -> dict[str, Any] | None:
    state = app.get_state(config)
    for task in getattr(state, "tasks", ()) or ():
        interrupts = getattr(task, "interrupts", ()) or ()
        for item in interrupts:
            value = getattr(item, "value", item)
            if isinstance(value, dict):
                return value
            return {"raw": value, "message": STATUS_DRAFT_READY}
    interrupts = getattr(state, "interrupts", None)
    if interrupts:
        first = interrupts[0]
        value = getattr(first, "value", first)
        if isinstance(value, dict):
            return value
    return None


def _plain_from_messages(result: dict[str, Any]) -> str:
    return _format_answer_from_state(result)


async def start_run_async(
    query: str,
    *,
    thread_id: str | None = None,
    use_mcp: bool = True,
) -> dict[str, Any]:
    """Start a HITL graph run. Returns status completed|interrupted|error."""
    tid = thread_id or str(uuid.uuid4())
    config = _config(tid)
    try:
        app = await _ensure_app(use_mcp=use_mcp)
        result = await app.ainvoke(_initial_state(query), config)
        payload = _interrupt_payload(app, config)
        if payload is not None:
            email = payload.get("pending_email") or {}
            whatsapp = payload.get("pending_whatsapp") or {}
            sections = []
            if wants_immediate_send(query):
                sections.append(STATUS_CANNOT_SEND_WITHOUT_APPROVE)
            sections.extend(
                [
                    format_email_draft(email) if email else "",
                    format_whatsapp_draft(whatsapp) if whatsapp else "",
                ]
            )
            preview = assemble_final_answer(
                sections=sections,
                status_line=STATUS_DRAFT_READY,
            )
            return {
                "ok": True,
                "status": "interrupted",
                "thread_id": tid,
                "payload": payload,
                "answer": preview,
            }
        return {
            "ok": True,
            "status": "completed",
            "thread_id": tid,
            "payload": None,
            "answer": _plain_from_messages(result if isinstance(result, dict) else {}),
            "result": result,
        }
    except LLMOutageError as exc:
        return {
            "ok": False,
            "status": "error",
            "thread_id": tid,
            "answer": outage_plain_text(exc),
        }
    except Exception as exc:  # noqa: BLE001
        try:
            app = _APP_CACHE.get("mcp" if use_mcp else "local") or _APP_CACHE.get("mcp") or _APP_CACHE.get("local")
            if app is not None:
                payload = _interrupt_payload(app, config)
                if payload is not None:
                    email = payload.get("pending_email") or {}
                    whatsapp = payload.get("pending_whatsapp") or {}
                    sections = []
                    if wants_immediate_send(query):
                        sections.append(STATUS_CANNOT_SEND_WITHOUT_APPROVE)
                    sections.extend(
                        [
                            format_email_draft(email) if email else "",
                            format_whatsapp_draft(whatsapp) if whatsapp else "",
                        ]
                    )
                    preview = assemble_final_answer(
                        sections=sections,
                        status_line=STATUS_DRAFT_READY,
                    )
                    return {
                        "ok": True,
                        "status": "interrupted",
                        "thread_id": tid,
                        "payload": payload,
                        "answer": preview,
                    }
        except Exception:  # noqa: BLE001
            pass
        logger.exception("start_run failed")
        return {
            "ok": False,
            "status": "error",
            "thread_id": tid,
            "answer": f"The assistant could not complete this request: {exc}",
        }


async def resume_run_async(
    thread_id: str,
    *,
    decision: str,
    email: dict[str, Any] | None = None,
    whatsapp: dict[str, Any] | None = None,
    use_mcp: bool = True,
) -> dict[str, Any]:
    """Resume after human Approve/Reject."""
    config = _config(thread_id)
    resume_payload = {
        "decision": decision,
        "email": email or {},
        "whatsapp": whatsapp or {},
    }
    try:
        app = await _ensure_app(use_mcp=use_mcp)
        result = await app.ainvoke(Command(resume=resume_payload), config)
        payload = _interrupt_payload(app, config)
        if payload is not None:
            return {
                "ok": True,
                "status": "interrupted",
                "thread_id": thread_id,
                "payload": payload,
                "answer": STATUS_DRAFT_READY,
            }
        return {
            "ok": True,
            "status": "completed",
            "thread_id": thread_id,
            "answer": _plain_from_messages(result if isinstance(result, dict) else {}),
            "result": result,
        }
    except LLMOutageError as exc:
        return {
            "ok": False,
            "status": "error",
            "thread_id": thread_id,
            "answer": outage_plain_text(exc),
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("resume_run failed")
        return {
            "ok": False,
            "status": "error",
            "thread_id": thread_id,
            "answer": f"Could not resume human review: {exc}",
        }


def start_run(
    query: str,
    *,
    thread_id: str | None = None,
    use_mcp: bool = True,
) -> dict[str, Any]:
    """Sync entry for Streamlit. Always uses Unified MCP by default."""
    return _run_async(
        start_run_async(query, thread_id=thread_id, use_mcp=use_mcp)
    )


def resume_run(
    thread_id: str,
    *,
    decision: str,
    email: dict[str, Any] | None = None,
    whatsapp: dict[str, Any] | None = None,
    use_mcp: bool = True,
) -> dict[str, Any]:
    return _run_async(
        resume_run_async(
            thread_id,
            decision=decision,
            email=email,
            whatsapp=whatsapp,
            use_mcp=use_mcp,
        )
    )


def get_interrupt_payload(thread_id: str, *, use_mcp: bool = True) -> dict[str, Any] | None:
    app = _APP_CACHE.get("mcp" if use_mcp else "local")
    if app is None:
        return None
    return _interrupt_payload(app, _config(thread_id))


def new_thread_id() -> str:
    return str(uuid.uuid4())
