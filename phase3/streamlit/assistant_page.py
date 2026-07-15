"""Page: Brand Intelligence Assistant (HITL chatbot)."""

from __future__ import annotations

import re
from datetime import datetime

import streamlit as st

from phase3.agents.cache import get_tool_cache
from phase3.agents.hitl import new_thread_id, resume_run, start_run
from phase3.agents.sql_intents import extract_lookup_snippet, is_post_lookup_query
from phase3.config import (
    alerts_enabled,
    get_email_settings,
    get_gemini_api_key,
    get_gemini_fallback_model,
    get_ollama_settings,
    get_tool_cache_ttl,
    get_twilio_settings,
)


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = new_thread_id()
    if "awaiting_approval" not in st.session_state:
        st.session_state.awaiting_approval = False
    if "pending_email" not in st.session_state:
        st.session_state.pending_email = {}
    if "pending_whatsapp" not in st.session_state:
        st.session_state.pending_whatsapp = {}
    if "findings" not in st.session_state:
        st.session_state.findings = ""
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = []
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = new_thread_id()
    if "last_classified_text" not in st.session_state:
        st.session_state.last_classified_text = ""


def _chat_title(messages: list[dict]) -> str:
    for msg in messages:
        if msg.get("role") == "user" and str(msg.get("content") or "").strip():
            text = str(msg["content"]).strip().replace("\n", " ")
            return text[:48] + ("..." if len(text) > 48 else "")
    return "New chat"


def _snapshot_active_chat() -> None:
    messages = list(st.session_state.messages or [])
    if not messages:
        return
    chat_id = st.session_state.active_chat_id
    entry = {
        "id": chat_id,
        "title": _chat_title(messages),
        "messages": messages,
        "thread_id": st.session_state.thread_id,
        "findings": st.session_state.findings,
        "awaiting_approval": st.session_state.awaiting_approval,
        "pending_email": dict(st.session_state.pending_email or {}),
        "pending_whatsapp": dict(st.session_state.pending_whatsapp or {}),
        "last_classified_text": st.session_state.last_classified_text or "",
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    sessions = list(st.session_state.chat_sessions or [])
    for i, item in enumerate(sessions):
        if item.get("id") == chat_id:
            sessions[i] = entry
            st.session_state.chat_sessions = sessions
            return
    sessions.insert(0, entry)
    st.session_state.chat_sessions = sessions[:30]


def _load_chat(chat_id: str) -> None:
    _snapshot_active_chat()
    for item in st.session_state.chat_sessions:
        if item.get("id") != chat_id:
            continue
        st.session_state.active_chat_id = chat_id
        st.session_state.messages = list(item.get("messages") or [])
        st.session_state.thread_id = item.get("thread_id") or new_thread_id()
        st.session_state.findings = item.get("findings") or ""
        st.session_state.awaiting_approval = bool(item.get("awaiting_approval"))
        st.session_state.pending_email = dict(item.get("pending_email") or {})
        st.session_state.pending_whatsapp = dict(item.get("pending_whatsapp") or {})
        st.session_state.last_classified_text = item.get("last_classified_text") or ""
        if not st.session_state.last_classified_text:
            st.session_state.last_classified_text = _infer_last_classified_text(
                st.session_state.messages
            )
        return


def _start_new_chat() -> None:
    _snapshot_active_chat()
    st.session_state.active_chat_id = new_thread_id()
    st.session_state.messages = []
    st.session_state.thread_id = new_thread_id()
    st.session_state.awaiting_approval = False
    st.session_state.pending_email = {}
    st.session_state.pending_whatsapp = {}
    st.session_state.findings = ""
    st.session_state.last_classified_text = ""


def _infer_last_classified_text(messages: list[dict]) -> str:
    """Recover post text from the latest classify turn in chat history."""
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue
        content = str(msg.get("content") or "")
        if not re.search(r"\b(predict|classify|sentiment\s+for)\b", content, re.I):
            continue
        quotes = re.findall(r"[\"']([^\"']{3,300})[\"']", content)
        if quotes:
            return max(quotes, key=len).strip()
        m = re.search(
            r"predict\s+(?:the\s+)?sentiment\s+for\s+(.+)$",
            content,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(1).strip(" \t\"'")
    return ""


def _remember_classified_text(prompt: str, answer: str) -> None:
    if "Predicted sentiment" not in (answer or ""):
        return
    quotes = re.findall(r"[\"']([^\"']{3,300})[\"']", prompt)
    if quotes:
        st.session_state.last_classified_text = max(quotes, key=len).strip()
        return
    m = re.search(
        r"predict\s+(?:the\s+)?sentiment\s+for\s+(.+)$",
        prompt,
        flags=re.IGNORECASE,
    )
    if m:
        st.session_state.last_classified_text = m.group(1).strip(" \t\"'")


def _enrich_followup_prompt(prompt: str) -> str:
    """Attach prior classified text so SQL can resolve 'the above text' look-ups."""
    if not is_post_lookup_query(prompt):
        return prompt
    if extract_lookup_snippet(prompt):
        return prompt
    prior = (st.session_state.last_classified_text or "").strip()
    if not prior:
        prior = _infer_last_classified_text(st.session_state.messages)
        if prior:
            st.session_state.last_classified_text = prior
    if not prior:
        return prompt
    return f'{prompt.strip()}\n\n[Previous classified text: "{prior}"]'


def _sidebar() -> None:
    ollama = get_ollama_settings()
    gemini_model = get_gemini_fallback_model()
    has_gemini = bool(get_gemini_api_key())
    st.sidebar.header("Brand Assistant")
    st.sidebar.write(f"**Primary:** Ollama `{ollama['model']}`")
    st.sidebar.write(f"**Fallback:** Gemini `{gemini_model}`")
    st.sidebar.write(
        f"**Gemini key:** {'set' if has_gemini else 'missing (set GOOGLE_API_KEY)'}"
    )
    st.sidebar.write("**Tools:** Unified MCP")
    st.sidebar.write(f"**ALERTS_ENABLED:** `{alerts_enabled()}`")
    st.sidebar.write(f"**Cache TTL:** {get_tool_cache_ttl()}s")

    if st.sidebar.button("New chat", use_container_width=True, type="primary"):
        _start_new_chat()
        st.rerun()

    st.sidebar.subheader("Chat history")
    sessions = list(st.session_state.chat_sessions or [])
    if not sessions:
        st.sidebar.caption("No past chats yet.")
    else:
        for item in sessions:
            chat_id = item.get("id")
            title = item.get("title") or "Chat"
            stamp = item.get("updated_at") or ""
            is_active = chat_id == st.session_state.active_chat_id
            cols = st.sidebar.columns([4, 1])
            if cols[0].button(
                title,
                key=f"hist_{chat_id}",
                use_container_width=True,
                disabled=is_active,
                help=stamp or None,
            ):
                _load_chat(chat_id)
                st.rerun()
            if cols[1].button("X", key=f"del_{chat_id}", help="Delete chat"):
                st.session_state.chat_sessions = [
                    s for s in sessions if s.get("id") != chat_id
                ]
                if chat_id == st.session_state.active_chat_id:
                    _start_new_chat()
                st.rerun()

    if st.sidebar.button("Clear tool cache"):
        get_tool_cache().clear()
        st.sidebar.success("Cache cleared.")

    with st.sidebar.expander("System role (summary)"):
        st.caption(
            "Brand Intelligence Assistant - sentiment, ML metrics, predictions, "
            "and Email/WhatsApp drafts. Sends only after Approve when alerts are enabled."
        )


def _alerts_banner() -> None:
    if not alerts_enabled():
        st.info(
            "Alerts are disabled (`ALERTS_ENABLED=false`). "
            "Approve will log drafts only - no real send."
        )
        return
    email = get_email_settings()
    twilio = get_twilio_settings()
    missing = []
    if not email.get("user") or not email.get("password") or not email.get("default_to"):
        missing.append("SMTP / ALERT_EMAIL_TO")
    if not twilio.get("account_sid") or not twilio.get("auth_token"):
        missing.append("Twilio (optional for WhatsApp)")
    if missing:
        st.warning(
            "ALERTS_ENABLED is true, but credentials may be incomplete: "
            + ", ".join(missing)
        )


def _render_hitl_panel() -> None:
    st.subheader("Human review")
    st.caption("Edit the drafts if needed, then Approve or Reject.")
    email = dict(st.session_state.pending_email or {})
    whatsapp = dict(st.session_state.pending_whatsapp or {})

    with st.form("hitl_form"):
        st.markdown("**Email**")
        email_to = st.text_input("To", value=str(email.get("to") or ""))
        email_subject = st.text_input(
            "Subject", value=str(email.get("subject") or "Brand Sentiment Alert")
        )
        email_body = st.text_area("Body", value=str(email.get("body") or ""), height=160)

        st.markdown("**WhatsApp**")
        wa_to = st.text_input("WhatsApp To", value=str(whatsapp.get("to") or ""))
        wa_body = st.text_area(
            "WhatsApp Body", value=str(whatsapp.get("body") or ""), height=120
        )

        col1, col2 = st.columns(2)
        approve = col1.form_submit_button("Approve", type="primary")
        reject = col2.form_submit_button("Reject")

    if approve or reject:
        decision = "approve" if approve else "reject"
        with st.spinner("Resuming supervisor…"):
            result = resume_run(
                st.session_state.thread_id,
                decision=decision,
                email={
                    "to": email_to,
                    "subject": email_subject,
                    "body": email_body,
                },
                whatsapp={"to": wa_to, "body": wa_body},
                use_mcp=True,
            )
        answer = result.get("answer") or ""
        st.session_state.messages.append({"role": "assistant", "content": answer})
        if result.get("status") == "interrupted":
            payload = result.get("payload") or {}
            st.session_state.awaiting_approval = True
            st.session_state.pending_email = payload.get("pending_email") or {}
            st.session_state.pending_whatsapp = payload.get("pending_whatsapp") or {}
        else:
            st.session_state.awaiting_approval = False
            st.session_state.pending_email = {}
            st.session_state.pending_whatsapp = {}
        _snapshot_active_chat()
        st.rerun()


def render_assistant_page() -> None:
    _init_state()
    st.title("Brand Intelligence Assistant")
    st.write(
        "Ask about sentiment, model metrics, predictions, or draft Email/WhatsApp alerts. "
        "Alert sends require human approval."
    )
    _sidebar()
    _alerts_banner()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.awaiting_approval:
        _render_hitl_panel()

    prompt = st.chat_input("Ask about sentiment, metrics, predictions, or alerts…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not st.session_state.awaiting_approval:
            st.session_state.thread_id = new_thread_id()

        run_prompt = _enrich_followup_prompt(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Running agents via Unified MCP…"):
                result = start_run(
                    run_prompt,
                    thread_id=st.session_state.thread_id,
                    use_mcp=True,
                )
            answer = result.get("answer") or "No response."
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.findings = answer
            _remember_classified_text(prompt, answer)

            if result.get("status") == "interrupted":
                payload = result.get("payload") or {}
                st.session_state.awaiting_approval = True
                st.session_state.pending_email = payload.get("pending_email") or {}
                st.session_state.pending_whatsapp = payload.get("pending_whatsapp") or {}
            else:
                st.session_state.awaiting_approval = False
            _snapshot_active_chat()
            if result.get("status") == "interrupted":
                st.rerun()
