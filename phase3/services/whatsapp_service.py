"""WhatsApp draft / send via Twilio. Drafts logged unless ALERTS_ENABLED=true."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from phase3.config import (
    ALERTS_LOG_DIR,
    alert_created_at,
    alert_log_stamp,
    alerts_enabled,
    get_twilio_settings,
)

logger = logging.getLogger(__name__)


def _log_draft(payload: dict[str, Any]) -> Path:
    ALERTS_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = ALERTS_LOG_DIR / f"whatsapp_draft_{alert_log_stamp()}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def draft_whatsapp(to: str, body: str) -> dict[str, Any]:
    settings = get_twilio_settings()
    to_addr = to or settings["default_to"]
    payload = {
        "channel": "whatsapp",
        "status": "draft",
        "to": to_addr,
        "body": body,
        "created_at": alert_created_at(),
        "alerts_enabled": alerts_enabled(),
    }
    path = _log_draft(payload)
    payload["log_path"] = str(path)
    return {"ok": True, "mode": "draft", "draft": payload}


def send_whatsapp(to: str, body: str) -> dict[str, Any]:
    draft = draft_whatsapp(to, body)
    if not alerts_enabled():
        draft["message"] = (
            "ALERTS_ENABLED is false — WhatsApp drafted/logged only, not sent."
        )
        return draft

    settings = get_twilio_settings()
    to_addr = to or settings["default_to"]
    if not settings["account_sid"] or not settings["auth_token"] or not to_addr:
        return {
            "ok": False,
            "error": "Missing TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN/ALERT_WHATSAPP_TO",
            "draft": draft.get("draft"),
        }

    try:
        from twilio.rest import Client

        client = Client(settings["account_sid"], settings["auth_token"])
        if not to_addr.startswith("whatsapp:"):
            to_addr = f"whatsapp:{to_addr}"
        message = client.messages.create(
            from_=settings["from_whatsapp"],
            to=to_addr,
            body=body,
        )
        draft["draft"]["status"] = "sent"
        draft["draft"]["sid"] = message.sid
        _log_draft(draft["draft"])
        return {"ok": True, "mode": "sent", "sid": message.sid, "to": to_addr}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Twilio WhatsApp send failed")
        return {"ok": False, "error": str(exc), "draft": draft.get("draft")}
