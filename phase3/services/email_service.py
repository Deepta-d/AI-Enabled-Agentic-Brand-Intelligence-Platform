"""Email draft / send (Gmail SMTP). Drafts are logged unless ALERTS_ENABLED=true."""

from __future__ import annotations

import json
import logging
import smtplib
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from phase3.config import (
    ALERTS_LOG_DIR,
    alert_created_at,
    alert_log_stamp,
    alerts_enabled,
    get_email_settings,
)

logger = logging.getLogger(__name__)


def _log_draft(payload: dict[str, Any]) -> Path:
    ALERTS_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = ALERTS_LOG_DIR / f"email_draft_{alert_log_stamp()}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def draft_email(to: str, subject: str, body: str) -> dict[str, Any]:
    settings = get_email_settings()
    to_addr = to or settings["default_to"]
    payload = {
        "channel": "email",
        "status": "draft",
        "to": to_addr,
        "subject": subject,
        "body": body,
        "created_at": alert_created_at(),
        "alerts_enabled": alerts_enabled(),
    }
    path = _log_draft(payload)
    payload["log_path"] = str(path)
    return {"ok": True, "mode": "draft", "draft": payload}


def send_email(to: str, subject: str, body: str) -> dict[str, Any]:
    """Send only when ALERTS_ENABLED=true; otherwise save draft."""
    draft = draft_email(to, subject, body)
    if not alerts_enabled():
        draft["message"] = (
            "ALERTS_ENABLED is false — email drafted/logged only, not sent."
        )
        return draft

    settings = get_email_settings()
    to_addr = to or settings["default_to"]
    if not settings["user"] or not settings["password"] or not to_addr:
        return {
            "ok": False,
            "error": "Missing SMTP_USER/SMTP_PASSWORD/ALERT_EMAIL_TO (or to=)",
            "draft": draft.get("draft"),
        }

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = settings["from_addr"] or settings["user"]
    msg["To"] = to_addr

    try:
        with smtplib.SMTP(settings["host"], int(settings["port"])) as server:
            server.starttls()
            server.login(settings["user"], settings["password"])
            server.send_message(msg)
        draft["draft"]["status"] = "sent"
        _log_draft(draft["draft"])
        return {"ok": True, "mode": "sent", "to": to_addr, "subject": subject}
    except smtplib.SMTPAuthenticationError as exc:
        logger.exception("SMTP authentication failed")
        return {
            "ok": False,
            "error": (
                "Gmail rejected the username/password. Use a Google App Password "
                "(not your normal Gmail password): "
                "https://myaccount.google.com/apppasswords — "
                f"details: {exc}"
            ),
            "draft": draft.get("draft"),
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("SMTP send failed")
        return {"ok": False, "error": str(exc), "draft": draft.get("draft")}
