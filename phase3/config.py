"""Phase 3 configuration (extends Phase 1 .env)."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from phase1.src.config import PROJECT_ROOT, get_database_url

load_dotenv(PROJECT_ROOT / ".env")

ARTIFACTS_DIR = PROJECT_ROOT / "phase2" / "artifacts"
ALERTS_LOG_DIR = PROJECT_ROOT / "phase3" / "logs"
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_v1.joblib"
METRICS_BEST_PATH = ARTIFACTS_DIR / "metrics_best_v1.json"
COMPARISON_PATH = ARTIFACTS_DIR / "comparison_v1.json"


def now_local() -> datetime:
    """Current time in the machine's local timezone (with offset)."""
    return datetime.now().astimezone()


def alert_log_stamp() -> str:
    """Filename-safe local stamp, e.g. 20260711T161450-0400."""
    return now_local().strftime("%Y%m%dT%H%M%S%z")


def alert_created_at() -> str:
    """ISO local timestamp with offset, e.g. 2026-07-11T16:14:50-04:00."""
    return now_local().isoformat(timespec="seconds")


def alerts_enabled() -> bool:
    return os.getenv("ALERTS_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def get_email_settings() -> dict[str, str]:
    return {
        "host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
        "port": os.getenv("SMTP_PORT", "587"),
        "user": os.getenv("SMTP_USER", os.getenv("GMAIL_USER", "")),
        "password": os.getenv("SMTP_PASSWORD", os.getenv("GMAIL_APP_PASSWORD", "")),
        "from_addr": os.getenv("SMTP_FROM", os.getenv("GMAIL_USER", "")),
        "default_to": os.getenv("ALERT_EMAIL_TO", ""),
    }


def get_twilio_settings() -> dict[str, str]:
    return {
        "account_sid": os.getenv("TWILIO_ACCOUNT_SID", ""),
        "auth_token": os.getenv("TWILIO_AUTH_TOKEN", ""),
        "from_whatsapp": os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886"),
        "default_to": os.getenv("ALERT_WHATSAPP_TO", ""),
    }


def get_fred_api_key() -> str:
    return os.getenv("FRED_API_KEY", "").strip()


def get_ollama_settings() -> dict[str, str]:
    return {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        "model": os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
    }


def get_gemini_api_key() -> str:
    return (
        os.getenv("GOOGLE_API_KEY", "").strip()
        or os.getenv("GEMINI_API_KEY", "").strip()
    )


def get_gemini_fallback_model() -> str:
    return os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"


def get_tool_cache_ttl() -> int:
    raw = os.getenv("TOOL_CACHE_TTL_SECONDS", "300").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 300


def get_llm_provider() -> str:
    """Resolve LLM backend: explicit LLM_PROVIDER, else cloud keys, else local Ollama."""
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in {"ollama", "openai", "anthropic"}:
        return explicit
    if os.getenv("ANTHROPIC_API_KEY", "").strip():
        return "anthropic"
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai"
    # Free local default (Ollama installed on this machine)
    return "ollama"


__all__ = [
    "PROJECT_ROOT",
    "ARTIFACTS_DIR",
    "ALERTS_LOG_DIR",
    "BEST_MODEL_PATH",
    "METRICS_BEST_PATH",
    "COMPARISON_PATH",
    "get_database_url",
    "alerts_enabled",
    "now_local",
    "alert_log_stamp",
    "alert_created_at",
    "get_email_settings",
    "get_twilio_settings",
    "get_fred_api_key",
    "get_ollama_settings",
    "get_gemini_api_key",
    "get_gemini_fallback_model",
    "get_tool_cache_ttl",
    "get_llm_provider",
]
