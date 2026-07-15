"""LLM builders with temperature 0 and Ollama → Gemini 2.5 Flash failover."""

from __future__ import annotations

import logging
from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from phase3.config import (
    get_gemini_api_key,
    get_gemini_fallback_model,
    get_llm_provider,
    get_ollama_settings,
)

logger = logging.getLogger(__name__)

LLM_TEMPERATURE = 0.0

_FAILOVER_MARKERS = (
    "10061",
    "actively refused",
    "connection refused",
    "connecterror",
    "timeout",
    "timed out",
    "429",
    "rate limit",
    "quota",
    "resource exhausted",
    "unavailable",
    "failed to connect",
    "connection error",
    "name or service not known",
)


class LLMOutageError(RuntimeError):
    """Raised when primary and fallback LLMs are both unavailable."""


def build_primary_llm() -> BaseChatModel:
    """Primary: Ollama (default) or configured cloud provider."""
    provider = get_llm_provider()
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        settings = get_ollama_settings()
        logger.info(
            "Primary LLM Ollama model=%s base_url=%s temp=%s",
            settings["model"],
            settings["base_url"],
            LLM_TEMPERATURE,
        )
        return ChatOllama(
            model=settings["model"],
            base_url=settings["base_url"],
            temperature=LLM_TEMPERATURE,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model="claude-sonnet-4-20250514", temperature=LLM_TEMPERATURE)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini", temperature=LLM_TEMPERATURE)
    raise RuntimeError(
        "No primary LLM configured. Set LLM_PROVIDER=ollama with Ollama running."
    )


def build_fallback_llm() -> BaseChatModel | None:
    """Fallback: Gemini 2.5 Flash via free Google AI Studio API key."""
    api_key = get_gemini_api_key()
    if not api_key:
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = get_gemini_fallback_model()
    logger.info("Fallback LLM Gemini model=%s temp=%s", model, LLM_TEMPERATURE)
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=LLM_TEMPERATURE,
    )


def build_llm() -> BaseChatModel:
    """Compatibility alias: returns primary LLM."""
    return build_primary_llm()


def _should_failover(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _FAILOVER_MARKERS)


def invoke_with_failover(messages: Sequence[BaseMessage]) -> AIMessage:
    """Invoke primary; on outage/rate-limit, switch once to Gemini 2.5 Flash."""
    primary = build_primary_llm()
    try:
        response = primary.invoke(list(messages))
        content = getattr(response, "content", response)
        return response if isinstance(response, AIMessage) else AIMessage(content=str(content))
    except Exception as primary_exc:  # noqa: BLE001
        if not _should_failover(primary_exc):
            raise
        logger.warning(
            "Primary LLM failed (%s); attempting Gemini 2.5 Flash fallback",
            primary_exc,
        )
        fallback = build_fallback_llm()
        if fallback is None:
            raise LLMOutageError(
                "Primary LLM is unavailable and GOOGLE_API_KEY is not set. "
                "Get a free key at https://aistudio.google.com/apikey and add "
                "GOOGLE_API_KEY to .env, or start Ollama and retry."
            ) from primary_exc
        try:
            response = fallback.invoke(list(messages))
            content = getattr(response, "content", response)
            logger.info("Using Gemini fallback successfully")
            return (
                response
                if isinstance(response, AIMessage)
                else AIMessage(content=str(content))
            )
        except Exception as fallback_exc:  # noqa: BLE001
            raise LLMOutageError(
                "Both primary LLM and Gemini 2.5 Flash fallback failed. "
                "Check Ollama and GOOGLE_API_KEY, then retry. "
                "No metrics were invented."
            ) from fallback_exc


def outage_plain_text(exc: BaseException) -> str:
    if isinstance(exc, LLMOutageError):
        return str(exc)
    return (
        "The language model is temporarily unavailable. "
        "Start Ollama or set GOOGLE_API_KEY for Gemini 2.5 Flash fallback, then retry. "
        "No metrics were invented."
    )
