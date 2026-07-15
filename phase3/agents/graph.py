"""LangGraph star-topology supervisor with MCP tools, cache, failover, and HITL."""

from __future__ import annotations

import json
import logging
import operator
import re
from typing import Annotated, Any, Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from phase3.agents.cache import get_tool_cache, user_requests_refresh
from phase3.agents.formatters import (
    STATUS_APPROVED_DRAFT_ONLY,
    STATUS_APPROVED_SENT,
    STATUS_CANNOT_SEND_WITHOUT_APPROVE,
    STATUS_DRAFT_ONLY,
    STATUS_DRAFT_READY,
    STATUS_REJECTED,
    assemble_final_answer,
    format_comparison,
    format_email_draft,
    format_metrics,
    format_post_lookup,
    format_prediction,
    format_query_rows,
    format_sentiment_summary,
    format_whatsapp_draft,
    unwrap_tool_payload,
)
from phase3.agents.llm import (
    LLMOutageError,
    build_llm,
    invoke_with_failover,
    outage_plain_text,
)
from phase3.agents.prompts import (
    EMAIL_BODY_PROMPT,
    IDENTITY_INTRO,
    LABEL_CAPABILITY_ANSWER,
    MYSQL_TOOL_HELP,
    SCHEMA_SQL_DECLINE,
    SUPERVISOR_ROUTING_PROMPT,
    WHATSAPP_BODY_PROMPT,
)
from phase3.agents.sql_intents import (
    SCHEMA_HINT,
    TABLE_SAMPLE_PLANS,
    COLUMN_ABOUT_BLURBS,
    classify_sql_intent,
    detect_table_contents_request,
    extract_country_filter,
    followup_sql_for_intent,
    mentioned_tables,
    table_count_plan,
)
from phase3.agents.tools import EMAIL_TOOLS, ML_TOOLS, SQL_TOOLS, WHATSAPP_TOOLS
from phase3.config import alerts_enabled, get_email_settings, get_twilio_settings

logger = logging.getLogger(__name__)

WORKERS = ("sql_agent", "ml_agent", "email_agent", "whatsapp_agent")


class GraphState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    visited: Annotated[list[str], operator.add]
    pending_email: dict[str, Any]
    pending_whatsapp: dict[str, Any]
    hitl_decision: str
    bypass_cache: bool
    final_text: str
    cache_notes: Annotated[list[str], operator.add]


def _local_tool_buckets() -> dict[str, list[BaseTool]]:
    return {
        "sql_agent": list(SQL_TOOLS),
        "ml_agent": list(ML_TOOLS),
        "email_agent": list(EMAIL_TOOLS),
        "whatsapp_agent": list(WHATSAPP_TOOLS),
    }


def _user_text(messages: Sequence[BaseMessage]) -> str:
    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.content:
            return str(msg.content)
    return ""


def is_sentiment_data_query(text: str) -> bool:
    """True when the user wants DB sentiment counts/distribution (not ML model metrics)."""
    lower = text.lower()
    if not re.search(r"\bsentiment\b", lower):
        return False
    # Explicit model/performance wording → ML path instead
    if any(
        k in lower
        for k in (
            "model",
            "validation",
            "winner",
            "won",
            "comparison",
            "compare",
            "f1",
            "accuracy",
            "artifact",
            "best_v1",
            "which model",
            "best model",
            "best ml",
        )
    ):
        return False
    return True


def is_model_metrics_query(text: str) -> bool:
    """True when the user asks which model won / about ML evaluation metrics."""
    lower = text.lower()
    # "sentiment metrics" / "summarize sentiment" → SQL distribution, not model comparison
    if is_sentiment_data_query(text):
        return False
    markers = (
        "best model",
        "best ml",
        "which model",
        "what model",
        "won",
        "winner",
        "validation",
        "evaluation",
        "eval",
        "model metric",
        "model metrics",
        "ml metric",
        "ml metrics",
        "comparison",
        "compare",
        "f1",
        "accuracy",
        "artifact",
        "performance",
    )
    if any(m in lower for m in markers):
        return True
    # Bare "metrics" only counts as ML when paired with model/ml/val context
    if re.search(r"\bmetrics?\b", lower) and any(
        k in lower for k in ("model", "ml ", " ml", "val ", "test ", "phase 2", "phase2")
    ):
        return True
    return False


def is_label_capability_query(text: str) -> bool:
    """True when asking which sentiment labels the model can predict (not classify text)."""
    lower = (text or "").lower().strip()
    if not lower:
        return False
    # Has an actual snippet to score → real classify path
    if re.search(r"""['"][^'"]{3,}['"]""", text):
        return False
    if re.search(
        r"\b(what|which|list|other|supported|available)\b.{0,40}\b"
        r"(sentiments?|labels?|classes?|categories)\b",
        lower,
    ):
        return True
    if re.search(
        r"\b(sentiments?|labels?|classes?)\b.{0,40}\b"
        r"(can you|do you|you can|classify|predict|support|handle)\b",
        lower,
    ):
        return True
    if re.search(r"\bother\s+sentiments?\b", lower):
        return True
    return False


def _has_classify_intent(text: str) -> bool:
    """True only for classify/predict-on-text intents (not 'prediction' evaluation talk)."""
    if is_label_capability_query(text):
        return False
    lower = text.lower()
    # Word-boundary: "predict" in "prediction" must NOT match.
    if re.search(r"\b(predict|classify|label)\b", lower):
        # "what can you classify" without a text sample is capability, not a job
        if re.search(r"\b(what|which|other)\b", lower) and not re.search(
            r"""['"][^'"]{3,}['"]""", text
        ):
            if re.search(r"\b(sentiments?|labels?|classes?)\b", lower):
                return False
        return True
    has_snippet = bool(re.search(r"""['"][^'"]{3,}['"]""", text))
    if has_snippet and any(
        k in lower for k in ("sentiment", "positive", "negative", "neutral")
    ):
        return True
    # "Is <text> Negative or Positive?" without quotes still counts if polarity asked.
    if re.search(r"\b(positive|negative|neutral)\b", lower) and re.search(
        r"\b(is|or|classify|label|tell)\b", lower
    ):
        if not re.search(r"\b(count|summary|summarize|distribution|metrics?)\b", lower):
            return True
    return False


def alert_fact_workers(text: str) -> list[str]:
    """Workers that must run before drafting Email/WhatsApp so bodies use real facts."""
    lower = text.lower()
    facts: list[str] = []
    wants_sql = any(
        k in lower
        for k in (
            "sql",
            "mysql",
            "database",
            "analytics",
            "dataset",
            "sentiment",
            "username",
            "platform",
            "country",
            "hashtag",
            "phase 1",
            "phase1",
            "posts",
            "agreement",
        )
    )
    wants_ml = any(
        k in lower
        for k in (
            "model",
            "ml ",
            "metric",
            "f1",
            "validation",
            "evaluation",
            "artifact",
            "phase 2",
            "phase2",
            "predict",
            "comparison",
            "winner",
            "fred",
        )
    )
    if wants_sql:
        facts.append("sql_agent")
    if wants_ml:
        facts.append("ml_agent")
    # Generic alert with no topic → at least sentiment SQL facts
    if not facts:
        facts.append("sql_agent")
    return facts


def _has_data_or_alert_intent(text: str) -> bool:
    """True when the user is asking for analytics, ML, or outbound alerts."""
    lower = text.lower()
    if any(
        k in lower
        for k in (
            "email",
            "e-mail",
            "mail",
            "alert",
            "draft",
            "notify",
            "whatsapp",
            "twilio",
            "sms",
        )
    ):
        return True
    if is_model_metrics_query(text) or _has_classify_intent(text):
        return True
    if re.search(
        r"\b(username|user\s*name|users|authors?|handles?|customers?|platform|country|hashtag|"
        r"hash\s*tag|retweets?|likes?|posts?|dataset|table|column|schema|sentiment|"
        r"mysql|sql|database|analytics|summary|summarize|summarise|count|agreement|"
        r"model|metric|f1|predict|classify|artifact|fred|validation|evaluation|"
        r"positive|negative|neutral|how\s+many)\b",
        lower,
    ):
        return True
    if extract_country_filter(text):
        return True
    return False


def is_identity_query(text: str) -> bool:
    """True for self-intro / capability / help meta questions (not analytics)."""
    if _has_data_or_alert_intent(text):
        return False
    lower = (text or "").lower().strip()
    if not lower:
        return False
    patterns = (
        r"\bwho\s+are\s+you\b",
        r"\bwhat\s+are\s+you\b",
        r"\bwhat(?:'s|\s+is)\s+your\s+name\b",
        r"\bwhat(?:'s|\s+is)\s+your\s+responsibility\b",
        r"\bwhat\s+are\s+your\s+(?:responsibilities|capabilities|abilities)\b",
        r"\byour\s+responsibility\b",
        r"\bwhat\s+do\s+you\s+do\b",
        r"\bwhat\s+can\s+you\s+do\b",
        r"\bhow\s+(?:can|do)\s+you\s+help\b",
        r"\bhow\s+can\s+i\s+use\s+you\b",
        r"\bintroduce\s+yourself\b",
        r"\btell\s+me\s+about\s+yourself\b",
        r"\bare\s+you\s+(?:a\s+)?(?:chatbot|bot|assistant|ai)\b",
        r"^(?:please\s+)?help(?:\s+me)?(?:\s+please)?\s*[?.!]*$",
        r"^(?:hi|hello|hey)\s*[?.!]*$",
    )
    return any(re.search(p, lower) for p in patterns)


def detect_mysql_tool_topic(text: str) -> str | None:
    """Return a canonical mysql_* tool name if the user asks about that tool."""
    natural = (text or "").lower()
    compact = re.sub(r"[^a-z0-9]+", "", natural)

    if "prediction" in natural and "agreement" in natural:
        return "mysql_prediction_agreement"
    if "mysql" in natural and "sentiment" in natural and "summary" in natural:
        return "mysql_sentiment_summary"
    if "mysql" in natural and "latest" in natural and "metric" in natural:
        return "mysql_latest_metrics"
    if "mysql_query" in natural or "mysqlquery" in compact:
        return "mysql_query"

    # Underscore / glued forms
    for name in (
        "mysql_prediction_agreement",
        "mysql_sentiment_summary",
        "mysql_latest_metrics",
        "mysql_query",
    ):
        if name.replace("_", "") in compact:
            return name
    return None


def is_mysql_tool_query(text: str) -> bool:
    return detect_mysql_tool_topic(text) is not None


def is_schema_query(text: str) -> bool:
    """True when the user asks for table/column schema (declined for end users)."""
    lower = (text or "").lower()
    return bool(
        re.search(
            r"\b(schema(?:\s+table)?|table\s+schema|show\s+schema|"
            r"describe\s+tables?|table\s+structure|columns?\s+(?:list|of)|"
            r"what\s+(?:are\s+)?(?:the\s+)?columns?|ddl|create\s+table)\b",
            lower,
        )
    )


def is_sql_meta_query(text: str) -> bool:
    """True for questions about DB name / which SQL was run (declined for end users)."""
    lower = (text or "").lower()
    if is_mysql_tool_query(text):
        # Tool "what is mysql_*" is answered separately; pure SQL/schema meta is not.
        if is_schema_query(text):
            return True
        return False
    if is_schema_query(text):
        return True
    if re.search(
        r"\b(database\s+name|name\s+of\s+(?:the\s+)?database|which\s+database|"
        r"what\s+database|db\s+name)\b",
        lower,
    ):
        return True
    if re.search(
        r"\b(what\s+sql|sql\s+(?:analytics|queries?)(?:\s+were)?\s+(?:performed|executed|run)|"
        r"queries?\s+(?:were\s+)?(?:executed|run|performed)|"
        r"analytics\s+were\s+performed|what\s+analytics(?:\s+were)?\s+performed|"
        r"which\s+sql\s+analytics|what\s+tables|which\s+tables|"
        r"show\s+(?:me\s+)?(?:the\s+)?sql|list\s+(?:the\s+)?(?:sql\s+)?queries)\b",
        lower,
    ):
        return True
    return False


def plan_from_query(text: str) -> list[str]:
    """Deterministic worker plan so small local LLMs cannot skip required steps."""
    lower = text.lower()
    plan: list[str] = []

    wants_email = any(
        k in lower for k in ("email", "e-mail", "mail", "alert", "draft", "notify")
    )
    wants_wa = any(k in lower for k in ("whatsapp", "twilio", "sms"))
    wants_alert = wants_email or wants_wa

    # Identity / help / small-talk → no SQL/ML tools (supervisor replies with intro).
    if is_identity_query(text):
        return []

    # Which labels can we classify? → short capability answer (no ml_predict).
    if is_label_capability_query(text) and not wants_alert:
        return []

    # Explain / run a specific mysql_* tool.
    if is_mysql_tool_query(text) and not wants_alert:
        return ["sql_agent"]

    # Table/column schema → declined (no schema dump).
    if is_schema_query(text) and not wants_alert:
        return []

    # DB name / SQL executed / analytics inventory → declined.
    if is_sql_meta_query(text) and not wants_alert:
        return []

    # Pure classification requests should not also dump SQL sentiment summaries.
    if is_predict_only_query(text) and not wants_alert:
        return ["ml_agent"]

    # Sentiment counts / distribution → SQL only (not ML model comparison).
    if is_sentiment_data_query(text) and not wants_alert:
        return ["sql_agent"]

    # Specific country ("from USA", "customers in India") → SQL.
    if extract_country_filter(text) and not wants_alert and not _has_classify_intent(text):
        return ["sql_agent"]

    # Model winner / evaluation questions → ML agent only (unless DB/alert-oriented).
    if (
        is_model_metrics_query(text)
        and not wants_alert
        and not any(k in lower for k in ("mysql", "sql", "database", "stored metric"))
    ):
        return ["ml_agent"]

    # Column / dataset field questions → SQL only (unless drafting an alert).
    if re.search(
        r"\b(username|user\s*name|users|authors?|handles?|customers?|platform|country|hashtag|hash\s*tag|"
        r"retweets?|likes?|posts?|dataset|table|column|schema|how\s+many)\b",
        lower,
    ) and not is_model_metrics_query(text) and not _has_classify_intent(text):
        if not wants_alert:
            return ["sql_agent"]

    wants_sql = any(
        k in lower
        for k in (
            "sentiment",
            "mysql",
            "sql",
            "summary",
            "summarize",
            "summarise",
            "count",
            "database",
            "dataset",
            "username",
            "platform",
            "country",
            "hashtag",
            "posts",
            "customers",
            "customer",
            "agreement",
            "analytics",
            "phase 1",
            "phase1",
            "how many",
        )
    )
    wants_ml = any(
        k in lower
        for k in (
            "model",
            "ml ",
            "ml_",
            "f1",
            "comparison",
            "artifact",
            "fred",
            "won",
            "winner",
            "validation",
            "evaluation",
            "metric",
            "predict",
            "classify",
            "phase 2",
            "phase2",
        )
    ) or _has_classify_intent(text)

    if wants_ml and is_model_metrics_query(text) and not wants_alert:
        wants_sql = any(
            k in lower for k in ("mysql", "sql", "database", "stored metric", "count", "summarize sentiment")
        )
    if wants_sql and not wants_ml and not is_model_metrics_query(text) and not wants_alert:
        wants_ml = False

    for worker in WORKERS:
        if worker in lower and worker not in plan:
            plan.append(worker)

    if wants_sql and "sql_agent" not in plan:
        plan.append("sql_agent")
    if wants_ml and "ml_agent" not in plan:
        plan.append("ml_agent")
    if wants_email and "email_agent" not in plan:
        plan.append("email_agent")
    if wants_wa and "whatsapp_agent" not in plan:
        plan.append("whatsapp_agent")

    # Alerts must gather real Phase 1/2 facts before drafting.
    if wants_alert:
        facts = alert_fact_workers(text)
        ordered: list[str] = []
        for step in facts + [w for w in plan if w not in facts]:
            if step not in ordered:
                ordered.append(step)
        # Ensure outbound agents are last
        for out in ("email_agent", "whatsapp_agent"):
            if out in ordered:
                ordered = [x for x in ordered if x != out] + [out]
        plan = ordered

    # Unmapped non-analytics chatter → empty plan; supervisor returns the intro help text.
    if not plan and not _has_data_or_alert_intent(text):
        return []
    if not plan:
        plan = ["sql_agent"]
    return plan


def is_predict_only_query(text: str) -> bool:
    """True when the user wants a sentiment classification, not model metrics."""
    lower = text.lower()
    has_snippet = bool(re.search(r"""['"][^'"]{3,}['"]""", text))

    # "best model from prediction and evaluation" is metrics QA, not classify.
    if is_model_metrics_query(text) and not has_snippet:
        return False
    if not _has_classify_intent(text):
        return False
    if any(
        k in lower
        for k in ("alert", "email", "whatsapp", "summary", "database", "mysql", "count")
    ):
        return False
    return True


def _extract_predict_snippets(user: str) -> list[str]:
    snippets = re.findall(r"[\"']([^\"']{3,200})[\"']", user)
    if snippets:
        return snippets
    # Fallback: text after "predict sentiment for"
    m = re.search(
        r"predict\s+(?:the\s+)?sentiment\s+for\s+(.+)$",
        user,
        flags=re.IGNORECASE,
    )
    if m:
        return [m.group(1).strip(" \t\"'")]
    # "Is <phrase> Negative or Positive?" without wrapping quotes on short phrases
    m2 = re.search(
        r"^\s*is\s+(.+?)\s+(?:negative|positive|neutral)\b",
        user,
        flags=re.IGNORECASE,
    )
    if m2:
        return [m2.group(1).strip(" \t\"'?")]
    return []


async def _ainvoke_ml_predict(
    tools: list[BaseTool],
    snippets: list[str],
    *,
    model_version: str = "best_v1",
    bypass_cache: bool = False,
) -> tuple[str, bool]:
    """Call ml_predict with MCP (texts_json) or local (text) signatures."""
    by_name = _tool_map(tools)
    tool = by_name.get("ml_predict")
    if tool is None:
        return json.dumps({"ok": False, "error": "ml_predict tool not found"}), False

    cache = get_tool_cache()
    cache_key_kwargs = {
        "snippets": snippets,
        "model_version": model_version,
    }
    if not bypass_cache:
        cached = cache.get("ml_predict", cache_key_kwargs)
        if cached is not None:
            return cached, True

    # Prefer batch MCP signature, then local single-text signature
    attempts: list[dict[str, Any]] = [
        {"texts_json": json.dumps(snippets), "model_version": model_version},
    ]
    # Local StructuredTool requires `text`
    for s in snippets[:5]:
        attempts.append({"text": s, "model_version": model_version})

    last_error = ""
    collected: list[str] = []
    used_local_loop = False

    for kwargs in attempts:
        if "text" in kwargs and len(snippets) > 1:
            used_local_loop = True
        try:
            logger.info("Invoking tool ml_predict kwargs=%s", kwargs)
            try:
                result = await tool.ainvoke(kwargs)
            except Exception:  # noqa: BLE001 — sync-only tools
                result = tool.invoke(kwargs)
            text = result if isinstance(result, str) else json.dumps(result, default=str)
            if used_local_loop or "text" in kwargs:
                collected.append(text)
                if len(collected) >= len(snippets[:5]):
                    # Merge local single predictions
                    preds = []
                    for chunk in collected:
                        data = unwrap_tool_payload(chunk)
                        if isinstance(data, dict) and data.get("predictions"):
                            preds.extend(data["predictions"])
                        elif isinstance(data, dict) and data.get("ok") is False:
                            last_error = str(data.get("error") or chunk)
                            continue
                    merged = json.dumps(
                        {
                            "ok": True,
                            "model_version": model_version,
                            "predictions": preds,
                        },
                        default=str,
                    )
                    cache.set("ml_predict", merged, cache_key_kwargs)
                    return merged, False
                continue
            cache.set("ml_predict", text, cache_key_kwargs)
            return text, False
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            logger.warning("ml_predict attempt failed (%s): %s", kwargs.keys(), exc)
            continue

    return (
        json.dumps(
            {
                "ok": False,
                "error": last_error or "ml_predict failed for both texts_json and text signatures",
            }
        ),
        False,
    )


def parse_worker_choice(text: str) -> str | None:
    raw = (text or "").strip()
    if not raw:
        return None
    first = raw.splitlines()[0].strip().strip("`\"'").lower()
    tokens = (*WORKERS, "human_review", "send_alerts", "FINISH")
    for name in tokens:
        if first == name.lower():
            return name
    lower = raw.lower()
    for name in WORKERS:
        if re.search(rf"\b{re.escape(name)}\b", lower):
            return name
    if "human_review" in lower:
        return "human_review"
    if re.search(r"\bfinish\b", lower):
        return "FINISH"
    return None


def _tool_map(tools: list[BaseTool]) -> dict[str, BaseTool]:
    return {t.name: t for t in tools}


def _extract_draft_dict(payload: str) -> dict[str, Any]:
    data = unwrap_tool_payload(payload)
    if isinstance(data, dict):
        draft = data.get("draft")
        if isinstance(draft, dict):
            return draft
        if data.get("channel") in {"email", "whatsapp"}:
            return data
    return {}


async def _ainvoke_tool(
    tools: list[BaseTool],
    preferred_names: Sequence[str],
    *,
    bypass_cache: bool = False,
    **kwargs: Any,
) -> tuple[str, bool]:
    """Return (payload_str, from_cache)."""
    by_name = _tool_map(tools)
    cache = get_tool_cache()
    for name in preferred_names:
        tool = by_name.get(name)
        if tool is None:
            continue
        if not bypass_cache:
            cached = cache.get(name, kwargs)
            if cached is not None:
                return cached, True
        logger.info("Invoking tool %s kwargs=%s", name, kwargs)
        try:
            result = await tool.ainvoke(kwargs)
        except Exception:  # noqa: BLE001
            result = tool.invoke(kwargs)
        text = result if isinstance(result, str) else json.dumps(result, default=str)
        cache.set(name, text, kwargs)
        return text, False
    available = sorted(by_name)
    return (
        json.dumps(
            {
                "ok": False,
                "error": f"None of {list(preferred_names)} found",
                "available": available,
            }
        ),
        False,
    )


def _prior_findings(messages: Sequence[BaseMessage]) -> str:
    chunks: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content:
            text = str(msg.content)
            if text.startswith("[supervisor]->"):
                continue
            if text.startswith("Email draft:") or text.startswith("WhatsApp draft:"):
                continue
            chunks.append(text)
    return "\n\n".join(chunks[-8:])


def _findings_are_usable(findings: str) -> bool:
    text = (findings or "").strip()
    if not text or text == "No prior metrics available.":
        return False
    markers = (
        "Positive",
        "Negative",
        "Neutral",
        "unique usernames",
        "Validation winner",
        "Accuracy",
        "F1",
        "f1",
        "agreement",
        "best_v1",
        "Sentiment",
        "Usernames",
        "Posts by",
        "Model comparison",
        "Predicted sentiment",
        "Phase 2 artifacts",
    )
    return any(m in text for m in markers)


_INVENTED_ALERT_MARKERS = (
    "marketing campaign",
    "drive sales",
    "brand awareness",
    "budget will be shared",
    "unspecified period",
    "target audience",
    "[your name]",
    "top themes",
    "frequently used emoji",
    "out of 40",
    "out of 50",
    "enjoying nature",
    "fitness and wellness",
)


def wants_immediate_send(text: str) -> bool:
    """True when user asks to send now / skip draft / bypass approval."""
    lower = (text or "").lower()
    if not any(k in lower for k in ("email", "mail", "whatsapp", "alert", "send", "notify")):
        return False
    patterns = (
        r"\bsend\s+(?:it\s+|them\s+|the\s+alert\s+|the\s+email\s+|now\b)",
        r"\bsend\s+now\b",
        r"\bwithout\s+(?:showing\s+)?(?:me\s+)?(?:a\s+)?draft\b",
        r"\bskip\s+(?:the\s+)?draft\b",
        r"\bno\s+draft\b",
        r"\bbypass\s+approval\b",
        r"\bwithout\s+(?:my\s+)?approv",
        r"\bjust\s+send\b",
        r"\bsend\s+immediately\b",
    )
    return any(re.search(p, lower) for p in patterns)


def _body_looks_invented(body: str) -> bool:
    lower = body.lower()
    if any(m in lower for m in _INVENTED_ALERT_MARKERS):
        return True
    # Heuristic: emoji characters often mean invented marketing fluff
    if re.search(r"[\U0001F300-\U0001FAFF]", body):
        return True
    return False


def _alert_subject(user: str) -> str:
    lower = user.lower()
    if any(k in lower for k in ("sql", "analytics", "database", "mysql", "sentiment")):
        return "SQL / Sentiment Analytics Alert"
    if any(k in lower for k in ("model", "ml", "metric", "validation", "f1")):
        return "ML Model Performance Alert"
    return "Brand Intelligence Alert"


def _deterministic_alert_body(user: str, findings: str, *, channel: str = "email") -> str:
    facts = findings.strip() or "(No analytic findings were available.)"
    if channel == "whatsapp":
        return (
            "Brand Intelligence alert\n\n"
            f"{facts}\n\n"
            "Please review and Approve to send."
        )
    return (
        "Brand Intelligence alert based on the latest platform findings:\n\n"
        f"{facts}\n\n"
        "Please review these figures and Approve to send.\n"
        "- Brand Intelligence Assistant"
    )


async def _compose_alert_body(
    user: str,
    findings: str,
    *,
    channel: str = "email",
) -> str:
    grounded = _deterministic_alert_body(user, findings, channel=channel)
    # Skip-draft / send-now: never let the LLM invent fluff — stay on real findings only.
    if wants_immediate_send(user):
        return grounded
    prompt = EMAIL_BODY_PROMPT if channel == "email" else WHATSAPP_BODY_PROMPT
    draft_prompt = [
        SystemMessage(content=prompt),
        HumanMessage(
            content=(
                f"User request:\n{user}\n\n"
                f"Facts (use only these):\n{findings}\n"
            )
        ),
    ]
    try:
        body = str(invoke_with_failover(draft_prompt).content or "").strip()
    except LLMOutageError:
        return grounded

    if not body or _body_looks_invented(body) or not _findings_are_usable(findings):
        return grounded
    tokens = re.findall(r"[A-Za-z0-9_.-]{3,}", findings)
    skip = {
        "the",
        "and",
        "from",
        "with",
        "rows",
        "shown",
        "latest",
        "cached",
        "lookup",
        "this",
        "that",
        "for",
    }
    interesting = [t for t in tokens if t.lower() not in skip][:16]
    if interesting and not any(tok in body for tok in interesting):
        return grounded
    return body


async def _gather_alert_facts(
    buckets: dict[str, list[BaseTool]],
    user: str,
    *,
    bypass_cache: bool = False,
) -> str:
    """Collect SQL/ML facts for alert drafting when prior agents did not run."""
    sections: list[str] = []
    facts_needed = alert_fact_workers(user)
    lower = user.lower()

    if "sql_agent" in facts_needed:
        sql_tools = buckets.get("sql_agent") or []
        plan = classify_sql_intent(user)
        if plan and plan.intent == "username_summary" and plan.sql:
            payload, _ = await _ainvoke_tool(
                sql_tools,
                ["mysql_query"],
                bypass_cache=bypass_cache,
                sql=plan.sql,
                allow_write=False,
            )
            sections.append(format_query_rows(payload, title=plan.title))
            follow = followup_sql_for_intent(plan.intent)
            if follow and follow.sql:
                payload2, _ = await _ainvoke_tool(
                    sql_tools,
                    ["mysql_query"],
                    bypass_cache=bypass_cache,
                    sql=follow.sql,
                    allow_write=False,
                )
                sections.append(format_query_rows(payload2, title=follow.title))
        else:
            summary, _ = await _ainvoke_tool(
                sql_tools, ["mysql_sentiment_summary"], bypass_cache=bypass_cache
            )
            sections.append(format_sentiment_summary(summary))
            if any(k in lower for k in ("metric", "model", "agreement")):
                metrics, _ = await _ainvoke_tool(
                    sql_tools,
                    ["mysql_latest_metrics"],
                    bypass_cache=bypass_cache,
                    model_version="best_v1",
                )
                sections.append(format_metrics(metrics))

    if "ml_agent" in facts_needed:
        ml_tools = buckets.get("ml_agent") or []
        comparison, _ = await _ainvoke_tool(
            ml_tools, ["ml_get_comparison"], bypass_cache=bypass_cache
        )
        best, _ = await _ainvoke_tool(
            ml_tools, ["ml_get_best_metrics"], bypass_cache=bypass_cache
        )
        sections.append(format_comparison(comparison))
        sections.append(format_metrics(best, title="Best model artifact highlights"))

    return assemble_final_answer(sections=sections)


async def _llm_select_sql(user_question: str) -> str | None:
    """Ask the LLM for one safe SELECT; return None if unusable."""
    messages = [
        SystemMessage(
            content=(
                "You write ONE MySQL SELECT for the Brand Intelligence database. "
                "Read-only only (SELECT/SHOW/DESCRIBE/EXPLAIN). "
                "No INSERT/UPDATE/DELETE/DROP. Prefer LIMIT <= 50 for detail rows. "
                "Never invent multi-table JOINs just to show table contents — "
                "query one table at a time. If you must JOIN, qualify every column "
                "(table.column) so names like id are never ambiguous. "
                "Reply with SQL only — no markdown fences, no commentary.\n\n"
                f"{SCHEMA_HINT}"
            )
        ),
        HumanMessage(content=user_question),
    ]
    try:
        response = invoke_with_failover(messages)
        raw = str(response.content or "").strip()
    except LLMOutageError:
        return None
    raw = re.sub(r"^```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw).strip().rstrip(";")
    if not re.match(r"^\s*(SELECT|SHOW|DESCRIBE|DESC|EXPLAIN)\b", raw, re.I | re.S):
        return None
    if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT)\b", raw, re.I):
        return None
    return raw


def build_graph(
    tool_buckets: dict[str, list[BaseTool]] | None = None,
    *,
    hitl: bool = False,
    checkpointer: MemorySaver | None = None,
):
    """Build supervisor graph. Prefer MCP-derived tool_buckets when provided."""
    buckets = tool_buckets or _local_tool_buckets()
    for agent_name, tools in buckets.items():
        if not tools:
            logger.warning("Agent %s has no tools", agent_name)

    def supervisor_node(state: GraphState) -> dict:
        user = _user_text(state.get("messages") or [])
        plan = plan_from_query(user)
        visited = set(state.get("visited") or [])
        remaining = [step for step in plan if step not in visited]
        needs_hitl = hitl and any(
            w in plan for w in ("email_agent", "whatsapp_agent")
        )
        hitl_done = (state.get("hitl_decision") or "") in {"approve", "reject"}

        # Identity / schema-SQL decline / unmapped small-talk: finish without tool dumps.
        if not plan and not visited:
            if is_schema_query(user) or is_sql_meta_query(user):
                text = SCHEMA_SQL_DECLINE
            elif is_label_capability_query(user):
                text = LABEL_CAPABILITY_ANSWER
            else:
                text = IDENTITY_INTRO
            return {
                "next": "FINISH",
                "final_text": text,
                "messages": [AIMessage(content=text)],
            }

        if not remaining and needs_hitl and not hitl_done:
            choice = "human_review"
            return {
                "next": choice,
                "messages": [AIMessage(content=f"[supervisor]-> {choice}")],
            }
        if not remaining and needs_hitl and hitl_done and state.get("hitl_decision") == "approve":
            # After approval, send once then finish (visited send_alerts tracked via decision flow)
            if "send_alerts" not in visited:
                choice = "send_alerts"
                return {
                    "next": choice,
                    "messages": [AIMessage(content=f"[supervisor]-> {choice}")],
                }
            choice = "FINISH"
            return {
                "next": choice,
                "messages": [AIMessage(content=f"[supervisor]-> {choice}")],
            }
        if not remaining:
            choice = "FINISH"
            return {
                "next": choice,
                "messages": [AIMessage(content=f"[supervisor]-> {choice}")],
            }

        # For alert workflows, force next remaining worker (no LLM wander).
        if needs_hitl or any(w in remaining for w in ("email_agent", "whatsapp_agent", "sql_agent", "ml_agent")):
            if len(remaining) >= 1 and any(
                k in user.lower() for k in ("email", "alert", "draft", "whatsapp", "notify")
            ):
                choice = remaining[0]
                return {
                    "next": choice,
                    "messages": [AIMessage(content=f"[supervisor]-> {choice}")],
                }

        messages: list[BaseMessage] = [
            SystemMessage(content=SUPERVISOR_ROUTING_PROMPT),
            *(state.get("messages") or []),
            HumanMessage(
                content=(
                    f"Remaining required workers (in order): {', '.join(remaining)}. "
                    "Reply with ONLY the next worker name."
                )
            ),
        ]
        try:
            response = invoke_with_failover(messages)
            raw = str(response.content or "").strip()
        except LLMOutageError as exc:
            # Deterministic next step if LLM is down
            raw = remaining[0]
            logger.warning("Supervisor LLM outage; forcing %s (%s)", raw, exc)
        parsed = parse_worker_choice(raw)
        if parsed in remaining:
            choice = parsed
        else:
            choice = remaining[0]
            if parsed and parsed != choice:
                logger.warning(
                    "Overriding supervisor choice %r -> %r (plan remaining)",
                    parsed,
                    choice,
                )
        return {
            "next": choice,
            "messages": [AIMessage(content=f"[supervisor]-> {choice}")],
        }

    def route(
        state: GraphState,
    ) -> Literal[
        "sql_agent",
        "ml_agent",
        "email_agent",
        "whatsapp_agent",
        "human_review",
        "send_alerts",
        "__end__",
    ]:
        nxt = state.get("next", "FINISH")
        if nxt == "FINISH":
            return "__end__"
        return nxt  # type: ignore[return-value]

    async def sql_node(state: GraphState) -> dict:
        tools = buckets["sql_agent"]
        bypass = bool(state.get("bypass_cache"))
        user = _user_text(state.get("messages") or [])
        lower = user.lower()
        sections: list[str] = []
        notes: list[str] = []

        tool_topic = detect_mysql_tool_topic(user)
        if tool_topic:
            sections.append(MYSQL_TOOL_HELP.get(tool_topic, tool_topic))
            if tool_topic == "mysql_sentiment_summary":
                summary, c1 = await _ainvoke_tool(
                    tools, ["mysql_sentiment_summary"], bypass_cache=bypass
                )
                sections.append(format_sentiment_summary(summary, from_cache=c1))
                if c1:
                    notes.append("sql_cache_hit")
            elif tool_topic == "mysql_prediction_agreement":
                payload, c1 = await _ainvoke_tool(
                    tools,
                    ["mysql_prediction_agreement"],
                    bypass_cache=bypass,
                    model_version="best_v1",
                )
                sections.append(
                    format_query_rows(
                        payload,
                        title="Live prediction agreement (best_v1)",
                        from_cache=c1,
                    )
                )
                if c1:
                    notes.append("sql_cache_hit")
            elif tool_topic == "mysql_latest_metrics":
                metrics, c1 = await _ainvoke_tool(
                    tools,
                    ["mysql_latest_metrics"],
                    bypass_cache=bypass,
                    model_version="best_v1",
                )
                sections.append(format_metrics(metrics, from_cache=c1))
                if c1:
                    notes.append("sql_cache_hit")
            else:
                sections.append(
                    'To run it, ask with a SELECT, e.g. mysql_query: SHOW TABLES; '
                    'or "run SELECT platform, COUNT(*) n FROM social_posts GROUP BY platform".'
                )
            text = assemble_final_answer(sections=sections)
            return {
                "messages": [AIMessage(content=text)],
                "visited": ["sql_agent"],
                "cache_notes": notes,
                "final_text": text,
            }

        plan = classify_sql_intent(user)

        async def _run_select(title: str, sql: str) -> dict[str, Any]:
            payload, cached = await _ainvoke_tool(
                tools,
                ["mysql_query"],
                bypass_cache=bypass,
                sql=sql,
                allow_write=False,
            )
            sections.append(format_query_rows(payload, title=title, from_cache=cached))
            if cached:
                notes.append("sql_cache_hit")
            data = unwrap_tool_payload(payload) if isinstance(payload, str) else payload
            return data if isinstance(data, dict) else {"ok": False, "error": "query_failed"}

        async def _run_table_samples(tables: list[str]) -> None:
            """Safe per-table samples + counts (never one giant JOIN)."""
            for table in tables:
                sample = TABLE_SAMPLE_PLANS.get(table)
                if not sample:
                    continue
                count = table_count_plan(table)
                await _run_select(count.title, count.sql)
                await _run_select(sample.title, sample.sql)

        table_names = detect_table_contents_request(user)
        if table_names or (plan and plan.intent == "table_contents"):
            await _run_table_samples(table_names or list(TABLE_SAMPLE_PLANS))
        elif plan is None:
            # Ask LLM for a safe SELECT, then execute — never surface raw SQL/errors.
            sql = await _llm_select_sql(user)
            if sql:
                result = await _run_select("Query result", sql)
                if result.get("ok") is False:
                    logger.warning(
                        "LLM SELECT failed (hidden from user): %s",
                        result.get("error"),
                    )
                    # Drop the sanitized error section; retry safer samples or apologize.
                    if sections:
                        sections.pop()
                    fallback_tables = mentioned_tables(user)
                    if fallback_tables:
                        await _run_table_samples(fallback_tables)
                    else:
                        sections.append(
                            "I couldn't answer that cleanly. "
                            "Try asking about one table at a time "
                            "(social_posts, model_metrics, or model_predictions), "
                            "or about usernames, platforms, countries, or sentiment counts."
                        )
            else:
                # Safe default: describe what we can answer
                summary, c1 = await _ainvoke_tool(
                    tools, ["mysql_sentiment_summary"], bypass_cache=bypass
                )
                sections.append(format_sentiment_summary(summary, from_cache=c1))
                sections.append(
                    "I could not map that request to a specific column query. "
                    "Try asking about usernames, platforms, countries, hashtags, "
                    "or sentiment counts."
                )
                if c1:
                    notes.append("sql_cache_hit")
        elif plan.intent == "post_lookup":
            if not plan.sql:
                sections.append(
                    "I need the post text to look up its platform (or other fields). "
                    "Classify a post first, or quote the text in your question."
                )
            else:
                payload, c1 = await _ainvoke_tool(
                    tools,
                    ["mysql_query"],
                    bypass_cache=bypass,
                    sql=plan.sql,
                    allow_write=False,
                )
                sections.append(
                    format_post_lookup(payload, title=plan.title, from_cache=c1)
                )
                if c1:
                    notes.append("sql_cache_hit")
        elif plan.intent == "sentiment_summary":
            summary, c1 = await _ainvoke_tool(
                tools, ["mysql_sentiment_summary"], bypass_cache=bypass
            )
            sections.append(format_sentiment_summary(summary, from_cache=c1))
            if c1:
                notes.append("sql_cache_hit")
        elif plan.intent == "prediction_agreement":
            payload, c1 = await _ainvoke_tool(
                tools,
                ["mysql_prediction_agreement"],
                bypass_cache=bypass,
                model_version="best_v1",
            )
            sections.append(
                format_query_rows(payload, title=plan.title, from_cache=c1)
            )
            if c1:
                notes.append("sql_cache_hit")
        elif plan.intent == "db_metrics":
            metrics, c1 = await _ainvoke_tool(
                tools,
                ["mysql_latest_metrics"],
                bypass_cache=bypass,
                model_version="best_v1",
            )
            sections.append(format_metrics(metrics, from_cache=c1))
            if c1:
                notes.append("sql_cache_hit")
        else:
            blurb = COLUMN_ABOUT_BLURBS.get(plan.intent)
            if blurb:
                sections.append(blurb)
            if plan.sql:
                await _run_select(plan.title, plan.sql)
            follow = followup_sql_for_intent(plan.intent)
            if follow and follow.sql:
                await _run_select(follow.title, follow.sql)

        # Only attach DB metrics when the user explicitly asked for them.
        if re.search(r"\b(stored\s+metric|mysql\s+metric|database\s+metric)\b", lower):
            metrics, c2 = await _ainvoke_tool(
                tools,
                ["mysql_latest_metrics"],
                bypass_cache=bypass,
                model_version="best_v1",
            )
            sections.append(format_metrics(metrics, from_cache=c2))
            if c2:
                notes.append("sql_cache_hit")

        text = assemble_final_answer(sections=sections)
        return {
            "messages": [AIMessage(content=text)],
            "visited": ["sql_agent"],
            "cache_notes": notes,
            "final_text": text,
        }

    async def ml_node(state: GraphState) -> dict:
        tools = buckets["ml_agent"]
        bypass = bool(state.get("bypass_cache"))
        user = _user_text(state.get("messages") or [])
        lower = user.lower()
        sections: list[str] = []
        notes: list[str] = []
        predict_only = is_predict_only_query(user)
        snippets = _extract_predict_snippets(user)
        wants_metrics = is_model_metrics_query(user)
        wants_artifacts = any(k in lower for k in ("artifact", "files", "joblib", "what is saved"))
        wants_fred = any(k in lower for k in ("fred", "unemployment", "cpi", "macro", "economic"))

        # Classify path only for real predict/classify intent (never for model-QA wording).
        if (predict_only or snippets) and not (wants_metrics and not snippets):
            if not snippets:
                sections.append(
                    'Tell me the text to classify, e.g. Predict sentiment for "I love this product".'
                )
            else:
                pred, cp = await _ainvoke_ml_predict(
                    tools,
                    snippets,
                    model_version="best_v1",
                    bypass_cache=bypass,
                )
                sections.append(format_prediction(pred))
                if cp:
                    notes.append("ml_predict_cache_hit")

            if predict_only or (snippets and not wants_metrics):
                text = assemble_final_answer(sections=sections)
                return {
                    "messages": [AIMessage(content=text)],
                    "visited": ["ml_agent"],
                    "cache_notes": notes,
                    "final_text": text,
                }

        if wants_artifacts and not wants_metrics:
            arts, c0 = await _ainvoke_tool(
                tools, ["ml_list_artifacts"], bypass_cache=bypass
            )
            sections.append(format_query_rows(arts, title="Phase 2 artifacts", from_cache=c0))
            if c0:
                notes.append("ml_cache_hit")
            text = assemble_final_answer(sections=sections)
            return {
                "messages": [AIMessage(content=text)],
                "visited": ["ml_agent"],
                "cache_notes": notes,
                "final_text": text,
            }

        if wants_fred and not wants_metrics and not predict_only:
            series, c0 = await _ainvoke_tool(
                tools,
                ["fred_get_series"],
                bypass_cache=bypass,
                series_id="UNRATE",
                limit=6,
            )
            sections.append(
                format_query_rows(series, title="FRED series (UNRATE)", from_cache=c0)
            )
            if c0:
                notes.append("ml_cache_hit")
            text = assemble_final_answer(sections=sections)
            return {
                "messages": [AIMessage(content=text)],
                "visited": ["ml_agent"],
                "cache_notes": notes,
                "final_text": text,
            }

        # Default ML path: comparison + best metrics (only for model/metrics questions
        # or unmatched ML routing).
        comparison, c1 = await _ainvoke_tool(
            tools, ["ml_get_comparison"], bypass_cache=bypass
        )
        best, c2 = await _ainvoke_tool(
            tools, ["ml_get_best_metrics"], bypass_cache=bypass
        )
        sections.append(format_comparison(comparison, from_cache=c1))
        sections.append(
            format_metrics(best, title="Best model artifact highlights", from_cache=c2)
        )
        if c1 or c2:
            notes.append("ml_cache_hit")
        text = assemble_final_answer(sections=sections)
        return {
            "messages": [AIMessage(content=text)],
            "visited": ["ml_agent"],
            "cache_notes": notes,
            "final_text": text,
        }

    async def email_node(state: GraphState) -> dict:
        tools = buckets["email_agent"]
        user = _user_text(state.get("messages") or [])
        bypass = bool(state.get("bypass_cache"))
        findings = _prior_findings(state.get("messages") or [])
        if not _findings_are_usable(findings):
            findings = await _gather_alert_facts(buckets, user, bypass_cache=bypass)
        body = await _compose_alert_body(user, findings, channel="email")

        settings = get_email_settings()
        to_addr = settings.get("default_to") or ""
        subject = _alert_subject(user)
        result, _ = await _ainvoke_tool(
            tools,
            ["email_draft", "email_draft_alert"],
            bypass_cache=True,
            to=to_addr,
            subject=subject,
            body=body,
        )
        draft = _extract_draft_dict(result)
        if not draft:
            draft = {
                "to": to_addr,
                "subject": subject,
                "body": body,
                "channel": "email",
                "status": "draft",
            }
        else:
            draft["subject"] = draft.get("subject") or subject
            draft["body"] = body
        status = STATUS_DRAFT_READY if hitl else STATUS_DRAFT_ONLY
        text = format_email_draft(draft, status_line=status)
        if wants_immediate_send(user):
            text = assemble_final_answer(
                sections=[STATUS_CANNOT_SEND_WITHOUT_APPROVE, text]
            )
        return {
            "messages": [AIMessage(content=text)],
            "visited": ["email_agent"],
            "pending_email": draft,
            "final_text": text,
        }

    async def whatsapp_node(state: GraphState) -> dict:
        tools = buckets["whatsapp_agent"]
        user = _user_text(state.get("messages") or [])
        bypass = bool(state.get("bypass_cache"))
        findings = _prior_findings(state.get("messages") or [])
        if not _findings_are_usable(findings):
            findings = await _gather_alert_facts(buckets, user, bypass_cache=bypass)
        body = await _compose_alert_body(user, findings, channel="whatsapp")
        settings = get_twilio_settings()
        to_addr = settings.get("default_to") or ""
        result, _ = await _ainvoke_tool(
            tools,
            ["whatsapp_draft", "whatsapp_draft_alert"],
            bypass_cache=True,
            to=to_addr,
            body=body,
        )
        draft = _extract_draft_dict(result)
        if not draft:
            draft = {
                "to": to_addr,
                "body": body,
                "channel": "whatsapp",
                "status": "draft",
            }
        else:
            draft["body"] = body
        status = STATUS_DRAFT_READY if hitl else STATUS_DRAFT_ONLY
        text = format_whatsapp_draft(draft, status_line=status)
        if wants_immediate_send(user):
            text = assemble_final_answer(
                sections=[STATUS_CANNOT_SEND_WITHOUT_APPROVE, text]
            )
        return {
            "messages": [AIMessage(content=text)],
            "visited": ["whatsapp_agent"],
            "pending_whatsapp": draft,
            "final_text": text,
        }

    def human_review_node(state: GraphState) -> dict:
        payload = {
            "pending_email": state.get("pending_email") or {},
            "pending_whatsapp": state.get("pending_whatsapp") or {},
            "message": STATUS_DRAFT_READY,
        }
        resume_value = interrupt(payload)
        decision = "reject"
        email = dict(state.get("pending_email") or {})
        whatsapp = dict(state.get("pending_whatsapp") or {})
        if isinstance(resume_value, dict):
            decision = str(resume_value.get("decision") or "reject").strip().lower()
            if isinstance(resume_value.get("email"), dict):
                email.update(resume_value["email"])
            if isinstance(resume_value.get("whatsapp"), dict):
                whatsapp.update(resume_value["whatsapp"])
        elif isinstance(resume_value, str):
            decision = resume_value.strip().lower()

        if decision not in {"approve", "reject"}:
            decision = "reject"

        status = STATUS_DRAFT_READY
        if decision == "reject":
            status = STATUS_REJECTED
        elif decision == "approve" and not alerts_enabled():
            status = STATUS_APPROVED_DRAFT_ONLY

        return {
            "hitl_decision": decision,
            "pending_email": email,
            "pending_whatsapp": whatsapp,
            "messages": [
                AIMessage(content=f"[human_review] decision={decision}\n{status}")
            ],
            "final_text": status,
            "next": "send_alerts" if decision == "approve" else "FINISH",
        }

    async def send_alerts_node(state: GraphState) -> dict:
        decision = (state.get("hitl_decision") or "").lower()
        if decision != "approve":
            text = STATUS_REJECTED
            return {
                "messages": [AIMessage(content=text)],
                "visited": ["send_alerts"],
                "final_text": text,
                "next": "FINISH",
            }

        sections: list[str] = []
        if not alerts_enabled():
            text = STATUS_APPROVED_DRAFT_ONLY
            return {
                "messages": [AIMessage(content=text)],
                "visited": ["send_alerts"],
                "final_text": text,
                "next": "FINISH",
            }

        email = state.get("pending_email") or {}
        whatsapp = state.get("pending_whatsapp") or {}
        email_ok = False
        wa_ok = False

        if email.get("body") or email.get("subject"):
            tools = buckets["email_agent"]
            result, _ = await _ainvoke_tool(
                tools,
                ["email_send", "email_send_alert"],
                bypass_cache=True,
                to=email.get("to") or "",
                subject=email.get("subject") or "Brand Sentiment Alert",
                body=email.get("body") or "",
            )
            data = unwrap_tool_payload(result)
            if not isinstance(data, dict):
                data = {}
            mode = str(data.get("mode") or "")
            err = str(data.get("error") or "")
            if mode == "sent":
                email_ok = True
                sections.append("Email send: success.")
            elif err:
                sections.append(f"Email send failed: {err}")
            elif mode == "draft" or data.get("ok") is True:
                msg = data.get("message") or (
                    "Email logged as draft only. "
                    "Set ALERTS_ENABLED=true and use a Gmail App Password in SMTP_PASSWORD."
                )
                sections.append(str(msg))
            else:
                sections.append(
                    "Email send did not complete. "
                    "Check SMTP_USER / SMTP_PASSWORD (Gmail App Password) and ALERT_EMAIL_TO."
                )

        if whatsapp.get("body"):
            tools = buckets["whatsapp_agent"]
            result, _ = await _ainvoke_tool(
                tools,
                ["whatsapp_send", "whatsapp_send_alert"],
                bypass_cache=True,
                to=whatsapp.get("to") or "",
                body=whatsapp.get("body") or "",
            )
            data = unwrap_tool_payload(result)
            if not isinstance(data, dict):
                data = {}
            mode = str(data.get("mode") or "")
            err = str(data.get("error") or "")
            if mode == "sent":
                wa_ok = True
                sections.append("WhatsApp send: success.")
            elif err:
                sections.append(f"WhatsApp send failed: {err}")
            else:
                sections.append(
                    "WhatsApp: logged as draft (send not completed). "
                    "Check Twilio settings and ALERTS_ENABLED."
                )

        if email_ok and (wa_ok or not whatsapp.get("body")):
            status = STATUS_APPROVED_SENT
        elif wa_ok and not (email.get("body") or email.get("subject")):
            status = STATUS_APPROVED_SENT
        elif email_ok or wa_ok:
            status = "Partially sent — see details above."
        else:
            status = STATUS_APPROVED_DRAFT_ONLY
        text = assemble_final_answer(sections=sections, status_line=status)
        return {
            "messages": [AIMessage(content=text)],
            "visited": ["send_alerts"],
            "final_text": text,
            "next": "FINISH",
        }

    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("sql_agent", sql_node)
    graph.add_node("ml_agent", ml_node)
    graph.add_node("email_agent", email_node)
    graph.add_node("whatsapp_agent", whatsapp_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("send_alerts", send_alerts_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route,
        {
            "sql_agent": "sql_agent",
            "ml_agent": "ml_agent",
            "email_agent": "email_agent",
            "whatsapp_agent": "whatsapp_agent",
            "human_review": "human_review",
            "send_alerts": "send_alerts",
            "__end__": END,
        },
    )
    for agent_name in WORKERS:
        graph.add_edge(agent_name, "supervisor")
    # After human_review, supervisor decides send vs finish
    graph.add_edge("human_review", "supervisor")
    graph.add_edge("send_alerts", "supervisor")

    if checkpointer is None and hitl:
        checkpointer = MemorySaver()
    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


def _format_answer_from_state(result: dict[str, Any]) -> str:
    messages = result.get("messages") or []
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, AIMessage) or not msg.content:
            continue
        text = str(msg.content).strip()
        if text.startswith("[supervisor]->"):
            continue
        if text.startswith("[human_review]"):
            lines = text.splitlines()
            parts.append(lines[-1] if lines else text)
            continue
        parts.append(text)

    drafts = [
        p
        for p in parts
        if p.startswith("Email draft:") or p.startswith("WhatsApp draft:")
    ]
    if drafts:
        facts = [p for p in parts if p not in drafts]
        return assemble_final_answer(sections=[*facts[-2:], drafts[-1]])

    if result.get("final_text"):
        return str(result["final_text"])
    if not parts:
        return "(no agent output)"
    return "\n\n".join(parts[-6:])


def _initial_state(user_text: str) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=user_text)],
        "next": "",
        "visited": [],
        "pending_email": {},
        "pending_whatsapp": {},
        "hitl_decision": "",
        "bypass_cache": user_requests_refresh(user_text),
        "final_text": "",
        "cache_notes": [],
    }


async def run_query_via_mcp(user_text: str, *, hitl: bool = False) -> str:
    """Spawn Unified MCP, attach tools, run query (non-HITL by default for CLI)."""
    from phase3.agents.mcp_client import load_mcp_tools, partition_tools

    tools = await load_mcp_tools()
    buckets = partition_tools(tools)
    app = build_graph(tool_buckets=buckets, hitl=hitl)
    try:
        result = await app.ainvoke(_initial_state(user_text), {"recursion_limit": 24})
    except LLMOutageError as exc:
        return outage_plain_text(exc)
    return _format_answer_from_state(result)


def run_query(user_text: str, *, use_mcp: bool = True, hitl: bool = False) -> str:
    """Sync entrypoint. Default: tools from Unified MCP client, no HITL interrupt."""
    import asyncio

    if use_mcp:
        return asyncio.run(run_query_via_mcp(user_text, hitl=hitl))

    async def _local() -> str:
        logger.warning("Running with local LangChain tools (no MCP client)")
        app = build_graph(tool_buckets=None, hitl=hitl)
        try:
            result = await app.ainvoke(_initial_state(user_text), {"recursion_limit": 24})
        except LLMOutageError as exc:
            return outage_plain_text(exc)
        return _format_answer_from_state(result)

    return asyncio.run(_local())


# Re-export for compatibility
__all__ = [
    "GraphState",
    "WORKERS",
    "build_graph",
    "build_llm",
    "is_identity_query",
    "plan_from_query",
    "parse_worker_choice",
    "run_query",
    "run_query_via_mcp",
    "wants_immediate_send",
]
