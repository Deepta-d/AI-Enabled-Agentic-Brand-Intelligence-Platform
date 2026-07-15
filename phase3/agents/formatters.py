"""Plain-text formatters for Brand Intelligence Assistant responses."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

USER_SAFE_QUERY_ERROR = (
    "I couldn't load that table sample right now. "
    "Try asking for one table at a time (social_posts, model_metrics, or model_predictions)."
)


def unwrap_tool_payload(payload: str) -> Any:
    """Flatten MCP content-block wrappers into Python objects or text."""
    try:
        data = json.loads(payload)
    except Exception:  # noqa: BLE001
        return payload
    if isinstance(data, list) and data and isinstance(data[0], dict) and "text" in data[0]:
        texts = [str(item.get("text", "")) for item in data if isinstance(item, dict)]
        joined = "\n".join(t for t in texts if t)
        try:
            return json.loads(joined)
        except Exception:  # noqa: BLE001
            return joined
    return data


def sanitize_user_facing_error(error: Any) -> str:
    """Map driver/SQL dumps to a short plain message for end users."""
    text = str(error or "").strip()
    if text:
        logger.debug("Hiding query error from UI: %s", text[:500])
    return USER_SAFE_QUERY_ERROR


def _fmt_cell(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def format_post_lookup(
    payload: str | dict,
    *,
    title: str = "Matching post in the dataset",
    from_cache: bool = False,
) -> str:
    """Render a text→post lookup (platform, username, etc.)."""
    data = unwrap_tool_payload(payload) if isinstance(payload, str) else payload
    if not isinstance(data, dict):
        return f"{title}:\n- (unexpected result)"
    if data.get("ok") is False:
        err = data.get("error") or "query failed"
        logger.warning("Post lookup failed: %s", err)
        return f"{title}:\n- {sanitize_user_facing_error(err)}"

    rows = data.get("rows") or []
    if not rows:
        return (
            f"{title}:\n"
            "- No matching post found in social_posts for that text.\n"
            "- Tip: the model can still classify text that is not in the dataset."
        )

    lines = [f"{title}:"]
    for row in rows[:5]:
        if not isinstance(row, dict):
            lines.append(f"- {row}")
            continue
        platform = row.get("platform") or "(unknown)"
        username = row.get("username") or "(unknown)"
        country = row.get("country") or "(unknown)"
        sent = row.get("sentiment_group") or "?"
        preview = row.get("text_preview") or ""
        post_id = row.get("id", "?")
        lines.append(
            f"- Platform: {platform} (username {username}, country {country}, "
            f"sentiment {sent}, id {post_id})"
        )
        if preview:
            short = preview if len(str(preview)) <= 100 else str(preview)[:97] + "..."
            lines.append(f"  text: {short}")

    if len(rows) == 1 and isinstance(rows[0], dict):
        # Lead with a direct one-liner when there is a single match.
        platform = rows[0].get("platform") or "(unknown)"
        lines.insert(1, f"- That text appears on {platform} in the dataset.")

    if from_cache:
        lines.append("(From latest cached lookup.)")
    return "\n".join(lines)


def format_query_rows(
    payload: str | dict,
    *,
    title: str,
    from_cache: bool = False,
    limit: int = 40,
) -> str:
    """Render a generic mysql_query / SELECT result as plain bullets."""
    data = unwrap_tool_payload(payload) if isinstance(payload, str) else payload
    if not isinstance(data, dict):
        return f"{title}:\n- (unexpected result)"
    if data.get("ok") is False:
        err = data.get("error") or "query failed"
        logger.warning("Query failed for %r: %s", title, err)
        return f"{title}:\n- {sanitize_user_facing_error(err)}"

    if isinstance(data.get("files"), list):
        lines = [f"{title}:"]
        files = [str(f) for f in data["files"][:limit]]
        if not files:
            lines.append("- (none)")
        else:
            for name in files:
                lines.append(f"- {name}")
            if len(data["files"]) > limit:
                lines.append(f"- … and {len(data['files']) - limit} more")
        if from_cache:
            lines.append("(From latest cached lookup.)")
        return "\n".join(lines)

    rows = data.get("rows") or []
    columns = data.get("columns") or (list(rows[0].keys()) if rows and isinstance(rows[0], dict) else [])
    lines = [f"{title}:"]
    if not rows:
        lines.append("- (no rows)")
        if from_cache:
            lines.append("(From latest cached lookup.)")
        return "\n".join(lines)

    # Single-row aggregate: print as key/value bullets
    if len(rows) == 1 and isinstance(rows[0], dict) and len(rows[0]) <= 8:
        for key, value in rows[0].items():
            label = str(key).replace("_", " ")
            lines.append(f"- {label}: {_fmt_cell(value)}")
    else:
        for row in rows[:limit]:
            if not isinstance(row, dict):
                lines.append(f"- {row}")
                continue
            # Prefer first two columns as "name: count" when present
            cols = columns or list(row.keys())
            if len(cols) >= 2 and str(cols[-1]).lower() in {"n", "count", "total", "posts"}:
                name = row.get(cols[0], "?")
                n = row.get(cols[-1], "?")
                lines.append(f"- {name}: {_fmt_cell(n)}")
            else:
                parts = [f"{k}={_fmt_cell(row.get(k))}" for k in cols[:4]]
                lines.append(f"- {', '.join(parts)}")
        if len(rows) > limit:
            lines.append(f"- … and {len(rows) - limit} more")

    total = data.get("row_count")
    if isinstance(total, int) and total > 1 and len(rows) > 1:
        lines.insert(1, f"- Rows shown: {min(len(rows), limit)} of {total}")
    if from_cache:
        lines.append("(From latest cached lookup.)")
    return "\n".join(lines)


def format_sentiment_summary(payload: str | dict, *, from_cache: bool = False) -> str:
    data = unwrap_tool_payload(payload) if isinstance(payload, str) else payload
    rows = []
    if isinstance(data, dict):
        rows = data.get("rows") or []
    lines = ["Sentiment in the database:"]
    for row in rows:
        if not isinstance(row, dict):
            continue
        group = row.get("sentiment_group", "Unknown")
        n = row.get("n", row.get("count", "?"))
        lines.append(f"- {group}: {n}")
    if len(lines) == 1:
        lines.append("- (no rows returned)")
    if from_cache:
        lines.append("(From latest cached lookup.)")
    return "\n".join(lines)


def format_metrics(payload: str | dict, *, title: str = "Best model (best_v1) highlights", from_cache: bool = False) -> str:
    data = unwrap_tool_payload(payload) if isinstance(payload, str) else payload
    lines = [f"{title}:"]
    rows = []
    metrics: dict[str, Any] = {}
    if isinstance(data, dict):
        if "rows" in data:
            rows = data.get("rows") or []
        elif "data" in data and isinstance(data["data"], dict):
            metrics = data["data"].get("metrics") or data["data"]
        elif "metrics" in data:
            metrics = data["metrics"] or {}

    interesting = (
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
        "val_f1_macro",
        "val_accuracy",
    )
    if rows:
        by_name = {
            str(r.get("metric_name")): r.get("metric_value")
            for r in rows
            if isinstance(r, dict)
        }
        for key in interesting:
            if key in by_name:
                label = key.replace("_", " ").title()
                lines.append(f"- {label}: {_fmt_float(by_name[key])}")
    elif metrics:
        for key in interesting:
            if key in metrics:
                label = key.replace("_", " ").title()
                lines.append(f"- {label}: {_fmt_float(metrics[key])}")
    if len(lines) == 1:
        lines.append("- (no metrics available)")
    if from_cache:
        lines.append("(From latest cached lookup.)")
    return "\n".join(lines)


def format_comparison(payload: str | dict, *, from_cache: bool = False) -> str:
    data = unwrap_tool_payload(payload) if isinstance(payload, str) else payload
    lines = ["Model comparison:"]
    block = data
    if isinstance(data, dict) and "data" in data:
        block = data["data"]
    if isinstance(block, dict):
        winner = block.get("best_model_version") or block.get("best_source_version")
        if winner:
            lines.append(f"- Validation winner: {winner}")
        models = block.get("models") or {}
        if isinstance(models, dict):
            for name, m in models.items():
                if not isinstance(m, dict):
                    continue
                f1 = m.get("val_f1_macro", m.get("f1_macro"))
                acc = m.get("val_accuracy", m.get("accuracy"))
                parts = [name]
                if acc is not None:
                    parts.append(f"acc={_fmt_float(acc)}")
                if f1 is not None:
                    parts.append(f"f1_macro={_fmt_float(f1)}")
                lines.append(f"- {', '.join(parts)}")
    if len(lines) == 1:
        lines.append("- (comparison unavailable)")
    if from_cache:
        lines.append("(From latest cached lookup.)")
    return "\n".join(lines)


def format_prediction(payload: str | dict) -> str:
    data = unwrap_tool_payload(payload) if isinstance(payload, str) else payload
    preds = []
    if isinstance(data, dict):
        preds = data.get("predictions") or []
        version = data.get("model_version", "best_v1")
    else:
        version = "best_v1"
    lines = []
    for item in preds:
        if not isinstance(item, dict):
            continue
        text = item.get("text", "")
        label = item.get("predicted_sentiment", "?")
        conf = item.get("confidence")
        conf_s = f" (confidence about {_fmt_float(conf)})" if conf is not None else ""
        snippet = text if len(str(text)) <= 80 else str(text)[:77] + "..."
        lines.append(f'Predicted sentiment for "{snippet}": {label}{conf_s}, using {version}.')
    if not lines:
        return "No prediction available."
    return "\n".join(lines)


def format_email_draft(draft: dict[str, Any], *, status_line: str | None = None) -> str:
    lines = [
        "Email draft:",
        f"To: {draft.get('to') or '(default alert recipient)'}",
        f"Subject: {draft.get('subject') or 'Brand Sentiment Alert'}",
        "Body:",
        str(draft.get("body") or "").strip() or "(empty)",
    ]
    if status_line:
        lines.append(status_line)
    return "\n".join(lines)


def format_whatsapp_draft(draft: dict[str, Any], *, status_line: str | None = None) -> str:
    lines = [
        "WhatsApp draft:",
        f"To: {draft.get('to') or '(default alert recipient)'}",
        "Body:",
        str(draft.get("body") or "").strip() or "(empty)",
    ]
    if status_line:
        lines.append(status_line)
    return "\n".join(lines)


def strip_technical_dump(text: str) -> str:
    """Remove obvious JSON/object dumps from a blob before display."""
    cleaned = text.strip()
    if cleaned.startswith("{") or cleaned.startswith("["):
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict) and "rows" in obj:
                return format_sentiment_summary(obj)
            return "Results are ready. See the summary above."
        except Exception:  # noqa: BLE001
            pass
    # Drop lines that look like raw MCP wrappers
    out_lines = []
    for line in cleaned.splitlines():
        if re.search(r'"type":\s*"text"', line):
            continue
        out_lines.append(line)
    return "\n".join(out_lines).strip()


def assemble_final_answer(
    *,
    sections: list[str],
    status_line: str | None = None,
) -> str:
    parts = [s.strip() for s in sections if s and s.strip()]
    if status_line:
        parts.append(status_line.strip())
    if not parts:
        return "No results to show."
    return "\n\n".join(parts)


STATUS_DRAFT_READY = "Draft ready - waiting for human approval."
STATUS_APPROVED_SENT = "Approved and sent."
STATUS_REJECTED = "Rejected - not sent."
STATUS_DRAFT_ONLY = "Draft logged only (alerts disabled)."
STATUS_APPROVED_DRAFT_ONLY = "Approved - draft logged only (alerts disabled)."
STATUS_CANNOT_SEND_WITHOUT_APPROVE = (
    "I can't send alerts without your Approve. "
    "I've prepared a draft from the latest database findings for you to review."
)
