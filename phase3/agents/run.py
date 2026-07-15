"""Phase 3 agent runner (no Streamlit).

Usage:
    python -m phase3.agents.run --demo
    python -m phase3.agents.run --query "Summarize sentiment and draft an email alert"
    python -m phase3.agents.run --query "..." --no-mcp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3.config import alerts_enabled, get_llm_provider
from phase3.services import email_service, ml_service, mysql_service, whatsapp_service

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_demo() -> int:
    """Deterministic star-topology workflow without an LLM (services only)."""
    print("=== Phase 3 DEMO (no LLM, no Streamlit, no MCP) ===")
    print(f"ALERTS_ENABLED={alerts_enabled()}")

    print("\n[SQL Agent] sentiment summary")
    summary = mysql_service.sentiment_summary()
    print(json.dumps(summary, indent=2, default=str))

    print("\n[SQL Agent] best_v1 metrics")
    metrics = mysql_service.latest_metrics("best_v1")
    print(json.dumps(metrics, indent=2, default=str))

    print("\n[ML Agent] model comparison / winner")
    comparison = ml_service.get_model_comparison()
    print(json.dumps(comparison, indent=2, default=str)[:2000])

    print("\n[ML Agent] sample prediction")
    pred = ml_service.predict_texts(
        ["The product launch was amazing!", "Terrible customer support today."]
    )
    print(json.dumps(pred, indent=2, default=str))

    neg = summary.get("rows", [])
    neg_n = next((r["n"] for r in neg if r.get("sentiment_group") == "Negative"), 0)
    body = (
        f"Brand intelligence digest\n"
        f"- Negative posts in DB: {neg_n}\n"
        f"- Best model: {comparison.get('data', {}).get('best_model_version', 'n/a')}\n"
        f"- Sample prediction: {pred.get('predictions', [])}\n"
        f"(Draft only unless ALERTS_ENABLED=true)"
    )

    print("\n[Email Agent] draft alert")
    email = email_service.send_email(
        to="",
        subject="Sentiment Brand Alert (Phase 3 demo)",
        body=body,
    )
    print(json.dumps(email, indent=2, default=str))

    print("\n[WhatsApp Agent] draft alert")
    wa = whatsapp_service.send_whatsapp(to="", body=body[:500])
    print(json.dumps(wa, indent=2, default=str))

    print("\nDemo complete. Drafts are under phase3/logs/ when ALERTS_ENABLED=false.")
    return 0


async def _probe_mcp() -> list[str]:
    from phase3.agents.mcp_client import load_mcp_tools, partition_tools

    tools = await load_mcp_tools()
    buckets = partition_tools(tools)
    print("=== MCP client probe (Unified MCP) ===")
    for agent, bucket in buckets.items():
        names = [t.name for t in bucket]
        print(f"  {agent}: {names}")
    return [t.name for t in tools]


def run_agent_query(query: str, *, use_mcp: bool = True) -> int:
    from phase3.agents.graph import run_query

    print(f"LLM provider: {get_llm_provider()}")
    print(f"Tools source: {'Unified MCP client' if use_mcp else 'local LangChain tools'}")
    print(f"Query: {query}\n")
    try:
        answer = run_query(query, use_mcp=use_mcp)
    except Exception as exc:  # noqa: BLE001
        logger.error("Agent run failed: %s", exc)
        err = str(exc).lower()
        if "10061" in err or "actively refused" in err or "connection refused" in err:
            if get_llm_provider() == "ollama":
                logger.error(
                    "Ollama is not reachable. Start it (e.g. open the Ollama app, or "
                    "run: ollama serve), then retry. Check OLLAMA_BASE_URL in .env."
                )
            else:
                logger.error("A local service refused the connection — check MySQL/LLM endpoints.")
        elif use_mcp:
            logger.error(
                "Tip: ensure `python -m phase3.mcp.server` imports cleanly, "
                "or retry with --no-mcp for local tools."
            )
        return 1
    print(answer)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3 LangGraph agents (no Streamlit)")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run deterministic multi-agent service workflow without an LLM/MCP",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Natural-language request for the LangGraph supervisor (uses MCP tools by default)",
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="Use local LangChain tools instead of spawning Unified MCP",
    )
    parser.add_argument(
        "--probe-mcp",
        action="store_true",
        help="Start Unified MCP via client and list tools (no LLM)",
    )
    args = parser.parse_args(argv)

    if args.probe_mcp:
        try:
            names = asyncio.run(_probe_mcp())
            print(f"OK — {len(names)} tools from MCP client")
            return 0
        except Exception as exc:  # noqa: BLE001
            logger.error("MCP probe failed: %s", exc)
            return 1

    if args.demo:
        return run_demo()
    if not args.query:
        parser.error("Provide --query, --demo, or --probe-mcp")
    return run_agent_query(args.query, use_mcp=not args.no_mcp)


if __name__ == "__main__":
    raise SystemExit(main())
