"""MCP client that starts the Unified MCP server and loads LangChain tools."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from phase1.src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

SERVER_NAME = "unified"


def unified_mcp_connection() -> dict[str, Any]:
    """Stdio connection that spawns `python -m phase3.mcp.server`."""
    # Pass through current env so MySQL / FRED / SMTP keys reach the MCP child.
    env = {k: str(v) for k, v in os.environ.items()}
    return {
        SERVER_NAME: {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "phase3.mcp.server"],
            "cwd": str(PROJECT_ROOT),
            "env": env,
        }
    }


def create_mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(unified_mcp_connection())


async def load_mcp_tools(client: MultiServerMCPClient | None = None) -> list[BaseTool]:
    """Start Unified MCP (stdio) and return all tools as LangChain tools."""
    own_client = client is None
    if own_client:
        client = create_mcp_client()
    assert client is not None
    tools = await client.get_tools(server_name=SERVER_NAME)
    names = sorted(t.name for t in tools)
    logger.info("Loaded %s tools from Unified MCP: %s", len(tools), names)
    return tools


def partition_tools(tools: list[BaseTool]) -> dict[str, list[BaseTool]]:
    """Split MCP tools among star-topology specialist agents."""
    buckets: dict[str, list[BaseTool]] = {
        "sql_agent": [],
        "ml_agent": [],
        "email_agent": [],
        "whatsapp_agent": [],
    }
    for tool in tools:
        name = tool.name.lower()
        if name.startswith("mysql_"):
            buckets["sql_agent"].append(tool)
        elif name.startswith("ml_") or name.startswith("fred_"):
            buckets["ml_agent"].append(tool)
        elif name.startswith("email_"):
            buckets["email_agent"].append(tool)
        elif name.startswith("whatsapp_"):
            buckets["whatsapp_agent"].append(tool)
        else:
            logger.warning("Unassigned MCP tool (given to ml_agent): %s", tool.name)
            buckets["ml_agent"].append(tool)
    return buckets
