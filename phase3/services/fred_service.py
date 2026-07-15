"""FRED economic data enrichment (optional tool on the Unified MCP)."""

from __future__ import annotations

from typing import Any

import httpx

from phase3.config import get_fred_api_key

FRED_BASE = "https://api.stlouisfed.org/fred"


def fred_series_observations(
    series_id: str = "UNRATE",
    *,
    limit: int = 12,
    sort_order: str = "desc",
) -> dict[str, Any]:
    """Fetch recent observations for a FRED series (e.g. UNRATE, CPIAUCSL, GDPC1)."""
    api_key = get_fred_api_key()
    if not api_key:
        return {
            "ok": False,
            "error": "FRED_API_KEY not set in .env — get a free key at https://fred.stlouisfed.org/docs/api/api_key.html",
        }
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": sort_order,
        "limit": limit,
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{FRED_BASE}/series/observations", params=params)
            resp.raise_for_status()
            data = resp.json()
        obs = data.get("observations", [])
        return {
            "ok": True,
            "series_id": series_id,
            "observations": [
                {"date": o.get("date"), "value": o.get("value")} for o in obs
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def fred_search(search_text: str, *, limit: int = 5) -> dict[str, Any]:
    api_key = get_fred_api_key()
    if not api_key:
        return {
            "ok": False,
            "error": "FRED_API_KEY not set in .env",
        }
    params = {
        "search_text": search_text,
        "api_key": api_key,
        "file_type": "json",
        "limit": limit,
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{FRED_BASE}/series/search", params=params)
            resp.raise_for_status()
            data = resp.json()
        series = data.get("seriess", [])
        return {
            "ok": True,
            "results": [
                {
                    "id": s.get("id"),
                    "title": s.get("title"),
                    "units": s.get("units"),
                    "frequency": s.get("frequency"),
                }
                for s in series
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
