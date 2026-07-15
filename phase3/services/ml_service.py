"""ML artifact helpers for Unified MCP / ML Agent."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

# Avoid loky/multiprocessing deadlocks under MCP stdio on Windows.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

import joblib

from phase3.config import (
    ARTIFACTS_DIR,
    BEST_MODEL_PATH,
    COMPARISON_PATH,
    METRICS_BEST_PATH,
)

_MODEL_CACHE: dict[str, Any] = {}


def _load_pipeline(model_version: str) -> tuple[Any, Path] | tuple[None, None]:
    model_path = ARTIFACTS_DIR / f"{model_version}.joblib"
    if not model_path.exists():
        model_path = BEST_MODEL_PATH
    if not model_path.exists():
        return None, None
    key = str(model_path.resolve())
    if key not in _MODEL_CACHE:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _MODEL_CACHE[key] = joblib.load(model_path)
    return _MODEL_CACHE[key], model_path


def get_model_comparison() -> dict[str, Any]:
    if not COMPARISON_PATH.exists():
        return {"ok": False, "error": f"Missing {COMPARISON_PATH}. Run Phase 2 first."}
    return {"ok": True, "data": json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))}


def get_best_metrics() -> dict[str, Any]:
    path = METRICS_BEST_PATH
    if not path.exists():
        # fall back to winner name from comparison
        comparison = get_model_comparison()
        if not comparison.get("ok"):
            return comparison
        winner = comparison["data"].get("best_model_version", "best_v1")
        alt = ARTIFACTS_DIR / f"metrics_{winner}.json"
        if alt.exists():
            return {"ok": True, "data": json.loads(alt.read_text(encoding="utf-8"))}
        path = ARTIFACTS_DIR / "metrics_best_v1.json"
    if not path.exists():
        return {"ok": False, "error": "No metrics JSON found. Run Phase 2 pipeline."}
    return {"ok": True, "data": json.loads(path.read_text(encoding="utf-8"))}


def predict_texts(texts: list[str], model_version: str = "best_v1") -> dict[str, Any]:
    pipe, model_path = _load_pipeline(model_version)
    if pipe is None or model_path is None:
        return {"ok": False, "error": f"Model not found: {model_version}"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = pipe.predict(texts)
        confidences: list[float | None] = []
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(texts)
            confidences = [float(round(float(row.max()), 4)) for row in proba]
        else:
            confidences = [None] * len(texts)
    return {
        "ok": True,
        "model_version": model_path.stem,
        "predictions": [
            {"text": t, "predicted_sentiment": str(p), "confidence": c}
            for t, p, c in zip(texts, preds, confidences, strict=True)
        ],
    }


def list_artifacts() -> dict[str, Any]:
    if not ARTIFACTS_DIR.exists():
        return {"ok": False, "error": "artifacts dir missing"}
    files = sorted(p.name for p in ARTIFACTS_DIR.iterdir() if p.is_file())
    return {"ok": True, "files": files}
