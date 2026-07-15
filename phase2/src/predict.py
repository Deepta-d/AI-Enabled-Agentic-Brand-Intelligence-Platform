"""Score posts and write model_predictions to MySQL."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.pipeline import Pipeline

from phase1.src.config import get_database_url

logger = logging.getLogger(__name__)


def _confidence_and_labels(pipe: Pipeline, texts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    labels = pipe.predict(texts)
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(texts)
        confidence = proba.max(axis=1)
    else:
        confidence = np.full(len(labels), np.nan)
    return labels, confidence


def write_predictions(
    pipe: Pipeline,
    posts: pd.DataFrame,
    model_version: str,
    *,
    engine=None,
) -> int:
    """Replace predictions for model_version and insert new rows. Returns row count."""
    own_engine = engine is None
    if own_engine:
        engine = create_engine(get_database_url(), pool_pre_ping=True)

    labels, confidence = _confidence_and_labels(pipe, posts["text"])
    rows = [
        {
            "post_id": int(post_id),
            "model_version": model_version,
            "predicted_sentiment": str(pred),
            "confidence": None if np.isnan(conf) else float(round(float(conf), 4)),
        }
        for post_id, pred, conf in zip(posts["id"], labels, confidence, strict=True)
    ]

    try:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM model_predictions WHERE model_version = :v"),
                {"v": model_version},
            )
            conn.execute(
                text(
                    """
                    INSERT INTO model_predictions
                      (post_id, model_version, predicted_sentiment, confidence)
                    VALUES
                      (:post_id, :model_version, :predicted_sentiment, :confidence)
                    """
                ),
                rows,
            )
            count = conn.execute(
                text(
                    "SELECT COUNT(*) FROM model_predictions WHERE model_version = :v"
                ),
                {"v": model_version},
            ).scalar()
        logger.info("Wrote %s predictions for %s", count, model_version)
        return int(count or 0)
    finally:
        if own_engine:
            engine.dispose()


def write_all_predictions(
    pipelines: dict[str, Pipeline],
    posts: pd.DataFrame,
    model_versions: Iterable[str],
) -> dict[str, int]:
    engine = create_engine(get_database_url(), pool_pre_ping=True)
    counts: dict[str, int] = {}
    try:
        for version in model_versions:
            counts[version] = write_predictions(
                pipelines[version], posts, version, engine=engine
            )
    finally:
        engine.dispose()
    return counts
