"""Evaluate models on the test set and write model_metrics to MySQL."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine, text

from phase1.src.config import get_database_url
from phase2.src.data import ARTIFACTS_DIR, SplitData
from phase2.src.models import BEST_VERSION

logger = logging.getLogger(__name__)


def compute_test_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def save_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str],
    out_path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved confusion matrix -> %s", out_path)


def write_metrics_rows(
    model_version: str,
    metrics: dict[str, float],
    *,
    engine=None,
) -> None:
    own_engine = engine is None
    if own_engine:
        engine = create_engine(get_database_url(), pool_pre_ping=True)
    rows = [
        {"model_version": model_version, "metric_name": k, "metric_value": float(v)}
        for k, v in metrics.items()
    ]
    try:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM model_metrics WHERE model_version = :v"),
                {"v": model_version},
            )
            conn.execute(
                text(
                    """
                    INSERT INTO model_metrics (model_version, metric_name, metric_value)
                    VALUES (:model_version, :metric_name, :metric_value)
                    """
                ),
                rows,
            )
        logger.info("Wrote %s metrics for %s", len(rows), model_version)
    finally:
        if own_engine:
            engine.dispose()


def evaluate_all(
    pipelines: dict[str, Pipeline],
    split: SplitData,
    *,
    best_source_version: str,
    val_metrics_by_model: dict[str, dict[str, float]],
    run_id: str = "v1",
) -> dict[str, Any]:
    """Evaluate each model on test; write JSON + MySQL metrics."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = sorted(split.y_train.unique().tolist())
    summary: dict[str, Any] = {
        "run_id": run_id,
        "best_source_version": best_source_version,
        "models": {},
    }

    engine = create_engine(get_database_url(), pool_pre_ping=True)
    try:
        versions = list(pipelines.keys())
        for version in versions:
            pipe = pipelines[version]
            y_pred = pipe.predict(split.X_test)
            test_metrics = compute_test_metrics(split.y_test, y_pred)

            # Resolve validation metrics for best alias from the winning source model
            source_for_val = (
                best_source_version if version == BEST_VERSION else version
            )
            val_block = val_metrics_by_model.get(source_for_val, {})

            combined = {
                **test_metrics,
                "val_f1_macro": float(val_block.get("val_f1_macro", 0.0)),
                "val_accuracy": float(val_block.get("val_accuracy", 0.0)),
                "n_train": float(split.n_train),
                "n_val": float(split.n_val),
                "n_test": float(split.n_test),
                "is_best": 1.0
                if version == BEST_VERSION or version == best_source_version
                else 0.0,
            }

            report = classification_report(
                split.y_test, y_pred, zero_division=0, output_dict=True
            )
            summary["models"][version] = {
                "metrics": combined,
                "classification_report": report,
            }

            cm_path = ARTIFACTS_DIR / f"confusion_{version}.png"
            save_confusion_matrix(
                split.y_test,
                y_pred,
                labels,
                cm_path,
                title=f"Confusion — {version}",
            )

            metrics_path = ARTIFACTS_DIR / f"metrics_{version}.json"
            metrics_path.write_text(
                json.dumps(summary["models"][version], indent=2),
                encoding="utf-8",
            )

            write_metrics_rows(version, combined, engine=engine)
            logger.info(
                "%s test_acc=%.4f test_f1_macro=%.4f",
                version,
                combined["accuracy"],
                combined["f1_macro"],
            )
    finally:
        engine.dispose()

    summary_path = ARTIFACTS_DIR / f"eval_summary_{run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote eval summary -> %s", summary_path)
    return summary
