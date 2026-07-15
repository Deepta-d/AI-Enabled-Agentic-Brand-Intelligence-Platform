"""Train the three Phase 2 models (optional hyperparameter tuning), select best on val."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from phase2.src.data import ARTIFACTS_DIR, SplitData
from phase2.src.models import (
    BEST_VERSION,
    MODEL_VERSIONS,
    PARAM_GRIDS,
    build_pipelines,
    build_search_pipelines,
    finalize_tuned_pipeline,
)

logger = logging.getLogger(__name__)


def _val_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "val_accuracy": float(accuracy_score(y_true, y_pred)),
        "val_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "val_f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def _jsonable_params(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def _tune_one(
    name: str,
    pipe: Pipeline,
    split: SplitData,
    *,
    cv_folds: int,
    n_jobs: int,
) -> tuple[Pipeline, dict[str, Any]]:
    grid = PARAM_GRIDS[name]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    search = GridSearchCV(
        pipe,
        param_grid=grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )
    logger.info("Tuning %s (%s grid combinations, cv=%s) ...", name, _grid_size(grid), cv_folds)
    search.fit(split.X_train, split.y_train)
    best_params = _jsonable_params(search.best_params_)
    logger.info(
        "%s best CV f1_macro=%.4f params=%s",
        name,
        float(search.best_score_),
        best_params,
    )
    tuned = finalize_tuned_pipeline(name, search.best_estimator_)
    # Calibrated LinearSVC needs a final fit on full train after wrapping
    if name == "linearsvc_v1":
        tuned.fit(split.X_train, split.y_train)
    meta = {
        "best_params": best_params,
        "cv_f1_macro": float(search.best_score_),
        "cv_folds": cv_folds,
    }
    return tuned, meta


def _grid_size(grid: dict[str, list[Any]]) -> int:
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def train_all(
    split: SplitData,
    *,
    run_id: str = "v1",
    tune: bool = True,
    cv_folds: int = 3,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Fit all models on train (optionally with GridSearchCV), rank by val f1_macro."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "run_id": run_id,
        "tuned": tune,
        "n_train": split.n_train,
        "n_val": split.n_val,
        "n_test": split.n_test,
        "models": {},
    }

    best_name: str | None = None
    best_f1 = -1.0
    fitted: dict[str, Pipeline] = {}

    if tune:
        search_pipes = build_search_pipelines()
        for name in MODEL_VERSIONS:
            pipe, tune_meta = _tune_one(
                name,
                search_pipes[name],
                split,
                cv_folds=cv_folds,
                n_jobs=n_jobs,
            )
            y_val_pred = pipe.predict(split.X_val)
            metrics = _val_metrics(split.y_val, y_val_pred)
            results["models"][name] = {**metrics, **tune_meta}
            fitted[name] = pipe
            joblib.dump(pipe, ARTIFACTS_DIR / f"{name}.joblib")
            logger.info(
                "%s val_acc=%.4f val_f1_macro=%.4f",
                name,
                metrics["val_accuracy"],
                metrics["val_f1_macro"],
            )
            if metrics["val_f1_macro"] > best_f1:
                best_f1 = metrics["val_f1_macro"]
                best_name = name
    else:
        pipelines = build_pipelines()
        for name, pipe in pipelines.items():
            logger.info("Training %s (no tuning) ...", name)
            pipe.fit(split.X_train, split.y_train)
            y_val_pred = pipe.predict(split.X_val)
            metrics = _val_metrics(split.y_val, y_val_pred)
            results["models"][name] = metrics
            fitted[name] = pipe
            joblib.dump(pipe, ARTIFACTS_DIR / f"{name}.joblib")
            logger.info(
                "%s val_acc=%.4f val_f1_macro=%.4f",
                name,
                metrics["val_accuracy"],
                metrics["val_f1_macro"],
            )
            if metrics["val_f1_macro"] > best_f1:
                best_f1 = metrics["val_f1_macro"]
                best_name = name

    assert best_name is not None
    results["best_model_version"] = best_name
    results["best_alias"] = BEST_VERSION

    src = ARTIFACTS_DIR / f"{best_name}.joblib"
    dst = ARTIFACTS_DIR / f"{BEST_VERSION}.joblib"
    shutil.copyfile(src, dst)
    fitted[BEST_VERSION] = joblib.load(dst)
    logger.info(
        "Best model on validation: %s (f1_macro=%.4f) -> %s",
        best_name,
        best_f1,
        dst,
    )

    comparison_path = ARTIFACTS_DIR / f"comparison_{run_id}.json"
    comparison_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote comparison -> %s", comparison_path)

    return {
        "comparison": results,
        "pipelines": fitted,
        "best_model_version": best_name,
    }


def load_pipeline(model_version: str) -> Pipeline:
    path = ARTIFACTS_DIR / f"{model_version}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)
