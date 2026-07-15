"""Factory for the three Phase 2 TF-IDF classification pipelines + tuning grids."""

from __future__ import annotations

from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

MODEL_VERSIONS = ("logreg_v1", "linearsvc_v1", "multinomialnb_v1")
BEST_VERSION = "best_v1"


def _tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=1,
        sublinear_tf=True,
    )


def build_pipelines() -> dict[str, Pipeline]:
    """Return fresh unfitted pipelines keyed by model_version id."""
    return {
        "logreg_v1": Pipeline(
            [
                ("tfidf", _tfidf()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "linearsvc_v1": Pipeline(
            [
                ("tfidf", _tfidf()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(
                            class_weight="balanced", max_iter=8000, dual="auto"
                        ),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        ),
        "multinomialnb_v1": Pipeline(
            [
                ("tfidf", _tfidf()),
                ("clf", MultinomialNB()),
            ]
        ),
    }


def build_search_pipelines() -> dict[str, Pipeline]:
    """Pipelines used during GridSearchCV.

    LinearSVC is tuned without calibration (faster nested CV), then wrapped
    afterward so Phase 3 still gets predict_proba / confidence scores.
    """
    return {
        "logreg_v1": Pipeline(
            [
                ("tfidf", _tfidf()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "linearsvc_v1": Pipeline(
            [
                ("tfidf", _tfidf()),
                (
                    "clf",
                    LinearSVC(class_weight="balanced", max_iter=8000, dual="auto"),
                ),
            ]
        ),
        "multinomialnb_v1": Pipeline(
            [
                ("tfidf", _tfidf()),
                ("clf", MultinomialNB()),
            ]
        ),
    }


# Modest grids: train-only CV, then model selection on held-out validation.
PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "logreg_v1": {
        "tfidf__max_features": [3000, 5000, 8000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1.0, 10.0],
    },
    "linearsvc_v1": {
        "tfidf__max_features": [3000, 5000, 8000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1.0, 10.0],
    },
    "multinomialnb_v1": {
        "tfidf__max_features": [3000, 5000, 8000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__alpha": [0.1, 0.5, 1.0, 2.0],
    },
}


def finalize_tuned_pipeline(model_version: str, search_best: Pipeline) -> Pipeline:
    """Convert a tuned search pipeline into the production/export pipeline."""
    if model_version != "linearsvc_v1":
        return search_best

    tfidf = search_best.named_steps["tfidf"]
    svc = search_best.named_steps["clf"]
    return Pipeline(
        [
            ("tfidf", tfidf),
            (
                "clf",
                CalibratedClassifierCV(
                    estimator=svc,
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ]
    )
