"""Load social posts from MySQL and create stratified train/val/test splits."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, text

from phase1.src.config import PROJECT_ROOT, get_database_url

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = PROJECT_ROOT / "phase2" / "artifacts"


@dataclass
class SplitData:
    """Hold stratified train / validation / test partitions."""

    posts: pd.DataFrame
    X_train: pd.Series
    X_val: pd.Series
    X_test: pd.Series
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    ids_train: list[int]
    ids_val: list[int]
    ids_test: list[int]

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_val(self) -> int:
        return len(self.X_val)

    @property
    def n_test(self) -> int:
        return len(self.X_test)


def load_posts(engine=None) -> pd.DataFrame:
    """Load cleaned posts used for Phase 2 sentiment_group classification."""
    own_engine = engine is None
    if own_engine:
        engine = create_engine(get_database_url(), pool_pre_ping=True)
    try:
        query = text(
            """
            SELECT id, text, sentiment, sentiment_group, platform
            FROM social_posts
            WHERE text IS NOT NULL
              AND TRIM(text) <> ''
              AND sentiment_group IS NOT NULL
              AND TRIM(sentiment_group) <> ''
            ORDER BY id
            """
        )
        df = pd.read_sql(query, engine)
    finally:
        if own_engine:
            engine.dispose()

    df["text"] = df["text"].astype(str).str.strip()
    df["sentiment_group"] = df["sentiment_group"].astype(str).str.strip()
    df = df.loc[df["text"].ne("") & df["sentiment_group"].ne("")].reset_index(drop=True)
    logger.info(
        "Loaded %s posts from MySQL (classes: %s)",
        len(df),
        df["sentiment_group"].value_counts().to_dict(),
    )
    return df


def make_splits(
    posts: pd.DataFrame,
    *,
    random_state: int = 42,
    train_size: float = 0.70,
) -> SplitData:
    """Two-step stratified split: 70% train / 15% val / 15% test."""
    X = posts["text"]
    y = posts["sentiment_group"]
    ids = posts["id"]

    X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
        X,
        y,
        ids,
        train_size=train_size,
        stratify=y,
        random_state=random_state,
    )
    X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
        X_temp,
        y_temp,
        ids_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=random_state,
    )

    split = SplitData(
        posts=posts,
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        ids_train=[int(i) for i in ids_train.tolist()],
        ids_val=[int(i) for i in ids_val.tolist()],
        ids_test=[int(i) for i in ids_test.tolist()],
    )
    logger.info(
        "Split sizes — train=%s val=%s test=%s",
        split.n_train,
        split.n_val,
        split.n_test,
    )
    return split


def save_split_ids(split: SplitData, run_id: str = "v1") -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"split_ids_{run_id}.json"
    payload: dict[str, Any] = {
        "run_id": run_id,
        "n_train": split.n_train,
        "n_val": split.n_val,
        "n_test": split.n_test,
        "ids_train": split.ids_train,
        "ids_val": split.ids_val,
        "ids_test": split.ids_test,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote split ids -> %s", path)
    return path


def load_and_split(*, random_state: int = 42, run_id: str = "v1") -> SplitData:
    posts = load_posts()
    split = make_splits(posts, random_state=random_state)
    save_split_ids(split, run_id=run_id)
    return split
