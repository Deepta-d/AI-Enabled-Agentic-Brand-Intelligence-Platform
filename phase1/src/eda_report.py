"""CLI EDA summary for the raw sentiment CSV (no notebook required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow `python -m phase1.src.eda_report` from project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1.src.clean import clean_dataframe  # noqa: E402
from phase1.src.config import RAW_CSV_PATH  # noqa: E402


def main() -> None:
    raw = pd.read_csv(RAW_CSV_PATH)
    print("=== Raw dataset ===")
    print(f"Path: {RAW_CSV_PATH}")
    print(f"Shape: {raw.shape}")
    print(f"Columns: {list(raw.columns)}")
    print("\nDtypes:")
    print(raw.dtypes)
    print("\nMissing values:")
    print(raw.isna().sum())
    print(f"\nDuplicate rows: {raw.duplicated().sum()}")

    str_cols = ["Text", "Sentiment", "User", "Platform", "Hashtags", "Country"]
    print("\nWhitespace / trailing-space samples (before strip):")
    for col in str_cols:
        if col not in raw.columns:
            continue
        series = raw[col].astype(str)
        padded = (series != series.str.strip()).sum()
        print(f"  {col}: {padded} values with leading/trailing whitespace")

    print("\nPlatform value counts (raw):")
    print(raw["Platform"].astype(str).str.strip().value_counts())
    print("\nTop 15 sentiments (raw):")
    print(raw["Sentiment"].astype(str).str.strip().value_counts().head(15))

    print("\nEngagement describe:")
    print(raw[["Retweets", "Likes"]].describe())

    print("\nYear range:", raw["Year"].min(), "->", raw["Year"].max())

    cleaned = clean_dataframe(raw)
    print("\n=== After cleaning ===")
    print(f"Shape: {cleaned.shape}")
    print("\nSentiment group counts:")
    print(cleaned["sentiment_group"].value_counts())
    print("\nPlatform counts:")
    print(cleaned["platform"].value_counts())
    print("\nSample rows:")
    sample = cleaned.head(3).copy()
    sample["text"] = sample["text"].astype(str).str.encode("ascii", "replace").str.decode("ascii")
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
