"""Deterministic cleaning transforms for sentimentdataset.csv."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Fine-grained sentiment → Positive / Negative / Neutral
POSITIVE_SENTIMENTS = {
    "Positive",
    "Joy",
    "Excitement",
    "Contentment",
    "Gratitude",
    "Serenity",
    "Happy",
    "Hopeful",
    "Pride",
    "Elation",
    "Euphoria",
    "Enthusiasm",
    "Determination",
    "Playful",
    "Inspiration",
    "Happiness",
    "Hope",
    "Empowerment",
    "Inspired",
    "Admiration",
    "Calmness",
    "Compassion",
    "Tenderness",
    "Fulfillment",
    "Reverence",
    "Proud",
    "Grateful",
    "Compassionate",
    "Thrill",
    "Enchantment",
    "Love",
    "Amusement",
    "Anticipation",
    "Kind",
    "Empathetic",
    "Free-spirited",
    "Confident",
    "Satisfaction",
    "Accomplishment",
    "Harmony",
    "Creativity",
    "Wonder",
    "Adventure",
    "Enjoyment",
    "Affection",
    "Adoration",
    "Zest",
    "Whimsy",
    "Radiance",
    "Rejuvenation",
    "Coziness",
    "Resilience",
    "Exploration",
    "Captivation",
    "Tranquility",
    "Mischievous",
    "Overjoyed",
    "Motivation",
    "JoyfulReunion",
    "Blessed",
    "Appreciation",
    "Confidence",
    "Wonderment",
    "Optimism",
    "Intrigue",
    "PlayfulJoy",
    "Mindfulness",
    "DreamChaser",
    "Elegance",
    "FestiveJoy",
    "Freedom",
    "Dazzle",
    "Adrenaline",
    "ArtisticBurst",
    "CulinaryOdyssey",
    "Immersion",
    "Spark",
    "Marvel",
    "Positivity",
    "Kindness",
    "Friendship",
    "Success",
    "Amazement",
    "Romance",
    "Grandeur",
    "Energy",
    "Celebration",
    "Charm",
    "Ecstasy",
    "Colorful",
    "Hypnotic",
    "Connection",
    "Iconic",
    "Journey",
    "Engagement",
    "Touched",
    "Triumph",
    "Heartwarming",
    "Solace",
    "Breakthrough",
    "Joy in Baking",
    "Imagination",
    "Vibrancy",
    "Mesmerizing",
    "Culinary Adventure",
    "Winter Magic",
    "Thrilling Journey",
    "Nature's Beauty",
    "Celestial Wonder",
    "Creative Inspiration",
    "Runway Creativity",
    "Ocean's Freedom",
    "Relief",
    "Awe",
    "Acceptance",
    "Arousal",
    "Curiosity",
    "Surprise",
    "Nostalgia",
    "Reflection",
    "Contemplation",
    "Melodic",
    "InnerJourney",
    "Envisioning History",
    "Whispers of the Past",
    "Renewed Effort",
}

NEGATIVE_SENTIMENTS = {
    "Negative",
    "Despair",
    "Grief",
    "Loneliness",
    "Sad",
    "Embarrassment",
    "Embarrassed",
    "Frustration",
    "Regret",
    "Numbness",
    "Melancholy",
    "Hate",
    "Bad",
    "Disgust",
    "Bitterness",
    "Frustrated",
    "Betrayal",
    "Boredom",
    "Overwhelmed",
    "Desolation",
    "Bitter",
    "Shame",
    "Jealousy",
    "Resentment",
    "Fearful",
    "Jealous",
    "Devastated",
    "Envious",
    "Dismissive",
    "Anger",
    "Fear",
    "Sadness",
    "Disappointed",
    "Anxiety",
    "Intimidation",
    "Helplessness",
    "Envy",
    "Yearning",
    "Apprehensive",
    "Isolation",
    "Disappointment",
    "Sorrow",
    "Loss",
    "Suffering",
    "EmotionalStorm",
    "LostLove",
    "Exhaustion",
    "Darkness",
    "Desperation",
    "Ruins",
    "Heartache",
    "Solitude",
    "Heartbreak",
    "Suspense",
    "Obstacle",
    "Sympathy",
    "Pressure",
    "Miscalculation",
    "Challenge",
    "Confusion",
    "Indifference",
    "Ambivalence",
    "Bittersweet",
    "Pensive",
    "Emotion",
}

NEUTRAL_SENTIMENTS = {
    "Neutral",
}


def map_sentiment_group(sentiment: str) -> str:
    if sentiment in POSITIVE_SENTIMENTS:
        return "Positive"
    if sentiment in NEGATIVE_SENTIMENTS:
        return "Negative"
    if sentiment in NEUTRAL_SENTIMENTS:
        return "Neutral"
    logger.warning("Unknown sentiment label %r -> Neutral", sentiment)
    return "Neutral"


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].astype(str).str.strip()
        out.loc[out[col].isin(["nan", "None", "NaT", ""]), col] = pd.NA
    return out


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        c
        for c in df.columns
        if str(c).startswith("Unnamed") or str(c).strip() == ""
    ]
    return df.drop(columns=drop_cols, errors="ignore")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Phase 1 cleaning and return a MySQL-ready DataFrame."""
    # Preserve original CSV row index before dropping unnamed columns
    if "Unnamed: 0" in df.columns:
        source_ids = pd.to_numeric(df["Unnamed: 0"], errors="coerce")
    else:
        source_ids = pd.Series(range(len(df)), index=df.index)

    working = _drop_unnamed(df)
    working = _strip_strings(working)

    rename_map = {
        "Text": "text",
        "Sentiment": "sentiment",
        "Timestamp": "timestamp",
        "User": "username",
        "Platform": "platform",
        "Hashtags": "hashtags",
        "Retweets": "retweets",
        "Likes": "likes",
        "Country": "country",
        "Year": "year",
        "Month": "month",
        "Day": "day",
        "Hour": "hour",
    }
    working = working.rename(columns={k: v for k, v in rename_map.items() if k in working.columns})

    if "platform" in working.columns:
        working["platform"] = working["platform"].str.title()
        platform_aliases = {"X": "Twitter", "Tweet": "Twitter", "Fb": "Facebook", "Ig": "Instagram"}
        working["platform"] = working["platform"].replace(platform_aliases)

    if "timestamp" in working.columns:
        working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
        bad_ts = working["timestamp"].isna().sum()
        if bad_ts:
            logger.warning("Dropping %s rows with unparseable timestamps", bad_ts)
            working = working.dropna(subset=["timestamp"])

    for eng_col in ("retweets", "likes"):
        if eng_col in working.columns:
            working[eng_col] = pd.to_numeric(working[eng_col], errors="coerce").round().astype("Int64")

    if "hashtags" in working.columns:
        working["hashtags"] = (
            working["hashtags"]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        working.loc[working["hashtags"].isin(["nan", "<NA>", "None"]), "hashtags"] = pd.NA

    if "sentiment" in working.columns:
        working["sentiment"] = working["sentiment"].astype(str).str.strip()
        working["sentiment_group"] = working["sentiment"].map(map_sentiment_group)

    for part in ("year", "month", "day", "hour"):
        if part in working.columns:
            working[part] = pd.to_numeric(working[part], errors="coerce").astype("Int64")

    if "text" in working.columns:
        working["text"] = working["text"].astype(str).str.strip()
        empty_mask = working["text"].isin(["", "nan", "<NA>", "None"]) | working["text"].isna()
        dropped = int(empty_mask.sum())
        if dropped:
            logger.warning("Dropping %s rows with empty text", dropped)
            working = working.loc[~empty_mask]

    working.insert(0, "source_row_id", source_ids.reindex(working.index).astype("Int64"))

    column_order = [
        "source_row_id",
        "text",
        "sentiment",
        "sentiment_group",
        "timestamp",
        "username",
        "platform",
        "hashtags",
        "retweets",
        "likes",
        "country",
        "year",
        "month",
        "day",
        "hour",
    ]
    existing = [c for c in column_order if c in working.columns]
    return working[existing].reset_index(drop=True)


def load_and_clean(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    raw = pd.read_csv(path)
    return clean_dataframe(raw)


def save_cleaned_csv(df: pd.DataFrame, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
