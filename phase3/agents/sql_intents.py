"""Deterministic mapping from natural-language questions to MySQL SELECT plans."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SqlPlan:
    intent: str
    title: str
    sql: str


SCHEMA_HINT = """
Tables:
- social_posts(id, source_row_id, text, sentiment, sentiment_group, timestamp,
  username, platform, hashtags, retweets, likes, country, year, month, day, hour, created_at)
- model_metrics(model_version, metric_name, metric_value, ...)
- model_predictions(post_id, model_version, predicted_sentiment, ...)
""".strip()

KNOWN_TABLES = ("social_posts", "model_metrics", "model_predictions")

# Safe, fully-qualified sample queries — never JOIN across these for "table contents".
TABLE_SAMPLE_PLANS: dict[str, SqlPlan] = {
    "social_posts": SqlPlan(
        intent="table_sample_social_posts",
        title="social_posts (sample)",
        sql=(
            "SELECT id, username, platform, sentiment_group, "
            "LEFT(text, 80) AS text_preview, likes, country "
            "FROM social_posts ORDER BY id DESC LIMIT 5"
        ),
    ),
    "model_metrics": SqlPlan(
        intent="table_sample_model_metrics",
        title="model_metrics (sample)",
        sql=(
            "SELECT model_version, metric_name, metric_value "
            "FROM model_metrics ORDER BY model_version, metric_name LIMIT 20"
        ),
    ),
    "model_predictions": SqlPlan(
        intent="table_sample_model_predictions",
        title="model_predictions (sample)",
        sql=(
            "SELECT post_id, model_version, predicted_sentiment "
            "FROM model_predictions ORDER BY post_id DESC LIMIT 5"
        ),
    ),
}

_TABLE_CONTENTS_RE = re.compile(
    r"\b("
    r"contents?|whats?\s+in|what\s+is\s+in|what\s+are\s+(?:the\s+)?(?:contents?|in)|"
    r"show\s+(?:me\s+)?(?:the\s+)?(?:rows|data|sample)|"
    r"sample\s+(?:rows?|data)|preview|"
    r"rows?\s+(?:from|in)|data\s+(?:from|in)|"
    r"list\s+(?:the\s+)?(?:rows|columns?|fields?)"
    r")\b",
    re.IGNORECASE,
)


def mentioned_tables(text: str) -> list[str]:
    """Return known table names mentioned in the question (stable order)."""
    lower = (text or "").lower()
    found: list[str] = []
    for name in KNOWN_TABLES:
        if name in lower or name.replace("_", " ") in lower:
            found.append(name)
    return found


def table_count_plan(table: str) -> SqlPlan:
    """Row-count SELECT for a known table."""
    if table not in TABLE_SAMPLE_PLANS:
        raise ValueError(f"Unknown table: {table}")
    return SqlPlan(
        intent=f"table_count_{table}",
        title=f"{table} (row count)",
        sql=f"SELECT COUNT(*) AS n FROM {table}",
    )


def extract_lookup_snippet(text: str) -> str | None:
    """Pull a post text snippet from quotes or a Previous classified text marker."""
    m = re.search(
        r'\[?\s*Previous classified text:\s*"([^"]{3,300})"\s*\]?',
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    m = re.search(
        r'\[?\s*Previous classified text:\s*\'([^\']{3,300})\'\s*\]?',
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    quotes = re.findall(r"[\"']([^\"']{3,300})[\"']", text)
    if quotes:
        # Prefer the longest quote (usually the post body, not a short word)
        return max(quotes, key=len).strip()
    return None


def is_post_lookup_query(text: str) -> bool:
    """True when asking platform/username/etc. for a specific (prior) post text."""
    lower = (text or "").lower()
    asks_field = bool(
        re.search(r"\b(platform|username|country|hashtag|source|where)\b", lower)
    )
    if not asks_field:
        return False
    refers_prior = bool(
        re.search(
            r"\b(above|previous|prior|that text|this text|the text|that post|this post|"
            r"classified text|generated)\b",
            lower,
        )
    )
    has_marker = "previous classified text" in lower
    has_quote = bool(re.search(r"""['"][^'"]{3,}['"]""", text))
    return refers_prior or has_marker or (has_quote and asks_field)


def post_lookup_sql(snippet: str) -> str:
    """Safe LIKE lookup for a post text snippet (escaped)."""
    safe = snippet.replace("\\", "\\\\").replace("'", "''")
    needle = safe[:160]
    return f"""
    SELECT id, platform, username, country, sentiment_group,
           LEFT(text, 120) AS text_preview
    FROM social_posts
    WHERE text LIKE '%{needle}%'
    ORDER BY id
    LIMIT 5
    """.strip()


# Canonical country labels matched against social_posts.country (case-insensitive).
_COUNTRY_ALIAS_TO_CANON: dict[str, str] = {
    "usa": "USA",
    "us": "USA",
    "u.s": "USA",
    "u.s.": "USA",
    "u.s.a": "USA",
    "u.s.a.": "USA",
    "united states": "USA",
    "united states of america": "USA",
    "america": "USA",
    "uk": "UK",
    "u.k": "UK",
    "u.k.": "UK",
    "united kingdom": "UK",
    "britain": "UK",
    "great britain": "UK",
    "england": "UK",
    "canada": "Canada",
    "india": "India",
    "australia": "Australia",
    "germany": "Germany",
    "france": "France",
    "brazil": "Brazil",
    "mexico": "Mexico",
    "japan": "Japan",
    "china": "China",
    "spain": "Spain",
    "italy": "Italy",
    "nigeria": "Nigeria",
    "south africa": "South Africa",
    "uae": "UAE",
    "pakistan": "Pakistan",
}

# Short codes need location context so "tell us" / "users" never match.
_SHORT_COUNTRY_CONTEXT = re.compile(
    r"\b(?:from|in|of|to|for)\s+(?P<code>u\.?s\.?a?\.?|uk|u\.?k\.?|uae)\b|"
    r"\b(?P<code2>usa|uk|uae)\b",
    re.IGNORECASE,
)


def extract_country_filter(text: str) -> str | None:
    """Return a canonical country name if the question targets a specific country."""
    lower = (text or "").lower().strip()
    if not lower:
        return None

    # Longer multi-word aliases first (united states, south africa, …).
    for alias in sorted(_COUNTRY_ALIAS_TO_CANON, key=len, reverse=True):
        if len(alias) <= 3:
            continue
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            return _COUNTRY_ALIAS_TO_CANON[alias]

    m = _SHORT_COUNTRY_CONTEXT.search(lower)
    if m:
        raw = (m.group("code") or m.group("code2") or "").lower().rstrip(".")
        raw = raw.replace(".", "")
        if raw in {"us", "usa"}:
            return "USA"
        if raw in {"uk"}:
            return "UK"
        if raw == "uae":
            return "UAE"
        return _COUNTRY_ALIAS_TO_CANON.get(raw)

    # "from <Country Name>" for anything else that looks like a place label.
    m2 = re.search(
        r"\b(?:from|in)\s+(?P<name>[a-z][a-z][a-z\s\-]{0,40}?)"
        r"(?:\s+(?:posts?|users?|customers?|people|authors?|handles?)?\s*)?[?.!]?\s*$",
        lower,
    )
    if m2:
        name = re.sub(r"\s+", " ", m2.group("name")).strip(" -")
        # Drop trailing filler words mistakenly captured
        name = re.sub(
            r"\b(posts?|users?|customers?|people|authors?|handles?|there|are|is)\b",
            "",
            name,
        ).strip(" -")
        if len(name) >= 2:
            return _COUNTRY_ALIAS_TO_CANON.get(name) or name.title()
    return None


def wants_user_count(text: str) -> bool:
    """True when counting people/customers/users rather than posts."""
    lower = (text or "").lower()
    return bool(
        re.search(
            r"\b(customers?|users?|usernames?|people|authors?|handles?|accounts?)\b",
            lower,
        )
    )


def country_filter_plan(text: str, country: str) -> SqlPlan:
    """COUNT for a specific country (users and/or posts)."""
    safe = country.replace("\\", "\\\\").replace("'", "''")
    variants = {safe, safe.upper(), safe.title()}
    if country.upper() == "USA":
        variants.update({"USA", "US", "United States", "United States of America"})
    elif country.upper() == "UK":
        variants.update({"UK", "United Kingdom", "Britain", "Great Britain"})
    upper_vals = sorted({v.upper() for v in variants})
    in_list = ", ".join(f"'{v}'" for v in upper_vals)
    if wants_user_count(text):
        sql = f"""
        SELECT
          '{safe}' AS country,
          COUNT(DISTINCT CASE
            WHEN username IS NOT NULL AND TRIM(username) <> '' THEN username
          END) AS unique_users,
          COUNT(*) AS posts
        FROM social_posts
        WHERE UPPER(TRIM(country)) IN ({in_list})
        """.strip()
        title = f"Users / posts from {country}"
    else:
        sql = f"""
        SELECT
          '{safe}' AS country,
          COUNT(*) AS posts
        FROM social_posts
        WHERE UPPER(TRIM(country)) IN ({in_list})
        """.strip()
        title = f"Posts from {country}"
    return SqlPlan(intent="country_filter", title=title, sql=sql)


def detect_table_contents_request(text: str) -> list[str] | None:
    """
    If the user asks what is in one or more tables, return those table names.
    When contents-intent is clear but no table is named, return all known tables.
    """
    lower = (text or "").lower().strip()
    if not lower:
        return None
    # Meta "what tables exist" is handled elsewhere — not a contents dump.
    if re.search(r"\b(what|which)\s+tables?\b", lower) and not _TABLE_CONTENTS_RE.search(lower):
        return None

    tables = mentioned_tables(lower)
    if _TABLE_CONTENTS_RE.search(lower):
        return tables or list(KNOWN_TABLES)
    # "show social_posts table" / "social_posts, model_metrics table"
    if tables and re.search(r"\btables?\b", lower):
        return tables
    return None


def classify_sql_intent(text: str) -> SqlPlan | None:
    """Return a concrete SELECT plan for common analytics questions, or None."""
    lower = text.lower().strip()

    # Table-contents are multi-plan; callers should use detect_table_contents_request.
    if detect_table_contents_request(lower) is not None:
        return SqlPlan(
            intent="table_contents",
            title="Table contents",
            sql="",
        )

    # Platform/username of a specific prior/quoted post (not aggregate platform counts).
    if is_post_lookup_query(text):
        snippet = extract_lookup_snippet(text)
        if snippet:
            return SqlPlan(
                intent="post_lookup",
                title="Matching post in the dataset",
                sql=post_lookup_sql(snippet),
            )
        return SqlPlan(
            intent="post_lookup",
            title="Matching post in the dataset",
            sql="",
        )

    # Specific country filter ("how many customers from USA?") before aggregates.
    country = extract_country_filter(text)
    if country and (
        wants_user_count(text)
        or re.search(r"\b(how\s+many|count|number|posts?|from|in)\b", lower)
        or re.search(r"\b(countr(?:y|ies))\b", lower)
    ):
        return country_filter_plan(text, country)

    # Email / WhatsApp / alert draft companion → sentiment counts only (no LLM row dumps).
    if re.search(r"\b(email|e-mail|whatsapp|alert|draft|notify|send)\b", lower) and not re.search(
        r"\b(username|platform|country|hashtag|schema|contents?|model\s+metric|agreement)\b",
        lower,
    ):
        return SqlPlan(
            intent="sentiment_summary",
            title="Sentiment in the database",
            sql="",
        )

    if re.search(r"\b(username|user\s*name|users|authors?|handles?|customers?)\b", lower):
        # Aggregate username summary unless this is a prior-post / country lookup (above)
        return SqlPlan(
            intent="username_summary",
            title="Usernames in the dataset",
            sql="""
            SELECT
              COUNT(DISTINCT CASE WHEN username IS NOT NULL AND TRIM(username) <> '' THEN username END) AS unique_usernames,
              COUNT(*) AS total_posts,
              SUM(CASE WHEN username IS NULL OR TRIM(username) = '' THEN 1 ELSE 0 END) AS posts_missing_username
            FROM social_posts
            """.strip(),
        )

    if re.search(r"\b(platform|source|twitter|reddit|facebook|instagram)\b", lower) and not re.search(
        r"\b(sentiment|model|f1|accuracy)\b", lower
    ):
        return SqlPlan(
            intent="platform_summary",
            title="Posts by platform",
            sql="""
            SELECT COALESCE(NULLIF(TRIM(platform), ''), '(unknown)') AS platform, COUNT(*) AS n
            FROM social_posts
            GROUP BY COALESCE(NULLIF(TRIM(platform), ''), '(unknown)')
            ORDER BY n DESC
            """.strip(),
        )

    if re.search(r"\b(countr(?:y|ies)|location|region)\b", lower):
        return SqlPlan(
            intent="country_summary",
            title="Posts by country",
            sql="""
            SELECT COALESCE(NULLIF(TRIM(country), ''), '(unknown)') AS country, COUNT(*) AS n
            FROM social_posts
            GROUP BY COALESCE(NULLIF(TRIM(country), ''), '(unknown)')
            ORDER BY n DESC
            LIMIT 40
            """.strip(),
        )

    if re.search(r"\b(hashtags?|hash\s*tags?)\b", lower):
        return SqlPlan(
            intent="hashtag_summary",
            title="Hashtags column overview",
            sql="""
            SELECT
              COUNT(*) AS total_posts,
              SUM(CASE WHEN hashtags IS NOT NULL AND TRIM(hashtags) <> '' THEN 1 ELSE 0 END)
                AS posts_with_hashtags,
              SUM(CASE WHEN hashtags IS NULL OR TRIM(hashtags) = '' THEN 1 ELSE 0 END)
                AS posts_missing_hashtags,
              COUNT(DISTINCT CASE
                WHEN hashtags IS NOT NULL AND TRIM(hashtags) <> '' THEN hashtags
              END) AS distinct_hashtag_values
            FROM social_posts
            """.strip(),
        )

    if re.search(r"\b(agreement|agree|match(?:es|ed)?)\b", lower) and re.search(
        r"\b(predict|prediction|label|model)\b", lower
    ):
        return SqlPlan(
            intent="prediction_agreement",
            title="Prediction agreement",
            sql="",  # use specialized tool
        )

    if re.search(r"\b(stored|database|mysql|db)\b", lower) and re.search(
        r"\bmetric", lower
    ):
        return SqlPlan(
            intent="db_metrics",
            title="Stored model metrics",
            sql="",  # use specialized tool
        )

    # Sentiment / SQL analytics overview (live counts) — not meta "what was performed"
    if re.search(
        r"\b(what\s+sql|sql\s+analytics(?:\s+were)?\s+performed|analytics\s+were\s+performed|"
        r"database\s+name|name\s+of\s+(?:the\s+)?database|which\s+database|what\s+tables)\b",
        lower,
    ):
        return None

    if re.search(
        r"\b(sentiment|positive|negative|neutral|distribution|how\s+many\s+posts|"
        r"sql\s+analytics|analytics|phase\s*1)\b",
        lower,
    ) or (
        re.search(r"\b(summar(?:y|ize|ise)|overview|dataset)\b", lower)
        and not re.search(
            r"\b(username|user\s*name|platform|country|hashtag|model|metric|f1)\b",
            lower,
        )
    ):
        return SqlPlan(
            intent="sentiment_summary",
            title="Sentiment in the database",
            sql="",  # use specialized tool
        )

    # Top usernames if they ask for list/top/most
    if re.search(r"\b(top|most\s+active|list)\b", lower) and re.search(
        r"\b(user|author|handle)\b", lower
    ):
        return SqlPlan(
            intent="top_usernames",
            title="Top usernames by post count",
            sql="""
            SELECT username, COUNT(*) AS n
            FROM social_posts
            WHERE username IS NOT NULL AND TRIM(username) <> ''
            GROUP BY username
            ORDER BY n DESC
            LIMIT 30
            """.strip(),
        )

    return None


def followup_sql_for_intent(intent: str) -> SqlPlan | None:
    """Optional second query for richer answers."""
    if intent == "username_summary":
        return SqlPlan(
            intent="top_usernames",
            title="Top usernames by post count",
            sql="""
            SELECT username, COUNT(*) AS n
            FROM social_posts
            WHERE username IS NOT NULL AND TRIM(username) <> ''
            GROUP BY username
            ORDER BY n DESC
            LIMIT 20
            """.strip(),
        )
    if intent == "hashtag_summary":
        return SqlPlan(
            intent="hashtag_sample",
            title="Most common hashtag values",
            sql="""
            SELECT hashtags, COUNT(*) AS n
            FROM social_posts
            WHERE hashtags IS NOT NULL AND TRIM(hashtags) <> ''
            GROUP BY hashtags
            ORDER BY n DESC
            LIMIT 25
            """.strip(),
        )
    return None


COLUMN_ABOUT_BLURBS: dict[str, str] = {
    "hashtag_summary": (
        "The hashtags column on social_posts holds the hashtag text for each post "
        "(blank when the post has none)."
    ),
}
