-- Phase 1 analytics queries for MySQL Workbench
-- Run after: python -m phase1.src.load_mysql

USE sentiment_brand_intel;

-- ---------------------------------------------------------------------------
-- Volume by platform
-- ---------------------------------------------------------------------------
SELECT platform, COUNT(*) AS post_count
FROM social_posts
GROUP BY platform
ORDER BY post_count DESC;

-- ---------------------------------------------------------------------------
-- Fine-grained sentiment counts (top 20)
-- ---------------------------------------------------------------------------
SELECT sentiment, COUNT(*) AS post_count
FROM social_posts
GROUP BY sentiment
ORDER BY post_count DESC
LIMIT 20;

-- ---------------------------------------------------------------------------
-- Sentiment group (Positive / Negative / Neutral)
-- ---------------------------------------------------------------------------
SELECT sentiment_group, COUNT(*) AS post_count
FROM social_posts
GROUP BY sentiment_group
ORDER BY post_count DESC;

-- ---------------------------------------------------------------------------
-- Country distribution
-- ---------------------------------------------------------------------------
SELECT country, COUNT(*) AS post_count
FROM social_posts
GROUP BY country
ORDER BY post_count DESC;

-- ---------------------------------------------------------------------------
-- Daily volume and average engagement
-- ---------------------------------------------------------------------------
SELECT
  DATE(timestamp) AS post_date,
  COUNT(*) AS post_count,
  ROUND(AVG(likes), 2) AS avg_likes,
  ROUND(AVG(retweets), 2) AS avg_retweets
FROM social_posts
WHERE timestamp IS NOT NULL
GROUP BY DATE(timestamp)
ORDER BY post_date;

-- ---------------------------------------------------------------------------
-- Monthly volume and engagement
-- ---------------------------------------------------------------------------
SELECT
  year,
  month,
  COUNT(*) AS post_count,
  ROUND(AVG(likes), 2) AS avg_likes,
  ROUND(AVG(retweets), 2) AS avg_retweets
FROM social_posts
GROUP BY year, month
ORDER BY year, month;

-- ---------------------------------------------------------------------------
-- Sentiment mix over time (by month + sentiment_group)
-- ---------------------------------------------------------------------------
SELECT
  year,
  month,
  sentiment_group,
  COUNT(*) AS post_count
FROM social_posts
GROUP BY year, month, sentiment_group
ORDER BY year, month, sentiment_group;

-- ---------------------------------------------------------------------------
-- Platform x sentiment_group cross-tab
-- ---------------------------------------------------------------------------
SELECT
  platform,
  sentiment_group,
  COUNT(*) AS post_count
FROM social_posts
GROUP BY platform, sentiment_group
ORDER BY platform, post_count DESC;

-- ---------------------------------------------------------------------------
-- Simple hashtag search examples (Phase 1 — LIKE-based)
-- ---------------------------------------------------------------------------
SELECT COUNT(*) AS fitness_mentions
FROM social_posts
WHERE hashtags LIKE '%Fitness%';

SELECT id, platform, sentiment, LEFT(text, 80) AS text_preview, hashtags
FROM social_posts
WHERE hashtags LIKE '%Travel%'
LIMIT 20;

-- ---------------------------------------------------------------------------
-- Data-quality checks
-- ---------------------------------------------------------------------------
SELECT
  COUNT(*) AS total_rows,
  SUM(CASE WHEN text IS NULL OR TRIM(text) = '' THEN 1 ELSE 0 END) AS empty_text,
  SUM(CASE WHEN timestamp IS NULL THEN 1 ELSE 0 END) AS null_timestamp,
  SUM(CASE WHEN platform IS NULL THEN 1 ELSE 0 END) AS null_platform,
  SUM(CASE WHEN sentiment IS NULL THEN 1 ELSE 0 END) AS null_sentiment,
  SUM(CASE WHEN sentiment_group IS NULL THEN 1 ELSE 0 END) AS null_sentiment_group,
  SUM(CASE WHEN likes IS NULL THEN 1 ELSE 0 END) AS null_likes,
  SUM(CASE WHEN retweets IS NULL THEN 1 ELSE 0 END) AS null_retweets
FROM social_posts;

-- Rows where Year/Month/Day/Hour disagree with timestamp
SELECT COUNT(*) AS time_part_mismatches
FROM social_posts
WHERE timestamp IS NOT NULL
  AND (
    year <> YEAR(timestamp)
    OR month <> MONTH(timestamp)
    OR day <> DAY(timestamp)
    OR hour <> HOUR(timestamp)
  );
