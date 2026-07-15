-- Phase 2 verification queries (run after: python -m phase2.src.pipeline_run)
USE sentiment_brand_intel;

-- ---------------------------------------------------------------------------
-- Prediction counts by model_version
-- ---------------------------------------------------------------------------
SELECT model_version, COUNT(*) AS n_predictions
FROM model_predictions
GROUP BY model_version
ORDER BY model_version;

-- ---------------------------------------------------------------------------
-- Metrics pivot-style listing by model
-- ---------------------------------------------------------------------------
SELECT model_version, metric_name, metric_value
FROM model_metrics
ORDER BY model_version, metric_name;

-- ---------------------------------------------------------------------------
-- Which models are marked best (is_best = 1)
-- ---------------------------------------------------------------------------
SELECT model_version, metric_value AS is_best
FROM model_metrics
WHERE metric_name = 'is_best' AND metric_value = 1
ORDER BY model_version;

-- ---------------------------------------------------------------------------
-- Key test metrics side-by-side
-- ---------------------------------------------------------------------------
SELECT
  model_version,
  MAX(CASE WHEN metric_name = 'accuracy' THEN metric_value END) AS accuracy,
  MAX(CASE WHEN metric_name = 'f1_macro' THEN metric_value END) AS f1_macro,
  MAX(CASE WHEN metric_name = 'f1_weighted' THEN metric_value END) AS f1_weighted,
  MAX(CASE WHEN metric_name = 'val_f1_macro' THEN metric_value END) AS val_f1_macro,
  MAX(CASE WHEN metric_name = 'is_best' THEN metric_value END) AS is_best
FROM model_metrics
GROUP BY model_version
ORDER BY f1_macro DESC;

-- ---------------------------------------------------------------------------
-- Agreement: predicted vs true sentiment_group (by model)
-- ---------------------------------------------------------------------------
SELECT
  mp.model_version,
  COUNT(*) AS n,
  SUM(CASE WHEN mp.predicted_sentiment = sp.sentiment_group THEN 1 ELSE 0 END) AS matches,
  ROUND(
    100.0 * SUM(CASE WHEN mp.predicted_sentiment = sp.sentiment_group THEN 1 ELSE 0 END) / COUNT(*),
    2
  ) AS agreement_pct
FROM model_predictions mp
JOIN social_posts sp ON sp.id = mp.post_id
GROUP BY mp.model_version
ORDER BY agreement_pct DESC;

-- ---------------------------------------------------------------------------
-- Confusion-style counts for best_v1
-- ---------------------------------------------------------------------------
SELECT
  sp.sentiment_group AS actual,
  mp.predicted_sentiment AS predicted,
  COUNT(*) AS n
FROM model_predictions mp
JOIN social_posts sp ON sp.id = mp.post_id
WHERE mp.model_version = 'best_v1'
GROUP BY sp.sentiment_group, mp.predicted_sentiment
ORDER BY actual, predicted;

-- ---------------------------------------------------------------------------
-- Split size metrics
-- ---------------------------------------------------------------------------
SELECT model_version, metric_name, metric_value
FROM model_metrics
WHERE metric_name IN ('n_train', 'n_val', 'n_test')
ORDER BY model_version, metric_name;
