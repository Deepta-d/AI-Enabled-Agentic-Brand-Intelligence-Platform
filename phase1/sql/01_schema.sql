-- Phase 1 schema: Social Media Sentiment & Brand Intelligence Platform
-- Run in MySQL Workbench (or via phase1.src.load_mysql).

CREATE DATABASE IF NOT EXISTS sentiment_brand_intel
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE sentiment_brand_intel;

-- Cleaned social posts (source of truth for Phase 2 ML and SQL Agent)
CREATE TABLE IF NOT EXISTS social_posts (
  id INT NOT NULL AUTO_INCREMENT,
  source_row_id INT NULL,
  text TEXT NOT NULL,
  sentiment VARCHAR(64) NOT NULL,
  sentiment_group VARCHAR(16) NULL,
  timestamp DATETIME NULL,
  username VARCHAR(128) NULL,
  platform VARCHAR(32) NULL,
  hashtags TEXT NULL,
  retweets INT NULL,
  likes INT NULL,
  country VARCHAR(64) NULL,
  year INT NULL,
  month INT NULL,
  day INT NULL,
  hour INT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Phase 2 stub: model predictions written back after evaluation
CREATE TABLE IF NOT EXISTS model_predictions (
  id INT NOT NULL AUTO_INCREMENT,
  post_id INT NOT NULL,
  model_version VARCHAR(64) NOT NULL,
  predicted_sentiment VARCHAR(64) NOT NULL,
  confidence DECIMAL(6, 4) NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  CONSTRAINT fk_predictions_post
    FOREIGN KEY (post_id) REFERENCES social_posts (id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Phase 2 stub: evaluation metrics
CREATE TABLE IF NOT EXISTS model_metrics (
  id INT NOT NULL AUTO_INCREMENT,
  model_version VARCHAR(64) NOT NULL,
  metric_name VARCHAR(64) NOT NULL,
  metric_value DECIMAL(12, 6) NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
