-- Indexes for analytics and SQL Agent queries
USE sentiment_brand_intel;

CREATE INDEX idx_social_posts_platform ON social_posts (platform);
CREATE INDEX idx_social_posts_sentiment ON social_posts (sentiment);
CREATE INDEX idx_social_posts_sentiment_group ON social_posts (sentiment_group);
CREATE INDEX idx_social_posts_timestamp ON social_posts (timestamp);
CREATE INDEX idx_social_posts_country ON social_posts (country);

CREATE INDEX idx_model_predictions_post_id ON model_predictions (post_id);
CREATE INDEX idx_model_predictions_version ON model_predictions (model_version);
CREATE INDEX idx_model_metrics_version ON model_metrics (model_version);
