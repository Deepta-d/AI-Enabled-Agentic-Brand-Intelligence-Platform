# AI enabled Agentic Platform for Social Media Sentiment & Brand Intelligence Platform

**Project 5 — Research Report**

---

## Table of Contents

| Section | Title |
|---------|-------|
| — | Abstract |
| 1 | Introduction |
| 1.1 | Background |
| 1.2 | Objectives |
| 1.3 | Scope |
| 2 | Related Work |
| 3 | Dataset & Methodology |
| 3.1 | SQL Data Model & Analytics |
| 3.2 | Exploratory Analysis & Cleaning |
| 3.3 | Machine Learning Models |
| 3.4 | Unified MCP & Agentic Supervisor |
| 3.5 | Streamlit Analytics & Brand Assistant |
| 4 | System Architecture |
| 5 | Results |
| 5.1 | Brand Intelligence KPIs |
| 5.2 | Sentiment & Platform Patterns |
| 5.3 | Model Comparison (Validation / Test) |
| 5.4 | Agent Routing & Tool Outcomes |
| 5.5 | HITL Alert Safety |
| 5.6 | Output Artifacts |
| 6 | Conclusion |
| 7 | Discussion |
| 7.1 | Product & Operational Insights |
| 7.2 | Limitations |
| 7.3 | Recommendations & Future Work |
| 8 | References |

---

## Abstract

This study presents an end-to-end AI-enabled agentic platform for social media sentiment and brand intelligence across three integrated phases: MySQL data integration and SQL analytics, classical text classification with train/validation/test selection, and a LangGraph star-topology supervisor over a Unified Model Context Protocol (MCP) tool server. Raw social posts are cleaned into Positive / Negative / Neutral labels, stored in MySQL (`sentiment_brand_intel.social_posts`), and used to train three TF-IDF classifiers (Logistic Regression, Calibrated Linear SVM, Multinomial Naive Bayes). The validation winner (`multinomialnb_v1`, aliased as `best_v1`) achieved 90.0% test accuracy and approximately 0.594 macro-F1 on a held-out stratified test set. Phase 3 exposes MySQL, ML, Email (Gmail SMTP), WhatsApp (Twilio), and optional FRED tools through MCP; a Streamlit Brand Assistant answers analytics and prediction questions with deterministic intent routing, tool caching, Ollama-primary / Gemini-fallback LLMs, and human-in-the-loop approval before any outbound alert is sent.

**Keywords:** Social media sentiment, brand intelligence, MySQL, TF-IDF, Multinomial Naive Bayes, LangGraph, Model Context Protocol (MCP), Streamlit, HITL alerts, Ollama, Gemini

---

## 1. Introduction

### 1.1 Background

Brands continuously monitor public conversation across social platforms to detect shifts in sentiment, emerging hashtags, and engagement patterns. Raw feeds are noisy: fine-grained emotion labels differ by source, timestamps and usernames are inconsistent, and operational teams need both reproducible analytics and natural-language access to grounded metrics—not invented marketing copy.

Pairing a relational analytics layer with classical NLP classifiers and an agentic assistant gives a full stack: SQL for trustworthy aggregates, ML for on-demand classification with known quality metrics, and a supervisor that orchestrates specialist tools (SQL, ML, Email, WhatsApp) under explicit human approval for outbound messaging.

### 1.2 Objectives

1. Ingest and clean social posts into MySQL with a three-class `sentiment_group` label and validated indexes.
2. Deliver SQL analytics covering platforms, countries, hashtags, volume, engagement, and sentiment mix.
3. Train and compare three TF-IDF classifiers with GridSearchCV on train only; select on validation `f1_macro`; report on held-out test.
4. Write predictions and metrics back to MySQL and persist joblib / JSON artifacts for agents.
5. Expose MySQL, ML, Email, WhatsApp, and optional FRED through one Unified MCP server.
6. Build a LangGraph star supervisor with deterministic routing, tool caching, and LLM failover (Ollama → Gemini).
7. Ship a Streamlit UI with Analytics (KPIs/charts) and Brand Assistant (HITL Approve/Reject for alerts).

### 1.3 Scope

Phases 1–3 cover SQL BI and cleaning, machine learning, and the agentic MCP + Streamlit layer. The analytical grain is one cleaned post row in `social_posts`. Fine-grained source sentiments are coarsened to Positive / Negative / Neutral. Live continuous scrapers and deep transformer fine-tuning are out of scope; optional future work includes durable graph checkpoints, richer models, and DirectQuery-style live feeds.

---

## 2. Related Work

Brand and social listening systems commonly combine: (1) relational storage of post-level engagement attributes; (2) lexicon or classical ML sentiment classifiers before neural deployments; (3) BI dashboards for volume and mix; and (4) increasingly, LLM agents that call tools rather than free-form invent facts.

- **Sentiment coarsening** — Mapping rich emotion taxonomies into Positive / Negative / Neutral for stable reporting and modeling.
- **TF-IDF + linear / NB models** — Strong baselines for short social text with transparent artifacts and low latency inference.
- **Multi-agent orchestration** — Supervisor–worker patterns (here LangGraph star topology) keep specialists isolated and auditable.
- **Tool protocols** — MCP standardizes tool exposure so Streamlit, CLI, and IDE clients share one server.
- **HITL for outbound actions** — Draft-first alerting with explicit Approve prevents accidental customer or partner spam.

This project extends coursework-style ETL + ML into an operable agent platform: validation-gated model selection, MySQL write-back for agreement queries, deterministic SQL/ML intent routing so small local models cannot skip tools, and dual locks (`ALERTS_ENABLED` + human Approve) on sends.

---

## 3. Dataset & Methodology

| Attribute | Detail |
|-----------|--------|
| Source | Project social sentiment CSV (multi-platform posts) |
| Local files | `dataset/sentimentdataset.csv` → `cleaned_sentimentdataset.csv` |
| Database | MySQL `sentiment_brand_intel` |
| Core table | `social_posts` (cleaned post grain) |
| ML tables | `model_predictions`, `model_metrics` |
| Target | `sentiment_group` ∈ {Positive, Negative, Neutral} |
| Split | Stratified 70% / 15% / 15% train / val / test (seed 42) |
| n_train / n_val / n_test | 512 / 110 / 110 (run v1) |
| Champion | `multinomialnb_v1` → `best_v1` alias |
| Test accuracy / macro-F1 | 0.900 / ≈ 0.594 |
| UI | Streamlit Analytics + Brand Assistant |
| Agents | LangGraph supervisor + SQL / ML / Email / WhatsApp |
| Tools protocol | Unified MCP (stdio) + local LangChain fallback |

**Data quality decisions:** strip whitespace; parse timestamps into year/month/day/hour fields; map fine-grained emotions to three sentiment groups; preserve platform, username, country, hashtags, likes, and retweets for SQL analytics and follow-up post lookups by text snippet.

### 3.1 SQL Data Model & Analytics

| Step | Script / file | Output |
|------|---------------|--------|
| Schema | `phase1/sql/01_schema.sql` | Tables + types |
| Indexes | `phase1/sql/02_indexes.sql` | platform, sentiment, country, … |
| Load | `phase1/src/load_mysql.py` | Populate `social_posts` |
| Analytics | `phase1/sql/03_analytics_queries.sql` | Platform/country/hashtag SQL |
| ML verify | `phase2/sql/04_ml_verification.sql` | Metrics & agreement checks |

**Schema grain:**

| Table | Grain | Key columns |
|-------|-------|-------------|
| `social_posts` | 1 row / post | id, text, sentiment, sentiment_group, username, platform, hashtags, likes, retweets, country, timestamp parts |
| `model_predictions` | 1 row / post × model | post_id, model_version, predicted_sentiment, confidence |
| `model_metrics` | 1 row / metric | model_version, metric_name, metric_value |

Phase 3 maps natural-language analytics to deterministic SELECT plans in `phase3/agents/sql_intents.py` (sentiment summary, platform/country/username aggregates, country filters such as “from USA”, hashtag column overview, and post lookup by previously classified text).

### 3.2 Exploratory Analysis & Cleaning

| Step | Notebook / script | Output |
|------|-------------------|--------|
| EDA | `phase1/notebooks/01_eda.ipynb`, `eda_report.py` | Distributions & DQ notes |
| Clean | `phase1/src/clean.py` | `cleaned_sentimentdataset.csv` |
| Load | `phase1/src/load_mysql.py` | MySQL `social_posts` + printouts |
| ML EDA | `phase2/notebooks/02_ml_eda.ipynb` | Class balance / text length |

EDA focuses on platform volume, sentiment-group mix, country concentration, engagement (likes/retweets), and temporal fields. Cleaning is intentionally conservative so Phase 2 and Phase 3 agents share the same source of truth in MySQL.

### 3.3 Machine Learning Models

Pipeline (`python -m phase2.src.pipeline_run --run-id v1`): load MySQL → stratified split → GridSearchCV on train only (3-fold stratified) → score validation → alias validation winner as `best_v1` → predict all posts → evaluate test → write MySQL + artifacts.

| Version | Classifier | Role |
|---------|------------|------|
| `logreg_v1` | Logistic Regression (TF-IDF) | Interpretable linear baseline |
| `linearsvc_v1` | Calibrated Linear SVM | Margin classifier + probabilities |
| `multinomialnb_v1` | Multinomial Naive Bayes | Validation winner / champion |
| `best_v1` | Copy of validation winner | Default inference for agents |

Feature pipeline: TF-IDF unigrams (tuned `max_features`, typically 3000) with classifier-specific regularisation (C / alpha). Selection criterion is validation `f1_macro` to avoid majority-class collapse; final reported numbers use the held-out test set.

**Validation comparison (run v1):**

| Model (val) | Val accuracy | Val F1-macro |
|-------------|--------------|--------------|
| `logreg_v1` | 88.2% | 0.600 |
| `linearsvc_v1` | 91.8% | 0.621 |
| `multinomialnb_v1` (best) | 94.5% | 0.638 |

**Champion test metrics (`best_v1`):**

| Champion test metric | Value |
|----------------------|-------|
| Accuracy | 90.0% |
| F1-macro | ≈ 0.594 |
| F1-weighted | ≈ 0.890 |
| Positive F1 (support 75) | ≈ 0.930 |
| Negative F1 (support 33) | ≈ 0.852 |
| Neutral F1 (support 2) | 0.0 (insufficient support) |

### 3.4 Unified MCP & Agentic Supervisor

Phase 3 exposes one FastMCP server (`unified-sentiment-mcp`) with tool buckets consumed by LangGraph workers:

| Worker | Tools |
|--------|-------|
| `sql_agent` | `mysql_query`, `mysql_sentiment_summary`, `mysql_prediction_agreement`, `mysql_latest_metrics` |
| `ml_agent` | `ml_list_artifacts`, `ml_get_comparison`, `ml_get_best_metrics`, `ml_predict`, `fred_*` |
| `email_agent` | `email_draft` / `email_send` (Gmail SMTP) |
| `whatsapp_agent` | `whatsapp_draft` / `whatsapp_send` (Twilio) |

**Star topology.** The supervisor is the only hub: workers return to the supervisor and never call each other. Planning uses `plan_from_query` so predict-only, metrics, SQL column questions, country filters, and alert workflows take deterministic paths. Optional LLM routing uses Ollama (`llama3.2:latest`, temperature 0) with Gemini 2.5 Flash failover. Tool results cache with TTL (default 300s). Alert drafting pulls live SQL/ML facts before composing copy.

**HITL path.** After drafts, the graph interrupts; Streamlit shows editable Email/WhatsApp forms; Approve resumes send (only if `ALERTS_ENABLED=true`); Reject cancels. CLI `--query` drafts and logs without interrupt.

```
                    ┌─────────────┐
                    │  supervisor │◄──────────────────┐
                    └──────┬──────┘                   │
           ┌───────────────┼───────────────┐          │
           ▼               ▼               ▼          │
      sql_agent       ml_agent      email_agent       │
           │               │       whatsapp_agent     │
           └───────────────┴───────────────┘──────────┘
                              │
                    human_review → (approve) → send_alerts → END
```

### 3.5 Streamlit Analytics & Brand Assistant

| Surface | Capabilities |
|---------|--------------|
| Analytics page | KPIs (posts, positive %, negative %, agreement), Plotly sentiment pie, platform donut, top countries |
| Brand Assistant | Natural-language Q&A, classify text, dataset lookups, grounded alert drafts, Approve/Reject, chat history |
| Follow-ups | Remembers last classified text for “platform of the above post” style lookups |

Recommended launch: `streamlit run phase3/ui/app.py` (persistent MCP runtime avoids respawning stdio per chat turn).

---

## 4. System Architecture

| Layer | Tools |
|-------|-------|
| Database | MySQL 8.x, PyMySQL / SQLAlchemy |
| ETL / cleaning | Python, pandas |
| ML | scikit-learn, joblib, TF-IDF pipelines |
| Agents | LangChain, LangGraph (`StateGraph`, `interrupt`, `MemorySaver`) |
| Tools protocol | MCP FastMCP, langchain-mcp-adapters |
| LLM | Ollama primary; Gemini fallback; optional OpenAI/Anthropic |
| UI | Streamlit, Plotly |
| Outbound | Gmail SMTP, Twilio WhatsApp |
| Enrichment | FRED API (optional macro series) |
| Outputs | CSV, joblib, JSON metrics, confusion PNGs, alert JSON logs |

**End-to-end flow:**

```
Raw CSV → EDA/clean → MySQL → train/val/test ML → artifacts + DB write-back
       → Unified MCP tools → LangGraph star supervisor → Streamlit (Analytics + Brand Assistant)
```

---

## 5. Results

### 5.1 Brand Intelligence KPIs

After Phase 1 load and Phase 2 run `v1`, the operating dataset is multi-platform (Instagram, Twitter, Facebook observed in production queries) with ternary sentiment labels. Exact row totals depend on the cleaned CSV load; Analytics caches KPI queries for 300 seconds.

| KPI | Observed (run v1 / Live DB) |
|-----|-----------------------------|
| Train / Val / Test posts (ML split) | 512 / 110 / 110 |
| Champion model | `multinomialnb_v1` (`best_v1`) |
| Test accuracy | 90.0% |
| Test F1-macro | ≈ 0.594 |
| Validation F1-macro (selection) | ≈ 0.638 |
| Inference default for assistants | `best_v1.joblib` |
| Tool cache TTL | 300 seconds (default) |

### 5.2 Sentiment & Platform Patterns

SQL and Streamlit analytics surface platform volume, sentiment-group mix, and country concentration. Column intents (e.g., hashtags overview) return fill rates and top values rather than schema dumps. Country questions such as “how many customers from USA?” resolve to filtered user/post counts instead of generic help text. Post follow-ups after classification look up `platform` / `username` for matching text in `social_posts`.

### 5.3 Model Comparison (Validation / Test)

| Model | Val Acc | Val F1-macro | Selected |
|-------|---------|--------------|----------|
| `logreg_v1` | 88.2% | 0.600 | No |
| `linearsvc_v1` | 91.8% | 0.621 | No |
| `multinomialnb_v1` | 94.5% | 0.638 | Yes → `best_v1` |

**Interpretation.** MultinomialNB led validation F1-macro and was aliased to `best_v1`. Test accuracy is high (90%), but macro-F1 is limited by extremely rare Neutral support on the test slice (n=2), which is an important operational caveat when presenting three-class capability.

| Class (test) | Precision | Recall | F1 | Support |
|--------------|-----------|--------|-----|---------|
| Positive | 0.890 | 0.973 | 0.930 | 75 |
| Negative | 0.929 | 0.788 | 0.852 | 33 |
| Neutral | 0.000 | 0.000 | 0.000 | 2 |

### 5.4 Agent Routing & Tool Outcomes

- Predict / classify with quoted text → `ml_agent` / `ml_predict` only (no unnecessary SQL dumps).
- Sentiment counts / platforms / countries / hashtags / usernames → `sql_agent` with typed plans.
- Model winner / metrics wording → `ml_agent` comparison artifacts.
- Alerts → gather SQL/ML facts first, then draft Email/WhatsApp; never send without HITL when Streamlit is used.
- Schema / “what SQL was performed” meta-questions are declined with a safe capability message.

### 5.5 HITL Alert Safety

| Control | Behavior |
|---------|----------|
| Default | Draft + JSON log under `phase3/logs/` |
| Streamlit interrupt | Editable drafts; Approve / Reject required |
| `ALERTS_ENABLED=false` | Approve still logs only — no real SMTP/Twilio send |
| `ALERTS_ENABLED=true` + Approve | Send using configured SMTP/Twilio credentials |
| Grounding | Bodies built from tool findings; marketing fluff filtered |

### 5.6 Output Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| Cleaned CSV | `dataset/cleaned_sentimentdataset.csv` | Post-level clean load |
| Models | `phase2/artifacts/*_v1.joblib` | Tuned pipelines + `best_v1` |
| Comparison | `phase2/artifacts/comparison_v1.json` | Val metrics & winner |
| Test metrics | `phase2/artifacts/metrics_best_v1.json` | Holdout report |
| Confusion charts | `phase2/artifacts/` | PNG evaluation figures |
| MCP server | `phase3/mcp/server.py` | Unified tool surface |
| Supervisor | `phase3/agents/graph.py` | Star routing + workers |
| Streamlit app | `phase3/ui/app.py` | Analytics + Assistant |
| Alert logs | `phase3/logs/` | Draft/send audit files |

---

## 6. Conclusion

This report documents a three-phase AI-enabled agentic platform for social media sentiment and brand intelligence. Phase 1 delivers a validated MySQL store and SQL analytics; Phase 2 trains three TF-IDF classifiers and promotes `multinomialnb_v1` to `best_v1` (90% test accuracy; macro-F1 constrained by Neutral scarcity); Phase 3 wraps MySQL, ML, Email, and WhatsApp in Unified MCP under a LangGraph star supervisor with Streamlit Analytics and a HITL Brand Assistant. Deterministic intents, tool caching, and dual send locks make the stack suitable for demo-to-operations brand monitoring workflows.

---

## 7. Discussion

### 7.1 Product & Operational Insights

1. **Three-class reporting beats raw emotion strings** for dashboarding and agreement queries against `best_v1`.
2. **Validation-gated selection** prevents train-set optimism; MultinomialNB won on val F1-macro.
3. **Agents must be tool-bound** — free-form LLM answers invent metrics; SQL/ML tools ground replies.
4. **HITL is non-negotiable for outbound** — drafts without Approve (and `ALERTS_ENABLED`) never become customer-facing messages.
5. **Follow-up context matters** — “platform of the above text” needs remembered post text, not platform aggregates.
6. **Failover LLMs** keep routing available when local Ollama is down, while temperature 0 reduces tool-selection jitter.

### 7.2 Limitations

- Neutral class is rare on the holdout slice (support = 2), so macro-F1 understates Positive/Negative quality.
- Classical TF-IDF models miss deeper context, sarcasm, and multilingual content.
- Dataset is project CSV-scale, not a live firehose; no continuous scraper in scope.
- LangGraph `MemorySaver` is process-local; Streamlit restarts can require new thread IDs.
- SMTP/Twilio deliverability depends on operator credentials and provider policies.
- Intent routing is regex/heuristic hybrid—novel phrasings may still need expansion.
- FRED enrichment is optional and adjacent to core brand sentiment, not causal of sentiment shifts.

### 7.3 Recommendations & Future Work

| Priority | Action |
|----------|--------|
| High | Collect more Neutral-labeled posts; rebalance or use cost-sensitive training |
| High | Add fairness / platform-slice metrics on champion predictions |
| Medium | Upgrade embeddings or fine-tuned transformers while keeping MCP `ml_predict` API |
| Medium | Persistent checkpointer (SQLite/Postgres) for multi-user HITL sessions |
| Medium | Scheduled Phase 1/2 refresh jobs feeding Analytics cache invalidation |
| Low | Live platform APIs → MySQL CDC for near-real-time listening |
| Low | DirectQuery-style dashboards mirroring MySQL KPIs outside Streamlit |

Near-term GenAI extensions can deepen natural-language explanations of confusion-matrix errors and campaign-level narratives, always grounded in `mysql_*` / `ml_*` tool outputs rather than unconstrained generation.

---

## 8. References

1. Pang, B., & Lee, L. (2008). Opinion Mining and Sentiment Analysis. *Foundations and Trends in Information Retrieval*.
2. Bird, S., Klein, E., & Loper, E. — NLTK / classical NLP pipelines for short text baselines.
3. Pedregosa et al. — scikit-learn: Machine Learning in Python (TF-IDF, GridSearchCV, calibration).
4. LangChain / LangGraph documentation — supervisor graphs, interrupts, checkpointing.
5. Model Context Protocol (MCP) specification — tool servers over stdio/SSE.
6. Streamlit documentation — multipage apps, chat elements, forms for HITL.
7. Ollama documentation — local LLM serving; Google AI Studio — Gemini API keys.
8. MySQL 8.0 Reference Manual — indexing, aggregations, LIKE filtering for analytics.
9. Twilio WhatsApp / Gmail SMTP documentation — outbound messaging integrations.
10. Federal Reserve Economic Data (FRED) API — optional macro enrichment series.
11. Project repository README — Social Media Sentiment & Brand Intelligence Platform (Phases 1–3).
