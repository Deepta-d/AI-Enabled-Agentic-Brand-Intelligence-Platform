# Social Media Sentiment & Brand Intelligence Platform

AI-enabled agentic platform for social media sentiment and brand intelligence.

## Phase 1: Data Pipeline

Phase 1 loads raw social posts from CSV, runs Python EDA/cleaning, stores results in MySQL, and provides Workbench analytics SQL for Phase 2 ML and the SQL Agent.

```
Raw CSV → Python EDA / Cleaning → SQL Queries → MySQL
```

### Prerequisites

- Python 3.10+
- Local MySQL Server + MySQL Workbench
- Project dataset at `dataset/sentimentdataset.csv`

### Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r phase1/requirements.txt
```

2. Copy environment template and set MySQL credentials:

```bash
copy .env.example .env
```

Edit `.env`:

```
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password_here
MYSQL_DATABASE=sentiment_brand_intel
```

3. (Optional) Create the schema manually in MySQL Workbench by running:

- `phase1/sql/01_schema.sql`
- `phase1/sql/02_indexes.sql`

`python -m phase1.src.load_mysql` also applies these scripts automatically.

### Run EDA

CLI summary:

```bash
python -m phase1.src.eda_report
```

Notebook: open `phase1/notebooks/01_eda.ipynb` in Cursor/VS Code and select the `.venv` kernel
(`ipykernel` is installed with Phase 1 requirements).

> **Windows note:** The full `jupyter` meta-package can fail with long-path errors under `.venv`.
> Core packages + `ipykernel` are enough for Phase 1. To enable long paths system-wide, see:
> https://pip.pypa.io/warnings/enable-long-paths

### Clean and load into MySQL

From the project root:

```bash
python -m phase1.src.load_mysql
```

This will:

1. Apply schema + indexes (if needed)
2. Clean the raw CSV (strip whitespace, parse timestamps, map `sentiment_group`, etc.)
3. Write `dataset/cleaned_sentimentdataset.csv`
4. Replace rows in `social_posts` (prediction/metrics stub tables are left intact)
5. Print validation counts by `platform` and `sentiment_group`

### Analytics in MySQL Workbench

Open and run `phase1/sql/03_analytics_queries.sql` after the load. Queries include:

- Counts by platform, sentiment, sentiment_group, country
- Daily/monthly volume and engagement averages
- Sentiment mix over time
- Simple hashtag `LIKE` searches
- Data-quality checks

### Key tables

| Table | Purpose |
|-------|---------|
| `social_posts` | Cleaned posts (Phase 1 source of truth) |
| `model_predictions` | Phase 2 stub for prediction write-back |
| `model_metrics` | Phase 2 stub for evaluation metrics |

### Project layout (Phase 1)

```
dataset/
  sentimentdataset.csv
  cleaned_sentimentdataset.csv   # created by load script
phase1/
  notebooks/01_eda.ipynb
  sql/01_schema.sql
  sql/02_indexes.sql
  sql/03_analytics_queries.sql
  src/config.py
  src/clean.py
  src/eda_report.py
  src/load_mysql.py
  requirements.txt
.env.example
```

## Phase 2: ML Pipeline (3 models + train/val/test)

Phase 2 reads `social_posts` from MySQL, splits data **70% train / 15% validation / 15% test** (stratified on `sentiment_group`), trains three TF-IDF classifiers with **GridSearchCV hyperparameter tuning on train only**, selects the best on validation (`f1_macro`), evaluates on test, and writes predictions/metrics back to MySQL for the Phase 3 ML Agent.

```
MySQL → train/val/test split → train 3 models → validate/select → predict → evaluate → MySQL
```

### Models

| Version | Classifier |
|---------|------------|
| `logreg_v1` | Logistic Regression |
| `linearsvc_v1` | Calibrated Linear SVM |
| `multinomialnb_v1` | Multinomial Naive Bayes |
| `best_v1` | Copy of the validation winner |

### Setup

```bash
.\.venv\Scripts\activate
pip install -r phase2/requirements.txt
```

Requires Phase 1 data already loaded (`python -m phase1.src.load_mysql`).

### Run the full pipeline

```bash
python -m phase2.src.pipeline_run --run-id v1
# Optional: skip tuning
python -m phase2.src.pipeline_run --run-id v1 --no-tune
```

This will:

1. Load posts from MySQL and create stratified train/val/test splits (same seed as before: 42)
2. Tune TF-IDF + classifier hyperparameters with stratified GridSearchCV on **train only**
3. Score tuned models on validation; pick the best by `f1_macro` and save `best_v1.joblib`
4. Write/replace `model_predictions` for all three models + `best_v1`
5. Evaluate on the held-out test set and write/replace `model_metrics`
6. Save artifacts under `phase2/artifacts/` (Phase 3 MCP/`ml_*` tools read these same paths)

### Notebook

Open `phase2/notebooks/02_ml_eda.ipynb` in Cursor and select the `.venv` kernel.

### Verify in MySQL Workbench

Run `phase2/sql/04_ml_verification.sql` to compare metrics, agreement rates, and the selected best model.

### Phase 2 layout

```
phase2/
  requirements.txt
  notebooks/02_ml_eda.ipynb
  sql/04_ml_verification.sql
  src/
    data.py
    models.py
    train.py
    predict.py
    evaluate.py
    pipeline_run.py
  artifacts/          # joblib models, metrics JSON, confusion matrices
```

## Phase 3: Agentic Layer + Unified MCP + Streamlit HITL

Phase 3 adds a **LangGraph Supervisor** (star topology) with **ML**, **SQL**, **Email**, and **WhatsApp** agents. All agents use one **Unified MCP server** that exposes:

`MySQL | ML Pipeline | Email (Gmail SMTP) | WhatsApp (Twilio) | FRED (optional enrichment)`

The **Brand Intelligence Assistant** is a Streamlit chatbot with human-in-the-loop (interrupt / resume) before outbound alerts.

### Alert safety

- Default: **draft + log** under `phase3/logs/`
- Streamlit: drafts pause for **Approve / Reject**; real SMTP/Twilio sends only when `ALERTS_ENABLED=true` **and** the operator Approves
- CLI `--query`: drafts only (no HITL interrupt)

### LLM primary + fallback

- **Primary:** Ollama (`OLLAMA_MODEL`, default `llama3.2:latest`), temperature `0`
- **Fallback:** Gemini 2.5 Flash (`GEMINI_FALLBACK_MODEL=gemini-2.5-flash`) using a free [Google AI Studio](https://aistudio.google.com/apikey) key in `GOOGLE_API_KEY`
- Tool results are cached for `TOOL_CACHE_TTL_SECONDS` (default 300) to reduce repeat LLM/tool work
- User-facing answers are **plain text** (no raw JSON / MCP wrappers)

### Setup

```bash
.\.venv\Scripts\activate
pip install -r phase3/requirements.txt
```

Copy new keys from `.env.example` into your `.env` (SMTP/Twilio/FRED/`GOOGLE_API_KEY` as needed).

### Run Streamlit Brand Assistant (recommended)

```bash
# Ollama should be running for primary LLM
streamlit run phase3/ui/app.py
```

### Run Unified MCP server (stdio)

```bash
python -m phase3.mcp.server
```

Example Cursor MCP config snippet:

```json
{
  "mcpServers": {
    "unified-sentiment": {
      "command": "D:\\\\Social Media Sentiment & Brand Intelligence Platform\\\\.venv\\\\Scripts\\\\python.exe",
      "args": ["-m", "phase3.mcp.server"],
      "cwd": "D:\\\\Social Media Sentiment & Brand Intelligence Platform"
    }
  }
}
```

### Run agents (CLI, no HITL)

Deterministic demo (no LLM required):

```bash
python -m phase3.agents.run --demo
```

```bash
ollama list
python -m phase3.agents.run --query "Summarize sentiment metrics and draft an email alert"
```

`.env` settings:

```
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:latest
GOOGLE_API_KEY=
GEMINI_FALLBACK_MODEL=gemini-2.5-flash
TOOL_CACHE_TTL_SECONDS=300
ALERTS_ENABLED=false
```

Probe MCP tools:

```bash
python -m phase3.agents.run --probe-mcp
```

Fallback to local LangChain tools (no MCP process):

```bash
python -m phase3.agents.run --query "..." --no-mcp
```

### Phase 3 layout

```
phase3/
  requirements.txt
  config.py
  ui/app.py              # Streamlit Brand Assistant (HITL)
  mcp/server.py          # Unified MCP (stdio)
  services/              # MySQL, ML, Email, WhatsApp, FRED
  agents/
    prompts.py           # Brand Assistant system prompt
    llm.py               # Ollama → Gemini failover (temp 0)
    cache.py             # Tool TTL cache
    formatters.py        # Plain-text responses
    tools.py             # local LangChain tools (fallback / --no-mcp)
    mcp_client.py        # MultiServerMCPClient -> Unified MCP
    graph.py             # LangGraph star supervisor + HITL nodes
    hitl.py              # start_run / resume_run helpers
    run.py               # CLI (--demo / --query / --probe-mcp)
  logs/                  # alert drafts
```
