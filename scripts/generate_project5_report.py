"""Generate Project 5 research report PDF (mirrors Project 4 structure)."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(__file__).resolve().parent.parent / "PROJECT_5_RESEARCH_REPORT.pdf"
TITLE = (
    "AI enabled Agentic Platform for Social Media Sentiment "
    "& Brand Intelligence Platform"
)


def styles():
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle(
            "TTitle",
            parent=base["Title"],
            fontSize=16,
            leading=20,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "TSub",
            parent=base["Normal"],
            fontSize=11,
            alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "TH1",
            parent=base["Heading1"],
            fontSize=13,
            spaceBefore=14,
            spaceAfter=8,
            textColor=colors.HexColor("#1a365d"),
        ),
        "h2": ParagraphStyle(
            "TH2",
            parent=base["Heading2"],
            fontSize=11,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#2c5282"),
        ),
        "h3": ParagraphStyle(
            "TH3",
            parent=base["Heading3"],
            fontSize=10,
            spaceBefore=8,
            spaceAfter=4,
            textColor=colors.HexColor("#2d3748"),
        ),
        "body": ParagraphStyle(
            "TBody",
            parent=base["Normal"],
            fontSize=10,
            leading=13,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "toc": ParagraphStyle(
            "TToc",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            spaceAfter=2,
        ),
        "caption": ParagraphStyle(
            "TCap",
            parent=base["Normal"],
            fontSize=9,
            leading=11,
            textColor=colors.HexColor("#4a5568"),
            spaceAfter=8,
        ),
        "kw": ParagraphStyle(
            "TKw",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            spaceBefore=6,
            spaceAfter=10,
        ),
    }
    return s


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#4a5568"))
    page = canvas.getPageNumber()
    canvas.drawCentredString(letter[0] / 2, 0.55 * inch, f"{page}")
    canvas.restoreState()


def table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5282")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#f7fafc")],
                ),
            ]
        )
    )
    return t


def bullets(items, style):
    return ListFlowable(
        [ListItem(Paragraph(i, style), leftIndent=12, bulletColor=colors.HexColor("#2c5282")) for i in items],
        bulletType="bullet",
        start="•",
        leftIndent=18,
        spaceAfter=8,
    )


def build():
    s = styles()
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.75 * inch,
        title=TITLE,
        author="Project 5",
    )
    story = []
    P = lambda t, st="body": Paragraph(t, s[st])

    # Cover / TOC
    story.append(P(TITLE, "title"))
    story.append(P("Project 5 — Research Report", "subtitle"))
    story.append(P("Table of Contents", "h1"))
    toc = [
        ("—", "Abstract", "1"),
        ("1", "Introduction", "2"),
        ("1.1", "Background", "2"),
        ("1.2", "Objectives", "2"),
        ("1.3", "Scope", "2"),
        ("2", "Related Work", "3"),
        ("3", "Dataset & Methodology", "3"),
        ("3.1", "SQL Data Model & Analytics", "4"),
        ("3.2", "Exploratory Analysis & Cleaning", "5"),
        ("3.3", "Machine Learning Models", "6"),
        ("3.4", "Unified MCP & Agentic Supervisor", "7"),
        ("3.5", "Streamlit Analytics & Brand Assistant", "8"),
        ("4", "System Architecture", "9"),
        ("5", "Results", "9"),
        ("5.1", "Brand Intelligence KPIs", "9"),
        ("5.2", "Sentiment & Platform Patterns", "10"),
        ("5.3", "Model Comparison (Validation / Test)", "10"),
        ("5.4", "Agent Routing & Tool Outcomes", "11"),
        ("5.5", "HITL Alert Safety", "11"),
        ("5.6", "Output Artifacts", "12"),
        ("6", "Conclusion", "12"),
        ("7", "Discussion", "13"),
        ("7.1", "Product & Operational Insights", "13"),
        ("7.2", "Limitations", "13"),
        ("7.3", "Recommendations & Future Work", "14"),
        ("8", "References", "15"),
    ]
    for num, title, page in toc:
        label = f"{num} {title}" if num != "—" else title
        story.append(P(f"{label}{'.' * max(2, 55 - len(label))} {page}", "toc"))

    story.append(PageBreak())

    # Abstract
    story.append(P("Abstract", "h1"))
    story.append(
        P(
            "This study presents an end-to-end AI-enabled agentic platform for social media "
            "sentiment and brand intelligence across three integrated phases: MySQL data "
            "integration and SQL analytics, classical text classification with train/"
            "validation/test selection, and a LangGraph star-topology supervisor over a "
            "Unified Model Context Protocol (MCP) tool server. Raw social posts are cleaned "
            "into Positive / Negative / Neutral labels, stored in MySQL "
            "(<font face='Courier'>sentiment_brand_intel.social_posts</font>), and used to "
            "train three TF-IDF classifiers (Logistic Regression, Calibrated Linear SVM, "
            "Multinomial Naive Bayes). The validation winner "
            "(<font face='Courier'>multinomialnb_v1</font>, aliased as "
            "<font face='Courier'>best_v1</font>) achieved 90.0% test accuracy and "
            "approximately 0.594 macro-F1 on a held-out stratified test set. Phase 3 exposes "
            "MySQL, ML, Email (Gmail SMTP), WhatsApp (Twilio), and optional FRED tools through "
            "MCP; a Streamlit Brand Assistant answers analytics and prediction questions with "
            "deterministic intent routing, tool caching, Ollama-primary / Gemini-fallback LLMs, "
            "and human-in-the-loop approval before any outbound alert is sent."
        )
    )
    story.append(
        P(
            "<b>Keywords:</b> Social media sentiment, brand intelligence, MySQL, TF-IDF, "
            "Multinomial Naive Bayes, LangGraph, Model Context Protocol (MCP), Streamlit, "
            "HITL alerts, Ollama, Gemini",
            "kw",
        )
    )

    # 1 Introduction
    story.append(P("1. Introduction", "h1"))
    story.append(P("1.1 Background", "h2"))
    story.append(
        P(
            "Brands continuously monitor public conversation across social platforms to detect "
            "shifts in sentiment, emerging hashtags, and engagement patterns. Raw feeds are "
            "noisy: fine-grained emotion labels differ by source, timestamps and usernames are "
            "inconsistent, and operational teams need both reproducible analytics and natural-"
            "language access to grounded metrics—not invented marketing copy."
        )
    )
    story.append(
        P(
            "Pairing a relational analytics layer with classical NLP classifiers and an "
            "agentic assistant gives a full stack: SQL for trustworthy aggregates, ML for "
            "on-demand classification with known quality metrics, and a supervisor that "
            "orchestrates specialist tools (SQL, ML, Email, WhatsApp) under explicit "
            "human approval for outbound messaging."
        )
    )

    story.append(P("1.2 Objectives", "h2"))
    story.append(
        bullets(
            [
                "Ingest and clean social posts into MySQL with a three-class <font face='Courier'>sentiment_group</font> label and validated indexes.",
                "Deliver SQL analytics covering platforms, countries, hashtags, volume, engagement, and sentiment mix.",
                "Train and compare three TF-IDF classifiers with GridSearchCV on train only; select on validation <font face='Courier'>f1_macro</font>; report on held-out test.",
                "Write predictions and metrics back to MySQL and persist joblib / JSON artifacts for agents.",
                "Expose MySQL, ML, Email, WhatsApp, and optional FRED through one Unified MCP server.",
                "Build a LangGraph star supervisor with deterministic routing, tool caching, and LLM failover (Ollama → Gemini).",
                "Ship a Streamlit UI with Analytics (KPIs/charts) and Brand Assistant (HITL Approve/Reject for alerts).",
            ],
            s["body"],
        )
    )

    story.append(P("1.3 Scope", "h2"))
    story.append(
        P(
            "Phases 1–3 cover SQL BI and cleaning, machine learning, and the agentic MCP + "
            "Streamlit layer. The analytical grain is one cleaned post row in "
            "<font face='Courier'>social_posts</font>. Fine-grained source sentiments are "
            "coarsened to Positive / Negative / Neutral. Live continuous scrapers and deep "
            "transformer fine-tuning are out of scope; optional future work includes durable "
            "graph checkpoints, richer models, and DirectQuery-style live feeds."
        )
    )

    story.append(PageBreak())

    # 2 Related Work
    story.append(P("2. Related Work", "h1"))
    story.append(
        P(
            "Brand and social listening systems commonly combine: (1) relational storage of "
            "post-level engagement attributes; (2) lexicon or classical ML sentiment "
            "classifiers before neural deployments; (3) BI dashboards for volume and mix; and "
            "(4) increasingly, LLM agents that call tools rather than free-form invent facts."
        )
    )
    story.append(
        bullets(
            [
                "<b>Sentiment coarsening</b> — Mapping rich emotion taxonomies into Positive / Negative / Neutral for stable reporting and modeling.",
                "<b>TF-IDF + linear / NB models</b> — Strong baselines for short social text with transparent artifacts and low latency inference.",
                "<b>Multi-agent orchestration</b> — Supervisor–worker patterns (here LangGraph star topology) keep specialists isolated and auditable.",
                "<b>Tool protocols</b> — MCP standardizes tool exposure so Streamlit, CLI, and IDE clients share one server.",
                "<b>HITL for outbound actions</b> — Draft-first alerting with explicit Approve prevents accidental customer or partner spam.",
            ],
            s["body"],
        )
    )
    story.append(
        P(
            "This project extends coursework-style ETL + ML into an operable agent platform: "
            "validation-gated model selection, MySQL write-back for agreement queries, "
            "deterministic SQL/ML intent routing so small local models cannot skip tools, and "
            "dual locks (<font face='Courier'>ALERTS_ENABLED</font> + human Approve) on sends."
        )
    )

    # 3 Dataset & Methodology
    story.append(P("3. Dataset & Methodology", "h1"))
    story.append(
        table(
            [
                ["Attribute", "Detail"],
                ["Source", "Project social sentiment CSV (multi-platform posts)"],
                ["Local files", "dataset/sentimentdataset.csv → cleaned_sentimentdataset.csv"],
                ["Database", "MySQL sentiment_brand_intel"],
                ["Core table", "social_posts (cleaned post grain)"],
                ["ML tables", "model_predictions, model_metrics"],
                ["Target", "sentiment_group ∈ {Positive, Negative, Neutral}"],
                ["Split", "Stratified 70% / 15% / 15% train / val / test (seed 42)"],
                ["n_train / n_val / n_test", "512 / 110 / 110 (run v1)"],
                ["Champion", "multinomialnb_v1 → best_v1 alias"],
                ["Test accuracy / macro-F1", "0.900 / ≈ 0.594"],
                ["UI", "Streamlit Analytics + Brand Assistant"],
                ["Agents", "LangGraph supervisor + SQL / ML / Email / WhatsApp"],
                ["Tools protocol", "Unified MCP (stdio) + local LangChain fallback"],
            ],
            col_widths=[1.6 * inch, 4.6 * inch],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        P(
            "<b>Data quality decisions:</b> strip whitespace; parse timestamps into year/"
            "month/day/hour fields; map fine-grained emotions to three sentiment groups; "
            "preserve platform, username, country, hashtags, likes, and retweets for SQL "
            "analytics and follow-up post lookups by text snippet."
        )
    )

    story.append(P("3.1 SQL Data Model & Analytics", "h2"))
    story.append(
        table(
            [
                ["Step", "Script / file", "Output"],
                ["Schema", "phase1/sql/01_schema.sql", "Tables + types"],
                ["Indexes", "phase1/sql/02_indexes.sql", "platform, sentiment, country, …"],
                ["Load", "phase1/src/load_mysql.py", "Populate social_posts"],
                ["Analytics", "phase1/sql/03_analytics_queries.sql", "Platform/country/hashtag SQL"],
                ["ML verify", "phase2/sql/04_ml_verification.sql", "Metrics & agreement checks"],
            ],
            col_widths=[1.1 * inch, 2.5 * inch, 2.6 * inch],
        )
    )
    story.append(Spacer(1, 6))
    story.append(P("Schema grain:", "body"))
    story.append(
        table(
            [
                ["Table", "Grain", "Key columns"],
                [
                    "social_posts",
                    "1 row / post",
                    "id, text, sentiment, sentiment_group, username, platform, hashtags, likes, retweets, country, timestamp parts",
                ],
                [
                    "model_predictions",
                    "1 row / post × model",
                    "post_id, model_version, predicted_sentiment, confidence",
                ],
                [
                    "model_metrics",
                    "1 row / metric",
                    "model_version, metric_name, metric_value",
                ],
            ],
            col_widths=[1.4 * inch, 1.3 * inch, 3.5 * inch],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        P(
            "Phase 3 maps natural-language analytics to deterministic SELECT plans in "
            "<font face='Courier'>phase3/agents/sql_intents.py</font> (sentiment summary, "
            "platform/country/username aggregates, country filters such as “from USA”, "
            "hashtag column overview, and post lookup by previously classified text)."
        )
    )

    story.append(PageBreak())
    story.append(P("3.2 Exploratory Analysis & Cleaning", "h2"))
    story.append(
        table(
            [
                ["Step", "Notebook / script", "Output"],
                ["EDA", "phase1/notebooks/01_eda.ipynb, eda_report.py", "Distributions & DQ notes"],
                ["Clean", "phase1/src/clean.py", "cleaned_sentimentdataset.csv"],
                ["Load", "phase1/src/load_mysql.py", "MySQL social_posts + printouts"],
                ["ML EDA", "phase2/notebooks/02_ml_eda.ipynb", "Class balance / text length"],
            ],
            col_widths=[1.0 * inch, 3.2 * inch, 2.0 * inch],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        P(
            "EDA focuses on platform volume, sentiment-group mix, country concentration, "
            "engagement (likes/retweets), and temporal fields. Cleaning is intentionally "
            "conservative so Phase 2 and Phase 3 agents share the same source of truth in MySQL."
        )
    )

    story.append(P("3.3 Machine Learning Models", "h2"))
    story.append(
        P(
            "Pipeline (<font face='Courier'>python -m phase2.src.pipeline_run --run-id v1</font>): "
            "load MySQL → stratified split → GridSearchCV on train only (3-fold stratified) → "
            "score validation → alias validation winner as <font face='Courier'>best_v1</font> → "
            "predict all posts → evaluate test → write MySQL + artifacts."
        )
    )
    story.append(
        table(
            [
                ["Version", "Classifier", "Role"],
                ["logreg_v1", "Logistic Regression (TF-IDF)", "Interpretable linear baseline"],
                ["linearsvc_v1", "Calibrated Linear SVM", "Margin classifier + probabilities"],
                ["multinomialnb_v1", "Multinomial Naive Bayes", "Validation winner / champion"],
                ["best_v1", "Copy of validation winner", "Default inference for agents"],
            ],
            col_widths=[1.5 * inch, 2.5 * inch, 2.2 * inch],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        P(
            "Feature pipeline: TF-IDF unigrams (tuned <font face='Courier'>max_features</font>, "
            "typically 3000) with classifier-specific regularisation "
            "(C / alpha). Selection criterion is validation <font face='Courier'>f1_macro</font> "
            "to avoid majority-class collapse; final reported numbers use the held-out test set."
        )
    )
    story.append(
        table(
            [
                ["Model (val)", "Val accuracy", "Val F1-macro"],
                ["logreg_v1", "88.2%", "0.600"],
                ["linearsvc_v1", "91.8%", "0.621"],
                ["multinomialnb_v1 (best)", "94.5%", "0.638"],
            ],
            col_widths=[2.2 * inch, 1.5 * inch, 1.5 * inch],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        table(
            [
                ["Champion test metric", "Value"],
                ["Accuracy", "90.0%"],
                ["F1-macro", "≈ 0.594"],
                ["F1-weighted", "≈ 0.890"],
                ["Positive F1 (support 75)", "≈ 0.930"],
                ["Negative F1 (support 33)", "≈ 0.852"],
                ["Neutral F1 (support 2)", "0.0 (insufficient support)"],
            ],
            col_widths=[3.0 * inch, 2.5 * inch],
        )
    )

    story.append(PageBreak())
    story.append(P("3.4 Unified MCP & Agentic Supervisor", "h2"))
    story.append(
        P(
            "Phase 3 exposes one FastMCP server (<font face='Courier'>unified-sentiment-mcp</font>) "
            "with tool buckets consumed by LangGraph workers:"
        )
    )
    story.append(
        table(
            [
                ["Worker", "Tools"],
                ["sql_agent", "mysql_query, mysql_sentiment_summary, mysql_prediction_agreement, mysql_latest_metrics"],
                ["ml_agent", "ml_list_artifacts, ml_get_comparison, ml_get_best_metrics, ml_predict, fred_*"],
                ["email_agent", "email_draft / email_send (Gmail SMTP)"],
                ["whatsapp_agent", "whatsapp_draft / whatsapp_send (Twilio)"],
            ],
            col_widths=[1.4 * inch, 4.8 * inch],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        P(
            "<b>Star topology.</b> The supervisor is the only hub: workers return to the "
            "supervisor and never call each other. Planning uses "
            "<font face='Courier'>plan_from_query</font> so predict-only, metrics, SQL "
            "column questions, country filters, and alert workflows take deterministic "
            "paths. Optional LLM routing uses Ollama (<font face='Courier'>llama3.2:latest</font>, "
            "temperature 0) with Gemini 2.5 Flash failover. Tool results cache with TTL "
            "(default 300s). Alert drafting pulls live SQL/ML facts before composing copy."
        )
    )
    story.append(
        P(
            "<b>HITL path.</b> After drafts, the graph interrupts; Streamlit shows editable "
            "Email/WhatsApp forms; Approve resumes send (only if "
            "<font face='Courier'>ALERTS_ENABLED=true</font>); Reject cancels. CLI "
            "<font face='Courier'>--query</font> drafts and logs without interrupt."
        )
    )

    story.append(P("3.5 Streamlit Analytics & Brand Assistant", "h2"))
    story.append(
        table(
            [
                ["Surface", "Capabilities"],
                [
                    "Analytics page",
                    "KPIs (posts, positive %, negative %, agreement), Plotly sentiment pie, platform donut, top countries",
                ],
                [
                    "Brand Assistant",
                    "Natural-language Q&A, classify text, dataset lookups, grounded alert drafts, Approve/Reject, chat history",
                ],
                [
                    "Follow-ups",
                    "Remembers last classified text for “platform of the above post” style lookups",
                ],
            ],
            col_widths=[1.5 * inch, 4.7 * inch],
        )
    )
    story.append(
        P(
            "Recommended launch: <font face='Courier'>streamlit run phase3/ui/app.py</font> "
            "(persistent MCP runtime avoids respawning stdio per chat turn)."
        )
    )

    # 4 Architecture
    story.append(P("4. System Architecture", "h1"))
    story.append(
        table(
            [
                ["Layer", "Tools"],
                ["Database", "MySQL 8.x, PyMySQL / SQLAlchemy"],
                ["ETL / cleaning", "Python, pandas"],
                ["ML", "scikit-learn, joblib, TF-IDF pipelines"],
                ["Agents", "LangChain, LangGraph (StateGraph, interrupt, MemorySaver)"],
                ["Tools protocol", "MCP FastMCP, langchain-mcp-adapters"],
                ["LLM", "Ollama primary; Gemini fallback; optional OpenAI/Anthropic"],
                ["UI", "Streamlit, Plotly"],
                ["Outbound", "Gmail SMTP, Twilio WhatsApp"],
                ["Enrichment", "FRED API (optional macro series)"],
                ["Outputs", "CSV, joblib, JSON metrics, confusion PNGs, alert JSON logs"],
            ],
            col_widths=[1.5 * inch, 4.7 * inch],
        )
    )

    story.append(PageBreak())

    # 5 Results
    story.append(P("5. Results", "h1"))
    story.append(P("5.1 Brand Intelligence KPIs", "h2"))
    story.append(
        P(
            "After Phase 1 load and Phase 2 run <font face='Courier'>v1</font>, the operating "
            "dataset is multi-platform (Instagram, Twitter, Facebook observed in production "
            "queries) with ternary sentiment labels. Exact row totals depend on the cleaned "
            "CSV load; Analytics caches KPI queries for 300 seconds."
        )
    )
    story.append(
        table(
            [
                ["KPI", "Observed (run v1 / Live DB)"],
                ["Train / Val / Test posts (ML split)", "512 / 110 / 110"],
                ["Champion model", "multinomialnb_v1 (best_v1)"],
                ["Test accuracy", "90.0%"],
                ["Test F1-macro", "≈ 0.594"],
                ["Validation F1-macro (selection)", "≈ 0.638"],
                ["Inference default for assistants", "best_v1.joblib"],
                ["Tool cache TTL", "300 seconds (default)"],
            ],
            col_widths=[3.0 * inch, 3.2 * inch],
        )
    )

    story.append(P("5.2 Sentiment & Platform Patterns", "h2"))
    story.append(
        P(
            "SQL and Streamlit analytics surface platform volume, sentiment-group mix, and "
            "country concentration. Column intents (e.g., hashtags overview) return fill "
            "rates and top values rather than schema dumps. Country questions such as "
            "“how many customers from USA?” resolve to filtered user/post counts instead of "
            "generic help text. Post follow-ups after classification look up "
            "<font face='Courier'>platform</font> / <font face='Courier'>username</font> for "
            "matching text in <font face='Courier'>social_posts</font>."
        )
    )

    story.append(P("5.3 Model Comparison (Validation / Test)", "h2"))
    story.append(
        table(
            [
                ["Model", "Val Acc", "Val F1-macro", "Selected"],
                ["logreg_v1", "88.2%", "0.600", "No"],
                ["linearsvc_v1", "91.8%", "0.621", "No"],
                ["multinomialnb_v1", "94.5%", "0.638", "Yes → best_v1"],
            ],
            col_widths=[1.8 * inch, 1.2 * inch, 1.4 * inch, 1.6 * inch],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        P(
            "<b>Interpretation.</b> MultinomialNB led validation F1-macro and was aliased to "
            "<font face='Courier'>best_v1</font>. Test accuracy is high (90%), but macro-F1 "
            "is limited by extremely rare Neutral support on the test slice (n=2), which is "
            "an important operational caveat when presenting three-class capability."
        )
    )
    story.append(
        table(
            [
                ["Class (test)", "Precision", "Recall", "F1", "Support"],
                ["Positive", "0.890", "0.973", "0.930", "75"],
                ["Negative", "0.929", "0.788", "0.852", "33"],
                ["Neutral", "0.000", "0.000", "0.000", "2"],
            ],
            col_widths=[1.4 * inch, 1.1 * inch, 1.1 * inch, 1.0 * inch, 1.0 * inch],
        )
    )

    story.append(P("5.4 Agent Routing & Tool Outcomes", "h2"))
    story.append(
        bullets(
            [
                "Predict / classify with quoted text → <font face='Courier'>ml_agent</font> / <font face='Courier'>ml_predict</font> only (no unnecessary SQL dumps).",
                "Sentiment counts / platforms / countries / hashtags / usernames → <font face='Courier'>sql_agent</font> with typed plans.",
                "Model winner / metrics wording → <font face='Courier'>ml_agent</font> comparison artifacts.",
                "Alerts → gather SQL/ML facts first, then draft Email/WhatsApp; never send without HITL when Streamlit is used.",
                "Schema / “what SQL was performed” meta-questions are declined with a safe capability message.",
            ],
            s["body"],
        )
    )

    story.append(PageBreak())
    story.append(P("5.5 HITL Alert Safety", "h2"))
    story.append(
        table(
            [
                ["Control", "Behavior"],
                ["Default", "Draft + JSON log under phase3/logs/"],
                ["Streamlit interrupt", "Editable drafts; Approve / Reject required"],
                ["ALERTS_ENABLED=false", "Approve still logs only — no real SMTP/Twilio send"],
                ["ALERTS_ENABLED=true + Approve", "Send using configured SMTP/Twilio credentials"],
                ["Grounding", "Bodies built from tool findings; marketing fluff filtered"],
            ],
            col_widths=[2.0 * inch, 4.2 * inch],
        )
    )

    story.append(P("5.6 Output Artifacts", "h2"))
    story.append(
        table(
            [
                ["Artifact", "Location", "Description"],
                ["Cleaned CSV", "dataset/cleaned_sentimentdataset.csv", "Post-level clean load"],
                ["Models", "phase2/artifacts/*_v1.joblib", "Tuned pipelines + best_v1"],
                ["Comparison", "phase2/artifacts/comparison_v1.json", "Val metrics & winner"],
                ["Test metrics", "phase2/artifacts/metrics_best_v1.json", "Holdout report"],
                ["Confusion charts", "phase2/artifacts/", "PNG evaluation figures"],
                ["MCP server", "phase3/mcp/server.py", "Unified tool surface"],
                ["Supervisor", "phase3/agents/graph.py", "Star routing + workers"],
                ["Streamlit app", "phase3/ui/app.py", "Analytics + Assistant"],
                ["Alert logs", "phase3/logs/", "Draft/send audit files"],
            ],
            col_widths=[1.4 * inch, 2.4 * inch, 2.4 * inch],
        )
    )

    # 6 Conclusion
    story.append(P("6. Conclusion", "h1"))
    story.append(
        P(
            "This report documents a three-phase AI-enabled agentic platform for social media "
            "sentiment and brand intelligence. Phase 1 delivers a validated MySQL store and "
            "SQL analytics; Phase 2 trains three TF-IDF classifiers and promotes "
            "<font face='Courier'>multinomialnb_v1</font> to <font face='Courier'>best_v1</font> "
            "(90% test accuracy; macro-F1 constrained by Neutral scarcity); Phase 3 wraps MySQL, "
            "ML, Email, and WhatsApp in Unified MCP under a LangGraph star supervisor with "
            "Streamlit Analytics and a HITL Brand Assistant. Deterministic intents, tool "
            "caching, and dual send locks make the stack suitable for demo-to-operations brand "
            "monitoring workflows."
        )
    )

    # 7 Discussion
    story.append(P("7. Discussion", "h1"))
    story.append(P("7.1 Product & Operational Insights", "h2"))
    story.append(
        bullets(
            [
                "<b>Three-class reporting beats raw emotion strings</b> for dashboarding and agreement queries against <font face='Courier'>best_v1</font>.",
                "<b>Validation-gated selection</b> prevents train-set optimism; MultinomialNB won on val F1-macro.",
                "<b>Agents must be tool-bound</b> — free-form LLM answers invent metrics; SQL/ML tools ground replies.",
                "<b>HITL is non-negotiable for outbound</b> — drafts without Approve (and ALERTS_ENABLED) never become customer-facing messages.",
                "<b>Follow-up context matters</b> — “platform of the above text” needs remembered post text, not platform aggregates.",
                "<b>Failover LLMs</b> keep routing available when local Ollama is down, while temperature 0 reduces tool-selection jitter.",
            ],
            s["body"],
        )
    )

    story.append(P("7.2 Limitations", "h2"))
    story.append(
        bullets(
            [
                "Neutral class is rare on the holdout slice (support = 2), so macro-F1 understates Positive/Negative quality.",
                "Classical TF-IDF models miss deeper context, sarcasm, and multilingual content.",
                "Dataset is project CSV-scale, not a live firehose; no continuous scraper in scope.",
                "LangGraph MemorySaver is process-local; Streamlit restarts can require new thread IDs.",
                "SMTP/Twilio deliverability depends on operator credentials and provider policies.",
                "Intent routing is regex/heuristic hybrid—novel phrasings may still need expansion.",
                "FRED enrichment is optional and adjacent to core brand sentiment, not causal of sentiment shifts.",
            ],
            s["body"],
        )
    )

    story.append(PageBreak())
    story.append(P("7.3 Recommendations & Future Work", "h2"))
    story.append(
        table(
            [
                ["Priority", "Action"],
                ["High", "Collect more Neutral-labeled posts; rebalance or use cost-sensitive training"],
                ["High", "Add fairness / platform-slice metrics on champion predictions"],
                ["Medium", "Upgrade embeddings or fine-tuned transformers while keeping MCP ml_predict API"],
                ["Medium", "Persistent checkpointer (SQLite/Postgres) for multi-user HITL sessions"],
                ["Medium", "Scheduled Phase 1/2 refresh jobs feeding Analytics cache invalidation"],
                ["Low", "Live platform APIs → MySQL CDC for near-real-time listening"],
                ["Low", "DirectQuery-style dashboards mirroring MySQL KPIs outside Streamlit"],
            ],
            col_widths=[1.1 * inch, 5.1 * inch],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        P(
            "Near-term GenAI extensions can deepen natural-language explanations of "
            "confusion-matrix errors and campaign-level narratives, always grounded in "
            "mysql_* / ml_* tool outputs rather than unconstrained generation."
        )
    )

    # 8 References
    story.append(P("8. References", "h1"))
    refs = [
        "Pang, B., & Lee, L. (2008). Opinion Mining and Sentiment Analysis. Foundations and Trends in Information Retrieval.",
        "Bird, S., Klein, E., & Loper, E. — NLTK / classical NLP pipelines for short text baselines.",
        "Pedregosa et al. — scikit-learn: Machine Learning in Python (TF-IDF, GridSearchCV, calibration).",
        "LangChain / LangGraph documentation — supervisor graphs, interrupts, checkpointing.",
        "Model Context Protocol (MCP) specification — tool servers over stdio/SSE.",
        "Streamlit documentation — multipage apps, chat elements, forms for HITL.",
        "Ollama documentation — local LLM serving; Google AI Studio — Gemini API keys.",
        "MySQL 8.0 Reference Manual — indexing, aggregations, LIKE filtering for analytics.",
        "Twilio WhatsApp / Gmail SMTP documentation — outbound messaging integrations.",
        "Federal Reserve Economic Data (FRED) API — optional macro enrichment series.",
        "Project repository README — Social Media Sentiment & Brand Intelligence Platform (Phases 1–3).",
    ]
    for i, r in enumerate(refs, 1):
        story.append(P(f"{i}. {r}", "caption"))

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
