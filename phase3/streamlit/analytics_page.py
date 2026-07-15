"""Page: Brand Intelligence Analytics dashboard."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from phase3.ui.analytics_data import (
    load_agreement_pct,
    load_country_rows,
    load_platform_rows,
    load_sentiment_rows,
    load_val_accuracy,
    rows_to_frame,
    sentiment_kpis,
)

_SENTIMENT_COLORS = {
    "Positive": "#2ecc71",
    "Negative": "#e74c3c",
    "Neutral": "#95a5a6",
}

_PLATFORM_COLORS = ["#3498db", "#9b59b6", "#1abc9c", "#f39c12", "#e67e22", "#34495e"]


def _inject_kpi_styles() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background: linear-gradient(160deg, rgba(40, 48, 64, 0.95), rgba(24, 28, 38, 0.98));
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 1rem 1.1rem 0.85rem 1.1rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
        }
        div[data-testid="stMetric"] label {
            color: rgba(255, 255, 255, 0.72) !important;
            font-size: 0.86rem !important;
            letter-spacing: 0.02em;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #f5f7fa !important;
            font-size: 1.65rem !important;
            font-weight: 650 !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            font-size: 0.85rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _pie_chart(df, *, names: str, values: str, title: str, hole: float = 0.0, color_map=None):
    if df.empty:
        return None
    color_kwargs = {}
    if color_map:
        color_kwargs["color_discrete_map"] = color_map
    else:
        color_kwargs["color_discrete_sequence"] = _PLATFORM_COLORS
    fig = px.pie(
        df,
        names=names,
        values=values,
        hole=hole,
        title=title,
        **color_kwargs,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="%{label}<br>%{value:,} (%{percent})<extra></extra>",
    )
    fig.update_layout(
        margin=dict(t=48, b=16, l=16, r=16),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            xanchor="center",
            x=0.5,
            title_text="",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eaed"),
        height=380,
        showlegend=True,
    )
    return fig


def render_analytics_page() -> None:
    _inject_kpi_styles()
    st.title("Brand Intelligence Analytics")
    st.caption("Live snapshot from MySQL social posts and model metrics.")

    sentiment_rows = load_sentiment_rows()
    kpis = sentiment_kpis(sentiment_rows)
    agreement = load_agreement_pct("best_v1")
    val_acc = load_val_accuracy()
    fourth_label = "Model agreement %"
    fourth_value = f"{agreement:.1f}%" if agreement is not None else "—"
    fourth_delta = "best_v1 vs labels"
    if agreement is None and val_acc is not None:
        fourth_label = "Val accuracy"
        fourth_value = f"{val_acc:.1%}" if val_acc <= 1 else f"{val_acc:.1f}"
        fourth_delta = "best model"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total posts", f"{kpis['total']:,}", delta="all platforms")
    c2.metric(
        "Positive %",
        f"{kpis['positive_pct']:.1f}%",
        delta=f"{kpis['positive']:,} posts",
    )
    c3.metric(
        "Negative %",
        f"{kpis['negative_pct']:.1f}%",
        delta=f"{kpis['negative']:,} posts",
        delta_color="inverse",
    )
    c4.metric(fourth_label, fourth_value, delta=fourth_delta)

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment distribution")
        sent_df = rows_to_frame(sentiment_rows, index_col="sentiment_group")
        if sent_df.empty:
            st.info("No sentiment rows available.")
        else:
            fig = _pie_chart(
                sent_df,
                names="sentiment_group",
                values="n",
                title="Share by sentiment",
                hole=0.0,
                color_map=_SENTIMENT_COLORS,
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Posts by platform")
        plat_df = rows_to_frame(load_platform_rows(), index_col="platform")
        if plat_df.empty:
            st.info("No platform rows available.")
        else:
            fig = _pie_chart(
                plat_df,
                names="platform",
                values="n",
                title="Share by platform",
                hole=0.58,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top countries")
    country_df = rows_to_frame(load_country_rows(10), index_col="country")
    if country_df.empty:
        st.caption("No country breakdown available.")
    else:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=country_df["n"],
                    y=country_df["country"],
                    orientation="h",
                    marker_color="#5dade2",
                    hovertemplate="%{y}: %{x:,}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            margin=dict(t=16, b=16, l=16, r=16),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8eaed"),
            height=360,
            xaxis=dict(title="Posts", gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(title="", autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)
