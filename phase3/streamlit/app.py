"""Streamlit entry: Analytics + Brand Intelligence Assistant."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from phase3.ui.analytics_page import render_analytics_page
from phase3.ui.assistant_page import render_assistant_page


def main() -> None:
    st.set_page_config(
        page_title="Brand Intelligence",
        page_icon="📊",
        layout="wide",
    )
    analytics = st.Page(
        render_analytics_page,
        title="Analytics",
        icon=":material/analytics:",
        default=True,
    )
    assistant = st.Page(
        render_assistant_page,
        title="Brand Assistant",
        icon=":material/smart_toy:",
    )
    nav = st.navigation([analytics, assistant])
    nav.run()


if __name__ == "__main__":
    main()
