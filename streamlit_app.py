# Streamlit Cloud entrypoint for RetailPulse
# Streamlit Cloud can use this root-level file as the main app path.

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import dashboard.app  # noqa: F401
except Exception as exc:
    st.set_page_config(page_title="RetailPulse", page_icon="📊")
    st.title("RetailPulse Dashboard Failed to Load")
    st.error("Unable to import dashboard.app")
    st.exception(exc)
