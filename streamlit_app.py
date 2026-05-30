# Streamlit Cloud entrypoint for RetailPulse
# Streamlit Cloud can use this root-level file as the main app path.

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="RetailPulse", page_icon="📊")
st.title("RetailPulse Dashboard Entrypoint")
st.write("Root path:", str(ROOT))
st.write("sys.path[0]:", sys.path[0])

try:
    import dashboard.app  # noqa: F401
    st.write("Imported dashboard.app successfully")
except Exception as exc:
    st.title("RetailPulse Dashboard Failed to Load")
    st.error("Unable to import dashboard.app")
    st.exception(exc)
