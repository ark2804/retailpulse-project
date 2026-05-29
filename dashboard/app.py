"""
RetailPulse – Streamlit Dashboard (F06)
Multi-page interactive analytics with what-if sliders and CSV exports.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="RetailPulse Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED = Path("data/processed")
REPORTS = Path("reports")

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(ttl=300)
def load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def metric_card(label: str, value, delta=None):
    st.metric(label=label, value=value, delta=delta)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.shields.io/badge/RetailPulse-v2.0-blue", use_column_width=True)
st.sidebar.title("🛒 RetailPulse")
st.sidebar.caption("AI-Powered Retail Analytics")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📈 Demand Forecast", "👥 Customer Segments",
     "🔴 Churn Risk", "📦 Inventory", "⚙️ What-If Analysis"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Data refreshes every 5 minutes")

# ── Pages ─────────────────────────────────────────────────────────────────────

if page == "🏠 Overview":
    st.title("📊 RetailPulse – Executive Dashboard")
    st.markdown("*End-to-End Retail Analytics powered by AI*")

    # Load metrics
    metrics = load_json(REPORTS / "metrics.json") or {}

    col1, col2, col3, col4 = st.columns(4)

    forecast_metrics = metrics.get("forecasting", {})
    churn_metrics = metrics.get("churn", {})
    seg_metrics = metrics.get("segmentation", {})
    inv_metrics = metrics.get("inventory", {})

    with col1:
        mape = forecast_metrics.get("mape_pct", "N/A")
        st.metric("Forecast MAPE", f"{mape}%" if mape != "N/A" else "N/A",
                  delta="≤12% target" if isinstance(mape, float) and mape <= 12 else None)
    with col2:
        auc = churn_metrics.get("auc_roc", "N/A")
        st.metric("Churn AUC-ROC", auc,
                  delta="≥0.88 target" if isinstance(auc, float) and auc >= 0.88 else None)
    with col3:
        n_cust = seg_metrics.get("n_customers", "N/A")
        st.metric("Customers Segmented", f"{n_cust:,}" if isinstance(n_cust, int) else n_cust)
    with col4:
        reorder = inv_metrics.get("reorder_count", "N/A")
        st.metric("Products Need Reorder", reorder)

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📦 Pipeline Status")
        pipeline_info = metrics.get("pipeline", {})
        elapsed = pipeline_info.get("elapsed_seconds", "?")
        n_ds = pipeline_info.get("datasets_processed", "?")
        st.success(f"✅ Pipeline ran in {elapsed}s – processed {n_ds} datasets")

        drift = metrics.get("drift", {})
        status = drift.get("overall_status", "Unknown")
        color = "🟢" if "Stable" in status else ("🟡" if "Moderate" in status else "🔴")
        st.info(f"{color} Data Drift: **{status}**")

    with col_b:
        st.subheader("🎯 Business Impact Targets")
        st.markdown("""
        | Metric | Target | Status |
        |--------|--------|--------|
        | Demand MAPE | ≤ 12% | 🎯 |
        | Churn AUC-ROC | ≥ 0.88 | 🎯 |
        | Stockout reduction | 30–50% | 📌 |
        | Revenue improvement | 15–25% | 📌 |
        """)

    # Segment chart if available
    seg_df = load_csv(REPORTS / "customer_segments.csv")
    if seg_df is not None and "segment" in seg_df.columns:
        st.subheader("👥 Customer Segment Distribution")
        seg_counts = seg_df["segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        try:
            import plotly.express as px
            fig = px.pie(seg_counts, names="Segment", values="Count",
                         color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(seg_counts.set_index("Segment"))


elif page == "📈 Demand Forecast":
    st.title("📈 Demand Forecast (30-Day Horizon)")

    fc_df = load_csv(REPORTS / "forecast_30d.csv")
    if fc_df is None:
        st.warning("No forecast data found. Run the pipeline first.")
    else:
        fc_df["ds"] = pd.to_datetime(fc_df["ds"])

        if "series_id" in fc_df.columns:
            series_list = fc_df["series_id"].unique().tolist()
            selected = st.selectbox("Select Product/Series", series_list[:100])
            plot_df = fc_df[fc_df["series_id"] == selected]
        else:
            plot_df = fc_df

        st.markdown(f"Showing forecast for **{len(plot_df)}** days")

        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_df["ds"], y=plot_df["yhat"],
                                     name="Ensemble Forecast", line=dict(color="#2196F3", width=2)))
            if "yhat_prophet" in plot_df.columns:
                fig.add_trace(go.Scatter(x=plot_df["ds"], y=plot_df["yhat_prophet"],
                                         name="Prophet", line=dict(dash="dot", color="#FF9800")))
            if "yhat_stat" in plot_df.columns:
                fig.add_trace(go.Scatter(x=plot_df["ds"], y=plot_df["yhat_stat"],
                                         name="Statistical", line=dict(dash="dash", color="#9C27B0")))
            fig.update_layout(title="30-Day Demand Forecast", xaxis_title="Date",
                               yaxis_title="Predicted Demand", height=450)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.line_chart(plot_df.set_index("ds")[["yhat"]])

        st.markdown("### 📥 Download Forecast")
        st.download_button("Download CSV", fc_df.to_csv(index=False).encode(),
                           "forecast_30d.csv", "text/csv")


elif page == "👥 Customer Segments":
    st.title("👥 Customer Segmentation")

    seg_df = load_csv(REPORTS / "customer_segments.csv")
    if seg_df is None:
        st.warning("No segmentation data. Run the pipeline first.")
    else:
        if "segment" in seg_df.columns:
            segs = seg_df["segment"].value_counts()
            cols = st.columns(min(len(segs), 6))
            for i, (seg, cnt) in enumerate(segs.items()):
                with cols[i % len(cols)]:
                    st.metric(seg, f"{cnt:,}")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("RFM Score Distribution")
            if "rfm_score" in seg_df.columns:
                try:
                    import plotly.express as px
                    if "segment" in seg_df.columns:
                        fig = px.histogram(seg_df, x="rfm_score", color="segment",
                                           nbins=15, title="RFM Score by Segment")
                    else:
                        fig = px.histogram(seg_df, x="rfm_score", nbins=15)
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.bar_chart(seg_df["rfm_score"].value_counts().sort_index())

        with col2:
            st.subheader("Monetary vs Frequency")
            if "monetary" in seg_df.columns and "frequency" in seg_df.columns:
                try:
                    import plotly.express as px
                    sample = seg_df.sample(min(2000, len(seg_df)), random_state=42)
                    color_col = "segment" if "segment" in sample.columns else None
                    fig = px.scatter(sample, x="frequency", y="monetary", color=color_col,
                                     title="Customer Value Map", opacity=0.6)
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.dataframe(seg_df[["frequency", "monetary"]].head(100))

        st.markdown("---")
        st.dataframe(seg_df.head(500), use_container_width=True)
        st.download_button("Download Segments CSV", seg_df.to_csv(index=False).encode(),
                           "customer_segments.csv", "text/csv")


elif page == "🔴 Churn Risk":
    st.title("🔴 Customer Churn Risk Dashboard")

    churn_df = load_csv(REPORTS / "churn_scores.csv")
    if churn_df is None:
        st.warning("No churn data. Run the pipeline first.")
    else:
        if "churn_probability" in churn_df.columns:
            col1, col2, col3 = st.columns(3)
            with col1:
                high_risk = (churn_df["churn_probability"] >= 0.6).sum()
                st.metric("🔴 High Risk Customers", f"{high_risk:,}")
            with col2:
                med_risk = ((churn_df["churn_probability"] >= 0.3) &
                            (churn_df["churn_probability"] < 0.6)).sum()
                st.metric("🟡 Medium Risk", f"{med_risk:,}")
            with col3:
                low_risk = (churn_df["churn_probability"] < 0.3).sum()
                st.metric("🟢 Low Risk", f"{low_risk:,}")

            st.markdown("---")
            try:
                import plotly.express as px
                fig = px.histogram(churn_df, x="churn_probability", nbins=30,
                                   title="Churn Probability Distribution",
                                   color_discrete_sequence=["#F44336"])
                fig.add_vline(x=0.3, line_dash="dash", annotation_text="Medium Risk")
                fig.add_vline(x=0.6, line_dash="dash", annotation_text="High Risk",
                              line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(churn_df["churn_probability"].value_counts(bins=20))

        risk_filter = st.selectbox("Filter by Risk Tier",
                                   ["All", "High Risk", "Medium Risk", "Low Risk"])
        if risk_filter != "All" and "risk_tier" in churn_df.columns:
            display_df = churn_df[churn_df["risk_tier"] == risk_filter]
        else:
            display_df = churn_df

        st.dataframe(display_df.head(500), use_container_width=True)
        st.download_button("Download Churn Scores", display_df.to_csv(index=False).encode(),
                           "churn_scores.csv", "text/csv")


elif page == "📦 Inventory":
    st.title("📦 Inventory Optimization")

    inv_df = load_csv(REPORTS / "inventory_recommendations.csv")
    if inv_df is None:
        st.warning("No inventory data. Run the pipeline first.")
    else:
        if "status" in inv_df.columns:
            col1, col2, col3 = st.columns(3)
            with col1:
                critical = inv_df["status"].str.contains("CRITICAL", na=False).sum()
                st.metric("🚨 Critical", critical)
            with col2:
                reorder = inv_df["status"].str.contains("REORDER", na=False).sum()
                st.metric("⚠️ Reorder Now", reorder)
            with col3:
                ok = inv_df["status"].str.contains("OK", na=False).sum()
                st.metric("✅ OK", ok)

            try:
                import plotly.express as px
                status_counts = inv_df["status"].value_counts().reset_index()
                status_counts.columns = ["Status", "Count"]
                fig = px.bar(status_counts, x="Status", y="Count",
                             color="Status", title="Inventory Status Overview",
                             color_discrete_map={
                                 "OK": "#4CAF50",
                                 "REORDER NOW": "#FF9800",
                                 "CRITICAL – Below Safety Stock": "#F44336",
                                 "OVERSTOCK": "#2196F3"
                             })
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(inv_df["status"].value_counts())

        st.dataframe(inv_df, use_container_width=True)
        st.download_button("Download Recommendations", inv_df.to_csv(index=False).encode(),
                           "inventory_recommendations.csv", "text/csv")


elif page == "⚙️ What-If Analysis":
    st.title("⚙️ What-If Analysis & Scenario Modelling")

    st.subheader("Demand Sensitivity")
    base_demand = st.slider("Base Daily Demand (units)", 10, 1000, 100)
    growth_rate = st.slider("Demand Growth Rate (%/month)", -20, 50, 5) / 100
    horizon = st.slider("Forecast Horizon (days)", 7, 90, 30)

    days = np.arange(horizon)
    monthly_growth = (1 + growth_rate) ** (days / 30)
    projected = base_demand * monthly_growth
    noise = np.random.default_rng(42).normal(0, base_demand * 0.05, horizon)
    projected = np.maximum(projected + noise, 0)

    fc_df = pd.DataFrame({"Day": days, "Projected Demand": np.round(projected, 1)})

    try:
        import plotly.express as px
        fig = px.line(fc_df, x="Day", y="Projected Demand",
                      title=f"Projected Demand: {base_demand} units/day, {growth_rate*100:.0f}%/month growth")
        fig.update_traces(fill="tozeroy", fillcolor="rgba(33,150,243,0.1)")
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.line_chart(fc_df.set_index("Day"))

    st.markdown("---")
    st.subheader("Inventory What-If")

    col1, col2, col3 = st.columns(3)
    with col1:
        lead_time = st.number_input("Lead Time (days)", 1, 60, 7)
    with col2:
        service_level = st.selectbox("Service Level", ["90% (Z=1.28)", "95% (Z=1.65)", "99% (Z=2.33)"])
        z_map = {"90% (Z=1.28)": 1.28, "95% (Z=1.65)": 1.65, "99% (Z=2.33)": 2.33}
        z = z_map[service_level]
    with col3:
        demand_std = st.number_input("Demand Std Dev", 0.0, 200.0, float(base_demand * 0.15))

    safety_stock = z * demand_std * np.sqrt(lead_time)
    rop = base_demand * lead_time + safety_stock
    eoq = np.sqrt(2 * base_demand * 365 * 50 / 2)

    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.metric("Safety Stock", f"{safety_stock:.0f} units")
    with res_col2:
        st.metric("Reorder Point", f"{rop:.0f} units")
    with res_col3:
        st.metric("EOQ", f"{eoq:.0f} units")
