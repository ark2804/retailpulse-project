"""
F02 – Customer Segmentation
RFM scoring + K-Means (primary) / DBSCAN (density-based) clustering.
Falls back to pure numpy if sklearn not installed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEGMENT_LABELS = {
    0: "Champions",
    1: "Loyal Customers",
    2: "Potential Loyalists",
    3: "At Risk",
    4: "Hibernating",
    5: "Lost",
}

BUSINESS_DESCRIPTIONS = {
    "Champions": "Bought recently, buy often, spend the most. Reward them.",
    "Loyal Customers": "Buy regularly. Offer loyalty programs.",
    "Potential Loyalists": "Recent customers with decent frequency. Nurture with offers.",
    "At Risk": "Above-average customers who haven't bought recently. Reactivate urgently.",
    "Hibernating": "Low R, F, M scores. May be slipping away.",
    "Lost": "Lowest scores. Attempt win-back campaigns.",
}


def compute_rfm(sales_df: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary from a sales DataFrame."""
    # Detect columns
    date_col = _find_col(sales_df, ["order_date", "date", "ds", "transaction_date", "invoice_date"])
    cust_col = _find_col(sales_df, ["customer_id", "customerid", "user_id"])
    rev_col = _find_col(sales_df, ["revenue", "amount", "total", "sales_value", "unit_price"])

    if date_col is None or cust_col is None:
        raise ValueError("Cannot find date or customer_id columns for RFM")

    df = sales_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if reference_date is None:
        reference_date = df[date_col].max() + pd.Timedelta(days=1)

    rfm = df.groupby(cust_col).agg(
        recency=(date_col, lambda x: (reference_date - x.max()).days),
        frequency=(date_col, "count"),
    ).reset_index()

    if rev_col:
        mon = df.groupby(cust_col)[rev_col].sum().reset_index()
        mon.columns = [cust_col, "monetary"]
        rfm = rfm.merge(mon, on=cust_col)
    else:
        rfm["monetary"] = rfm["frequency"]

    rfm.columns = ["customer_id", "recency", "frequency", "monetary"]

    # Score 1-5 (higher=better)
    rfm["r_score"] = pd.qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    return rfm


def segment_customers(
    rfm: pd.DataFrame,
    n_clusters: int = 6,
    method: str = "kmeans",
) -> pd.DataFrame:
    """Cluster customers and attach segment labels."""
    features = rfm[["recency", "frequency", "monetary"]].copy()

    # Normalize
    means = features.mean()
    stds = features.std().replace(0, 1)
    X = (features - means) / stds

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans, DBSCAN

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        if method == "dbscan":
            model = DBSCAN(eps=0.5, min_samples=5)
            labels = model.fit_predict(X_scaled)
            # Map -1 (noise) to a new cluster
            labels = np.where(labels == -1, labels.max() + 1, labels)
            n_found = len(set(labels))
            print(f"   DBSCAN found {n_found} clusters")
        else:
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)

        rfm = rfm.copy()
        rfm["cluster"] = labels

    except ImportError:
        # Numpy fallback: simple quantile-based segmentation
        print("   scikit-learn not found, using numpy quantile segmentation")
        rfm = rfm.copy()
        rfm["cluster"] = pd.qcut(rfm["rfm_score"], q=min(n_clusters, 5), labels=False, duplicates="drop")

    # Assign human-readable segment names
    cluster_means = rfm.groupby("cluster")["rfm_score"].mean().sort_values(ascending=False)
    label_map = {cluster: SEGMENT_LABELS.get(i, f"Segment {i}") for i, cluster in enumerate(cluster_means.index)}
    rfm["segment"] = rfm["cluster"].map(label_map)
    rfm["segment_description"] = rfm["segment"].map(BUSINESS_DESCRIPTIONS)

    return rfm


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def run_segmentation(
    sales_df: pd.DataFrame,
    config: dict | None = None,
    output_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    config = config or {}
    n_clusters = config.get("segmentation_clusters", 6)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n🔵 Computing RFM scores...")
    rfm = compute_rfm(sales_df)
    print(f"   RFM computed for {len(rfm):,} customers")

    print(f"🔵 Clustering with K-Means (k={n_clusters})...")
    segmented = segment_customers(rfm, n_clusters=n_clusters)

    seg_counts = segmented["segment"].value_counts()
    print("   Segment distribution:")
    for seg, cnt in seg_counts.items():
        print(f"     {seg}: {cnt:,} customers")

    out_path = output_dir / "customer_segments.csv"
    segmented.to_csv(out_path, index=False)
    print(f"✅ Segmentation saved → {out_path}")

    # Also write to reports
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    segmented.to_csv(reports_dir / "customer_segments.csv", index=False)

    return segmented
