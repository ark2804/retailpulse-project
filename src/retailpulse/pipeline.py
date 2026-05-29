"""
RetailPulse main pipeline orchestrator.
Usage:
    python run_pipeline.py                    # scan data/raw
    python run_pipeline.py --generate-sample  # generate & run on sample data
    python run_pipeline.py --raw-dir data/cleaned
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import pandas as pd

from retailpulse.preprocessing import ingest_directory, load_config, write_quality_report
from retailpulse.segmentation import run_segmentation
from retailpulse.forecasting import run_forecasting
from retailpulse.churn import run_churn
from retailpulse.inventory import run_inventory
from retailpulse.drift import run_drift_detection
from retailpulse.mlops import log_experiment, write_metrics_summary


def _banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="RetailPulse pipeline")
    parser.add_argument("--generate-sample", action="store_true",
                        help="Generate synthetic sample data before running")
    parser.add_argument("--raw-dir", default="data/raw",
                        help="Directory containing raw input files")
    parser.add_argument("--processed-dir", default="data/processed",
                        help="Output directory for processed files")
    parser.add_argument("--skip-forecast", action="store_true")
    parser.add_argument("--skip-churn", action="store_true")
    parser.add_argument("--skip-inventory", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    config = load_config()
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    all_metrics: dict = {}

    # ── 0. Generate sample data ───────────────────────────────────────────────
    if args.generate_sample:
        _banner("STEP 0 – Generating synthetic retail data")
        from retailpulse.sample_data import generate_all
        generate_all(output_dir=args.raw_dir)

    # ── 1. Ingest & Clean ─────────────────────────────────────────────────────
    _banner("STEP 1 – Data Ingestion & Cleaning")
    datasets = ingest_directory(args.raw_dir, config)

    if not datasets:
        print("❌ No datasets found. Run with --generate-sample or add files to data/raw/")
        return 1

    write_quality_report(datasets, processed_dir)
    print(f"\n✅ Loaded {len(datasets)} datasets: {list(datasets.keys())}")

    # Identify dataset roles
    sales_df = _find_dataset(datasets, ["sales", "transaction", "order", "retail"])
    churn_df = _find_dataset(datasets, ["churn", "customer"])
    inventory_df = _find_dataset(datasets, ["inventory", "stock"])

    # ── 2. Customer Segmentation ──────────────────────────────────────────────
    _banner("STEP 2 – Customer Segmentation (RFM + K-Means)")
    seg_df = None
    if sales_df is not None:
        try:
            seg_df = run_segmentation(sales_df, config, processed_dir)
            seg_counts = seg_df["segment"].value_counts().to_dict() if seg_df is not None else {}
            all_metrics["segmentation"] = {"n_customers": len(seg_df), "segments": seg_counts}
            log_experiment("segmentation", {"n_clusters": config.get("segmentation_clusters", 6)},
                           {"n_customers": len(seg_df)})
        except Exception as e:
            print(f"⚠️  Segmentation error: {e}")
    else:
        print("⚠️  No sales dataset found – skipping segmentation")

    # ── 3. Demand Forecasting ─────────────────────────────────────────────────
    forecast_df = None
    if not args.skip_forecast:
        _banner("STEP 3 – Demand Forecasting (Prophet + LSTM Ensemble)")
        if sales_df is not None:
            try:
                result = run_forecasting(sales_df, config, processed_dir)
                if isinstance(result, tuple):
                    forecast_df, mape = result
                else:
                    forecast_df, mape = result, float("nan")
                all_metrics["forecasting"] = {"mape_pct": round(mape, 2),
                                               "horizon_days": config.get("forecast_horizon_days", 30)}
                log_experiment("forecasting", {"horizon": config.get("forecast_horizon_days", 30)},
                               {"mape": mape})
            except Exception as e:
                print(f"⚠️  Forecasting error: {e}")
        else:
            print("⚠️  No sales dataset – skipping forecasting")

    # ── 4. Churn Prediction ───────────────────────────────────────────────────
    if not args.skip_churn:
        _banner("STEP 4 – Churn Prediction")
        df_for_churn = churn_df if churn_df is not None else (
            _build_churn_from_sales(sales_df, seg_df, config) if sales_df is not None else None
        )
        if df_for_churn is not None:
            try:
                _, churn_artifacts = run_churn(df_for_churn, config, processed_dir)
                all_metrics["churn"] = churn_artifacts["metrics"]
                log_experiment("churn_prediction", {"model": churn_artifacts["model_name"]},
                               churn_artifacts["metrics"])
            except Exception as e:
                print(f"⚠️  Churn error: {e}")
        else:
            print("⚠️  No churn/customer dataset – skipping churn")

    # ── 5. Inventory Optimization ─────────────────────────────────────────────
    if not args.skip_inventory:
        _banner("STEP 5 – Inventory Optimization")
        try:
            inv_recs = run_inventory(inventory_df, forecast_df, config, processed_dir)
            if inv_recs is not None and not inv_recs.empty:
                all_metrics["inventory"] = {
                    "products_assessed": len(inv_recs),
                    "reorder_count": int(inv_recs["status"].str.contains("REORDER|CRITICAL").sum()),
                }
        except Exception as e:
            print(f"⚠️  Inventory error: {e}")

    # ── 6. Drift Detection ────────────────────────────────────────────────────
    _banner("STEP 6 – Drift Detection")
    if sales_df is not None and len(sales_df) > 100:
        try:
            half = len(sales_df) // 2
            numeric_cols = list(sales_df.select_dtypes(include=[np.number]).columns[:8])
            drift_report = run_drift_detection(
                sales_df.iloc[:half],
                sales_df.iloc[half:],
                numeric_cols=numeric_cols,
            )
            all_metrics["drift"] = drift_report
        except Exception as e:
            print(f"⚠️  Drift detection error: {e}")

    # ── 7. Write consolidated metrics ─────────────────────────────────────────
    _banner("STEP 7 – Writing Metrics Summary")
    elapsed = time.time() - start_time
    all_metrics["pipeline"] = {
        "elapsed_seconds": round(elapsed, 1),
        "datasets_processed": len(datasets),
    }
    write_metrics_summary(all_metrics)

    _banner("✅ RetailPulse Pipeline Complete")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Outputs in: reports/ and data/processed/")
    print("\nTo launch the dashboard:")
    print("  streamlit run dashboard/app.py")
    return 0


def _find_dataset(datasets: dict[str, pd.DataFrame], keywords: list[str]) -> pd.DataFrame | None:
    """Find the first dataset whose name contains any keyword."""
    for kw in keywords:
        for name, df in datasets.items():
            if kw in name.lower():
                return df
    return None


def _build_churn_from_sales(
    sales_df: pd.DataFrame,
    seg_df: pd.DataFrame | None,
    config: dict,
) -> pd.DataFrame | None:
    """Synthesize a churn dataset from sales + segmentation data."""
    try:
        date_col = next((c for c in sales_df.columns if "date" in c or c == "ds"), None)
        cust_col = next((c for c in sales_df.columns if "customer" in c or "user" in c), None)
        rev_col = next((c for c in sales_df.columns if "revenue" in c or "amount" in c), None)

        if date_col is None or cust_col is None:
            return None

        sales_df = sales_df.copy()
        sales_df[date_col] = pd.to_datetime(sales_df[date_col])
        cutoff = sales_df[date_col].max()
        inactivity = config.get("churn_inactivity_days", 60)

        last_txn = sales_df.groupby(cust_col)[date_col].max().reset_index()
        last_txn.columns = ["customer_id", "last_order_date"]
        last_txn["days_since_last_order"] = (cutoff - last_txn["last_order_date"]).dt.days
        last_txn["churn"] = (last_txn["days_since_last_order"] > inactivity).astype(int)

        freq = sales_df.groupby(cust_col).size().reset_index(name="frequency")
        last_txn = last_txn.merge(freq, on="customer_id", how="left")

        if rev_col:
            mon = sales_df.groupby(cust_col)[rev_col].sum().reset_index(name="monetary")
            last_txn = last_txn.merge(mon, on="customer_id", how="left")

        if seg_df is not None and "customer_id" in seg_df.columns:
            last_txn = last_txn.merge(
                seg_df[["customer_id", "recency", "rfm_score", "r_score", "f_score", "m_score"]],
                on="customer_id", how="left"
            )

        return last_txn
    except Exception as e:
        print(f"   Could not build churn from sales: {e}")
        return None
