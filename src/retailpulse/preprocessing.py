"""
F01 – Data Ingestion & Cleaning
Handles CSV, XLSX, JSON, Parquet, ZIP with flexible column aliasing from config.json.
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "config.json"


def load_config() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def _read_file(path: Path) -> pd.DataFrame | None:
    """Read a single file into a DataFrame."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, low_memory=False)
        elif suffix in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        elif suffix == ".parquet":
            return pd.read_parquet(path)
        elif suffix == ".json":
            return pd.read_json(path)
        elif suffix == ".zip":
            # Expand zip and read first CSV inside
            with zipfile.ZipFile(path) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if csv_names:
                    with zf.open(csv_names[0]) as f:
                        return pd.read_csv(f, low_memory=False)
    except Exception as e:
        print(f"  ⚠️  Could not read {path.name}: {e}")
    return None


def _resolve_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    """Find first column in df that matches any alias (case-insensitive)."""
    lower_cols = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lower_cols:
            return lower_cols[alias.lower()]
    return None


def _standardize_columns(df: pd.DataFrame, col_aliases: dict[str, list[str]]) -> pd.DataFrame:
    """Rename columns to standard names based on aliases."""
    rename_map = {}
    for std_name, aliases in col_aliases.items():
        found = _resolve_column(df, aliases)
        if found and found not in rename_map.values():
            rename_map[found] = std_name
    return df.rename(columns=rename_map)


def clean_dataframe(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """Core cleaning: drop duplicates, fix dtypes, handle missing values."""
    original_shape = df.shape

    # Drop complete duplicates
    df = df.drop_duplicates()

    # Normalize column names
    df.columns = [re.sub(r"[^a-zA-Z0-9_]", "_", c.strip().lower()) for c in df.columns]

    # Drop columns that are >80% null
    null_pct = df.isnull().mean()
    drop_cols = null_pct[null_pct > 0.8].index.tolist()
    if drop_cols:
        print(f"   Dropping mostly-null columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # Parse date columns
    for col in df.columns:
        if any(kw in col for kw in ["date", "time", "week", "day", "ds"]):
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
            except Exception:
                pass

    # Fill numeric nulls with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Fill string nulls with "Unknown"
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df[col] = df[col].fillna("Unknown")

    print(f"   {name}: {original_shape} → {df.shape} after cleaning")
    return df


def ingest_directory(raw_dir: str | Path, config: dict | None = None) -> dict[str, pd.DataFrame]:
    """
    Scan a directory, read all supported files, apply column standardization and cleaning.
    Returns a dict of {label: DataFrame}.
    """
    raw_dir = Path(raw_dir)
    if config is None:
        config = load_config()

    col_aliases = config.get("column_aliases", {})
    excluded = config.get("excluded_file_patterns", [])
    max_rows = config.get("max_rows_per_file", 30000)

    extensions = {".csv", ".xlsx", ".xls", ".parquet", ".json", ".zip"}
    files = [f for f in sorted(raw_dir.iterdir())
             if f.is_file() and f.suffix.lower() in extensions
             and not any(ex in f.stem for ex in excluded)]

    datasets: dict[str, pd.DataFrame] = {}

    for fpath in files:
        print(f"\n📂 Reading: {fpath.name}")
        df = _read_file(fpath)
        if df is None or df.empty:
            continue

        # Sample if too large
        if max_rows and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
            print(f"   Sampled to {max_rows:,} rows")

        df = _standardize_columns(df, col_aliases)
        df = clean_dataframe(df, fpath.stem)
        label = fpath.stem.lower().replace(" ", "_")
        datasets[label] = df

    return datasets


def write_quality_report(datasets: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    """Write data_quality_report.json summarising each dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    for name, df in datasets.items():
        report[name] = {
            "rows": len(df),
            "columns": list(df.columns),
            "null_counts": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "numeric_summary": df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else {},
        }
    out = output_dir / "data_quality_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"📋 Data quality report → {out}")
