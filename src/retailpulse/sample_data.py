"""
Generate realistic synthetic retail datasets for RetailPulse.
Covers: sales transactions, customer demographics, inventory, churn labels.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

CATEGORIES = ["Electronics", "Clothing", "Food & Beverage", "Home & Garden", "Sports", "Beauty"]
STORES = [f"STORE_{i:03d}" for i in range(1, 11)]
N_PRODUCTS = 80
N_CUSTOMERS = 3000
N_DAYS = 365 * 2  # 2 years of daily data


def _make_products() -> pd.DataFrame:
    products = []
    for i in range(1, N_PRODUCTS + 1):
        cat = random.choice(CATEGORIES)
        price = round(rng.uniform(5, 500), 2)
        products.append({
            "product_id": f"SKU_{i:04d}",
            "category": cat,
            "base_price": price,
            "cost": round(price * rng.uniform(0.3, 0.65), 2),
            "reorder_point": int(rng.integers(10, 50)),
            "lead_time_days": int(rng.integers(3, 14)),
        })
    return pd.DataFrame(products)


def _make_customers() -> pd.DataFrame:
    customers = []
    for i in range(1, N_CUSTOMERS + 1):
        age = int(rng.integers(18, 75))
        income = round(rng.normal(55000, 20000), 0)
        income = max(15000, income)
        customers.append({
            "customer_id": f"CUST_{i:05d}",
            "age": age,
            "gender": random.choice(["M", "F", "Other"]),
            "income": int(income),
            "region": random.choice(["North", "South", "East", "West", "Central"]),
            "signup_date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(rng.integers(0, 400))),
            "preferred_category": random.choice(CATEGORIES),
        })
    return pd.DataFrame(customers)


def _make_sales(products: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Generate 2 years of daily sales transactions with seasonality and trend."""
    start = pd.Timestamp("2023-01-01")
    dates = pd.date_range(start, periods=N_DAYS, freq="D")

    rows = []
    prod_ids = products["product_id"].tolist()
    cust_ids = customers["customer_id"].tolist()
    price_map = products.set_index("product_id")["base_price"].to_dict()

    for date in dates:
        # Seasonality: higher sales in Nov-Dec
        seasonal_factor = 1.0
        if date.month in [11, 12]:
            seasonal_factor = 1.8
        elif date.month in [6, 7, 8]:
            seasonal_factor = 1.2
        elif date.month in [1, 2]:
            seasonal_factor = 0.7

        # Weekend boost
        if date.dayofweek >= 5:
            seasonal_factor *= 1.3

        n_txn = int(rng.poisson(120 * seasonal_factor))
        for _ in range(n_txn):
            prod = random.choice(prod_ids)
            qty = int(rng.integers(1, 6))
            store = random.choice(STORES)
            cust = random.choice(cust_ids)
            unit_price = round(price_map[prod] * rng.uniform(0.9, 1.1), 2)
            rows.append({
                "transaction_id": f"TXN_{date.strftime('%Y%m%d')}_{len(rows):06d}",
                "order_date": date,
                "customer_id": cust,
                "product_id": prod,
                "store_id": store,
                "quantity": qty,
                "unit_price": unit_price,
                "revenue": round(qty * unit_price, 2),
                "discount_pct": round(rng.uniform(0, 0.25), 2),
            })

    df = pd.DataFrame(rows)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def _make_inventory(products: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, prod in products.iterrows():
        for store in STORES:
            stock = int(rng.integers(0, 200))
            rows.append({
                "product_id": prod["product_id"],
                "store_id": store,
                "stock": stock,
                "reorder_point": prod["reorder_point"],
                "lead_time_days": prod["lead_time_days"],
                "last_updated": pd.Timestamp("2024-12-31"),
            })
    return pd.DataFrame(rows)


def _make_churn(customers: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """Build churn labels: customers inactive for 60+ days are churned."""
    last_txn = sales.groupby("customer_id")["order_date"].max().reset_index()
    last_txn.columns = ["customer_id", "last_order_date"]
    cutoff = pd.Timestamp("2024-10-01")
    df = customers[["customer_id", "age", "gender", "income", "region"]].copy()
    df = df.merge(last_txn, on="customer_id", how="left")
    df["days_since_last_order"] = (cutoff - df["last_order_date"]).dt.days
    df["days_since_last_order"] = df["days_since_last_order"].fillna(999)
    df["churn"] = (df["days_since_last_order"] > 60).astype(int)
    # Add RFM-like features
    freq = sales.groupby("customer_id").size().reset_index(name="frequency")
    mon = sales.groupby("customer_id")["revenue"].sum().reset_index(name="monetary")
    df = df.merge(freq, on="customer_id", how="left")
    df = df.merge(mon, on="customer_id", how="left")
    df["frequency"] = df["frequency"].fillna(0)
    df["monetary"] = df["monetary"].fillna(0)
    return df


def generate_all(output_dir: str | Path = "data/raw") -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔧 Generating products...")
    products = _make_products()
    products_path = output_dir / "products_catalog.csv"
    products.to_csv(products_path, index=False)

    print("👥 Generating customers...")
    customers = _make_customers()
    customers_path = output_dir / "customers.csv"
    customers.to_csv(customers_path, index=False)

    print("🛒 Generating sales transactions (this may take a moment)...")
    sales = _make_sales(products, customers)
    sales_path = output_dir / "sales_transactions.csv"
    sales.to_csv(sales_path, index=False)

    print("📦 Generating inventory...")
    inventory = _make_inventory(products)
    inventory_path = output_dir / "inventory.csv"
    inventory.to_csv(inventory_path, index=False)

    print("🔮 Generating churn dataset...")
    churn = _make_churn(customers, sales)
    churn_path = output_dir / "churn_dataset.csv"
    churn.to_csv(churn_path, index=False)

    paths = {
        "products": products_path,
        "customers": customers_path,
        "sales": sales_path,
        "inventory": inventory_path,
        "churn": churn_path,
    }
    print(f"✅ Sample data written to {output_dir}/")
    for k, v in paths.items():
        size_mb = v.stat().st_size / 1e6
        print(f"   {k}: {v.name} ({size_mb:.1f} MB, {pd.read_csv(v).shape[0]:,} rows)")
    return paths
