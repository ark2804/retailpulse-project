"""
F05 – Inventory Optimization
Computes safety stock, reorder point, and economic order quantity (EOQ)
using forecasted demand and lead times.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def compute_safety_stock(
    daily_demand_mean: float,
    daily_demand_std: float,
    lead_time_days: int,
    service_level_z: float = 1.65,  # 95% service level
) -> float:
    """Safety Stock = Z × σ_demand × √(lead_time)"""
    return service_level_z * daily_demand_std * np.sqrt(lead_time_days)


def compute_reorder_point(
    daily_demand_mean: float,
    lead_time_days: int,
    safety_stock: float,
) -> float:
    """ROP = (demand_mean × lead_time) + safety_stock"""
    return daily_demand_mean * lead_time_days + safety_stock


def compute_eoq(
    annual_demand: float,
    ordering_cost: float = 50.0,
    holding_cost_per_unit: float = 2.0,
) -> float:
    """EOQ = √(2 × D × S / H)"""
    if holding_cost_per_unit <= 0 or annual_demand <= 0:
        return 0
    return np.sqrt(2 * annual_demand * ordering_cost / holding_cost_per_unit)


def run_inventory(
    inventory_df: pd.DataFrame | None,
    forecast_df: pd.DataFrame | None,
    config: dict | None = None,
    output_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    config = config or {}
    z = config.get("inventory_service_level_z", 1.65)
    default_lead = config.get("default_lead_time_days", 7)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n📦 Computing inventory recommendations...")

    rows = []

    # If we have forecast data, derive demand stats from it
    if forecast_df is not None and not forecast_df.empty:
        series_col = "series_id" if "series_id" in forecast_df.columns else None
        if series_col:
            for sid, grp in forecast_df.groupby(series_col):
                yhat = grp["yhat"].values
                demand_mean = float(np.mean(yhat))
                demand_std = float(np.std(yhat))

                # Get lead time from inventory or use default
                lead_time = default_lead
                current_stock = None
                if inventory_df is not None:
                    prod_inv = inventory_df[
                        inventory_df.get("product_id", inventory_df.get(inventory_df.columns[0], pd.Series())).astype(str) == str(sid)
                    ]
                    if not prod_inv.empty:
                        if "lead_time_days" in prod_inv.columns:
                            lead_time = int(prod_inv["lead_time_days"].iloc[0])
                        if "stock" in prod_inv.columns:
                            current_stock = float(prod_inv["stock"].sum())

                ss = compute_safety_stock(demand_mean, demand_std, lead_time, z)
                rop = compute_reorder_point(demand_mean, lead_time, ss)
                eoq = compute_eoq(demand_mean * 365)

                status = "OK"
                if current_stock is not None:
                    if current_stock < ss:
                        status = "CRITICAL – Below Safety Stock"
                    elif current_stock < rop:
                        status = "REORDER NOW"
                    elif current_stock > eoq * 3:
                        status = "OVERSTOCK"

                rows.append({
                    "product_id": sid,
                    "forecast_mean_daily_demand": round(demand_mean, 2),
                    "forecast_demand_std": round(demand_std, 2),
                    "lead_time_days": lead_time,
                    "safety_stock": round(ss, 1),
                    "reorder_point": round(rop, 1),
                    "eoq": round(eoq, 1),
                    "current_stock": round(current_stock, 1) if current_stock else "N/A",
                    "status": status,
                    "recommended_order_qty": round(max(eoq, rop - (current_stock or 0)), 1),
                })
        else:
            # Single aggregate series
            yhat = forecast_df["yhat"].values
            demand_mean = float(np.mean(yhat))
            demand_std = float(np.std(yhat))
            ss = compute_safety_stock(demand_mean, demand_std, default_lead, z)
            rop = compute_reorder_point(demand_mean, default_lead, ss)
            eoq = compute_eoq(demand_mean * 365)
            rows.append({
                "product_id": "aggregate",
                "forecast_mean_daily_demand": round(demand_mean, 2),
                "forecast_demand_std": round(demand_std, 2),
                "lead_time_days": default_lead,
                "safety_stock": round(ss, 1),
                "reorder_point": round(rop, 1),
                "eoq": round(eoq, 1),
                "current_stock": "N/A",
                "status": "OK",
                "recommended_order_qty": round(eoq, 1),
            })

    elif inventory_df is not None and not inventory_df.empty:
        # No forecast – use inventory data directly
        for _, row in inventory_df.iterrows():
            prod = row.get("product_id", row.get(inventory_df.columns[0], "unknown"))
            stock = float(row.get("stock", row.get("inventory", 0)))
            lead = int(row.get("lead_time_days", default_lead))
            rop_val = float(row.get("reorder_point", stock * 0.2))
            status = "REORDER NOW" if stock < rop_val else "OK"
            rows.append({
                "product_id": prod,
                "current_stock": round(stock, 1),
                "reorder_point": round(rop_val, 1),
                "lead_time_days": lead,
                "status": status,
                "recommended_order_qty": round(max(0, rop_val * 2 - stock), 1),
            })
    else:
        print("   ⚠️  No inventory or forecast data available")
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    reorder_count = (result["status"].str.contains("REORDER|CRITICAL")).sum()
    print(f"   {len(result)} products assessed, {reorder_count} need reordering")

    out_path = output_dir / "inventory_recommendations.csv"
    result.to_csv(out_path, index=False)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    result.to_csv(reports_dir / "inventory_recommendations.csv", index=False)

    print(f"✅ Inventory recommendations → {out_path}")
    return result
