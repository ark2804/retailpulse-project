"""
F03 – Demand Forecasting
Prophet + LSTM (PyTorch) ensemble with statistical fallback.
Produces 30-day ahead forecasts per product-store series.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROPHET_WEIGHT = 0.45
LSTM_WEIGHT = 0.35
STAT_WEIGHT = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# Statistical baseline (always available)
# ─────────────────────────────────────────────────────────────────────────────

def _statistical_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    """Simple seasonal naive + linear trend forecast."""
    vals = series.values.astype(float)
    if len(vals) < 2:
        return np.full(horizon, vals[-1] if len(vals) else 0)

    # Linear trend
    x = np.arange(len(vals))
    slope, intercept = np.polyfit(x, vals, 1)

    # Seasonal component (7-day period if enough data)
    seasonal = np.zeros(7)
    if len(vals) >= 14:
        for i in range(7):
            seasonal[i] = np.mean(vals[i::7]) - np.mean(vals)

    future_x = np.arange(len(vals), len(vals) + horizon)
    trend = slope * future_x + intercept
    seas = np.array([seasonal[i % 7] for i in range(horizon)])
    forecast = trend + seas
    return np.maximum(forecast, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Prophet wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _prophet_forecast(ts: pd.DataFrame, horizon: int) -> np.ndarray | None:
    try:
        from prophet import Prophet
        m = Prophet(
            seasonality_mode="multiplicative",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95,
        )
        m.fit(ts[["ds", "y"]])
        future = m.make_future_dataframe(periods=horizon)
        forecast = m.predict(future)
        return forecast["yhat"].values[-horizon:]
    except Exception as e:
        print(f"     Prophet failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LSTM wrapper (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def _lstm_forecast(series: pd.Series, horizon: int, lookback: int = 30) -> np.ndarray | None:
    try:
        import torch
        import torch.nn as nn

        vals = series.values.astype(np.float32)
        if len(vals) < lookback + horizon:
            return None

        # Normalise
        mean, std = vals.mean(), vals.std() + 1e-8
        norm = (vals - mean) / std

        # Build sequences
        X, y = [], []
        for i in range(len(norm) - lookback):
            X.append(norm[i:i + lookback])
            y.append(norm[i + lookback])
        X = torch.tensor(np.array(X)).unsqueeze(-1)  # (N, L, 1)
        y = torch.tensor(np.array(y)).unsqueeze(-1)

        class LSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(1, 32, num_layers=2, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(32, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        model = LSTMModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        model.train()
        for _ in range(30):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        # Predict
        model.eval()
        with torch.no_grad():
            inp = torch.tensor(norm[-lookback:]).unsqueeze(0).unsqueeze(-1)
            preds = []
            for _ in range(horizon):
                out = model(inp)
                preds.append(out.item())
                inp = torch.cat([inp[:, 1:, :], out.unsqueeze(1)], dim=1)

        return np.array(preds) * std + mean

    except Exception as e:
        print(f"     LSTM failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_forecast(
    ts: pd.DataFrame,
    horizon: int = 30,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Produce ensemble forecast for a single time series.
    ts: DataFrame with columns [ds, y]
    Returns DataFrame with [ds, yhat, yhat_prophet, yhat_lstm, yhat_stat]
    """
    weights = weights or {"prophet": PROPHET_WEIGHT, "lstm": LSTM_WEIGHT, "statistical": STAT_WEIGHT}

    series = ts.set_index("ds")["y"].sort_index()
    future_dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")

    stat_pred = _statistical_forecast(series, horizon)
    prophet_pred = _prophet_forecast(ts.sort_values("ds"), horizon)
    lstm_pred = _lstm_forecast(series, horizon)

    # Weighted ensemble
    total_w = 0.0
    ensemble = np.zeros(horizon)

    ensemble += stat_pred * weights["statistical"]
    total_w += weights["statistical"]

    if prophet_pred is not None:
        ensemble += prophet_pred * weights["prophet"]
        total_w += weights["prophet"]
    else:
        ensemble += stat_pred * weights["prophet"]
        total_w += weights["prophet"]

    if lstm_pred is not None:
        ensemble += lstm_pred * weights["lstm"]
        total_w += weights["lstm"]
    else:
        ensemble += stat_pred * weights["lstm"]
        total_w += weights["lstm"]

    ensemble = np.maximum(ensemble / total_w, 0)

    result = pd.DataFrame({
        "ds": future_dates,
        "yhat": np.round(ensemble, 2),
        "yhat_prophet": np.round(prophet_pred, 2) if prophet_pred is not None else np.nan,
        "yhat_lstm": np.round(lstm_pred, 2) if lstm_pred is not None else np.nan,
        "yhat_stat": np.round(stat_pred, 2),
    })
    return result


def _compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def run_forecasting(
    sales_df: pd.DataFrame,
    config: dict | None = None,
    output_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    config = config or {}
    horizon = config.get("forecast_horizon_days", 30)
    max_series = config.get("max_forecast_series", 50)
    weights = config.get("forecast_ensemble_weights", {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect columns
    date_col = next((c for c in sales_df.columns if "date" in c or c == "ds"), None)
    prod_col = next((c for c in sales_df.columns if "product" in c or "item" in c or "sku" in c), None)
    qty_col = next((c for c in sales_df.columns if "quantity" in c or "qty" in c or "sales" in c or "demand" in c), None)

    if date_col is None or qty_col is None:
        raise ValueError("Cannot find date or quantity columns for forecasting")

    sales_df = sales_df.copy()
    sales_df[date_col] = pd.to_datetime(sales_df[date_col])

    # Aggregate daily
    if prod_col:
        series_groups = sales_df.groupby([prod_col, pd.Grouper(key=date_col, freq="D")])[qty_col].sum().reset_index()
        series_groups.columns = ["series_id", "ds", "y"]
        all_series = series_groups["series_id"].unique()[:max_series]
    else:
        daily = sales_df.groupby(pd.Grouper(key=date_col, freq="D"))[qty_col].sum().reset_index()
        daily.columns = ["ds", "y"]
        daily["series_id"] = "total"
        series_groups = daily
        all_series = ["total"]

    print(f"\n📈 Forecasting {len(all_series)} series × {horizon} days...")

    all_forecasts = []
    mape_scores = []

    for i, sid in enumerate(all_series):
        if prod_col:
            ts = series_groups[series_groups["series_id"] == sid][["ds", "y"]].sort_values("ds")
        else:
            ts = series_groups[["ds", "y"]].sort_values("ds")

        ts["y"] = ts["y"].fillna(0)
        if len(ts) < 14:
            continue

        # Hold out last horizon days for MAPE
        if len(ts) > horizon * 2:
            train = ts.iloc[:-horizon]
            test_actual = ts["y"].values[-horizon:]
        else:
            train = ts
            test_actual = None

        fc = ensemble_forecast(train, horizon=horizon, weights=weights)
        fc["series_id"] = sid

        if test_actual is not None:
            mape = _compute_mape(test_actual, fc["yhat"].values)
            mape_scores.append(mape)

        all_forecasts.append(fc)

        if (i + 1) % 20 == 0 or i == len(all_series) - 1:
            print(f"   Completed {i + 1}/{len(all_series)} series")

    if not all_forecasts:
        print("⚠️  No forecasts generated")
        return pd.DataFrame()

    result = pd.concat(all_forecasts, ignore_index=True)

    # Metrics
    avg_mape = float(np.nanmean(mape_scores)) if mape_scores else float("nan")
    print(f"   Average MAPE: {avg_mape:.1f}%")

    # Save
    out_path = output_dir / "forecast_30d.csv"
    result.to_csv(out_path, index=False)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    result.to_csv(reports_dir / "forecast_30d.csv", index=False)

    print(f"✅ Forecast saved → {out_path}")
    return result, avg_mape
