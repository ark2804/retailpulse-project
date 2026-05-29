"""
Prometheus metrics exporter for RetailPulse.
Reads reports/metrics.json and exposes as Prometheus gauges.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def serve(port: int = 9108):
    try:
        from prometheus_client import start_http_server, Gauge, REGISTRY
        import prometheus_client

        metrics_file = Path("reports/metrics.json")

        gauges = {
            "forecast_mape": Gauge("retailpulse_forecast_mape", "Demand forecast MAPE %"),
            "churn_auc": Gauge("retailpulse_churn_auc_roc", "Churn model AUC-ROC"),
            "churn_f1": Gauge("retailpulse_churn_f1", "Churn model F1 score"),
            "pipeline_elapsed": Gauge("retailpulse_pipeline_elapsed_seconds", "Pipeline runtime"),
            "datasets_processed": Gauge("retailpulse_datasets_processed", "Number of datasets"),
        }

        start_http_server(port)
        print(f"Prometheus metrics server running on :{port}/metrics")

        while True:
            if metrics_file.exists():
                with open(metrics_file) as f:
                    m = json.load(f)
                mape = m.get("forecasting", {}).get("mape_pct")
                if mape is not None:
                    gauges["forecast_mape"].set(float(mape))
                auc = m.get("churn", {}).get("auc_roc")
                if auc is not None:
                    gauges["churn_auc"].set(float(auc))
                f1 = m.get("churn", {}).get("f1_score")
                if f1 is not None:
                    gauges["churn_f1"].set(float(f1))
                elapsed = m.get("pipeline", {}).get("elapsed_seconds")
                if elapsed is not None:
                    gauges["pipeline_elapsed"].set(float(elapsed))
                n_ds = m.get("pipeline", {}).get("datasets_processed")
                if n_ds is not None:
                    gauges["datasets_processed"].set(float(n_ds))
            time.sleep(30)

    except ImportError:
        print("prometheus_client not installed. Run: pip install prometheus-client")


if __name__ == "__main__":
    serve()
