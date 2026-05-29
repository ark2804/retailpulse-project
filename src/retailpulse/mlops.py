"""
MLOps – MLflow experiment tracking, model registry, and metrics logging.
"""
from __future__ import annotations

import json
from pathlib import Path


def log_experiment(
    experiment_name: str,
    params: dict,
    metrics: dict,
    artifacts: list[str] | None = None,
    model=None,
    model_name: str = "",
) -> str | None:
    """Log a training run to MLflow if available."""
    try:
        import mlflow

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

            if artifacts:
                for art in artifacts:
                    p = Path(art)
                    if p.exists():
                        mlflow.log_artifact(str(p))

            if model is not None and model_name:
                try:
                    mlflow.sklearn.log_model(model, model_name)
                except Exception:
                    try:
                        mlflow.xgboost.log_model(model, model_name)
                    except Exception:
                        pass

            run_id = run.info.run_id
            print(f"   MLflow run logged: {run_id}")
            return run_id

    except ImportError:
        print("   MLflow not installed – skipping experiment tracking")
        # Write fallback metrics to JSON
        out = Path("reports") / "mlflow_fallback.json"
        existing = []
        if out.exists():
            with open(out) as f:
                existing = json.load(f)
        existing.append({"experiment": experiment_name, "params": params, "metrics": metrics})
        with open(out, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        return None
    except Exception as e:
        print(f"   MLflow error: {e}")
        return None


def write_metrics_summary(all_metrics: dict, output_dir: str | Path = "reports") -> None:
    """Write a consolidated metrics.json for Prometheus and dashboard."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "metrics.json"
    with open(out, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"✅ Metrics summary → {out}")
