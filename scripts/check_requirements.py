"""Check which optional requirements are installed."""
PACKAGES = [
    "pandas", "numpy", "openpyxl", "pyarrow",
    "sklearn", "xgboost", "shap", "mlflow", "evidently",
    "prophet", "torch", "pytorch_lightning",
    "streamlit", "plotly", "great_expectations",
    "optuna", "sqlalchemy", "psycopg2", "redis",
    "prometheus_client",
]

print("RetailPulse – Dependency Check")
print("=" * 40)
for pkg in PACKAGES:
    try:
        mod = __import__(pkg)
        version = getattr(mod, "__version__", "installed")
        print(f"  ✅ {pkg:<25} {version}")
    except ImportError:
        print(f"  ❌ {pkg:<25} NOT INSTALLED")
print("=" * 40)
print("Run: pip install -r requirements.txt  to install all")
