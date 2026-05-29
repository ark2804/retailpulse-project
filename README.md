# 📊 RetailPulse – AI-Powered Customer Analytics & Demand Forecasting Platform

> *Predictive Demand • Customer Segmentation • Churn Analysis • Inventory Optimization*

**Author:** Zidio Development | **Version:** 2.0 – Industry Edition | **Date:** March 2026

---

## Project Overview

RetailPulse is an end-to-end data science platform that ingests retail sales, customer, and inventory data to deliver:
- **30-day demand forecasts** (Prophet + LSTM ensemble, MAPE ≤ 12%)
- **RFM customer segmentation** (K-Means / DBSCAN, 6–8 segments)
- **Churn prediction** (XGBoost with SHAP, AUC-ROC ≥ 0.88)
- **Inventory optimization** (Safety stock, EOQ, reorder recommendations)
- **Drift monitoring** (PSI-based + Evidently AI)
- **Interactive Streamlit dashboard** with what-if analysis

## Project Structure

```
retailpulse/
├── src/retailpulse/          # Core pipeline modules
│   ├── pipeline.py           # Main orchestrator
│   ├── sample_data.py        # Synthetic data generator
│   ├── preprocessing.py      # F01 – Data ingestion & cleaning
│   ├── segmentation.py       # F02 – RFM + K-Means/DBSCAN
│   ├── forecasting.py        # F03 – Prophet + LSTM ensemble
│   ├── churn.py              # F04 – Churn prediction (XGBoost)
│   ├── inventory.py          # F05 – Inventory optimization (EOQ)
│   ├── drift.py              # Drift detection (PSI + Evidently)
│   ├── mlops.py              # MLflow experiment tracking
│   └── database.py           # PostgreSQL + Redis layer
├── dashboard/app.py          # F06 – Streamlit dashboard
├── scripts/
│   ├── serve_metrics.py      # Prometheus exporter
│   └── check_requirements.py # Dependency checker
├── notebooks/01_eda.ipynb    # Exploratory Data Analysis
├── monitoring/prometheus.yml  # Prometheus scrape config
├── k8s/deployment.yaml       # Kubernetes manifests
├── .github/workflows/ci.yml  # GitHub Actions CI/CD
├── data/raw/                 # Input datasets (place Kaggle files here)
├── data/processed/           # Cleaned datasets & features
├── models/                   # Pickled model artifacts
├── reports/                  # Metrics, forecasts, recommendations
├── Dockerfile
├── docker-compose.yml
├── config.json
└── requirements.txt
```

## Quick Start

### Option 1 – Generate sample data and run

```bash
# Install core dependencies
pip install pandas numpy scikit-learn

# Generate synthetic data + run pipeline
python run_pipeline.py --generate-sample

# Launch dashboard
streamlit run dashboard/app.py
```

### Option 2 – Use your own Kaggle datasets

Place CSV/XLSX/Parquet files in `data/raw/`, then:

```bash
python run_pipeline.py
```

### Option 3 – Docker (full stack)

```bash
docker-compose up
# App:       http://localhost:8501
# Prometheus: http://localhost:9090
# Grafana:   http://localhost:3000
```

## Full Installation

```bash
pip install -r requirements.txt
```

Check what's available:

```bash
python scripts/check_requirements.py
```

## Supported Datasets

The flexible column aliasing in `config.json` handles schemas from:
- M5 Forecasting Accuracy
- Rossmann Store Sales
- Online Retail II (UCI)
- Retailrocket E-Commerce
- Telco Customer Churn
- Customer Personality Analysis
- Supply Chain / Inventory datasets

## Configuration

Edit `config.json` to tune:

| Key | Default | Description |
|-----|---------|-------------|
| `forecast_horizon_days` | 30 | Days ahead to forecast |
| `segmentation_clusters` | 6 | K-Means clusters |
| `churn_inactivity_days` | 60 | Days inactive → churned |
| `inventory_service_level_z` | 1.65 | Safety stock Z-score (95%) |
| `max_forecast_series` | 250 | Cap on series to forecast |
| `enable_prophet` | true | Use Prophet model |
| `enable_lstm` | true | Use LSTM model |

## Requirement Coverage

| ID | Requirement | Module |
|----|------------|--------|
| F01 | Data ingestion & cleaning | `preprocessing.py` |
| F02 | RFM + K-Means/DBSCAN segmentation | `segmentation.py` |
| F03 | Prophet + LSTM ensemble forecasting | `forecasting.py` |
| F04 | XGBoost churn prediction + SHAP | `churn.py` |
| F05 | EOQ inventory optimization | `inventory.py` |
| F06 | Streamlit dashboard | `dashboard/app.py` |
| MLOps | MLflow, Evidently, Prometheus/Grafana | `mlops.py`, `drift.py` |
| Deploy | Docker, Kubernetes, GitHub Actions | `Dockerfile`, `k8s/` |

## Deliverables

After running the pipeline, check `reports/`:

- `metrics.json` – consolidated model metrics
- `forecast_30d.csv` – 30-day demand forecast
- `customer_segments.csv` – RFM segmentation output
- `churn_scores.csv` – per-customer churn risk
- `inventory_recommendations.csv` – reorder recommendations
- `drift_report.json` – PSI-based drift monitor
- `data_quality_report.json` – data quality summary

---

*RetailPulse – Crafted with precision and modern data science principles · Zidio Development · March 2026*
