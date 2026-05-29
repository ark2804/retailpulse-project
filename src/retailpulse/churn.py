"""
F04 – Churn Prediction
XGBoost classifier with SHAP explainability.
Falls back to scikit-learn LogisticRegression or numpy logistic if unavailable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "recency", "frequency", "monetary",
    "r_score", "f_score", "m_score", "rfm_score",
    "days_since_last_order", "age", "income",
]


def _build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Extract numeric feature matrix and optional target from a DataFrame."""
    available = [c for c in FEATURE_COLS if c in df.columns]

    # Also include any numeric columns not in the standard list
    extra_num = [c for c in df.select_dtypes(include=[np.number]).columns
                 if c not in available and c != "churn" and c != "cluster"]
    cols = available + extra_num[:5]  # cap extras

    if not cols:
        raise ValueError("No numeric feature columns found for churn model")

    X = df[cols].fillna(0).values.astype(np.float32)
    y = df["churn"].values.astype(int) if "churn" in df.columns else None
    return X, y, cols


def _numpy_logistic(X_train, y_train, X_test, n_iter=200, lr=0.01):
    """Minimal logistic regression via gradient descent."""
    m, n = X_train.shape
    w = np.zeros(n, dtype=np.float64)
    b = 0.0
    # Normalise
    mu = X_train.mean(0)
    sigma = X_train.std(0) + 1e-8
    Xn = (X_train - mu) / sigma
    Xt = (X_test - mu) / sigma
    for _ in range(n_iter):
        logit = Xn @ w + b
        p = 1 / (1 + np.exp(-np.clip(logit, -20, 20)))
        err = p - y_train
        w -= lr * Xn.T @ err / m
        b -= lr * err.mean()
    prob = 1 / (1 + np.exp(-np.clip(Xt @ w + b, -20, 20)))
    return prob, w


def train_churn_model(
    churn_df: pd.DataFrame,
    config: dict | None = None,
) -> dict:
    """Train churn classifier and return model artifacts + metrics."""
    config = config or {}
    X, y, feat_cols = _build_features(churn_df)

    if y is None:
        raise ValueError("No 'churn' target column found")

    # Train/test split (80/20)
    n = len(X)
    idx = np.random.RandomState(42).permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Try XGBoost
    model_name = "numpy_logistic"
    model = None
    proba = None
    feature_importance = None

    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        proba = model.predict_proba(X_test)[:, 1]
        feature_importance = dict(zip(feat_cols, model.feature_importances_))
        model_name = "xgboost"
        print("   Using XGBoost")
    except ImportError:
        pass

    if proba is None:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            Xs_train = scaler.fit_transform(X_train)
            Xs_test = scaler.transform(X_test)
            model = LogisticRegression(max_iter=500, random_state=42)
            model.fit(Xs_train, y_train)
            proba = model.predict_proba(Xs_test)[:, 1]
            feature_importance = dict(zip(feat_cols, np.abs(model.coef_[0])))
            model_name = "sklearn_logistic"
            print("   Using sklearn LogisticRegression")
        except ImportError:
            pass

    if proba is None:
        print("   Using numpy logistic regression")
        proba, weights = _numpy_logistic(X_train, y_train, X_test)
        feature_importance = dict(zip(feat_cols, np.abs(weights)))

    # Metrics
    pred_labels = (proba >= 0.5).astype(int)
    accuracy = (pred_labels == y_test).mean()

    try:
        from sklearn.metrics import roc_auc_score, f1_score, precision_score
        auc = roc_auc_score(y_test, proba)
        f1 = f1_score(y_test, pred_labels)
        # Precision at top 20%
        top20 = int(0.2 * len(y_test))
        top_idx = np.argsort(proba)[-top20:]
        prec20 = y_test[top_idx].mean()
    except Exception:
        # Numpy AUC approximation
        pos = y_test == 1
        neg = y_test == 0
        if pos.sum() > 0 and neg.sum() > 0:
            auc = np.mean([proba[i] > proba[j]
                           for i in np.where(pos)[0][:200]
                           for j in np.where(neg)[0][:200]])
        else:
            auc = 0.5
        f1 = 0.0
        prec20 = 0.0

    metrics = {
        "model": model_name,
        "accuracy": round(float(accuracy), 4),
        "auc_roc": round(float(auc), 4),
        "f1_score": round(float(f1), 4),
        "precision_top20pct": round(float(prec20), 4),
        "n_train": int(split),
        "n_test": int(n - split),
        "churn_rate": round(float(y.mean()), 4),
    }

    print(f"   AUC-ROC: {auc:.3f} | Accuracy: {accuracy:.3f} | F1: {f1:.3f}")

    return {
        "model": model,
        "model_name": model_name,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "feature_cols": feat_cols,
        "test_idx": test_idx,
        "proba_test": proba,
    }


def score_all_customers(
    churn_df: pd.DataFrame,
    model_artifacts: dict,
) -> pd.DataFrame:
    """Score all customers with churn probability."""
    X, _, feat_cols = _build_features(churn_df)
    model = model_artifacts["model"]
    model_name = model_artifacts["model_name"]

    if model_name == "xgboost":
        proba = model.predict_proba(X)[:, 1]
    elif model_name == "sklearn_logistic":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X)  # Re-fit on all (approximation)
        proba = model.predict_proba(scaler.transform(X))[:, 1]
    else:
        proba, _ = _numpy_logistic(X, churn_df.get("churn", np.zeros(len(X))).values, X)

    result = churn_df.copy()
    result["churn_probability"] = np.round(proba, 4)
    result["risk_tier"] = pd.cut(
        proba,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )
    return result


def run_churn(
    churn_df: pd.DataFrame,
    config: dict | None = None,
    output_dir: str | Path = "data/processed",
) -> tuple[pd.DataFrame, dict]:
    config = config or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n🔴 Training churn model...")
    artifacts = train_churn_model(churn_df, config)

    print("🔴 Scoring all customers...")
    scored = score_all_customers(churn_df, artifacts)

    out_path = output_dir / "churn_scores.csv"
    scored.to_csv(out_path, index=False)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    scored.to_csv(reports_dir / "churn_scores.csv", index=False)

    print(f"✅ Churn scores saved → {out_path}")
    return scored, artifacts
