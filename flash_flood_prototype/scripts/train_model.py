#!/usr/bin/env python3
"""
Flash Flood Risk Model Training Script
Production-ready training pipeline with cross-validation, hyperparameter tuning,
model comparison, SHAP explainability, and comprehensive experiment tracking.
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "train.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

RISK_LABELS = ["LOW", "MEDIUM", "HIGH"]
RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
INV_RISK_ORDER = {v: k for k, v in RISK_ORDER.items()}
N_SPLITS = 5
RANDOM_STATE = 42
N_JOBS = -1


def generate_synthetic_data(n_samples: int = 5000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate realistic synthetic training data with spatial/temporal correlations."""
    rng = np.random.default_rng(seed)
    n = n_samples

    base_rain = rng.gamma(2.5, 10, n).clip(0, 150)
    rain_1h = base_rain * rng.uniform(0.8, 1.3, n) + rng.normal(0, 5, n)
    rain_3h = rain_1h * rng.uniform(1.8, 3.0, n) + rng.normal(0, 10, n)
    rain_6h = rain_3h * rng.uniform(1.5, 2.3, n) + rng.normal(0, 15, n)
    rain_24h = rain_6h * rng.uniform(2.2, 4.5, n) + rng.normal(0, 30, n)

    rain_1h = np.clip(rain_1h, 0, 200)
    rain_3h = np.clip(rain_3h, 0, 400)
    rain_6h = np.clip(rain_6h, 0, 600)
    rain_24h = np.clip(rain_24h, 0, 1200)

    soil = rng.beta(3, 2, n) * 100
    slope = rng.uniform(2, 60, n)
    elevation = rng.uniform(200, 3500, n)
    river_dist = rng.uniform(30, 5000, n)
    historical = rng.poisson(1.5, n).clip(0, 15)

    spatial_cluster = rng.integers(0, 5, n)
    cluster_bias = np.array([0.05, -0.03, 0.08, -0.06, 0.02])[spatial_cluster]

    risk_score = (
        0.30 * np.clip(rain_1h / 120, 0, 1)
        + 0.22 * np.clip(rain_3h / 200, 0, 1)
        + 0.18 * np.clip(rain_6h / 350, 0, 1)
        + 0.15 * np.clip(rain_24h / 700, 0, 1)
        + 0.12 * (soil / 100)
        + 0.08 * np.clip(slope / 50, 0, 1)
        + 0.06 * np.clip((1500 - river_dist) / 1500, 0, 1)
        + 0.05 * (historical / 15)
        + 0.03 * np.clip((elevation - 2000) / 1500, -1, 1)
        + cluster_bias
        + rng.normal(0, 0.04, n)
    )

    y = pd.cut(
        risk_score,
        bins=[-np.inf, 0.35, 0.60, np.inf],
        labels=RISK_LABELS,
    ).astype(str)

    df = pd.DataFrame({
        "rain_1h_mm": rain_1h,
        "rain_3h_mm": rain_3h,
        "rain_6h_mm": rain_6h,
        "rain_24h_mm": rain_24h,
        "soil_moisture_pct": soil,
        "slope_deg": slope,
        "elevation_m": elevation,
        "distance_to_river_m": river_dist,
        "historical_events": historical,
        "risk": y,
    })

    log.info(f"Generated {len(df)} samples | Class distribution:\n{df['risk'].value_counts()}")
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean training data."""
    initial_len = len(df)
    df = df.dropna()
    df = df[(df["rain_1h_mm"] >= 0) & (df["rain_1h_mm"] <= 200)]
    df = df[(df["soil_moisture_pct"] >= 0) & (df["soil_moisture_pct"] <= 100)]
    df = df[(df["slope_deg"] >= 0) & (df["slope_deg"] <= 90)]
    df = df[(df["distance_to_river_m"] >= 0)]
    df = df[df["risk"].isin(RISK_LABELS)]
    if len(df) < initial_len:
        log.warning(f"Dropped {initial_len - len(df)} invalid rows")
    return df


def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    features = [c for c in df.columns if c != "risk"]
    X = df[features]
    y = df["risk"]
    return X, y, features


def encode_labels(y: pd.Series) -> np.ndarray:
    """Encode string labels to integers."""
    return np.array([RISK_ORDER[v] for v in y])


def decode_labels(y_encoded: np.ndarray) -> np.ndarray:
    """Decode integer labels to strings."""
    return np.array([INV_RISK_ORDER[v] for v in y_encoded])


def get_models() -> dict[str, Pipeline]:
    """Return base model pipelines for comparison."""
    base_models = {
        "logistic": LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=2,
            min_samples_split=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),
    }
    if HAS_XGB:
        base_models["xgboost"] = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbosity=0,
        )

    models = {}
    for name, clf in base_models.items():
        models[name] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])
    return models


def cv_evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Evaluate model with stratified cross-validation."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    y_encoded = encode_labels(y)

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": make_scorer(f1_score, average="macro"),
        "f1_weighted": make_scorer(f1_score, average="weighted"),
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
    }

    results = {}
    for name, scorer in scoring.items():
        scores = cross_val_score(model, X, y_encoded, cv=cv, scoring=scorer, n_jobs=N_JOBS)
        results[f"{name}_mean"] = float(np.mean(scores))
        results[f"{name}_std"] = float(np.std(scores))

    # ROC-AUC with probabilities
    try:
        y_proba_cv = cross_val_predict(model, X, y_encoded, cv=cv, method="predict_proba", n_jobs=N_JOBS)
        roc_auc = roc_auc_score(y_encoded, y_proba_cv, multi_class="ovr", average="macro")
        results["roc_auc_ovr_mean"] = float(roc_auc)
        results["roc_auc_ovr_std"] = 0.0
    except Exception as e:
        log.warning(f"ROC-AUC computation failed: {e}")
        results["roc_auc_ovr_mean"] = 0.0
        results["roc_auc_ovr_std"] = 0.0

    y_pred_encoded_cv = cross_val_predict(model, X, y_encoded, cv=cv, n_jobs=N_JOBS)
    y_pred_cv = decode_labels(y_pred_encoded_cv)
    results["confusion_matrix"] = confusion_matrix(y, y_pred_cv, labels=RISK_LABELS).tolist()

    return results


def optimize_hyperparams(model_name: str, X: pd.DataFrame, y: pd.Series, n_trials: int = 30) -> dict:
    """Optimize hyperparameters using Optuna."""
    if not HAS_OPTUNA:
        log.warning("Optuna not available, skipping hyperparameter optimization")
        return {}

    y_encoded = encode_labels(y)

    def objective(trial: optuna.Trial) -> float:
        if model_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 8, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
            clf = RandomForestClassifier(**params, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=N_JOBS)
        elif model_name == "xgboost" and HAS_XGB:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            }
            clf = xgb.XGBClassifier(**params, objective="multi:softprob", num_class=3, eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0)
        else:
            return 0.0

        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        score = cross_val_score(pipeline, X, y_encoded, cv=cv, scoring=make_scorer(f1_score, average="macro"), n_jobs=N_JOBS)
        return float(np.mean(score))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log.info(f"Best params for {model_name}: {study.best_params} (macro F1: {study.best_value:.4f})")
    return study.best_params


def compute_shap_values(model: Pipeline, X: pd.DataFrame, feature_names: list[str]) -> dict[str, Any]:
    """Compute SHAP values for model explainability."""
    if not HAS_SHAP:
        log.warning("SHAP not available, skipping explainability")
        return {}

    try:
        clf = model.named_steps["clf"]
        scaler = model.named_steps["scaler"]
        X_scaled = scaler.transform(X)

        if hasattr(clf, "feature_importances_"):
            explainer = shap.TreeExplainer(clf)
        else:
            explainer = shap.PermutationExplainer(clf.predict_proba, X_scaled)

        shap_values = explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        else:
            mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 2))

        feature_importance = dict(zip(feature_names, mean_abs_shap.tolist()))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

        return {
            "feature_importance_shap": feature_importance,
            "expected_value": explainer.expected_value.tolist() if hasattr(explainer.expected_value, "tolist") else float(explainer.expected_value),
        }
    except Exception as e:
        log.warning(f"SHAP computation failed: {e}")
        return {}


def compute_feature_importance(model: Pipeline, feature_names: list[str]) -> dict[str, float]:
    """Extract feature importance from the model."""
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importance = np.mean(np.abs(clf.coef_), axis=0)
    else:
        return {}
    return dict(sorted(zip(feature_names, importance.tolist()), key=lambda x: x[1], reverse=True))


def train_final_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    best_params: dict | None = None,
) -> tuple[Pipeline, dict]:
    """Train final model on full dataset."""
    models = get_models()
    base_model = models[model_name]

    if best_params:
        clf = base_model.named_steps["clf"]
        clf.set_params(**best_params)

    y_encoded = encode_labels(y)
    base_model.fit(X, y_encoded)

    shap_results = compute_shap_values(base_model, X, features)
    fi_results = compute_feature_importance(base_model, features)

    combined_results = {**shap_results, "feature_importance_model": fi_results}

    return base_model, combined_results


def save_artifacts(
    model: Pipeline,
    features: list[str],
    metrics: dict,
    shap_results: dict,
    model_name: str,
    best_params: dict,
    args: argparse.Namespace,
) -> Path:
    """Save model, metadata, and explainability artifacts."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"flood_model_{model_name}_{timestamp}.joblib"
    meta_filename = f"model_meta_{model_name}_{timestamp}.json"

    model_path = MODELS_DIR / model_filename
    meta_path = MODELS_DIR / meta_filename

    # Save model with label mapping for decoding predictions
    model_bundle = {
        "model": model,
        "features": features,
        "label_order": RISK_ORDER,
        "inv_label_order": INV_RISK_ORDER,
        "classes": RISK_LABELS,
    }
    joblib.dump(model_bundle, model_path)

    meta = {
        "model_name": model_name,
        "model_file": model_filename,
        "timestamp": timestamp,
        "training_samples": int(metrics.get("train_samples", 0)),
        "test_samples": int(metrics.get("test_samples", 0)),
        "features": features,
        "n_features": len(features),
        "classes": RISK_LABELS,
        "label_order": RISK_ORDER,
        "cv_metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "confusion_matrix": metrics.get("confusion_matrix", []),
        "confusion_matrix_labels": RISK_LABELS,
        "best_params": best_params,
        "shap_feature_importance": shap_results.get("feature_importance_shap", {}),
        "model_feature_importance": shap_results.get("feature_importance_model", {}),
        "config": {
            "n_splits": N_SPLITS,
            "random_state": RANDOM_STATE,
            "n_samples": args.n_samples,
            "optimize": args.optimize,
            "n_trials": args.n_trials,
        },
        "warning": "Synthetic demonstration dataset. Not for operational use.",
    }

    meta_path.write_text(json.dumps(meta, indent=2))

    latest_model = MODELS_DIR / "flood_model.joblib"
    latest_meta = MODELS_DIR / "model_meta.json"
    joblib.dump(model_bundle, latest_model)
    latest_meta.write_text(json.dumps(meta, indent=2))

    log.info(f"Saved model: {model_path}")
    log.info(f"Saved metadata: {meta_path}")
    log.info(f"Updated symlinks: {latest_model}, {latest_meta}")

    return model_path


def print_summary(metrics: dict, model_name: str, X_test: pd.DataFrame, y_test: pd.Series, model: Pipeline):
    """Print training summary to console."""
    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY - {model_name.upper()}")
    print(f"{'='*60}")
    print(f"CV Accuracy:      {metrics.get('accuracy_mean', 0):.4f} ± {metrics.get('accuracy_std', 0):.4f}")
    print(f"CV F1 (macro):    {metrics.get('f1_macro_mean', 0):.4f} ± {metrics.get('f1_macro_std', 0):.4f}")
    print(f"CV F1 (weighted): {metrics.get('f1_weighted_mean', 0):.4f} ± {metrics.get('f1_weighted_std', 0):.4f}")
    print(f"CV Precision:     {metrics.get('precision_macro_mean', 0):.4f} ± {metrics.get('precision_macro_std', 0):.4f}")
    print(f"CV Recall:        {metrics.get('recall_macro_mean', 0):.4f} ± {metrics.get('recall_macro_std', 0):.4f}")
    print(f"CV ROC-AUC (OVR): {metrics.get('roc_auc_ovr_mean', 0):.4f} ± {metrics.get('roc_auc_ovr_std', 0):.4f}")

    y_pred_encoded = model.predict(X_test)
    y_pred = decode_labels(y_pred_encoded)
    print(f"\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred, labels=RISK_LABELS, zero_division=0))

    cm = metrics.get("confusion_matrix", [])
    if cm:
        print(f"Confusion Matrix (rows=true, cols=pred):")
        print(pd.DataFrame(cm, index=RISK_LABELS, columns=RISK_LABELS).to_string())


def main():
    parser = argparse.ArgumentParser(description="Train flash flood risk prediction model")
    parser.add_argument("--n-samples", type=int, default=5000, help="Number of synthetic samples to generate")
    parser.add_argument("--model", choices=["logistic", "random_forest", "xgboost", "auto"], default="auto", help="Model to train (auto = compare all)")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization with Optuna")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed")
    parser.add_argument("--save-data", action="store_true", help="Save generated training data to CSV")
    args = parser.parse_args()

    log.info(f"Starting training with args: {vars(args)}")

    df = generate_synthetic_data(args.n_samples, args.seed)
    df = validate_data(df)

    if args.save_data:
        data_path = DATA_DIR / "training_data.csv"
        df.to_csv(data_path, index=False)
        log.info(f"Saved training data to {data_path}")

    X, y, features = get_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    models = get_models()
    if args.model != "auto":
        models = {args.model: models[args.model]}

    best_model_name = None
    best_score = -1
    best_metrics = None
    best_params = {}

    for name, model in models.items():
        log.info(f"\n{'='*50}")
        log.info(f"Evaluating {name}...")
        log.info(f"{'='*50}")

        if args.optimize and HAS_OPTUNA:
            params = optimize_hyperparams(name, X_train, y_train, args.n_trials)
        else:
            params = {}

        metrics = cv_evaluate(model, X_train, y_train)
        metrics["train_samples"] = len(X_train)
        metrics["test_samples"] = len(X_test)

        score = metrics.get("f1_macro_mean", 0)
        log.info(f"{name} CV macro F1: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model_name = name
            best_metrics = metrics
            best_params = params

    log.info(f"\nBest model: {best_model_name} (macro F1: {best_score:.4f})")

    final_model, shap_results = train_final_model(
        best_model_name, X_train, y_train, features, best_params
    )

    print_summary(best_metrics, best_model_name, X_test, y_test, final_model)

    save_artifacts(final_model, features, best_metrics, shap_results, best_model_name, best_params, args)

    log.info("Training complete!")


if __name__ == "__main__":
    main()