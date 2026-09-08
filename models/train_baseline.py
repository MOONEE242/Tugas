"""Baseline ML model: XGBoost multiclass classifier (NO_TRADE=0, BUY=1, SELL=2).

Deliberately the *first* model to train (see README "Jangan mulai dari Deep
Learning") - only after this is robust out-of-sample does it make sense to
move on to LSTM/GRU/Transformer.

Usage:
    python -m models.train_baseline --dataset data/processed/XAUUSD_M15_dataset.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

from features.feature_engineering import feature_columns
from models.dataset import split_by_ratio
from utils.config import REPO_ROOT


def train(
    dataset: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    model_out: Path | None = None,
) -> dict:
    feat_cols = feature_columns(dataset)
    split = split_by_ratio(dataset, train_frac=train_frac, val_frac=val_frac)

    x_train, y_train = split.train[feat_cols], split.train["target"]
    x_val, y_val = split.val[feat_cols], split.val["target"]
    x_test, y_test = split.test[feat_cols], split.test["target"]

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    metrics = {}
    for name, x, y in [("val", x_val, y_val), ("test", x_test, y_test)]:
        if len(x) == 0:
            continue
        preds = model.predict(x)
        report = classification_report(
            y, preds, labels=[0, 1, 2], target_names=["NO_TRADE", "BUY", "SELL"], output_dict=True, zero_division=0
        )
        metrics[name] = {
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y, preds, labels=[0, 1, 2]).tolist(),
            "n_samples": int(len(x)),
        }

    metrics["expectancy_test"] = _expectancy_from_predictions(model, x_test, split.test)
    metrics["feature_columns"] = feat_cols
    metrics["train_val_test_sizes"] = [len(split.train), len(split.val), len(split.test)]

    if model_out is not None:
        model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "feature_columns": feat_cols}, model_out)

    return metrics


def _expectancy_from_predictions(model, x_test: pd.DataFrame, test_rows: pd.DataFrame, min_confidence: float = 0.6) -> dict:
    """Sanity metric: if we only 'traded' when the model's top-class
    confidence >= min_confidence and it predicted BUY/SELL, what would the
    realized expectancy (in R-multiples via future_return) have looked
    like? This is NOT a substitute for backtest/backtest.py (no
    spread/slippage/position sizing here) - just a quick baseline signal.
    """
    if len(x_test) == 0:
        return {}
    proba = model.predict_proba(x_test)
    preds = proba.argmax(axis=1)
    confidence = proba.max(axis=1)

    tradeable = (preds != 0) & (confidence >= min_confidence)
    returns = test_rows["future_return"].to_numpy()[tradeable]
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return {"n_trades": 0}

    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    win_rate = len(wins) / len(returns)
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    return {
        "n_trades": int(len(returns)),
        "win_rate": float(win_rate),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": float(expectancy),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline XGBoost model")
    parser.add_argument("--dataset", required=True, help="Path to *_dataset.parquet from feature_engineering.build_dataset")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--model-out", default=None, help="Where to save the trained model (joblib)")
    args = parser.parse_args()

    dataset = pd.read_parquet(args.dataset)
    model_out = Path(args.model_out) if args.model_out else REPO_ROOT / "models" / "xgboost" / "baseline.joblib"

    metrics = train(dataset, train_frac=args.train_frac, val_frac=args.val_frac, model_out=model_out)
    print(json.dumps(metrics, indent=2, default=str))
    print(f"\nModel saved to: {model_out}")


if __name__ == "__main__":
    main()
