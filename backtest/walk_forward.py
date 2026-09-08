"""Walk-forward validation: train/val/test windows slide forward in time.

Backtest overfitting is the biggest risk in trading ML - a single lucky
train/test split can look spectacular. Walk-forward re-trains and
re-evaluates across many rolling windows, so a robust edge has to hold up
repeatedly on data the model has never seen, not just once.
"""
from __future__ import annotations

import pandas as pd
from xgboost import XGBClassifier

from backtest.backtest import run_backtest
from features.feature_engineering import feature_columns
from models.dataset import walk_forward_windows
from strategy.confluence import rule_based_signal


def _direction_from_model(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    feat_cols = feature_columns(train)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    model.fit(train[feat_cols], train["target"])
    preds = model.predict(test[feat_cols])

    direction = pd.Series("NO_TRADE", index=test.index)
    direction[preds == 1] = "BUY"
    direction[preds == 2] = "SELL"
    return direction


def run_walk_forward(dataset: pd.DataFrame, config: dict, use_model: bool = True) -> pd.DataFrame:
    wf_cfg = config["walk_forward"]
    label_cfg = config["labeling"]
    bt_cfg = config["backtest"]
    results = []

    windows = walk_forward_windows(
        dataset,
        train_years=wf_cfg["train_years"],
        val_months=wf_cfg["val_months"],
        test_months=wf_cfg["test_months"],
        step_months=wf_cfg["step_months"],
    )

    for i, split in enumerate(windows):
        if len(split.train) < 200 or len(split.test) < 20:
            continue

        train_fold = pd.concat([split.train, split.val])
        direction = _direction_from_model(train_fold, split.test) if use_model else rule_based_signal(split.test, min_score=0.6)

        atr, close = split.test["atr"], split.test["close"]
        sl = pd.Series(index=split.test.index, dtype=float)
        tp = pd.Series(index=split.test.index, dtype=float)
        buy, sell = direction == "BUY", direction == "SELL"
        sl[buy] = close[buy] - atr[buy] * label_cfg["sl_atr_multiple"]
        tp[buy] = close[buy] + atr[buy] * label_cfg["tp_atr_multiple"]
        sl[sell] = close[sell] + atr[sell] * label_cfg["sl_atr_multiple"]
        tp[sell] = close[sell] - atr[sell] * label_cfg["tp_atr_multiple"]

        result = run_backtest(
            split.test,
            direction,
            sl,
            tp,
            timeframe=config["execution_timeframe"],
            initial_cash=bt_cfg["initial_cash"],
            fee_pct=bt_cfg["fee_pct"],
            slippage_pct=bt_cfg["slippage_pct"],
        )

        row = result.stats.to_dict()
        row.update(
            {
                "window": i,
                "train_start": train_fold.index.min(),
                "train_end": train_fold.index.max(),
                "test_start": split.test.index.min(),
                "test_end": split.test.index.max(),
            }
        )
        results.append(row)

    return pd.DataFrame(results)
