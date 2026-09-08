"""Turn OHLCV + strategy signals into a numeric ML dataset with labels.

Pipeline:
    OHLCV --> strategy.ict.compute_ict_features --> strategy.confluence.compute_confluence
          --> base price features --> triple_barrier_label --> dataset.parquet

Target encoding (matches the spec in the task): 0 = NO_TRADE/LOSS,
1 = BUY WIN, 2 = SELL WIN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.confluence import compute_confluence
from strategy.ict import compute_ict_features


def base_price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["return"] = df["close"].pct_change()
    out["log_return"] = np.log(df["close"] / df["close"].shift(1))
    out["range"] = df["high"] - df["low"]
    out["body"] = (df["close"] - df["open"]).abs()
    out["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    out["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    return out


def triple_barrier_label(
    df: pd.DataFrame,
    bias: pd.Series,
    atr: pd.Series,
    tp_atr_multiple: float = 2.0,
    sl_atr_multiple: float = 1.0,
    max_holding_bars: int = 48,
) -> pd.DataFrame:
    """Label each bar by simulating the hypothetical trade the confluence
    bias implies (BUY if bias=1, SELL if bias=-1, no simulation if bias=0),
    with entry=close, SL/TP from ATR multiples, and a max-bars vertical
    barrier. Whichever barrier is touched first wins; if SL and TP are both
    touched within the same bar we conservatively assume SL first.

    Runs an O(n * max_holding_bars) scan - fine for M15/H1/H4/D1, will be
    slow on very large M1 datasets; vectorize/numba if that becomes the
    bottleneck.
    """
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    atr_v = atr.to_numpy()
    bias_v = bias.to_numpy()
    n = len(df)

    label = np.zeros(n, dtype=int)
    future_return = np.full(n, np.nan)
    bars_to_outcome = np.full(n, np.nan)

    for i in range(n):
        b = bias_v[i]
        a = atr_v[i]
        if b == 0 or np.isnan(a) or a <= 0:
            continue

        entry = close[i]
        if b == 1:
            tp, sl = entry + a * tp_atr_multiple, entry - a * sl_atr_multiple
        else:
            tp, sl = entry - a * tp_atr_multiple, entry + a * sl_atr_multiple

        end = min(n, i + 1 + max_holding_bars)
        outcome = 0
        exit_j = end - 1
        for j in range(i + 1, end):
            if b == 1:
                if low[j] <= sl:
                    outcome, exit_j = 0, j
                    break
                if high[j] >= tp:
                    outcome, exit_j = 1, j
                    break
            else:
                if high[j] >= sl:
                    outcome, exit_j = 0, j
                    break
                if low[j] <= tp:
                    outcome, exit_j = 2, j
                    break

        label[i] = outcome
        if exit_j > i:
            future_return[i] = (close[exit_j] - entry) / entry * (1 if b == 1 else -1)
            bars_to_outcome[i] = exit_j - i

    out = pd.DataFrame(index=df.index)
    out["target"] = label
    out["future_return"] = future_return
    out["bars_to_outcome"] = bars_to_outcome
    return out


def build_dataset(
    df: pd.DataFrame,
    config: dict,
    secondary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Full feature engineering pipeline for one symbol/timeframe.

    `df` must be a cleaned OHLCV DataFrame (see data/pipeline.clean_ohlc).
    `config` is the loaded config.yaml dict (see utils.config.load_config).
    """
    ict_cfg = config["ict"]
    label_cfg = config["labeling"]

    ict_features = compute_ict_features(
        df,
        swing_lookback=ict_cfg["swing_lookback"],
        fvg_min_size_atr=ict_cfg["fvg_min_size_atr"],
        order_block_lookback=ict_cfg["order_block_lookback"],
        atr_period=label_cfg["atr_period"],
        sessions=config.get("sessions"),
        killzones=config.get("killzones"),
        secondary_df=secondary_df,
    )
    confluence = compute_confluence(ict_features, weights=config.get("confluence", {}).get("weights"))
    base = base_price_features(df)

    labels = triple_barrier_label(
        df,
        bias=confluence["confluence_bias"],
        atr=ict_features["atr"],
        tp_atr_multiple=label_cfg["tp_atr_multiple"],
        sl_atr_multiple=label_cfg["sl_atr_multiple"],
        max_holding_bars=label_cfg["max_holding_bars"],
    )

    ohlcv = df[["open", "high", "low", "close", "volume"]]
    dataset = pd.concat([ohlcv, base, ict_features, confluence, labels], axis=1)
    dataset = dataset.dropna(subset=["atr"])  # drop ATR warm-up period
    return dataset


# Columns that describe the future (labels / leakage risk) - never feed
# these into the model as inputs.
LABEL_COLUMNS = ["target", "future_return", "bars_to_outcome"]

# Raw price/volume levels kept in the dataset for execution/backtest (entry
# price, SL/TP calc, plotting) but excluded from the ML feature set - an
# absolute price level is non-stationary and a poor, overfit-prone feature
# compared to the derived features (returns, ATR, structure, distances).
REFERENCE_COLUMNS = ["open", "high", "low", "close", "volume"]


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    non_numeric = dataset.select_dtypes(exclude=[np.number, bool]).columns.tolist()
    excluded = set(LABEL_COLUMNS) | set(REFERENCE_COLUMNS) | set(non_numeric)
    return [c for c in dataset.columns if c not in excluded]
