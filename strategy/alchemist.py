"""Alchemist expert signal.

Same disclaimer as strategy/msrn.py: "Alchemist" setup rules/entry
model/confirmation depend on the exact material you follow. This gives a
working default so the pipeline runs today; tighten `compute_alchemist_score`
with your real setup/entry/confirmation rules later.

Current heuristic (setup -> entry -> confirmation chain, ICT-flavoured):
  setup:         liquidity sweep against the higher-timeframe trend
  entry model:   MSS/BOS in the direction of that trend right after the sweep
  confirmation:  a fresh FVG in the same direction, and price in discount
                 (for longs) / premium (for shorts)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_alchemist_score(ict_features: pd.DataFrame) -> pd.DataFrame:
    f = ict_features

    setup_bull = f["liquidity_sweep_low"].astype(bool)
    setup_bear = f["liquidity_sweep_high"].astype(bool)

    entry_bull = f["mss_bullish"].astype(bool) | f["bos_bullish"].astype(bool)
    entry_bear = f["mss_bearish"].astype(bool) | f["bos_bearish"].astype(bool)

    confirm_bull = f["fvg_bullish"].astype(bool) & (f["premium_discount"] < 0)
    confirm_bear = f["fvg_bearish"].astype(bool) & (f["premium_discount"] > 0)

    # look back a small rolling window so setup/entry/confirmation don't all
    # have to land on the exact same candle
    window = 5
    setup_recent_bull = setup_bull.rolling(window, min_periods=1).max().astype(bool)
    setup_recent_bear = setup_bear.rolling(window, min_periods=1).max().astype(bool)
    entry_recent_bull = entry_bull.rolling(window, min_periods=1).max().astype(bool)
    entry_recent_bear = entry_bear.rolling(window, min_periods=1).max().astype(bool)

    bull_components = np.vstack(
        [setup_recent_bull.to_numpy(), entry_recent_bull.to_numpy(), confirm_bull.to_numpy()]
    ).astype(int)
    bear_components = np.vstack(
        [setup_recent_bear.to_numpy(), entry_recent_bear.to_numpy(), confirm_bear.to_numpy()]
    ).astype(int)

    bull_score = bull_components.sum(axis=0) / 3.0
    bear_score = bear_components.sum(axis=0) / 3.0

    out = pd.DataFrame(index=f.index)
    out["alchemist_score_bull"] = bull_score
    out["alchemist_score_bear"] = bear_score
    out["alchemist_score"] = np.maximum(bull_score, bear_score)
    out["alchemist_bias"] = np.select(
        [bull_score > bear_score, bear_score > bull_score], [1, -1], default=0
    )
    return out
