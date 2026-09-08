"""Combine ICT + MSRN + Alchemist + SNAR into one confluence score.

This is the "Feature Engine" box from the architecture diagram: each expert
strategy becomes a bull/bear score in [0, 1], then a weighted sum produces
`confluence_score_bull` / `confluence_score_bear`. This confluence score is
one of the model's *inputs* (a feature), not the final trading decision -
the final decision still goes through the ML/DL model and then the risk
engine (see execution/auto_entry.py).

A simple rule-based `rule_based_signal()` is also provided so backtest.py
and auto_entry.py have something runnable *before* a model is trained -
useful to sanity-check the feature pipeline end-to-end early.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.alchemist import compute_alchemist_score
from strategy.msrn import compute_msrn_score
from strategy.snar import compute_snar_score

DEFAULT_WEIGHTS = {
    "ict": 0.35,
    "msrn": 0.20,
    "alchemist": 0.20,
    "snar": 0.15,
    "session": 0.10,
}


def compute_ict_score(ict_features: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    f = ict_features
    bull_cols = ["bos_bullish", "mss_bullish", "fvg_bullish", "liquidity_sweep_low", "order_block_bullish"]
    bear_cols = ["bos_bearish", "mss_bearish", "fvg_bearish", "liquidity_sweep_high", "order_block_bearish"]

    bull_recent = f[bull_cols].rolling(window, min_periods=1).max()
    bear_recent = f[bear_cols].rolling(window, min_periods=1).max()

    bull_score = bull_recent.mean(axis=1)
    bear_score = bear_recent.mean(axis=1)

    out = pd.DataFrame(index=f.index)
    out["ict_score_bull"] = bull_score
    out["ict_score_bear"] = bear_score
    out["ict_score"] = np.maximum(bull_score, bear_score)
    out["ict_bias"] = np.select([bull_score > bear_score, bear_score > bull_score], [1, -1], default=0)
    return out


def compute_confluence(
    ict_features: pd.DataFrame,
    weights: dict | None = None,
) -> pd.DataFrame:
    """Compute all expert scores + the weighted confluence score.

    `ict_features` must be the output of `strategy.ict.compute_ict_features`
    (needs to include `killzone_active` if you want the session weight to
    matter - otherwise it's treated as 0).
    """
    weights = weights or DEFAULT_WEIGHTS

    ict_score = compute_ict_score(ict_features)
    msrn_score = compute_msrn_score(ict_features)
    alchemist_score = compute_alchemist_score(ict_features)
    snar_score = compute_snar_score(ict_features)

    session_score = (
        ict_features["killzone_active"] if "killzone_active" in ict_features.columns
        else pd.Series(0, index=ict_features.index)
    )

    confluence_bull = (
        weights["ict"] * ict_score["ict_score_bull"]
        + weights["msrn"] * msrn_score["msrn_score_bull"]
        + weights["alchemist"] * alchemist_score["alchemist_score_bull"]
        + weights["snar"] * snar_score["snar_score_bull"]
        + weights["session"] * session_score
    )
    confluence_bear = (
        weights["ict"] * ict_score["ict_score_bear"]
        + weights["msrn"] * msrn_score["msrn_score_bear"]
        + weights["alchemist"] * alchemist_score["alchemist_score_bear"]
        + weights["snar"] * snar_score["snar_score_bear"]
        + weights["session"] * session_score
    )

    out = pd.concat([ict_score, msrn_score, alchemist_score, snar_score], axis=1)
    out["confluence_score_bull"] = confluence_bull
    out["confluence_score_bear"] = confluence_bear
    out["confluence_score"] = np.maximum(confluence_bull, confluence_bear)
    out["confluence_bias"] = np.select(
        [confluence_bull > confluence_bear, confluence_bear > confluence_bull], [1, -1], default=0
    )
    return out


def rule_based_signal(confluence: pd.DataFrame, min_score: float = 0.6) -> pd.Series:
    """Naive rule-based BUY/SELL/NO_TRADE signal straight from confluence
    scores - a placeholder decision rule to smoke-test the pipeline before a
    real model exists. The production decision comes from models/, not this.
    """
    signal = pd.Series("NO_TRADE", index=confluence.index)
    buy = (confluence["confluence_bias"] == 1) & (confluence["confluence_score_bull"] >= min_score)
    sell = (confluence["confluence_bias"] == -1) & (confluence["confluence_score_bear"] >= min_score)
    signal[buy] = "BUY"
    signal[sell] = "SELL"
    return signal
