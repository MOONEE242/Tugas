"""MSRN expert signal.

IMPORTANT: "MSRN" isn't a single universally-defined framework the way ICT
is - the exact rules depend on the material you personally use. This module
gives you a working, config-driven scoring function built on top of the ICT
primitives (`strategy/ict.py`) so the whole pipeline (features -> dataset ->
model -> backtest) runs end-to-end today. Replace the scoring logic in
`compute_msrn_score()` with your own precise rules whenever you're ready -
the function signature (in: ICT feature DataFrame, out: score/bias columns)
is the only contract the rest of the system depends on.

Current heuristic: market-structure + range/network confluence -
  - structure aligned (trend == bias direction)
  - liquidity sweep just occurred against that bias
  - price sitting in the "network" of a recent order block
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_msrn_score(ict_features: pd.DataFrame) -> pd.DataFrame:
    f = ict_features
    close_in_bull_ob = (
        f["ob_bull_active_low"].notna()
        & (f["ob_bull_active_low"] <= f["last_swing_high"])
        & (f["trend"] >= 0)
    )
    close_in_bear_ob = (
        f["ob_bear_active_high"].notna()
        & (f["ob_bear_active_high"] >= f["last_swing_low"])
        & (f["trend"] <= 0)
    )

    bullish_votes = (
        (f["trend"] == 1).astype(int)
        + f["liquidity_sweep_low"]
        + f["mss_bullish"]
        + close_in_bull_ob.astype(int)
    )
    bearish_votes = (
        (f["trend"] == -1).astype(int)
        + f["liquidity_sweep_high"]
        + f["mss_bearish"]
        + close_in_bear_ob.astype(int)
    )

    max_votes = 4
    bullish_score = bullish_votes / max_votes
    bearish_score = bearish_votes / max_votes

    out = pd.DataFrame(index=f.index)
    out["msrn_score_bull"] = bullish_score
    out["msrn_score_bear"] = bearish_score
    out["msrn_score"] = np.maximum(bullish_score, bearish_score)
    out["msrn_bias"] = np.select(
        [bullish_score > bearish_score, bearish_score > bullish_score],
        [1, -1],
        default=0,
    )
    return out
