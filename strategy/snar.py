"""SNAR (Structure / Liquidity / Supply-Demand) expert signal.

Same disclaimer as msrn.py / alchemist.py: tune `compute_snar_score` to your
exact SNAR/SMC rules. Default heuristic combines:
  - structure (trend direction from BOS/MSS)
  - liquidity (recent sweep in the opposite direction of trend = fuel for a
    move back into structure)
  - supply/demand (order block zones acting as supply/demand)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_snar_score(ict_features: pd.DataFrame) -> pd.DataFrame:
    f = ict_features

    structure_bull = (f["trend"] == 1).astype(int)
    structure_bear = (f["trend"] == -1).astype(int)

    liquidity_bull = f["liquidity_sweep_low"].astype(int)
    liquidity_bear = f["liquidity_sweep_high"].astype(int)

    demand_zone_active = f["ob_bull_active_low"].notna().astype(int)
    supply_zone_active = f["ob_bear_active_high"].notna().astype(int)

    bull_score = (structure_bull + liquidity_bull + demand_zone_active) / 3.0
    bear_score = (structure_bear + liquidity_bear + supply_zone_active) / 3.0

    out = pd.DataFrame(index=f.index)
    out["snar_score_bull"] = bull_score
    out["snar_score_bear"] = bear_score
    out["snar_score"] = np.maximum(bull_score, bear_score)
    out["snar_bias"] = np.select(
        [bull_score > bear_score, bear_score > bull_score], [1, -1], default=0
    )
    return out
