"""ICT concepts translated into numeric features.

This is the module the whole project's README talks about: "ubah strategi
menjadi angka". Every function here takes an OHLCV DataFrame (DatetimeIndex,
columns open/high/low/close/volume) and returns numeric/boolean columns -
never plots, never "looks good"/"looks bad" judgments.

Implementation notes / simplifications (documented on purpose, not hidden):
- Swing highs/lows use a centered rolling max/min (fractal) of width
  `2*lookback+1`. This is a standard, fast approximation - ties can mark
  more than one bar as a swing point, which is acceptable for feature
  purposes.
- "Confirmed" swing values are revealed `lookback` bars after they occur
  (via `.shift(lookback)`) so no function here leaks future information
  into bar t's features - safe to use directly as ML features.
- BOS vs MSS: BOS = close breaks the last swing point *in the direction of
  the current trend* (continuation). MSS = close breaks the last swing
  point *against* the current trend (the first sign of a reversal). This
  matches the common ICT usage where MSS marks a potential shift and BOS
  confirms continuation.
- Order blocks are found by walking backward from a BOS/MSS trigger bar to
  the nearest opposite-colored candle - the simple, widely used heuristic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def find_swings(df: pd.DataFrame, lookback: int = 2) -> pd.DataFrame:
    """Detect fractal swing highs/lows and the last *confirmed* swing price
    known as of each bar (no lookahead)."""
    out = df.copy()
    window = 2 * lookback + 1

    roll_max = out["high"].rolling(window, center=True).max()
    roll_min = out["low"].rolling(window, center=True).min()

    out["swing_high"] = (out["high"] == roll_max) & roll_max.notna()
    out["swing_low"] = (out["low"] == roll_min) & roll_min.notna()

    swing_high_price = out["high"].where(out["swing_high"])
    swing_low_price = out["low"].where(out["swing_low"])

    # information about a swing at bar i is only "confirmed" `lookback`
    # bars later (once the surrounding window closes) -> shift forward.
    out["last_swing_high"] = swing_high_price.shift(lookback).ffill()
    out["last_swing_low"] = swing_low_price.shift(lookback).ffill()
    return out


# ---------------------------------------------------------------------------
# Market structure: BOS / MSS
# ---------------------------------------------------------------------------
def detect_bos_mss(df: pd.DataFrame, lookback: int = 2) -> pd.DataFrame:
    out = find_swings(df, lookback)
    close = out["close"].to_numpy()
    last_sh = out["last_swing_high"].to_numpy()
    last_sl = out["last_swing_low"].to_numpy()
    n = len(out)

    bos_up = np.zeros(n, dtype=bool)
    bos_down = np.zeros(n, dtype=bool)
    mss_up = np.zeros(n, dtype=bool)
    mss_down = np.zeros(n, dtype=bool)
    trend = np.zeros(n, dtype=int)

    current_trend = 0
    for i in range(n):
        sh, sl, c = last_sh[i], last_sl[i], close[i]
        if not np.isnan(sh) and c > sh:
            if current_trend <= 0:
                mss_up[i] = True
                current_trend = 1
            else:
                bos_up[i] = True
        if not np.isnan(sl) and c < sl:
            if current_trend >= 0:
                mss_down[i] = True
                current_trend = -1
            else:
                bos_down[i] = True
        trend[i] = current_trend

    out["bos_bullish"] = bos_up.astype(int)
    out["bos_bearish"] = bos_down.astype(int)
    out["mss_bullish"] = mss_up.astype(int)
    out["mss_bearish"] = mss_down.astype(int)
    out["bos"] = out["bos_bullish"] | out["bos_bearish"]
    out["mss"] = out["mss_bullish"] | out["mss_bearish"]
    out["trend"] = trend  # 1 = uptrend, -1 = downtrend, 0 = undefined yet
    return out


# ---------------------------------------------------------------------------
# Fair Value Gap
# ---------------------------------------------------------------------------
def detect_fvg(df: pd.DataFrame, atr: pd.Series, min_size_atr: float = 0.10) -> pd.DataFrame:
    high, low = df["high"], df["low"]
    prev2_high = high.shift(2)
    prev2_low = low.shift(2)

    raw_bullish = low > prev2_high
    raw_bearish = high < prev2_low

    bullish_size = (low - prev2_high).clip(lower=0)
    bearish_size = (prev2_low - high).clip(lower=0)

    min_size = atr * min_size_atr
    bullish = raw_bullish & (bullish_size >= min_size)
    bearish = raw_bearish & (bearish_size >= min_size)

    out = pd.DataFrame(index=df.index)
    out["fvg_bullish"] = bullish.astype(int)
    out["fvg_bearish"] = bearish.astype(int)
    out["fvg_bullish_size"] = bullish_size.where(bullish, 0.0)
    out["fvg_bearish_size"] = bearish_size.where(bearish, 0.0)
    out["fvg_bullish_top"] = low.where(bullish)
    out["fvg_bullish_bottom"] = prev2_high.where(bullish)
    out["fvg_bearish_top"] = prev2_low.where(bearish)
    out["fvg_bearish_bottom"] = high.where(bearish)
    return out


# ---------------------------------------------------------------------------
# Liquidity sweep (wick beyond a swing point, close back inside)
# ---------------------------------------------------------------------------
def detect_liquidity_sweep(df: pd.DataFrame, last_swing_high: pd.Series, last_swing_low: pd.Series) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    sweep_high = (high > last_swing_high) & (close < last_swing_high)
    sweep_low = (low < last_swing_low) & (close > last_swing_low)

    out = pd.DataFrame(index=df.index)
    out["liquidity_sweep_high"] = sweep_high.fillna(False).astype(int)
    out["liquidity_sweep_low"] = sweep_low.fillna(False).astype(int)
    return out


# ---------------------------------------------------------------------------
# Order blocks: last opposite-colored candle before a BOS/MSS impulse
# ---------------------------------------------------------------------------
def detect_order_blocks(df: pd.DataFrame, structure: pd.DataFrame, lookback: int = 15) -> pd.DataFrame:
    open_, close, high, low = df["open"], df["close"], df["high"], df["low"]
    bull_trigger = (structure["bos_bullish"] | structure["mss_bullish"]).to_numpy().astype(bool)
    bear_trigger = (structure["bos_bearish"] | structure["mss_bearish"]).to_numpy().astype(bool)
    is_bear_candle = (close < open_).to_numpy()
    is_bull_candle = (close > open_).to_numpy()
    high_v, low_v = high.to_numpy(), low.to_numpy()
    n = len(df)

    ob_bull = np.zeros(n, dtype=bool)
    ob_bear = np.zeros(n, dtype=bool)
    ob_bull_high = np.full(n, np.nan)
    ob_bull_low = np.full(n, np.nan)
    ob_bear_high = np.full(n, np.nan)
    ob_bear_low = np.full(n, np.nan)

    for i in np.where(bull_trigger)[0]:
        start = max(0, i - lookback)
        for j in range(i - 1, start - 1, -1):
            if is_bear_candle[j]:
                ob_bull[j] = True
                ob_bull_high[j] = high_v[j]
                ob_bull_low[j] = low_v[j]
                break

    for i in np.where(bear_trigger)[0]:
        start = max(0, i - lookback)
        for j in range(i - 1, start - 1, -1):
            if is_bull_candle[j]:
                ob_bear[j] = True
                ob_bear_high[j] = high_v[j]
                ob_bear_low[j] = low_v[j]
                break

    out = pd.DataFrame(index=df.index)
    out["order_block_bullish"] = ob_bull.astype(int)
    out["order_block_bearish"] = ob_bear.astype(int)
    # nearest still-active OB zone as of each bar (forward-filled)
    out["ob_bull_active_high"] = pd.Series(ob_bull_high, index=df.index).where(ob_bull).ffill()
    out["ob_bull_active_low"] = pd.Series(ob_bull_low, index=df.index).where(ob_bull).ffill()
    out["ob_bear_active_high"] = pd.Series(ob_bear_high, index=df.index).where(ob_bear).ffill()
    out["ob_bear_active_low"] = pd.Series(ob_bear_low, index=df.index).where(ob_bear).ffill()
    return out


# ---------------------------------------------------------------------------
# Premium / discount (position within current swing range)
# ---------------------------------------------------------------------------
def premium_discount(df: pd.DataFrame, last_swing_high: pd.Series, last_swing_low: pd.Series) -> pd.DataFrame:
    rng = last_swing_high - last_swing_low
    ratio = (df["close"] - last_swing_low) / rng.replace(0, np.nan)
    out = pd.DataFrame(index=df.index)
    # 0 = equilibrium, >0 = premium (sell zone), <0 = discount (buy zone)
    out["premium_discount"] = (ratio - 0.5).clip(-1.5, 1.5)
    return out


# ---------------------------------------------------------------------------
# Previous day / week high-low
# ---------------------------------------------------------------------------
def prev_day_week_levels(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.resample("1D", label="left", closed="left").agg({"high": "max", "low": "min"}).dropna()
    daily["previous_day_high"] = daily["high"].shift(1)
    daily["previous_day_low"] = daily["low"].shift(1)

    weekly = df.resample("1W-SUN", label="left", closed="left").agg({"high": "max", "low": "min"}).dropna()
    weekly["previous_week_high"] = weekly["high"].shift(1)
    weekly["previous_week_low"] = weekly["low"].shift(1)

    left = pd.DataFrame(index=df.index).reset_index()
    ts_col = left.columns[0]

    daily_r = daily[["previous_day_high", "previous_day_low"]].reset_index()
    daily_r.columns = [ts_col, "previous_day_high", "previous_day_low"]
    merged = pd.merge_asof(left.sort_values(ts_col), daily_r.sort_values(ts_col), on=ts_col)

    weekly_r = weekly[["previous_week_high", "previous_week_low"]].reset_index()
    weekly_r.columns = [ts_col, "previous_week_high", "previous_week_low"]
    merged = pd.merge_asof(merged.sort_values(ts_col), weekly_r.sort_values(ts_col), on=ts_col)

    merged = merged.set_index(ts_col)
    merged.index.name = df.index.name
    return merged[["previous_day_high", "previous_day_low", "previous_week_high", "previous_week_low"]]


# ---------------------------------------------------------------------------
# Session highs/lows + kill zones (hours in UTC)
# ---------------------------------------------------------------------------
def session_levels(df: pd.DataFrame, sessions: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    hour = df.index.hour
    date = df.index.normalize()

    for name, window in sessions.items():
        start, end = window["start_hour"], window["end_hour"]
        in_session = (hour >= start) & (hour < end) if start < end else (hour >= start) | (hour < end)

        sub_high = df["high"][in_session]
        sub_low = df["low"][in_session]
        sub_date = date[in_session]

        running_high = sub_high.groupby(sub_date).cummax()
        running_low = sub_low.groupby(sub_date).cummin()

        col_high = pd.Series(np.nan, index=df.index)
        col_low = pd.Series(np.nan, index=df.index)
        col_high.loc[running_high.index] = running_high.values
        col_low.loc[running_low.index] = running_low.values

        out[f"{name}_high"] = col_high.ffill()
        out[f"{name}_low"] = col_low.ffill()
    return out


def killzone_flags(df: pd.DataFrame, killzones: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    hour = df.index.hour
    any_active = np.zeros(len(df), dtype=bool)

    for name, window in killzones.items():
        start, end = window["start_hour"], window["end_hour"]
        flag = (hour >= start) & (hour < end) if start < end else (hour >= start) | (hour < end)
        out[f"killzone_{name}"] = flag.astype(int)
        any_active |= np.asarray(flag)

    out["killzone_active"] = any_active.astype(int)
    return out


# ---------------------------------------------------------------------------
# SMT divergence (needs a second, correlated symbol aligned to the same index)
# ---------------------------------------------------------------------------
def smt_divergence(df_primary: pd.DataFrame, df_secondary: pd.DataFrame, lookback: int = 2) -> pd.DataFrame:
    a = find_swings(df_primary, lookback)
    b = find_swings(df_secondary, lookback)
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]

    a_sh_change = a["last_swing_high"].diff()
    b_sh_change = b["last_swing_high"].diff()
    a_sl_change = a["last_swing_low"].diff()
    b_sl_change = b["last_swing_low"].diff()

    # primary makes a new higher high but secondary doesn't confirm -> bearish SMT
    smt_bearish = (a_sh_change > 0) & (b_sh_change <= 0)
    # primary makes a new lower low but secondary doesn't confirm -> bullish SMT
    smt_bullish = (a_sl_change < 0) & (b_sl_change >= 0)

    out = pd.DataFrame(index=idx)
    out["smt_bullish"] = smt_bullish.fillna(False).astype(int)
    out["smt_bearish"] = smt_bearish.fillna(False).astype(int)
    return out.reindex(df_primary.index).fillna(0).astype(int)


# ---------------------------------------------------------------------------
# Top-level: compute everything at once
# ---------------------------------------------------------------------------
def compute_ict_features(
    df: pd.DataFrame,
    swing_lookback: int = 2,
    fvg_min_size_atr: float = 0.10,
    order_block_lookback: int = 15,
    atr_period: int = 14,
    sessions: dict | None = None,
    killzones: dict | None = None,
    secondary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute the full ICT feature set for one symbol/timeframe.

    `secondary_df` (optional): OHLC of a correlated instrument (e.g. DXY vs
    XAUUSD, or EURUSD vs GBPUSD) to compute SMT divergence against.
    """
    atr = compute_atr(df, atr_period)
    structure = detect_bos_mss(df, swing_lookback)
    fvg = detect_fvg(df, atr, fvg_min_size_atr)
    liquidity = detect_liquidity_sweep(df, structure["last_swing_high"], structure["last_swing_low"])
    order_blocks = detect_order_blocks(df, structure, order_block_lookback)
    pd_zone = premium_discount(df, structure["last_swing_high"], structure["last_swing_low"])
    levels = prev_day_week_levels(df)

    parts = [
        atr.rename("atr"),
        structure[
            [
                "swing_high",
                "swing_low",
                "last_swing_high",
                "last_swing_low",
                "bos_bullish",
                "bos_bearish",
                "bos",
                "mss_bullish",
                "mss_bearish",
                "mss",
                "trend",
            ]
        ],
        fvg,
        liquidity,
        order_blocks,
        pd_zone,
        levels,
    ]

    if sessions:
        parts.append(session_levels(df, sessions))
    if killzones:
        parts.append(killzone_flags(df, killzones))
    if secondary_df is not None:
        parts.append(smt_divergence(df, secondary_df, swing_lookback))

    features = pd.concat(parts, axis=1)
    return features
