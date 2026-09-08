"""Cleaning / resampling / caching for OHLCV data.

Works with plain pandas so it runs on any platform (no MT5 dependency).
The typical flow is:

    MT5 (Windows)  --> raw CSV/parquet dump  --> pipeline.clean_ohlc()
                                              --> pipeline.save_processed()
    ... later, anywhere (Linux/Colab/CI) ...
    pipeline.load_processed("XAUUSD", "M15")  --> ready-to-use DataFrame
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.config import REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def load_ohlc_csv(path: str | Path, tz: str = "UTC") -> pd.DataFrame:
    """Load an OHLCV CSV with a 'timestamp' column (or first column as index)."""
    df = pd.read_csv(path)
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.rename(columns={ts_col: "timestamp"}).set_index("timestamp").sort_index()
    df.columns = [c.lower() for c in df.columns]
    return df


def clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Basic sanity cleaning: dedupe index, drop NaNs, enforce OHLC consistency."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib hilang: {missing}")

    out = df[~df.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=REQUIRED_COLUMNS)

    # buang bar yang secara struktural tidak valid (high < low, dsb)
    valid = (
        (out["high"] >= out["low"])
        & (out["high"] >= out["open"])
        & (out["high"] >= out["close"])
        & (out["low"] <= out["open"])
        & (out["low"] <= out["close"])
    )
    out = out[valid]
    return out


_RESAMPLE_RULE = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
    "W1": "1W",
}


def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample a lower-timeframe (e.g. M1) OHLCV dataframe up to `timeframe`."""
    rule = _RESAMPLE_RULE.get(timeframe)
    if rule is None:
        raise ValueError(f"Timeframe tidak dikenal: {timeframe}")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"])
    return out


def save_processed(df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{symbol}_{timeframe}.parquet"
    df.to_parquet(path)
    return path


def load_processed(symbol: str, timeframe: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} tidak ditemukan. Jalankan `python main.py fetch-data` dulu, "
            "atau taruh data historis (CSV) di data/raw/ lalu proses lewat pipeline."
        )
    return pd.read_parquet(path)
