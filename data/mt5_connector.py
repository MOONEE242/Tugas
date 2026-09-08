"""MT5 connection + historical data download.

The `MetaTrader5` PyPI package only works on Windows (it talks to a running
MT5 terminal over a local IPC bridge). This module is written so it can be
*imported* on any platform, but any call that actually touches MT5 raises a
clear `MT5UnavailableError` on non-Windows instead of failing on import.

Typical usage (on Windows, with the MT5 terminal open and logged in, or with
MT5_LOGIN/MT5_PASSWORD/MT5_SERVER env vars set for programmatic login):

    from data.mt5_connector import MT5Connector
    conn = MT5Connector()
    conn.connect()
    df = conn.get_ohlc("XAUUSD", "M15", n_bars=20000)
    conn.shutdown()
"""
from __future__ import annotations

import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from utils.config import get_secret, load_config

try:
    import MetaTrader5 as mt5  # type: ignore

    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False


class MT5UnavailableError(RuntimeError):
    """Raised when MT5 features are used on a platform/environment without
    the MetaTrader5 terminal + Python package available."""


# Maps our config timeframe strings to MT5 constants (resolved lazily so the
# module still imports fine when `mt5` is None).
_TIMEFRAME_NAMES = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
    "W1": "TIMEFRAME_W1",
}


def _require_mt5() -> None:
    if not MT5_AVAILABLE:
        raise MT5UnavailableError(
            "Package 'MetaTrader5' tidak tersedia di platform ini "
            f"({platform.system()}). MT5 Python API hanya jalan di Windows "
            "dengan terminal MT5 terinstall. Untuk development di Linux, "
            "gunakan data historis via data/pipeline.py (CSV/parquet), lalu "
            "jalankan fetch-data / live trading di mesin Windows."
        )


@dataclass
class MT5Credentials:
    login: int
    password: str
    server: str


def load_credentials_from_env() -> MT5Credentials:
    cfg = load_config()["mt5"]
    login = get_secret(cfg["login_env"])
    password = get_secret(cfg["password_env"])
    server = get_secret(cfg["server_env"])
    return MT5Credentials(login=int(login), password=password, server=server)


class MT5Connector:
    def __init__(self, credentials: Optional[MT5Credentials] = None):
        self._connected = False
        self._credentials = credentials

    def connect(self) -> None:
        _require_mt5()
        cfg = load_config()["mt5"]
        ok = mt5.initialize(
            path=cfg.get("terminal_path"),
            timeout=cfg.get("timeout_ms", 10000),
        )
        if not ok:
            raise MT5UnavailableError(f"mt5.initialize() gagal: {mt5.last_error()}")

        creds = self._credentials or load_credentials_from_env()
        authorized = mt5.login(creds.login, password=creds.password, server=creds.server)
        if not authorized:
            mt5.shutdown()
            raise MT5UnavailableError(f"mt5.login() gagal: {mt5.last_error()}")

        self._connected = True

    def shutdown(self) -> None:
        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            self._connected = False

    def __enter__(self) -> "MT5Connector":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def get_ohlc(
        self,
        symbol: str,
        timeframe: str,
        n_bars: int = 20000,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Download the last `n_bars` OHLCV candles for symbol/timeframe."""
        _require_mt5()
        if not self._connected:
            raise MT5UnavailableError("Belum connect(). Panggil conn.connect() dulu.")

        tf_name = _TIMEFRAME_NAMES.get(timeframe)
        if tf_name is None:
            raise ValueError(f"Timeframe tidak dikenal: {timeframe}")
        tf_const = getattr(mt5, tf_name)

        if not mt5.symbol_select(symbol, True):
            raise MT5UnavailableError(f"Symbol tidak ditemukan / gagal select: {symbol}")

        end = end or datetime.now(timezone.utc)
        rates = mt5.copy_rates_from(symbol, tf_const, end, n_bars)
        if rates is None or len(rates) == 0:
            raise MT5UnavailableError(
                f"copy_rates_from mengembalikan kosong untuk {symbol} {timeframe}: "
                f"{mt5.last_error()}"
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(
            columns={
                "time": "timestamp",
                "tick_volume": "volume",
            }
        )
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df[cols].set_index("timestamp").sort_index()
        return df

    def get_tick_data(self, symbol: str, n_ticks: int = 5000) -> pd.DataFrame:
        _require_mt5()
        if not self._connected:
            raise MT5UnavailableError("Belum connect(). Panggil conn.connect() dulu.")
        ticks = mt5.copy_ticks_from(
            symbol, datetime.now(timezone.utc), n_ticks, mt5.COPY_TICKS_ALL
        )
        if ticks is None:
            raise MT5UnavailableError(f"copy_ticks_from gagal: {mt5.last_error()}")
        df = pd.DataFrame(ticks)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df.set_index("time").sort_index()

    def get_account_info(self) -> dict:
        _require_mt5()
        if not self._connected:
            raise MT5UnavailableError("Belum connect(). Panggil conn.connect() dulu.")
        info = mt5.account_info()
        if info is None:
            raise MT5UnavailableError(f"account_info() gagal: {mt5.last_error()}")
        return info._asdict()
