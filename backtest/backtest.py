"""Historical backtest using VectorBT.

Turns a per-bar signal (BUY/SELL/NO_TRADE) plus SL/TP price levels into a
VectorBT Portfolio and reports the metrics that actually matter: win rate,
profit factor, expectancy, max drawdown, Sharpe/Sortino - not just "total
return", since high win rate or high total return alone can hide
overfitting or a fragile, low-frequency edge.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Package 'vectorbt' belum terinstall. Jalankan `pip install vectorbt` "
        "(lihat requirements.txt)."
    ) from exc

TIMEFRAME_TO_FREQ = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}


@dataclass
class BacktestResult:
    portfolio: "vbt.Portfolio"
    stats: pd.Series


def run_backtest(
    dataset: pd.DataFrame,
    direction: pd.Series,
    stop_loss: pd.Series,
    take_profit: pd.Series,
    timeframe: str,
    initial_cash: float = 10_000.0,
    fee_pct: float = 0.0002,
    slippage_pct: float = 0.0001,
) -> BacktestResult:
    close = dataset["close"]

    long_entries = (direction == "BUY").reindex(close.index, fill_value=False)
    short_entries = (direction == "SELL").reindex(close.index, fill_value=False)

    sl_pct = ((close - stop_loss).abs() / close).fillna(0.0)
    tp_pct = ((take_profit - close).abs() / close).fillna(0.0)

    freq = TIMEFRAME_TO_FREQ.get(timeframe, timeframe)

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=long_entries,
        short_entries=short_entries,
        sl_stop=sl_pct,
        tp_stop=tp_pct,
        init_cash=initial_cash,
        fees=fee_pct,
        slippage=slippage_pct,
        freq=freq,
    )

    trades = pf.trades
    n_trades = trades.count()
    stats = pd.Series(
        {
            "total_return_pct": pf.total_return() * 100,
            "win_rate_pct": trades.win_rate() * 100 if n_trades > 0 else np.nan,
            "profit_factor": trades.profit_factor() if n_trades > 0 else np.nan,
            "expectancy": trades.expectancy() if n_trades > 0 else np.nan,
            "max_drawdown_pct": pf.max_drawdown() * 100,
            "sharpe_ratio": pf.sharpe_ratio(),
            "sortino_ratio": pf.sortino_ratio(),
            "num_trades": n_trades,
        }
    )
    return BacktestResult(portfolio=pf, stats=stats)


def run_backtest_from_confluence(dataset: pd.DataFrame, config: dict, min_score: float = 0.6) -> BacktestResult:
    """Convenience wrapper: derive BUY/SELL/NO_TRADE straight from the
    rule-based confluence score (strategy.confluence.rule_based_signal) -
    useful to sanity-check the pipeline before a trained model exists."""
    from strategy.confluence import rule_based_signal

    signal = rule_based_signal(dataset, min_score=min_score)
    label_cfg = config["labeling"]
    atr, close = dataset["atr"], dataset["close"]

    sl = pd.Series(np.nan, index=dataset.index)
    tp = pd.Series(np.nan, index=dataset.index)
    buy, sell = signal == "BUY", signal == "SELL"
    sl[buy] = close[buy] - atr[buy] * label_cfg["sl_atr_multiple"]
    tp[buy] = close[buy] + atr[buy] * label_cfg["tp_atr_multiple"]
    sl[sell] = close[sell] + atr[sell] * label_cfg["sl_atr_multiple"]
    tp[sell] = close[sell] - atr[sell] * label_cfg["tp_atr_multiple"]

    bt_cfg = config["backtest"]
    return run_backtest(
        dataset,
        direction=signal,
        stop_loss=sl,
        take_profit=tp,
        timeframe=config["execution_timeframe"],
        initial_cash=bt_cfg["initial_cash"],
        fee_pct=bt_cfg["fee_pct"],
        slippage_pct=bt_cfg["slippage_pct"],
    )
