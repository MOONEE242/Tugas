"""Multi-entry auto trading loop.

Core rule (straight from the project brief): the bot is NOT limited to N
trades. Whenever a new bar produces a valid setup, it is evaluated again -
what limits entries is `risk_manager.allow_trade()`, never a trade counter:

    NEW BAR -> setup? -> model prediction -> RR check -> risk_manager.allow_trade()
        -> yes -> size position -> executor.send_market_order() -> register_trade_opened()
        -> no  -> log the reason, wait for the next bar

Two ways to drive this loop:
  - `run_paper_replay()`: iterate a precomputed feature dataset bar-by-bar
    (works anywhere, no MT5 needed) - this is Phase 9 "paper trading".
  - `run_live_loop()`: poll MT5 for new candles and process each one as it
    closes (Windows + MT5 terminal required) - this is Phase 10 "live/demo".
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import pandas as pd

from execution.mt5_executor import MT5Executor
from execution.signal import Prediction, Setup, TradeDecision
from risk.risk_manager import RiskManager


def build_setup_from_row(row: pd.Series, symbol: str, timestamp, bar_index: int, config: dict) -> Optional[Setup]:
    bias = row.get("confluence_bias", 0)
    atr = row.get("atr")
    if bias == 0 or atr is None or pd.isna(atr) or atr <= 0:
        return None

    entry = row["close"]
    label_cfg = config["labeling"]
    if bias == 1:
        direction = "BUY"
        sl = entry - atr * label_cfg["sl_atr_multiple"]
        tp = entry + atr * label_cfg["tp_atr_multiple"]
    else:
        direction = "SELL"
        sl = entry + atr * label_cfg["sl_atr_multiple"]
        tp = entry - atr * label_cfg["tp_atr_multiple"]

    return Setup(
        symbol=symbol,
        timestamp=timestamp,
        bar_index=bar_index,
        direction=direction,
        entry=float(entry),
        stop_loss=float(sl),
        take_profit=float(tp),
        confluence_score=float(row.get("confluence_score", 0.0)),
    )


def predict(model, row: pd.Series, feature_cols: list[str]) -> Prediction:
    """If `model` is None, fall back to the rule-based confluence score as a
    pseudo-probability (useful to smoke-test the loop before a model is
    trained). Once trained, pass a fitted sklearn/xgboost classifier with
    classes {0: NO_TRADE, 1: BUY, 2: SELL} and predict_proba()."""
    if model is None:
        bull = float(row.get("confluence_score_bull", 0.0))
        bear = float(row.get("confluence_score_bear", 0.0))
        no_trade = max(0.0, 1.0 - bull - bear)
        total = bull + bear + no_trade
        if total <= 0:
            return Prediction(p_buy=0.0, p_sell=0.0, p_no_trade=1.0)
        return Prediction(p_buy=bull / total, p_sell=bear / total, p_no_trade=no_trade / total)

    # a single-row pd.Series collapses to dtype=object once it mixes
    # int/float/bool columns, which xgboost/sklearn reject - force back to
    # float before predicting.
    x = row[feature_cols].to_frame().T.astype(float)
    proba = model.predict_proba(x)[0]
    classes = list(model.classes_)

    def get(label: int) -> float:
        return float(proba[classes.index(label)]) if label in classes else 0.0

    return Prediction(p_buy=get(1), p_sell=get(2), p_no_trade=get(0))


def process_bar(
    row: pd.Series,
    symbol: str,
    timestamp,
    bar_index: int,
    config: dict,
    model,
    feature_cols: list[str],
    risk_manager: RiskManager,
    executor: MT5Executor,
) -> Optional[TradeDecision]:
    """Evaluate one closed bar. Returns None if there is no setup at all
    (most bars), otherwise a TradeDecision explaining what happened -
    executed, or skipped with a reason."""
    setup = build_setup_from_row(row, symbol, timestamp, bar_index, config)
    if setup is None:
        return None

    prediction = predict(model, row, feature_cols)
    if prediction.direction != setup.direction:
        return TradeDecision(setup, prediction, False, "model direction disagrees with setup bias")

    allowed, reason = risk_manager.allow_trade(
        symbol=symbol,
        direction=setup.direction,
        confidence=prediction.confidence,
        rr=setup.rr,
        timestamp=timestamp,
        bar_index=bar_index,
    )
    if not allowed:
        return TradeDecision(setup, prediction, False, reason)

    lot = risk_manager.calculate_position_size(setup.entry, setup.stop_loss)
    if lot <= 0:
        return TradeDecision(setup, prediction, False, "calculated lot size is 0")

    result = executor.send_market_order(symbol, setup.direction, lot, setup.stop_loss, setup.take_profit)
    if result.success:
        risk_manager.register_trade_opened(symbol, setup.direction, timestamp, bar_index)
    return TradeDecision(setup, prediction, result.success, result.message, lot=lot)


def run_paper_replay(
    dataset: pd.DataFrame,
    symbol: str,
    config: dict,
    feature_cols: list[str],
    model=None,
    starting_equity: float = 10_000.0,
) -> pd.DataFrame:
    """Replay a precomputed feature dataset bar-by-bar through the exact
    same decision logic as live trading. Returns a log DataFrame - one row
    per bar that produced a setup, executed or not, with the reason."""
    from risk.risk_manager import RiskConfig

    risk_manager = RiskManager(RiskConfig.from_config(config), starting_equity=starting_equity)
    executor = MT5Executor(dry_run=True)

    log_rows = []
    for bar_index, (timestamp, row) in enumerate(dataset.iterrows()):
        decision = process_bar(row, symbol, timestamp, bar_index, config, model, feature_cols, risk_manager, executor)
        if decision is None:
            continue
        log_rows.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "direction": decision.setup.direction,
                "entry": decision.setup.entry,
                "sl": decision.setup.stop_loss,
                "tp": decision.setup.take_profit,
                "rr": decision.setup.rr,
                "confidence": decision.prediction.confidence,
                "executed": decision.allowed,
                "reason": decision.reason,
                "lot": decision.lot,
            }
        )
    return pd.DataFrame(log_rows)


def run_live_loop(
    symbol: str,
    timeframe: str,
    config: dict,
    feature_cols: list[str],
    model=None,
    executor: Optional[MT5Executor] = None,
    starting_equity: float = 10_000.0,
    lookback_bars: int = 2000,
    poll_seconds: int = 15,
    on_decision: Optional[Callable[[TradeDecision], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> None:
    """Live/demo loop: poll MT5 for new closed candles and run process_bar()
    on each new one. Requires Windows + MT5 terminal open & logged in.

    `executor` defaults to a DRY-RUN executor - pass
    `MT5Executor(dry_run=False)` explicitly once you intend to send real
    orders to a DEMO (then eventually live) account.
    """
    from data.mt5_connector import MT5Connector
    from features.feature_engineering import build_dataset
    from risk.risk_manager import RiskConfig

    executor = executor or MT5Executor(dry_run=True)
    risk_manager = RiskManager(RiskConfig.from_config(config), starting_equity=starting_equity)

    conn = MT5Connector()
    conn.connect()
    last_processed_ts = None
    try:
        while True:
            if stop_flag is not None and stop_flag():
                break

            raw = conn.get_ohlc(symbol, timeframe, n_bars=lookback_bars)
            dataset = build_dataset(raw, config)
            if len(dataset) == 0:
                time.sleep(poll_seconds)
                continue

            latest_ts = dataset.index[-1]
            if latest_ts != last_processed_ts:
                bar_index = len(dataset) - 1
                row = dataset.iloc[-1]
                decision = process_bar(
                    row, symbol, latest_ts, bar_index, config, model, feature_cols, risk_manager, executor
                )
                if decision is not None and on_decision is not None:
                    on_decision(decision)
                last_processed_ts = latest_ts

            time.sleep(poll_seconds)
    finally:
        conn.shutdown()
