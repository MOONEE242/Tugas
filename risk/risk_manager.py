"""Risk engine: the only thing allowed to say yes to a trade.

The model/strategy layer only ever produces a *signal* (direction +
confidence). Whether that signal actually becomes an order always goes
through `RiskManager.allow_trade()` first - see execution/auto_entry.py.

Design goals (per the project brief):
  - multi-entry per day is fine, bounded by risk/exposure, not a trade count
  - daily loss limit halts new entries, doesn't touch open ones
  - RR and AI-confidence thresholds are enforced here, not left to the caller
  - duplicate-signal protection: don't re-enter the exact same setup on the
    same bar/cooldown window
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def calculate_rr(entry: float, stop_loss: float, take_profit: float) -> float:
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    if risk <= 0:
        return 0.0
    return reward / risk


@dataclass
class OpenPosition:
    symbol: str
    direction: str  # "BUY" or "SELL"
    risk_pct: float
    opened_at: pd.Timestamp


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.25
    max_daily_loss_pct: float = 2.0
    max_total_open_risk_pct: float = 1.0
    max_open_positions: int = 5
    max_trades_per_day: int | None = None
    min_lot: float = 0.01
    max_lot: float = 5.0
    min_rr: float = 2.0
    min_confidence: float = 0.80
    duplicate_signal_cooldown_bars: int = 1

    @classmethod
    def from_config(cls, config: dict) -> "RiskConfig":
        risk_cfg = config["risk"]
        model_cfg = config["model"]
        return cls(
            risk_per_trade_pct=risk_cfg["risk_per_trade_pct"],
            max_daily_loss_pct=risk_cfg["max_daily_loss_pct"],
            max_total_open_risk_pct=risk_cfg["max_total_open_risk_pct"],
            max_open_positions=risk_cfg["max_open_positions"],
            max_trades_per_day=risk_cfg.get("max_trades_per_day"),
            min_lot=risk_cfg["min_lot"],
            max_lot=risk_cfg["max_lot"],
            min_rr=model_cfg["min_rr"],
            min_confidence=model_cfg["min_confidence"],
            duplicate_signal_cooldown_bars=risk_cfg["duplicate_signal_cooldown_bars"],
        )


class RiskManager:
    def __init__(self, config: RiskConfig, starting_equity: float):
        self.config = config
        self.equity = starting_equity
        self.day: pd.Timestamp | None = None
        self.daily_pnl_pct = 0.0
        self.trades_today = 0
        self.open_positions: list[OpenPosition] = []
        self._last_signal_bar: dict[tuple[str, str], int] = {}

    def _roll_day(self, timestamp) -> None:
        day = pd.Timestamp(timestamp).normalize()
        if self.day is None or day != self.day:
            self.day = day
            self.daily_pnl_pct = 0.0
            self.trades_today = 0

    def total_open_risk_pct(self) -> float:
        return sum(p.risk_pct for p in self.open_positions)

    def allow_trade(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        rr: float,
        timestamp,
        bar_index: int | None = None,
    ) -> tuple[bool, str]:
        """Returns (allowed, reason). Every rejection has an explicit reason
        so it can be logged/shown on the dashboard - never fail silently."""
        self._roll_day(timestamp)

        if confidence < self.config.min_confidence:
            return False, f"confidence {confidence:.2%} < min {self.config.min_confidence:.2%}"
        if rr < self.config.min_rr:
            return False, f"RR {rr:.2f} < min {self.config.min_rr:.2f}"
        if self.daily_pnl_pct <= -self.config.max_daily_loss_pct:
            return False, "daily loss limit reached"
        if len(self.open_positions) >= self.config.max_open_positions:
            return False, "max open positions reached"
        if self.total_open_risk_pct() + self.config.risk_per_trade_pct > self.config.max_total_open_risk_pct:
            return False, "max total open risk reached"
        if self.config.max_trades_per_day is not None and self.trades_today >= self.config.max_trades_per_day:
            return False, "max trades per day reached"

        if bar_index is not None:
            last_bar = self._last_signal_bar.get((symbol, direction))
            if last_bar is not None and (bar_index - last_bar) < self.config.duplicate_signal_cooldown_bars:
                return False, "duplicate signal cooldown"

        return True, "ok"

    def calculate_position_size(
        self,
        entry: float,
        stop_loss: float,
        contract_size: float = 100.0,
        lot_step: float = 0.01,
    ) -> float:
        """Simplified lot sizing: risk_amount / (stop_distance * contract_size).

        `contract_size` depends on the symbol (e.g. XAUUSD ~= 100 oz/lot,
        standard forex = 100,000 units/lot) - pass the real value from your
        broker's symbol spec (mt5.symbol_info(symbol).trade_contract_size)
        when wiring this to live execution.
        """
        stop_distance = abs(entry - stop_loss)
        if stop_distance <= 0:
            return 0.0
        risk_amount = self.equity * (self.config.risk_per_trade_pct / 100.0)
        raw_lot = risk_amount / (stop_distance * contract_size)
        lot = max(self.config.min_lot, min(self.config.max_lot, raw_lot))
        lot = round(lot / lot_step) * lot_step
        return round(lot, 2)

    def register_trade_opened(self, symbol: str, direction: str, timestamp, bar_index: int | None = None) -> None:
        self.open_positions.append(
            OpenPosition(symbol=symbol, direction=direction, risk_pct=self.config.risk_per_trade_pct, opened_at=pd.Timestamp(timestamp))
        )
        self.trades_today += 1
        if bar_index is not None:
            self._last_signal_bar[(symbol, direction)] = bar_index

    def register_trade_closed(self, symbol: str, direction: str, pnl_pct: float) -> None:
        """`pnl_pct` = trade P/L as % of equity (e.g. +0.5 or -0.25)."""
        for i, p in enumerate(self.open_positions):
            if p.symbol == symbol and p.direction == direction:
                del self.open_positions[i]
                break
        self.daily_pnl_pct += pnl_pct
        self.equity *= 1 + pnl_pct / 100.0
