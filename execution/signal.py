"""Shared data structures passed between strategy -> model -> risk -> execution."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Setup:
    """A candidate trade setup detected on a specific bar."""

    symbol: str
    timestamp: object  # pd.Timestamp
    bar_index: int
    direction: str  # "BUY" or "SELL"
    entry: float
    stop_loss: float
    take_profit: float
    confluence_score: float

    @property
    def rr(self) -> float:
        risk = abs(self.entry - self.stop_loss)
        reward = abs(self.take_profit - self.entry)
        return reward / risk if risk > 0 else 0.0


@dataclass
class Prediction:
    """Model output for a given Setup."""

    p_buy: float
    p_sell: float
    p_no_trade: float

    @property
    def direction(self) -> str:
        if self.p_buy >= self.p_sell and self.p_buy >= self.p_no_trade:
            return "BUY"
        if self.p_sell >= self.p_buy and self.p_sell >= self.p_no_trade:
            return "SELL"
        return "NO_TRADE"

    @property
    def confidence(self) -> float:
        return max(self.p_buy, self.p_sell, self.p_no_trade)


@dataclass
class TradeDecision:
    setup: Setup
    prediction: Prediction
    allowed: bool
    reason: str
    lot: float = 0.0
