"""Sends orders to MT5.

Windows-only at runtime (see data/mt5_connector.py for why the MetaTrader5
package can't run here) - but every call is dry-run-safe by default, so
`execution/auto_entry.py` can run its full paper-trading loop on any
platform. Only flip `dry_run=False` once you're pointed at a Windows
machine with the MT5 terminal open and logged into a DEMO account.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from data.mt5_connector import MT5_AVAILABLE, MT5UnavailableError

if MT5_AVAILABLE:
    import MetaTrader5 as mt5  # type: ignore


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[int]
    message: str
    dry_run: bool = False


class MT5Executor:
    def __init__(self, dry_run: bool = True, deviation: int = 20, magic: int = 20240001):
        self.dry_run = dry_run
        self.deviation = deviation
        self.magic = magic

    def send_market_order(
        self,
        symbol: str,
        direction: str,
        lot: float,
        sl: float,
        tp: float,
        comment: str = "ai-trading-bot",
    ) -> OrderResult:
        if self.dry_run:
            return OrderResult(
                True, None, f"[DRY-RUN] {direction} {lot} {symbol} SL={sl:.5f} TP={tp:.5f}", dry_run=True
            )

        if not MT5_AVAILABLE:
            raise MT5UnavailableError(
                "send_market_order dipanggil dengan dry_run=False tapi MetaTrader5 "
                "tidak tersedia di platform ini. Jalankan di Windows dengan MT5 terminal terbuka."
            )

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(False, None, f"symbol_info_tick gagal untuk {symbol}: {mt5.last_error()}")

        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if direction == "BUY" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None:
            return OrderResult(False, None, f"order_send gagal: {mt5.last_error()}")
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(False, result.order, f"order_send retcode={result.retcode}: {result.comment}")
        return OrderResult(True, result.order, "order sukses")

    def close_position(self, position_ticket: int) -> OrderResult:
        if self.dry_run:
            return OrderResult(True, position_ticket, f"[DRY-RUN] close position {position_ticket}", dry_run=True)
        if not MT5_AVAILABLE:
            raise MT5UnavailableError("close_position butuh MT5 di Windows.")

        positions = mt5.positions_get(ticket=position_ticket)
        if not positions:
            return OrderResult(False, None, f"posisi {position_ticket} tidak ditemukan")
        pos = positions[0]
        tick = mt5.symbol_info_tick(pos.symbol)
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": "ai-trading-bot close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(False, position_ticket, f"close gagal: {result}")
        return OrderResult(True, position_ticket, "close sukses")

    def get_open_positions(self, symbol: Optional[str] = None) -> list:
        if not MT5_AVAILABLE:
            raise MT5UnavailableError("get_open_positions butuh MT5 di Windows.")
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        return list(positions) if positions else []
