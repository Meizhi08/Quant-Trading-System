"""
风险管理模块。

功能：
  - 凯利公式仓位计算（半凯利，支持从历史交易动态计算参数）
  - 单日最大亏损限制
  - 最大回撤熔断（从历史最高点跌幅超阈值则暂停）
  - 单股最大仓位上限
  - 总仓位 & 持仓数量限制
"""

from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from loguru import logger

from config import settings


@dataclass
class PositionRecord:
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * (self.current_price or self.avg_cost)

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_cost) * self.quantity

    @property
    def pnl_pct(self) -> float:
        return (self.current_price / self.avg_cost - 1) if self.avg_cost else 0


class RiskManager:
    """
    实盘 & 回测共用的风控层。

    参数全部来自 settings，也可在构造时覆盖。
    """

    def __init__(
        self,
        initial_capital: float = settings.initial_cash,
        max_daily_loss_pct: float = settings.max_daily_loss_pct,
        max_position_pct: float = settings.max_position_pct,
        kelly_fraction: float = settings.kelly_fraction,
        max_positions: int = 10,
        max_drawdown_pct: float = 0.20,
    ):
        self.initial_capital = initial_capital
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_position_pct = max_position_pct
        self.kelly_fraction = kelly_fraction
        self.max_positions = max_positions
        self.max_drawdown_pct = max_drawdown_pct

        self._positions: dict[str, PositionRecord] = {}
        self._day_start_equity: float = initial_capital
        self._today: date = date.today()
        self._daily_pnl: float = 0.0
        self._peak_equity: float = initial_capital

    # ── 最大回撤熔断 ──────────────────────────────────────────────────────────

    def update_peak(self, equity: float) -> None:
        """每日收盘后调用，更新历史最高资产。"""
        if equity > self._peak_equity:
            self._peak_equity = equity

    def is_drawdown_breached(self, equity: float) -> bool:
        """从历史最高点回撤超过 max_drawdown_pct 则熔断。"""
        if self._peak_equity <= 0:
            return False
        drawdown = (self._peak_equity - equity) / self._peak_equity
        if drawdown >= self.max_drawdown_pct:
            logger.warning(
                f"[风控] 最大回撤熔断！当前回撤={drawdown:.2%} "
                f"(峰值={self._peak_equity:.0f}, 阈值={self.max_drawdown_pct:.2%})"
            )
            return True
        return False

    # ── 凯利公式 ──────────────────────────────────────────────────────────────

    @staticmethod
    def calc_kelly_params(trades: list[dict]) -> tuple[float, float, float]:
        """
        从历史成交记录动态计算凯利参数。
        返回 (win_rate, avg_win_pct, avg_loss_pct)。
        trades 中每条记录需有 side / price / avg_cost 字段。
        数据不足时返回保守默认值。
        """
        sells = [t for t in trades if t.get("side") == "SELL" and t.get("avg_cost")]
        if len(sells) < 5:
            return 0.50, 0.08, 0.05  # 不足5笔时用保守默认

        returns = [(t["price"] - t["avg_cost"]) / t["avg_cost"] for t in sells]
        wins  = [r for r in returns if r > 0]
        losses = [abs(r) for r in returns if r <= 0]

        win_rate  = len(wins) / len(returns)
        avg_win   = sum(wins)   / len(wins)   if wins   else 0.08
        avg_loss  = sum(losses) / len(losses) if losses else 0.05
        return win_rate, avg_win, avg_loss

    def kelly_position_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        返回建议仓位比例（0-1）。
        使用半凯利降低波动。

        Kelly% = W/L - (1-W)/(G/L)
        其中 W=胜率, G=平均盈利, L=平均亏损（取绝对值）
        """
        if avg_loss <= 0:
            return 0.0
        b = avg_win / avg_loss        # 赔率
        kelly = (win_rate * b - (1 - win_rate)) / b
        kelly = max(0.0, min(kelly, 1.0))
        half_kelly = kelly * self.kelly_fraction
        # 再受单股上限约束
        return min(half_kelly, self.max_position_pct)

    # ── 单日亏损检查 ──────────────────────────────────────────────────────────

    def record_daily_pnl(self, pnl: float) -> None:
        """每笔成交后更新当日 PnL。"""
        today = date.today()
        if today != self._today:
            self._today = today
            self._daily_pnl = 0.0
        self._daily_pnl += pnl

    def is_daily_loss_breached(self, equity: float | None = None) -> bool:
        """如果当日亏损超过初始资金的 max_daily_loss_pct，返回 True（触发熔断）。"""
        equity = equity or self.initial_capital
        loss_pct = -self._daily_pnl / equity
        if loss_pct >= self.max_daily_loss_pct:
            logger.warning(
                f"[风控] 单日亏损熔断！当日亏损={loss_pct:.2%} "
                f"(阈值={self.max_daily_loss_pct:.2%})"
            )
            return True
        return False

    # ── 仓位检查 ──────────────────────────────────────────────────────────────

    def can_open_position(self, symbol: str, value: float, total_equity: float) -> bool:
        """买入前检查：单股上限 & 持仓数量上限。"""
        if len(self._positions) >= self.max_positions and symbol not in self._positions:
            logger.warning(f"[风控] 持仓数量已达上限 {self.max_positions}")
            return False
        current_val = self._positions[symbol].market_value if symbol in self._positions else 0
        new_pct = (current_val + value) / total_equity
        if new_pct > self.max_position_pct:
            logger.warning(
                f"[风控] {symbol} 仓位上限！当前={new_pct:.2%} "
                f"(上限={self.max_position_pct:.2%})"
            )
            return False
        return True

    # ── 持仓更新 ──────────────────────────────────────────────────────────────

    def update_position(self, symbol: str, qty_delta: int, price: float) -> None:
        if symbol in self._positions:
            pos = self._positions[symbol]
            new_qty = pos.quantity + qty_delta
            if new_qty <= 0:
                del self._positions[symbol]
            else:
                if qty_delta > 0:
                    # 加仓：重新计算均价
                    total_cost = pos.avg_cost * pos.quantity + price * qty_delta
                    pos.avg_cost = total_cost / new_qty
                pos.quantity = new_qty
        elif qty_delta > 0:
            self._positions[symbol] = PositionRecord(
                symbol=symbol, quantity=qty_delta, avg_cost=price, current_price=price
            )

    def update_prices(self, prices: dict[str, float]) -> None:
        for symbol, price in prices.items():
            if symbol in self._positions:
                self._positions[symbol].current_price = price

    @property
    def positions(self) -> dict[str, PositionRecord]:
        return self._positions

    def portfolio_summary(self, cash: float) -> dict:
        total_mv = sum(p.market_value for p in self._positions.values())
        total_pnl = sum(p.unrealized_pnl for p in self._positions.values())
        return {
            "cash": cash,
            "market_value": total_mv,
            "total_equity": cash + total_mv,
            "unrealized_pnl": total_pnl,
            "daily_pnl": self._daily_pnl,
            "positions": [
                {
                    "symbol": p.symbol,
                    "qty": p.quantity,
                    "avg_cost": p.avg_cost,
                    "current_price": p.current_price,
                    "market_value": p.market_value,
                    "pnl_pct": f"{p.pnl_pct:.2%}",
                }
                for p in self._positions.values()
            ],
        }
