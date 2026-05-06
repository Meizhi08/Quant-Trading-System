"""回测结果指标计算。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    total_return: float       # 总收益率
    annual_return: float      # 年化收益率
    max_drawdown: float       # 最大回撤
    sharpe_ratio: float       # 夏普比率
    win_rate: float           # 胜率
    profit_factor: float      # 盈亏比
    total_trades: int         # 总交易次数
    avg_holding_days: float   # 平均持仓天数
    calmar_ratio: float       # 卡玛比率
    sortino_ratio: float      # 索提诺比率

    def to_dict(self) -> dict:
        return {
            "总收益率": f"{self.total_return:.2%}",
            "年化收益率": f"{self.annual_return:.2%}",
            "最大回撤": f"{self.max_drawdown:.2%}",
            "夏普比率": f"{self.sharpe_ratio:.3f}",
            "胜率": f"{self.win_rate:.2%}",
            "盈亏比": f"{self.profit_factor:.3f}",
            "卡玛比率": f"{self.calmar_ratio:.3f}",
            "索提诺比率": f"{self.sortino_ratio:.3f}",
            "总交易次数": self.total_trades,
            "平均持仓天数": f"{self.avg_holding_days:.1f}",
        }

    def __str__(self) -> str:
        lines = ["=" * 40, "  回测绩效报告", "=" * 40]
        for k, v in self.to_dict().items():
            lines.append(f"  {k:<12} {v}")
        lines.append("=" * 40)
        return "\n".join(lines)

    @classmethod
    def from_equity_curve(
        cls,
        equity: pd.Series,
        trades: list[dict],
        risk_free: float = 0.02,
        trading_days: int = 252,
    ) -> "BacktestMetrics":
        returns = equity.pct_change().dropna()
        total_days = len(equity)
        years = total_days / trading_days

        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        annual_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1

        # Max drawdown
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        max_drawdown = float(drawdown.min())

        # Sharpe
        excess = returns - risk_free / trading_days
        sharpe = float(excess.mean() / excess.std() * np.sqrt(trading_days)) if excess.std() else 0

        # Sortino
        downside = returns[returns < 0]
        sortino = float(
            excess.mean() / downside.std() * np.sqrt(trading_days)
        ) if len(downside) > 0 and downside.std() > 0 else 0

        # Calmar
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate & profit factor — 按时序配对每笔开平仓
        sorted_trades = sorted(trades, key=lambda t: t.get("date", ""))
        pnls: list[float] = []
        holding_days_list: list[float] = []
        open_buys: list[dict] = []
        for t in sorted_trades:
            if t["side"] == "BUY":
                open_buys.append(t)
            elif t["side"] == "SELL" and open_buys:
                b = open_buys.pop(0)
                pnls.append((t["price"] - b["price"]) * b["size"])
                try:
                    days = (pd.Timestamp(t["date"]) - pd.Timestamp(b["date"])).days
                    holding_days_list.append(float(days))
                except Exception:
                    pass

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) if pnls else 0
        profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")

        avg_holding = float(np.mean(holding_days_list)) if holding_days_list else 0.0

        return cls(
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(pnls),
            avg_holding_days=avg_holding,
            calmar_ratio=calmar,
            sortino_ratio=sortino,
        )
