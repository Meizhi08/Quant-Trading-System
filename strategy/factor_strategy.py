"""
因子策略 — 直接用多因子评分驱动买卖信号。

逻辑：
  factor_score >= buy_threshold  → BUY
  factor_score <= sell_threshold → SELL
  其余                           → HOLD

与 composite 策略的区别：信号来自 FactorEngine 的综合评分，
而不是多个指标投票。这样才能真正验证因子模型的有效性。
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from .base import BaseStrategy, Signal, SignalType
from factor import FactorEngine


class FactorStrategy(BaseStrategy):
    name = "Factor"

    def __init__(
        self,
        buy_threshold: float = 0.20,
        sell_threshold: float = -0.20,
        weights: dict[str, float] | None = None,
    ):
        super().__init__({
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
        })
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self._engine = FactorEngine(weights=weights)
        self._market_close: pd.Series | None = None  # S&P 500 close price for market filter

    def set_market_data(self, market_close: pd.Series) -> None:
        """Inject S&P 500 close price series for broad market trend filtering."""
        self._market_close = market_close

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _in_downtrend(self, df: pd.DataFrame) -> bool:
        """个股价格低于250日均线15%以上 → 结构性下跌。"""
        if len(df) < 250:
            return False
        ma250 = float(df["close"].rolling(250).mean().iloc[-1])
        return float(df["close"].iloc[-1]) < ma250 * 0.85

    def _market_in_downtrend(self, as_of_date) -> bool:
        """S&P 500 below 250-day MA → bear market, block all buys."""
        if self._market_close is None:
            return False
        series = self._market_close[self._market_close.index <= as_of_date]
        if len(series) < 250:
            return False
        ma250 = series.iloc[-250:].mean()
        return float(series.iloc[-1]) < float(ma250)

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        fs = self._engine.compute(df, symbol)
        score = fs.total_score
        price = float(df["close"].iloc[-1])
        last_date = df.index[-1]

        stock_down  = self._in_downtrend(df)
        market_down = self._market_in_downtrend(last_date)
        blocked     = stock_down or market_down

        if score >= self.buy_threshold and not blocked:
            sig = SignalType.BUY
        elif score <= self.sell_threshold:
            sig = SignalType.SELL
        else:
            sig = SignalType.HOLD

        note = ""
        if score >= self.buy_threshold and blocked:
            note = " [大盘过滤]" if market_down else " [个股过滤]"

        return Signal(
            symbol=symbol,
            signal=sig,
            price=price,
            timestamp=datetime.now(),
            strategy=self.name,
            confidence=abs(score),
            reason=f"score={score:+.3f} [{fs.grade}] {fs.reason}{note}",
            metadata={"factor_score": score, "grade": fs.grade, "factors": fs.factors},
        )
