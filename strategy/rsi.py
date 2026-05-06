"""RSI 策略。

默认参数：RSI(14)，超卖<30买入，超买>70卖出。
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, SignalType


class RSIStrategy(BaseStrategy):
    name = "RSI"

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__({"period": period, "oversold": oversold, "overbought": overbought})
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=self.period - 1, min_periods=self.period).mean()
        avg_loss = loss.ewm(com=self.period - 1, min_periods=self.period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_prev"] = df["rsi"].shift(1)
        return df

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        row = df.iloc[-1]
        rsi = float(row["rsi"])
        rsi_prev = float(row.get("rsi_prev", rsi))

        vol_ok = self._vol_confirmed(df)
        above_ma200 = self._uptrend_ma200(df)

        if rsi <= self.oversold and rsi_prev > self.oversold and vol_ok and above_ma200:
            sig = SignalType.BUY
            reason = f"RSI={rsi:.1f} 进入超卖区+量能确认，买入"
        elif rsi <= self.oversold and rsi_prev > self.oversold and (not vol_ok or not above_ma200):
            sig = SignalType.HOLD
            reason = f"RSI={rsi:.1f} 进入超卖区但{'量能不足' if not vol_ok else '趋势向下'}，观望"
        elif rsi >= self.overbought and rsi_prev < self.overbought:
            sig = SignalType.SELL
            reason = f"RSI={rsi:.1f} 进入超买区，卖出"
        elif rsi <= self.oversold and vol_ok and above_ma200:
            sig = SignalType.BUY
            reason = f"RSI={rsi:.1f} 持续超卖，持有多单"
        elif rsi <= self.oversold and (not vol_ok or not above_ma200):
            sig = SignalType.HOLD
            reason = f"RSI={rsi:.1f} 超卖但{'量能不足' if not vol_ok else '趋势向下'}，观望"
        elif rsi >= self.overbought:
            sig = SignalType.SELL
            reason = f"RSI={rsi:.1f} 持续超买，持有空单"
        else:
            sig = SignalType.HOLD
            reason = f"RSI={rsi:.1f} 中性区间"

        return Signal(
            symbol=symbol, signal=sig, price=float(row["close"]),
            timestamp=datetime.now(), strategy=self.name, reason=reason,
            metadata={"rsi": rsi},
        )
