"""KDJ 策略（随机指标）。

默认参数：K(9,3,3)，超卖K<20买入，超买K>80卖出。
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from .base import BaseStrategy, Signal, SignalType


class KDJStrategy(BaseStrategy):
    name = "KDJ"

    def __init__(self, period: int = 9, oversold: float = 20, overbought: float = 80):
        super().__init__({"period": period, "oversold": oversold, "overbought": overbought})
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        low_min = df["low"].rolling(self.period).min()
        high_max = df["high"].rolling(self.period).max()
        rsv = (df["close"] - low_min) / (high_max - low_min).replace(0, 1) * 100
        df["kdj_k"] = rsv.ewm(com=2, adjust=False).mean()
        df["kdj_d"] = df["kdj_k"].ewm(com=2, adjust=False).mean()
        df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]
        df["ma60"]  = df["close"].rolling(60).mean()
        df["ma120"] = df["close"].rolling(120).mean()
        return df

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        row  = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else row

        k, d, j = float(row["kdj_k"]), float(row["kdj_d"]), float(row["kdj_j"])
        k_prev, d_prev = float(prev["kdj_k"]), float(prev["kdj_d"])
        uptrend = row["ma60"] > row["ma120"]

        # K上穿D = 金叉买入
        kd_golden = k_prev < d_prev and k >= d
        # K下穿D = 死叉卖出
        kd_death  = k_prev > d_prev and k <= d

        vol_ok = self._vol_confirmed(df)

        if kd_golden and k < self.overbought and uptrend and vol_ok:
            sig = SignalType.BUY
            reason = f"KDJ金叉 K={k:.1f} D={d:.1f}+趋势向上+量能确认"
        elif kd_golden and k < self.overbought and uptrend and not vol_ok:
            sig = SignalType.HOLD
            reason = f"KDJ金叉但量能不足，观望"
        elif kd_golden and not uptrend:
            sig = SignalType.HOLD
            reason = f"KDJ金叉但趋势向下，观望"
        elif kd_death or j > 100:
            sig = SignalType.SELL
            reason = f"KDJ死叉/超买 K={k:.1f} J={j:.1f}"
        elif k < self.oversold and j < 0 and vol_ok:
            sig = SignalType.BUY
            reason = f"KDJ严重超卖+量能确认 K={k:.1f} J={j:.1f}"
        elif k < self.oversold:
            sig = SignalType.HOLD
            reason = f"KDJ超卖等待金叉/量能 K={k:.1f}"
        else:
            sig = SignalType.HOLD
            reason = f"KDJ中性 K={k:.1f} D={d:.1f}"

        return Signal(
            symbol=symbol, signal=sig, price=float(row["close"]),
            timestamp=datetime.now(), strategy=self.name, reason=reason,
            metadata={"k": k, "d": d, "j": j},
        )
