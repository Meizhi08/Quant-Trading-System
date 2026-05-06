"""布林带策略 (Bollinger Bands)。

默认：20日均线 ±2σ，价格触下轨买，触上轨卖。
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from .base import BaseStrategy, Signal, SignalType


class BollingerStrategy(BaseStrategy):
    name = "Bollinger"

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__({"period": period, "std_dev": std_dev})
        self.period = period
        self.std_dev = std_dev

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["bb_mid"] = df["close"].rolling(self.period).mean()
        std = df["close"].rolling(self.period).std()
        df["bb_upper"] = df["bb_mid"] + self.std_dev * std
        df["bb_lower"] = df["bb_mid"] - self.std_dev * std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        df["ma60"] = df["close"].rolling(60).mean()
        df["ma120"] = df["close"].rolling(120).mean()
        return df

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        row = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else row

        close = float(row["close"])
        upper = float(row["bb_upper"])
        lower = float(row["bb_lower"])
        pct = float(row["bb_pct"])
        uptrend = row["ma60"] > row["ma120"] and self._uptrend_ma200(df)

        vol_ok = self._vol_confirmed(df)

        if float(prev["close"]) < float(prev["bb_lower"]) and close >= lower and uptrend and vol_ok:
            sig = SignalType.BUY
            reason = f"下轨反弹+趋势向上+量能确认，%B={pct:.2f}"
        elif float(prev["close"]) < float(prev["bb_lower"]) and close >= lower and uptrend and not vol_ok:
            sig = SignalType.HOLD
            reason = f"下轨反弹但量能不足，%B={pct:.2f}"
        elif float(prev["close"]) > float(prev["bb_upper"]) and close <= upper and not uptrend:
            sig = SignalType.SELL
            reason = f"上轨回落+趋势向下，%B={pct:.2f}"
        elif close < lower:
            sig = SignalType.HOLD
            reason = f"突破下轨等待企稳，%B={pct:.2f}"
        elif close > upper:
            sig = SignalType.HOLD
            reason = f"突破上轨等待回落，%B={pct:.2f}"
        else:
            sig = SignalType.HOLD
            reason = f"布林带内，%B={pct:.2f}"

        return Signal(
            symbol=symbol, signal=sig, price=close,
            timestamp=datetime.now(), strategy=self.name, reason=reason,
            metadata={"bb_upper": upper, "bb_mid": float(row["bb_mid"]),
                      "bb_lower": lower, "bb_pct": pct},
        )
