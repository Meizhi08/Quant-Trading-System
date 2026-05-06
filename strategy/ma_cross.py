"""均线交叉策略 (Golden Cross / Death Cross)。

默认参数：MA5 × MA20，可在实例化时覆盖。
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from .base import BaseStrategy, Signal, SignalType


class MACrossStrategy(BaseStrategy):
    name = "MA_Cross"

    def __init__(self, fast: int = 5, slow: int = 20):
        super().__init__({"fast": fast, "slow": slow})
        self.fast = fast
        self.slow = slow

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df[f"ma{self.fast}"] = df["close"].rolling(self.fast).mean()
        df[f"ma{self.slow}"] = df["close"].rolling(self.slow).mean()
        df["ma60"] = df["close"].rolling(60).mean()
        df["ma120"] = df["close"].rolling(120).mean()
        df["cross"] = df[f"ma{self.fast}"] - df[f"ma{self.slow}"]
        df["cross_prev"] = df["cross"].shift(1)
        return df

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        row = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else row

        fast_above_slow = row[f"ma{self.fast}"] > row[f"ma{self.slow}"]
        golden_cross = (prev[f"ma{self.fast}"] < prev[f"ma{self.slow}"]) and fast_above_slow
        death_cross  = (prev[f"ma{self.fast}"] > prev[f"ma{self.slow}"]) and not fast_above_slow

        # 趋势过滤：MA60 > MA120，且 MA60 斜率为正（近10日在上升）
        ma60_slope = row["ma60"] > float(df["ma60"].iloc[-10]) if len(df) >= 10 else True
        uptrend = row["ma60"] > row["ma120"] and ma60_slope and self._uptrend_ma200(df)

        vol_ok = self._vol_confirmed(df)

        if golden_cross and uptrend and vol_ok:
            sig, reason = SignalType.BUY, f"MA{self.fast}上穿MA{self.slow}(金叉)+趋势+量能"
        elif golden_cross and uptrend:
            sig, reason = SignalType.BUY, f"MA{self.fast}上穿MA{self.slow}(金叉)+趋势"
        elif death_cross:
            sig, reason = SignalType.SELL, f"MA{self.fast}下穿MA{self.slow}(死叉)"
        else:
            sig, reason = SignalType.HOLD, "等待金叉"

        return Signal(
            symbol=symbol, signal=sig, price=float(row["close"]),
            timestamp=datetime.now(), strategy=self.name, reason=reason,
            metadata={f"ma{self.fast}": float(row[f"ma{self.fast}"]),
                      f"ma{self.slow}": float(row[f"ma{self.slow}"])},
        )
