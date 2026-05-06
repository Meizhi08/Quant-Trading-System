"""MACD 策略（Moving Average Convergence Divergence）。

默认参数：EMA12 / EMA26 / Signal9。
金叉买入（MACD上穿Signal），死叉卖出。
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from .base import BaseStrategy, Signal, SignalType


class MACDStrategy(BaseStrategy):
    name = "MACD"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__({"fast": fast, "slow": slow, "signal": signal})
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_fast = df["close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=self.signal_period, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["macd_hist_prev"] = df["macd_hist"].shift(1)
        df["ma60"] = df["close"].rolling(60).mean()
        df["ma120"] = df["close"].rolling(120).mean()
        return df

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        row = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else row

        macd = float(row["macd"])
        sig_line = float(row["macd_signal"])
        hist = float(row["macd_hist"])
        hist_prev = float(prev["macd_hist"])
        uptrend = row["ma60"] > row["ma120"] and self._uptrend_ma200(df)

        # 金叉：柱状图由负转正
        golden = hist_prev < 0 and hist >= 0
        # 死叉：柱状图由正转负
        death = hist_prev > 0 and hist <= 0

        vol_ok = self._vol_confirmed(df)

        if golden and uptrend and vol_ok:
            sig = SignalType.BUY
            reason = f"MACD金叉+趋势向上+量能确认 MACD={macd:.3f}"
        elif golden and uptrend and not vol_ok:
            sig = SignalType.HOLD
            reason = f"MACD金叉但量能不足，观望"
        elif golden and not uptrend:
            sig = SignalType.HOLD
            reason = f"MACD金叉但趋势向下，观望"
        elif death:
            sig = SignalType.SELL
            reason = f"MACD死叉 MACD={macd:.3f}"
        elif hist > 0 and macd > sig_line and uptrend and vol_ok:
            sig = SignalType.BUY
            reason = f"MACD多头持续+趋势向上"
        else:
            sig = SignalType.HOLD
            reason = f"MACD中性 hist={hist:.3f}"

        return Signal(
            symbol=symbol, signal=sig, price=float(row["close"]),
            timestamp=datetime.now(), strategy=self.name, reason=reason,
            metadata={"macd": macd, "signal": sig_line, "hist": hist},
        )
