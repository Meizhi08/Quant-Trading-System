"""
Stochastic Oscillator strategy — standard NA markets indicator.

Parameters: %K period=14, %K smoothing=3, %D period=3
Oversold: %K < 20 and %K crosses above %D  → BUY
Overbought: %K > 80 and %K crosses below %D → SELL
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from .base import BaseStrategy, Signal, SignalType


class StochasticStrategy(BaseStrategy):
    name = "Stochastic"

    def __init__(self, k_period: int = 14, k_smooth: int = 3,
                 oversold: float = 20, overbought: float = 80):
        super().__init__({"k_period": k_period, "k_smooth": k_smooth,
                          "oversold": oversold, "overbought": overbought})
        self.k_period   = k_period
        self.k_smooth   = k_smooth
        self.oversold   = oversold
        self.overbought = overbought

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        low_min  = df["low"].rolling(self.k_period).min()
        high_max = df["high"].rolling(self.k_period).max()
        raw_k = (df["close"] - low_min) / (high_max - low_min).replace(0, 1) * 100
        df["stoch_k"] = raw_k.rolling(self.k_smooth).mean()   # smoothed %K
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()        # %D (signal line)
        df["ma50"]  = df["close"].rolling(50).mean()
        df["ma200"] = df["close"].rolling(200).mean()
        return df

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < self.k_period + self.k_smooth + 3:
            return self._hold(df, symbol, "insufficient data")

        row  = df.iloc[-1]
        prev = df.iloc[-2]

        k, d = float(row["stoch_k"]), float(row["stoch_d"])
        k_prev, d_prev = float(prev["stoch_k"]), float(prev["stoch_d"])

        # Uptrend: price above MA50, MA50 above MA200
        uptrend = (float(row["close"]) > float(row.get("ma50", 0)) and
                   float(row.get("ma50", 0)) > float(row.get("ma200", 0)))

        k_cross_up   = k_prev < d_prev and k >= d   # %K crosses above %D (bullish)
        k_cross_down = k_prev > d_prev and k <= d   # %K crosses below %D (bearish)

        vol_ok = self._vol_confirmed(df)

        if k_cross_up and k < self.overbought and uptrend and vol_ok:
            sig    = SignalType.BUY
            reason = f"Stoch bullish cross %K={k:.1f} %D={d:.1f} + uptrend + volume"
        elif k_cross_up and k < self.overbought and uptrend:
            sig    = SignalType.BUY
            reason = f"Stoch bullish cross %K={k:.1f} %D={d:.1f} + uptrend"
        elif k_cross_up and not uptrend:
            sig    = SignalType.HOLD
            reason = f"Stoch bullish cross but downtrend — wait"
        elif k_cross_down or k > self.overbought:
            sig    = SignalType.SELL
            reason = f"Stoch bearish cross / overbought %K={k:.1f}"
        elif k < self.oversold and k_prev >= d_prev:
            sig    = SignalType.HOLD
            reason = f"Stoch oversold, waiting for cross %K={k:.1f}"
        else:
            sig    = SignalType.HOLD
            reason = f"Stoch neutral %K={k:.1f} %D={d:.1f}"

        return Signal(
            symbol=symbol, signal=sig, price=float(row["close"]),
            timestamp=datetime.now(), strategy=self.name, reason=reason,
            metadata={"stoch_k": round(k, 1), "stoch_d": round(d, 1)},
        )

    def _hold(self, df: pd.DataFrame, symbol: str, reason: str) -> Signal:
        return Signal(
            symbol=symbol, signal=SignalType.HOLD,
            price=float(df["close"].iloc[-1]),
            timestamp=datetime.now(), strategy=self.name, reason=reason,
        )
