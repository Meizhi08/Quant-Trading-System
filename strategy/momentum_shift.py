"""
Momentum Shift strategy — Python port of BigBeluga's "Momentum Shift" indicator.

Logic:
  mom = HMA(close - close[length], smooth)
  rising  = mom crosses above mom[2]  → BUY
  falling = mom crosses below mom[2]  → SELL

Extra strength filter (mirrors Pine Script):
  rising  + mom < 0  → momentum recovering from negative territory (stronger BUY)
  falling + mom > 0  → momentum dropping from positive territory (stronger SELL)
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, SignalType

_ROLL = 50  # rolling window for normalisation


class MomentumShiftStrategy(BaseStrategy):
    name = "MomentumShift"

    def __init__(self, length: int = 50, smooth: int = 50):
        super().__init__({"length": length, "smooth": smooth})
        self.length = length
        self.smooth  = smooth

    # ── Hull Moving Average ────────────────────────────────────────────────
    @staticmethod
    def _wma(series: pd.Series, period: int) -> pd.Series:
        weights = np.arange(1, period + 1, dtype=float)
        return series.rolling(period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

    def _hma(self, series: pd.Series, period: int) -> pd.Series:
        half       = max(period // 2, 1)
        sqrt_p     = max(int(period ** 0.5), 1)
        raw        = 2 * self._wma(series, half) - self._wma(series, period)
        return self._wma(raw, sqrt_p)

    # ── Indicators ─────────────────────────────────────────────────────────
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        price_change   = df["close"] - df["close"].shift(self.length)
        df["mom_shift"]   = self._hma(price_change, self.smooth)
        df["mom_shift_2"] = df["mom_shift"].shift(2)   # mom[2] in Pine Script
        return df

    # ── Continuous score ───────────────────────────────────────────────────
    def _continuous_score(self, df: pd.DataFrame) -> float:
        """
        Return a continuous score in [-1, +1] every bar (not just at crossovers).

        60% level  — how strong/positive is mom relative to its recent range
        40% accel  — is mom accelerating (rising) or decelerating (falling)
        """
        if "mom_shift" not in df.columns or len(df) < 10:
            return 0.0

        mom   = float(df["mom_shift"].iloc[-1])
        mom_2 = (float(df["mom_shift_2"].iloc[-1])
                 if "mom_shift_2" in df.columns
                 else float(df["mom_shift"].iloc[-3]))

        window = min(_ROLL, len(df))

        std = float(df["mom_shift"].rolling(window).std().iloc[-1])
        level = (float(np.clip(mom / std, -2.0, 2.0)) / 2.0
                 if std > 0 else float(np.sign(mom)) * 0.3)

        delta = mom - mom_2
        d_std = float(df["mom_shift"].diff(2).rolling(window).std().iloc[-1])
        accel = (float(np.clip(delta / d_std, -2.0, 2.0)) / 2.0
                 if d_std > 0 else float(np.sign(delta)) * 0.2)

        return float(np.clip(0.6 * level + 0.4 * accel, -1.0, 1.0))

    # ── Signal ─────────────────────────────────────────────────────────────
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        min_bars = self.length + self.smooth + 10
        if len(df) < min_bars:
            return self._make(df, symbol, SignalType.HOLD, "数据不足",
                              {"mom": 0.0, "continuous_score": 0.0})

        row  = df.iloc[-1]
        prev = df.iloc[-2]

        mom      = float(row["mom_shift"])
        mom_2    = float(row["mom_shift_2"])
        mom_p    = float(prev["mom_shift"])
        mom_p2   = float(prev["mom_shift_2"])

        # crossover / crossunder vs mom[2]
        rising  = (mom_p <= mom_p2) and (mom > mom_2)
        falling = (mom_p >= mom_p2) and (mom < mom_2)

        above_ma200 = self._uptrend_ma200(df)
        cont_score  = self._continuous_score(df)

        if rising and above_ma200:
            strength = "负区域反转" if mom < 0 else "动量上穿"
            sig    = SignalType.BUY
            reason = f"MomShift {strength} mom={mom:.2f} score={cont_score:+.2f}"
        elif falling:
            strength = "正区域下穿" if mom > 0 else "动量下穿"
            sig    = SignalType.SELL
            reason = f"MomShift {strength} mom={mom:.2f} score={cont_score:+.2f}"
        elif rising and not above_ma200:
            sig    = SignalType.HOLD
            reason = f"MomShift 上穿但价格低于MA200，观望 score={cont_score:+.2f}"
        else:
            sig    = SignalType.HOLD
            reason = f"MomShift 中性 mom={mom:.2f} score={cont_score:+.2f}"

        return self._make(df, symbol, sig, reason,
                          {"mom": round(mom, 4),
                           "continuous_score": round(cont_score, 4)})

    def _make(
        self,
        df: pd.DataFrame,
        symbol: str,
        sig: SignalType,
        reason: str,
        metadata: dict | None = None,
    ) -> Signal:
        return Signal(
            symbol=symbol, signal=sig,
            price=float(df["close"].iloc[-1]),
            timestamp=datetime.now(), strategy=self.name,
            reason=reason, metadata=metadata or {},
        )
