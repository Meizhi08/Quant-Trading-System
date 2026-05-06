"""
Composite strategy — weighted vote across technical indicators + TradingView.

Weights (use_tv=True):
  MA 0.12 / RSI 0.12 / Bollinger 0.12 / MACD 0.12 / Stochastic 0.12 / MomShift 0.15 / TV 0.25

MomentumShift gets slightly more weight — it captures trend turning points the others miss.
TV signal gets the highest single weight — it already aggregates 26 indicators.
Weekly TV confirmation: if daily signal is BUY but weekly TV is SELL, downgrade to HOLD.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from .base import BaseStrategy, Signal, SignalType
from .ma_cross import MACrossStrategy
from .rsi import RSIStrategy
from .bollinger import BollingerStrategy
from .macd import MACDStrategy
from .stochastic import StochasticStrategy
from .momentum_shift import MomentumShiftStrategy

_VOTE = {SignalType.BUY: 1.0, SignalType.HOLD: 0.0, SignalType.SELL: -1.0}


class CompositeStrategy(BaseStrategy):
    name = "Composite"

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        use_tv: bool = True,
        buy_threshold: float = 0.25,
        sell_threshold: float = -0.25,
    ):
        super().__init__()
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.use_tv = use_tv
        w = weights or {}

        if use_tv:
            self._strategies: list[tuple[BaseStrategy, float]] = [
                (MACrossStrategy(),       w.get("ma",    0.12)),
                (RSIStrategy(),           w.get("rsi",   0.12)),
                (BollingerStrategy(),     w.get("bb",    0.12)),
                (MACDStrategy(),          w.get("macd",  0.12)),
                (StochasticStrategy(),    w.get("stoch", 0.12)),
                (MomentumShiftStrategy(), w.get("mom",   0.15)),
            ]
            self._tv_weight = w.get("tv", 0.25)
        else:
            self._strategies = [
                (MACrossStrategy(),       w.get("ma",    0.15)),
                (RSIStrategy(),           w.get("rsi",   0.15)),
                (BollingerStrategy(),     w.get("bb",    0.15)),
                (MACDStrategy(),          w.get("macd",  0.15)),
                (StochasticStrategy(),    w.get("stoch", 0.15)),
                (MomentumShiftStrategy(), w.get("mom",   0.20)),
            ]
            self._tv_weight = 0.0

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        for strat, _ in self._strategies:
            df = strat.compute_indicators(df)
        return df

    def set_live_mode(self, live: bool = True) -> None:
        self.use_tv = live

    def _tv_vote(self, symbol: str, interval: str = "1d") -> tuple[SignalType, float]:
        """Fetch TradingView signal for a given timeframe."""
        try:
            from data.tv_signals import get_tv_signal
            tv = get_tv_signal(symbol, interval=interval)
            score = tv["score"]
            if score >= 0.5:
                return SignalType.BUY, score
            elif score <= -0.5:
                return SignalType.SELL, score
            else:
                return SignalType.HOLD, score
        except Exception:
            return SignalType.HOLD, 0.0

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        votes: list[tuple[str, SignalType, float]] = []

        for strat, weight in self._strategies:
            sig = strat.generate_signal(df, symbol)
            votes.append((strat.name, sig.signal, weight))

        tv_raw_score  = 0.0
        tv_weekly_sig = SignalType.HOLD

        if self.use_tv and self._tv_weight > 0:
            # Daily TV vote — counts toward score
            tv_daily_sig, tv_raw_score = self._tv_vote(symbol, "1d")
            votes.append(("TV_daily", tv_daily_sig, self._tv_weight))

            # Weekly TV confirmation — does NOT add weight, only blocks false signals
            tv_weekly_sig, _ = self._tv_vote(symbol, "1w")

        total_w = sum(w for _, _, w in votes)
        score   = sum(_VOTE[s] * w for _, s, w in votes) / total_w

        if score >= self.buy_threshold:
            # Weekly confirmation filter: block BUY if weekly trend is bearish
            if self.use_tv and tv_weekly_sig == SignalType.SELL:
                final = SignalType.HOLD
                weekly_note = " [weekly bearish — BUY blocked]"
            else:
                final = SignalType.BUY
                weekly_note = ""
        elif score <= self.sell_threshold:
            final = SignalType.SELL
            weekly_note = ""
        else:
            final = SignalType.HOLD
            weekly_note = ""

        buy_count  = sum(1 for _, s, _ in votes if s == SignalType.BUY)
        sell_count = sum(1 for _, s, _ in votes if s == SignalType.SELL)
        reasons    = " | ".join(f"{n}:{s.value}" for n, s, _ in votes)

        row = df.iloc[-1]
        return Signal(
            symbol=symbol, signal=final, price=float(row["close"]),
            timestamp=datetime.now(), strategy=self.name,
            confidence=abs(score),
            reason=f"[{buy_count}买/{sell_count}卖] {reasons}{weekly_note}",
            metadata={
                "score":       round(score, 4),
                "votes":       {n: s.value for n, s, _ in votes},
                "tv_raw":      round(tv_raw_score, 4),
                "tv_weekly":   tv_weekly_sig.value,
            },
        )
