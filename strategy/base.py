"""Strategy base class and Signal data model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    symbol: str
    signal: SignalType
    price: float
    timestamp: datetime
    strategy: str
    confidence: float = 1.0          # 0-1，AI策略给出概率
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"[{self.strategy}] {self.symbol} {self.signal.value} "
            f"@ {self.price:.2f}  conf={self.confidence:.0%}  {self.reason}"
        )


class BaseStrategy(ABC):
    """所有策略的抽象基类。

    子类只需实现 `compute_indicators` 和 `generate_signal`。
    """

    name: str = "base"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {}

    @abstractmethod
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """在 df 上计算并追加技术指标列，返回同一个 df。"""

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        """根据最新数据行生成交易信号。"""

    @staticmethod
    def _vol_confirmed(df: pd.DataFrame, period: int = 20, multiplier: float = 1.1) -> bool:
        """当前成交量是否高于N日均量的multiplier倍，用于过滤假突破。"""
        if "volume" not in df.columns or len(df) < period + 1:
            return True  # 数据不足时不过滤
        avg_vol = df["volume"].iloc[-(period + 1):-1].mean()
        cur_vol = df["volume"].iloc[-1]
        return bool(cur_vol >= avg_vol * multiplier)

    @staticmethod
    def _uptrend_ma200(df: pd.DataFrame) -> bool:
        """True if the latest close is above its 200-day MA (blocks BUY in bear markets).
        Returns True when data is insufficient so warm-up period is unaffected."""
        if len(df) < 200:
            return True
        ma200 = df["close"].rolling(200).mean().iloc[-1]
        if pd.isna(ma200):
            return True
        return bool(df["close"].iloc[-1] >= ma200)

    def run(self, df: pd.DataFrame, symbol: str) -> Signal:
        df = self.compute_indicators(df.copy())
        return self.generate_signal(df, symbol)

    def backtest_signals(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        """返回整段历史上每个交易日的 SignalType 序列（用于回测分析）。"""
        df = self.compute_indicators(df.copy())
        results = []
        for i in range(len(df)):
            sig = self.generate_signal(df.iloc[: i + 1], symbol)
            results.append(sig.signal)
        return pd.Series(results, index=df.index, name="signal")
