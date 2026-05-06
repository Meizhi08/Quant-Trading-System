"""
AI 综合信号策略 — 调用 Claude API 分析技术指标、资金面、情绪面，
返回 BUY / SELL / HOLD 信号及置信度。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd
import anthropic
from loguru import logger

from config import settings
from .base import BaseStrategy, Signal, SignalType
from .ma_cross import MACrossStrategy
from .rsi import RSIStrategy
from .bollinger import BollingerStrategy


_SYSTEM_PROMPT = """You are a professional quantitative analyst specializing in North American equity markets (NYSE, NASDAQ, TSX).
The user will provide a snapshot of technical indicators for a stock. Analyze them and give a trading recommendation.

Output format (strict JSON, no markdown code blocks):
{
  "signal": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0-1.0,
  "reason": "Concise analysis in English, under 100 words",
  "key_factors": ["factor1", "factor2", ...]
}
"""


class AISignalStrategy(BaseStrategy):
    name = "AI_Signal"

    def __init__(self, use_cache_signals: bool = True):
        super().__init__()
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._signal_cache: dict[str, Signal] = {}
        self._use_cache = use_cache_signals
        self._sub_strategies = [
            MACrossStrategy(),
            RSIStrategy(),
            BollingerStrategy(),
        ]

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        for s in self._sub_strategies:
            df = s.compute_indicators(df)
        # 成交量比率（5日均量）
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(5).mean()
        return df

    def _build_prompt(self, df: pd.DataFrame, symbol: str) -> str:
        row = df.iloc[-1]
        prev5_chg = float(df["close"].pct_change(5).iloc[-1] * 100)

        indicators: dict[str, Any] = {
            "symbol": symbol,
            "date": str(row.name.date() if hasattr(row.name, "date") else row.name),
            "close": float(row["close"]),
            "5d_change_pct": round(prev5_chg, 2),
            "MA5":   round(float(row.get("ma5", 0)), 2),
            "MA20":  round(float(row.get("ma20", 0)), 2),
            "RSI14": round(float(row.get("rsi", 50)), 2),
            "BB_upper": round(float(row.get("bb_upper", 0)), 2),
            "BB_mid":   round(float(row.get("bb_mid", 0)), 2),
            "BB_lower": round(float(row.get("bb_lower", 0)), 2),
            "BB_pct_b": round(float(row.get("bb_pct", 0.5)), 3),
            "volume_ratio_5d": round(float(row.get("volume_ratio", 1)), 2),
        }

        return f"Analyze the following NA equity indicator data and provide a trading recommendation:\n{json.dumps(indicators, indent=2)}"

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> Signal:
        cache_key = f"{symbol}_{str(df.index[-1])}"
        if self._use_cache and cache_key in self._signal_cache:
            return self._signal_cache[cache_key]

        prompt = self._build_prompt(df, symbol)
        row = df.iloc[-1]

        try:
            resp = self._client.messages.create(
                model=settings.claude_model,
                max_tokens=settings.claude_max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            data = json.loads(raw)
            sig = SignalType(data["signal"])
            confidence = float(data.get("confidence", 0.5))
            reason = data.get("reason", "")
            metadata = {"key_factors": data.get("key_factors", [])}
        except Exception as e:
            logger.error(f"AI signal error for {symbol}: {e}")
            sig, confidence, reason = SignalType.HOLD, 0.0, f"AI分析失败: {e}"
            metadata = {}

        signal = Signal(
            symbol=symbol, signal=sig, price=float(row["close"]),
            timestamp=datetime.now(), strategy=self.name,
            confidence=confidence, reason=reason, metadata=metadata,
        )
        if self._use_cache:
            self._signal_cache[cache_key] = signal
        return signal
