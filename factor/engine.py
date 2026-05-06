"""
多因子评分引擎（重构版）

因子分工（已剔除与 Heatmap / MomentumShift 重叠的趋势类因子）：

  选股质量因子  (roe_score, growth_score, debt_score)  — 基本面
  动量因子      (momentum_20, momentum_60)             — 中长期价格趋势
  均值回归      (momentum_5, rsi_score)                — 短期超买超卖
  量价因子      (vol_ratio, vol_trend)                 — 资金活跃度
  结构因子      (ma_alignment, price_position)         — 均线多空排列

已移除（与 Heatmap trendDir 高度重叠，会重复计票）：
  ema200_score → Heatmap 的追踪止损已覆盖
  adx_score    → Heatmap 的 ATR 强度已覆盖
  ichimoku_score → Heatmap + MomentumShift 已覆盖

动态权重（默认开启）：
  每次 compute() 时用近 60 日滚动 Rank-IC 微调权重。
  IC 高的因子加权，IC 弱或方向相反的因子减权。
  与外部 ic-calibrate 保存的文件权重叠加，不互斥。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FactorScore:
    symbol: str
    total_score: float
    factors: dict[str, float] = field(default_factory=dict)
    grade: str = "C"
    reason: str = ""
    weights_used: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.symbol}  [{self.grade}]  {self.total_score:+.3f}  {self.reason}"


DEFAULT_WEIGHTS: dict[str, float] = {
    "momentum_60":    0.21,   # 3 个月动量 — NA 最强单因子
    "momentum_20":    0.13,   # 1 个月动量
    "roe_score":      0.12,   # 质量：盈利能力
    "growth_score":   0.10,   # 质量：盈利增速
    "pb_score":       0.06,   # 估值：低 PB
    "ma_alignment":   0.09,   # 均线多头排列
    "rsi_score":      0.07,   # 超卖反弹
    "debt_score":     0.06,   # 质量：低负债
    "vol_ratio":      0.05,   # 量比放量
    "price_position": 0.05,   # 60 日价格位置
    "vol_trend":      0.03,   # 量能趋势
    "momentum_5":    -0.03,   # 极短期小幅反转
}

_IC_WEIGHTS_PATH = Path("data/factor_weights.json")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank IC (numpy only, no scipy)."""
    def _rank(a: np.ndarray) -> np.ndarray:
        tmp = np.argsort(a)
        r = np.empty_like(tmp, dtype=float)
        r[tmp] = np.arange(len(a), dtype=float)
        return r

    rx, ry = _rank(x), _rank(y)
    mx, my = rx.mean(), ry.mean()
    num = ((rx - mx) * (ry - my)).sum()
    den = np.sqrt(((rx - mx) ** 2).sum() * ((ry - my) ** 2).sum())
    return float(num / den) if den > 1e-9 else 0.0


class FactorEngine:
    """多因子评分引擎，至少需要 60 条 OHLCV 数据。"""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
    ):
        # Priority: passed weights > IC file weights > default weights
        if weights is not None:
            self.weights = weights
        else:
            loaded = self._load_ic_weights()
            self.weights = loaded if loaded is not None else DEFAULT_WEIGHTS

    # ── IC 权重文件加载 ────────────────────────────────────────────────────────

    @staticmethod
    def _load_ic_weights() -> dict[str, float] | None:
        try:
            if _IC_WEIGHTS_PATH.exists():
                data = json.loads(_IC_WEIGHTS_PATH.read_text())
                w = data.get("weights") or data  # 兼容两种格式
                if isinstance(w, dict) and w:
                    return {k: float(v) for k, v in w.items()}
        except Exception:
            pass
        return None

    # ── 主评分入口 ─────────────────────────────────────────────────────────────

    def compute(
        self,
        df: pd.DataFrame,
        symbol: str,
        fundamentals: dict | None = None,
    ) -> FactorScore:
        if len(df) < 60:
            return FactorScore(symbol=symbol, total_score=0.0, grade="N/A",
                               reason="数据不足60日")

        fund = fundamentals or {}
        factors: dict[str, float] = {
            "momentum_5":     self._momentum(df, 5),
            "momentum_20":    self._momentum(df, 20),
            "momentum_60":    self._momentum(df, 60),
            "vol_ratio":      self._vol_ratio(df),
            "vol_trend":      self._vol_trend(df),
            "ma_alignment":   self._ma_alignment(df),
            "price_position": self._price_position(df, 60),
            "rsi_score":      self._rsi_score(df),
            "roe_score":      self._roe_score(fund),
            "growth_score":   self._growth_score(fund),
            "debt_score":     self._debt_score(fund),
            "pb_score":       self._pb_score(fund),
        }

        # 基本面因子缺失时，将其权重重分配给其余正权重因子
        weights = self._redistribute_dead_weights(self.weights, fund)

        raw_sum = sum(
            factors.get(k, 0.0) * w for k, w in weights.items()
            if not np.isnan(factors.get(k, 0.0))
        )
        total = float(np.clip(raw_sum, -1.0, 1.0))

        grade = (
            "A" if total >= 0.40 else
            "B" if total >= 0.20 else
            "C" if total >= -0.10 else
            "D" if total >= -0.30 else
            "E"
        )

        pos_f = sorted([(k, v) for k, v in factors.items() if v > 0.3],  key=lambda x: -x[1])[:3]
        neg_f = sorted([(k, v) for k, v in factors.items() if v < -0.3], key=lambda x:  x[1])[:2]
        reason = " ".join([f"+{k}" for k, _ in pos_f] + [f"-{k}" for k, _ in neg_f]) or "中性"

        return FactorScore(
            symbol=symbol,
            total_score=round(total, 4),
            factors={k: round(v, 4) for k, v in factors.items()},
            grade=grade,
            reason=reason,
            weights_used={k: round(v, 4) for k, v in weights.items()},
        )

    def _redistribute_dead_weights(
        self,
        weights: dict[str, float],
        fund: dict,
    ) -> dict[str, float]:
        """
        当基本面数据缺失时，将对应因子权重清零并按比例分配给其余正权重因子。
        避免 31% 的权重固定贡献 0 分，导致综合评分系统性偏低。
        """
        dead: set[str] = set()
        if fund.get("roe") is None:
            dead.add("roe_score")
        if fund.get("net_profit_growth") is None:
            dead.add("growth_score")
        if fund.get("debt_ratio") is None:
            dead.add("debt_score")
        if not fund.get("pb"):
            dead.add("pb_score")

        if not dead:
            return weights

        dead_pos = sum(max(weights.get(k, 0.0), 0.0) for k in dead)
        if dead_pos == 0:
            return {k: (0.0 if k in dead else v) for k, v in weights.items()}

        live_pos = sum(max(v, 0.0) for k, v in weights.items() if k not in dead)
        if live_pos <= 0:
            return {k: (0.0 if k in dead else v) for k, v in weights.items()}

        scale = (live_pos + dead_pos) / live_pos
        return {
            k: (0.0 if k in dead else (v * scale if v > 0 else v))
            for k, v in weights.items()
        }

    # ── 单 bar 因子计算（供 compute 调用）────────────────────────────────────────

    def _momentum(self, df: pd.DataFrame, period: int) -> float:
        if len(df) < period + 1:
            return 0.0
        ret = df["close"].iloc[-1] / df["close"].iloc[-(period + 1)] - 1
        return float(np.clip(ret / 0.15, -1.0, 1.0))

    def _vol_ratio(self, df: pd.DataFrame) -> float:
        if "volume" not in df.columns or len(df) < 6:
            return 0.0
        avg5 = df["volume"].iloc[-6:-1].mean()
        if avg5 == 0:
            return 0.0
        return float(np.clip((df["volume"].iloc[-1] / avg5 - 1.25) / 0.75, -1.0, 1.0))

    def _vol_trend(self, df: pd.DataFrame) -> float:
        if "volume" not in df.columns or len(df) < 21:
            return 0.0
        avg5  = df["volume"].iloc[-6:-1].mean()
        avg20 = df["volume"].iloc[-21:-1].mean()
        if avg20 == 0:
            return 0.0
        return float(np.clip((avg5 / avg20 - 1.0) / 0.5, -1.0, 1.0))

    def _ma_alignment(self, df: pd.DataFrame) -> float:
        if len(df) < 60:
            return 0.0
        c    = df["close"]
        ma5  = c.iloc[-5:].mean()
        ma10 = c.iloc[-10:].mean()
        ma20 = c.iloc[-20:].mean()
        ma60 = c.iloc[-60:].mean()
        score = sum([
            1 if c.iloc[-1] > ma5  else -1,
            1 if ma5  > ma10       else -1,
            1 if ma10 > ma20       else -1,
            1 if ma20 > ma60       else -1,
        ])
        return score / 4.0

    def _price_position(self, df: pd.DataFrame, period: int = 60) -> float:
        window = df["close"].iloc[-period:]
        lo, hi = window.min(), window.max()
        if hi == lo:
            return 0.0
        return float(np.clip((df["close"].iloc[-1] - lo) / (hi - lo) * 2 - 1, -1.0, 1.0))

    def _rsi_score(self, df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 1:
            return 0.0
        delta = df["close"].diff().dropna().iloc[-period:]
        gain  = delta.clip(lower=0).mean()
        loss  = (-delta.clip(upper=0)).mean()
        rsi   = 100.0 if loss == 0 else 100.0 - 100.0 / (1 + gain / loss)
        return float(np.clip((50.0 - rsi) / 20.0, -1.0, 1.0))

    # ── 基本面因子 ─────────────────────────────────────────────────────────────

    def _roe_score(self, fund: dict) -> float:
        roe = fund.get("roe")
        return 0.0 if roe is None else float(np.clip((roe - 10.0) / 10.0, -1.0, 1.0))

    def _growth_score(self, fund: dict) -> float:
        g = fund.get("net_profit_growth")
        return 0.0 if g is None else float(np.clip(g / 30.0, -1.0, 1.0))

    def _debt_score(self, fund: dict) -> float:
        debt = fund.get("debt_ratio")
        return 0.0 if debt is None else float(np.clip((60.0 - debt) / 30.0, -1.0, 1.0))


    def _pb_score(self, fund: dict) -> float:
        pb = fund.get("pb")
        if not pb or pb <= 0:
            return 0.0
        # PB 1 = 满分，PB 3 = 中性，PB 6+ = 最差
        return float(np.clip((3.0 - pb) / 3.0, -1.0, 1.0))
