"""
统一策略 — 四路信号合成 + 大盘/个股趋势过滤。

信号来源及默认权重（回测 use_tv=False 时 TV 权重归零，其余三路自动归一）：
  factor     0.50  — FactorEngine（质量/动量/量价，已剔除与 Heatmap 重叠的趋势因子）
                     · 启动时自动加载 data/factor_weights.json（若存在）
                     · 每次评分用近 60 日滚动 IC 微调权重（动态权重）
  heatmap    0.25  — ATR 追踪止损 + Fibonacci 热力图（Zeiierman 移植）
  ms         0.10  — MomentumShift HMA 动量转折（BigBeluga 移植）
  tv         0.15  — TradingView 聚合信号（仅实盘启用）

过滤器（只阻止买入，不影响卖出）：
  个股低于 250 日均线 15% 以上
  S&P 500 低于 250 日均线（大盘熊市）
  周线 TV 看跌
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal, SignalType
from .momentum_shift import MomentumShiftStrategy
from . import heatmap_trail
from factor import FactorEngine

_VOTE = {SignalType.BUY: 1.0, SignalType.HOLD: 0.0, SignalType.SELL: -1.0}


class UnifiedStrategy(BaseStrategy):
    name = "Unified"

    def __init__(
        self,
        buy_threshold: float = 0.20,
        sell_threshold: float = -0.20,
        factor_weight: float = 0.50,
        heatmap_weight: float = 0.25,
        ms_weight: float = 0.10,
        tv_weight: float = 0.15,
        use_tv: bool = True,
        weights: dict[str, float] | None = None,
        use_risk_parity: bool = False,  # 新增：风险平价权重调整
    ):
        super().__init__({
            "buy_threshold":  buy_threshold,
            "sell_threshold": sell_threshold,
            "factor_weight":  factor_weight,
            "heatmap_weight": heatmap_weight,
            "ms_weight":      ms_weight,
            "tv_weight":      tv_weight,
        })
        self.buy_threshold   = buy_threshold
        self.sell_threshold  = sell_threshold
        self.factor_weight   = factor_weight
        self.heatmap_weight  = heatmap_weight
        self.ms_weight       = ms_weight
        self.tv_weight       = tv_weight
        self.use_tv          = use_tv
        self.use_risk_parity = use_risk_parity
        self._engine         = FactorEngine(weights=weights)
        self._ms             = MomentumShiftStrategy()
        self._market_close: pd.Series | None = None
        self._score_history: list[dict] = []  # 存储历史分数用于风险平价

    def set_market_data(self, market_close: pd.Series) -> None:
        self._market_close = market_close

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有需要的指标"""
        df = self._ms.compute_indicators(df)
        
        # 预计算 factor scores（如果需要）
        if len(df) >= 60:
            factor_data = self._engine.compute(df, "")
            df["factor_score"] = factor_data.total_score
        
        return df

    # ── 过滤器 ────────────────────────────────────────────────────────────────

    # 宽基指数ETF：用S&P500过滤自身是循环引用，跳过市场过滤
    _BROAD_ETF = {"SPY", "QQQ", "IWM", "VTI", "VOO", "EFA", "EEM",
                  "^GSPC", "^IXIC", "^DJI", "^GSPTSE"}

    def _stock_in_downtrend(self, df: pd.DataFrame, symbol: str = "") -> bool:
        if symbol in self._BROAD_ETF:
            return False
        if len(df) < 250:
            return False
        ma250 = float(df["close"].rolling(250).mean().iloc[-1])
        return float(df["close"].iloc[-1]) < ma250 * 0.85

    def _market_in_downtrend(self, as_of_date, symbol: str = "") -> bool:
        if symbol in self._BROAD_ETF:
            return False
        if self._market_close is None:
            return False
        series = self._market_close[self._market_close.index <= as_of_date]
        if len(series) < 250:
            return False
        return float(series.iloc[-1]) < float(series.iloc[-250:].mean())

    def _tv_score(self, symbol: str, interval: str = "1d") -> tuple[float, str]:
        try:
            from data.tv_signals import get_tv_signal
            tv = get_tv_signal(symbol, interval=interval)
            return float(tv["score"]), tv.get("recommendation", "NEUTRAL")
        except Exception:
            return 0.0, "NEUTRAL"

    # ── 风险平价权重调整 ──────────────────────────────────────────────────────

    def _get_risk_parity_weights(self, current_scores: dict) -> dict:
        """
        基于历史波动率调整权重（风险平价）
        让波动率高的信号源获得更低权重
        """
        if not self.use_risk_parity or len(self._score_history) < 20:
            return {
                "factor": self.factor_weight,
                "heatmap": self.heatmap_weight,
                "ms": self.ms_weight,
                "tv": self.tv_weight if self.use_tv else 0,
            }
        
        # 收集历史分数
        hist_df = pd.DataFrame(self._score_history[-60:])
        if len(hist_df) < 10:
            return {
                "factor": self.factor_weight,
                "heatmap": self.heatmap_weight,
                "ms": self.ms_weight,
                "tv": self.tv_weight if self.use_tv else 0,
            }
        
        # 计算波动率
        vol = hist_df[["factor", "heatmap", "ms"]].std()
        if self.use_tv:
            vol["tv"] = hist_df["tv"].std() if "tv" in hist_df else 1.0
        
        # 避免除零
        vol = vol.clip(lower=0.01)
        
        # 风险平价权重：与波动率成反比
        inv_vol = 1.0 / vol
        total = inv_vol.sum()
        weights = (inv_vol / total).to_dict()
        
        # 应用原始权重比例
        raw_weights = {
            "factor": self.factor_weight,
            "heatmap": self.heatmap_weight,
            "ms": self.ms_weight,
            "tv": self.tv_weight if self.use_tv else 0,
        }
        
        # 混合：70% 风险平价 + 30% 原始权重（避免过度波动）
        final = {}
        for k in weights:
            if raw_weights.get(k, 0) > 0:
                final[k] = weights[k] * 0.7 + (raw_weights[k] / sum(raw_weights.values())) * 0.3
        
        return final

    # ── 信号生成 ──────────────────────────────────────────────────────────────

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        """生成综合交易信号"""
        
        # 确保有动量指标
        if "mom_shift" not in df.columns:
            df = self._ms.compute_indicators(df.copy())

        # 1. Factor score  (-1 ~ +1)
        try:
            fs = self._engine.compute(df, symbol)
            factor_score = fs.total_score
            factor_reason = fs.reason
            factor_grade = fs.grade
        except Exception as e:
            factor_score = 0.0
            factor_reason = "N/A"
            factor_grade = "N/A"

        # 2. Heatmap score (-1 ~ +1)
        try:
            hm = heatmap_trail.compute(df)
            hm_score = hm["trend_dir"] * (hm["score"] / 10.0)
            hm_data = {
                "trend_dir": hm["trend_dir"],
                "strength": hm["score"],
                "sig_long": hm["sig_long"],
                "sig_short": hm["sig_short"],
                "trail": hm["trail"],
            }
        except Exception:
            hm = {"trend_dir": 1, "score": 5.0,
                  "sig_long": False, "sig_short": False,
                  "trail": float(df["close"].iloc[-1])}
            hm_score = 0.0
            hm_data = {"trend_dir": 1, "strength": 5.0, 
                       "sig_long": False, "sig_short": False,
                       "trail": float(df["close"].iloc[-1])}

        # 3. MomentumShift score — 获取连续分数 [-1, +1]
        try:
            ms_sig = self._ms.generate_signal(df, symbol)
            # 修复：正确获取连续分数
            continuous_score = ms_sig.metadata.get("continuous_score")
            if continuous_score is not None and isinstance(continuous_score, (int, float)):
                ms_score = float(continuous_score)
            else:
                # fallback: 从信号类型推断
                ms_score = _VOTE.get(ms_sig.signal, 0.0)
        except Exception:
            ms_score = 0.0

        # 4. TradingView score (实盘才有)
        tv_score, tv_daily_rec, tv_weekly_rec = 0.0, "NEUTRAL", "NEUTRAL"
        if self.use_tv:
            tv_score, tv_daily_rec = self._tv_score(symbol, "1d")
            _, tv_weekly_rec = self._tv_score(symbol, "1w")
            tv_score = max(-1.0, min(1.0, tv_score))  # 确保在 [-1,1] 范围

        # 5. 风险平价权重调整（可选）
        current_scores = {
            "factor": factor_score,
            "heatmap": hm_score,
            "ms": ms_score,
        }
        if self.use_tv:
            current_scores["tv"] = tv_score
        
        # 存储历史（用于风险平价计算）
        self._score_history.append(current_scores)
        if len(self._score_history) > 200:
            self._score_history.pop(0)
        
        # 计算动态权重
        risk_weights = self._get_risk_parity_weights(current_scores)
        
        # 6. 加权合成
        total_score = (
            factor_score * risk_weights.get("factor", self.factor_weight) +
            hm_score * risk_weights.get("heatmap", self.heatmap_weight) +
            ms_score * risk_weights.get("ms", self.ms_weight)
        )
        
        effective_tv_w = risk_weights.get("tv", self.tv_weight if self.use_tv else 0)
        if self.use_tv and effective_tv_w > 0:
            total_score += tv_score * effective_tv_w
            total_weight = sum(risk_weights.values())
        else:
            total_weight = sum(v for k, v in risk_weights.items() if k != "tv")
        
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0

        # 7. 过滤器（只阻止买入）
        last_date = df.index[-1]
        stock_down = self._stock_in_downtrend(df, symbol)
        market_down = self._market_in_downtrend(last_date, symbol)
        weekly_bear = self.use_tv and tv_weekly_rec == "SELL"
        buy_blocked = stock_down or market_down or weekly_bear

        # 8. 决策
        if final_score >= self.buy_threshold and not buy_blocked:
            sig = SignalType.BUY
        elif final_score <= self.sell_threshold:
            sig = SignalType.SELL
        else:
            sig = SignalType.HOLD

        # 热力图突破信号提升 confidence
        hm_confirm = (hm_data["sig_long"] and sig == SignalType.BUY) or \
                     (hm_data["sig_short"] and sig == SignalType.SELL)
        confidence = min(1.0, abs(final_score) * (1.15 if hm_confirm else 1.0))

        # 9. 原因说明（用于日志）
        block_reasons = []
        if final_score >= self.buy_threshold and buy_blocked:
            if market_down:
                block_reasons.append("大盘过滤")
            if stock_down:
                block_reasons.append("个股过滤")
            if weekly_bear:
                block_reasons.append("周线看跌")
        
        block_note = f" [{'/'.join(block_reasons)}]" if block_reasons else ""
        hm_tag = "⚡" if hm_confirm else ""
        
        # 动态权重显示
        weight_str = f"w=({risk_weights.get('factor',0):.2f},{risk_weights.get('heatmap',0):.2f},{risk_weights.get('ms',0):.2f}"
        if self.use_tv:
            weight_str += f",{risk_weights.get('tv',0):.2f}"
        weight_str += ")"
        
        reason = (
            f"score={final_score:+.3f} {weight_str} "
            f"f={factor_score:+.3f}({factor_grade}) "
            f"hm={hm_score:+.3f}(dir={hm_data['trend_dir']},s={hm_data['strength']:.0f}){hm_tag} "
            f"ms={ms_score:+.2f}"
        )
        
        if self.use_tv:
            reason += f" tv={tv_score:+.3f}({tv_daily_rec})"
        
        reason += f" | {factor_reason}{block_note}"

        return Signal(
            symbol=symbol,
            signal=sig,
            price=float(df["close"].iloc[-1]),
            timestamp=datetime.now(),
            strategy=self.name,
            confidence=round(confidence, 4),
            reason=reason[:200],  # 限制长度
            metadata={
                "score": round(final_score, 4),
                "factor_score": factor_score,
                "grade": factor_grade,
                "factors": fs.factors if 'fs' in dir() else {},
                "hm_score": round(hm_score, 4),
                "hm_trend": hm_data["trend_dir"],
                "hm_strength": hm_data["strength"],
                "hm_sig_long": hm_data["sig_long"],
                "hm_sig_short": hm_data["sig_short"],
                "hm_trail": hm_data["trail"],
                "ms_score": ms_score,
                "tv_daily": tv_score,
                "tv_weekly": tv_weekly_rec,
                "weights_used": risk_weights,
                "buy_blocked": buy_blocked,
                "stock_downtrend": stock_down,
                "market_downtrend": market_down,
                "weekly_bear": weekly_bear,
            },
        )

    def reset(self) -> None:
        """重置策略状态（用于新的回测）"""
        self._score_history = []
        if hasattr(self._engine, 'reset'):
            self._engine.reset()