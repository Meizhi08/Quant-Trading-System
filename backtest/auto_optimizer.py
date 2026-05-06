"""
自动策略优化器 — 对所有策略跑网格搜索，保存最优参数。

每次运行会：
1. 将数据分为训练期（前 N 个月）和样本外验证期（最后 3 个月）
2. 在训练期内网格搜索每个策略的最优参数
3. 用样本外期验证，只保存 OOS Sharpe 的权重
4. 结果保存到 data/best_params.json，composite 策略自动加载
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from strategy import (
    MACrossStrategy, RSIStrategy, BollingerStrategy,
    MACDStrategy, StochasticStrategy, MomentumShiftStrategy,
)
from .optimizer import ParameterOptimizer

_BEST_PARAMS_FILE = Path(__file__).parent.parent / "data" / "best_params.json"

_HOLDOUT_MONTHS = 3    # 最后 N 个月用于样本外验证
_MIN_TRAIN_MONTHS = 9  # 训练期至少 N 个月（总计至少 12 个月）

# 每个策略的参数搜索空间
_PARAM_GRIDS = {
    "ma": {
        "class": MACrossStrategy,
        "grid": {"fast": [5, 10, 15], "slow": [20, 30, 60]},
    },
    "rsi": {
        "class": RSIStrategy,
        "grid": {"period": [10, 14, 21], "oversold": [25, 30], "overbought": [70, 75]},
    },
    "bb": {
        "class": BollingerStrategy,
        "grid": {"period": [15, 20, 25], "std_dev": [1.8, 2.0, 2.2]},
    },
    "macd": {
        "class": MACDStrategy,
        "grid": {"fast": [10, 12], "slow": [24, 26], "signal": [7, 9]},
    },
    "stoch": {
        "class": StochasticStrategy,
        "grid": {"k_period": [9, 14, 21], "oversold": [15, 20], "overbought": [80, 85]},
    },
    "mom": {
        "class": MomentumShiftStrategy,
        "grid": {"length": [30, 50, 70], "smooth": [30, 50, 70]},
    },
}


class AutoOptimizer:
    def __init__(
        self,
        symbols: list[str],
        lookback_months: int = 12,
        metric: str = "sharpe_ratio",
        start: str | None = None,
        end: str | None = None,
    ):
        self.symbols = symbols
        # 至少需要 训练期 + holdout 期
        min_months = _MIN_TRAIN_MONTHS + _HOLDOUT_MONTHS
        self.lookback_months = max(lookback_months, min_months)
        self.metric = metric
        _end = date.fromisoformat(end) if end else date.today()
        _start = date.fromisoformat(start) if start else _end - timedelta(days=self.lookback_months * 30)
        self.start = str(_start)
        self.end = str(_end)

    # ── 样本外验证 ─────────────────────────────────────────────────────────────

    def _validate_holdout(
        self,
        strategy_class,
        params: dict,
        holdout_start: str,
        holdout_end: str,
    ) -> float:
        """在 holdout 期内跑回测，返回平均 Sharpe（最多取前2只股票）。"""
        from .engine import BacktestEngine

        try:
            strat = strategy_class(**params)
        except Exception:
            strat = strategy_class()

        sharpes: list[float] = []
        for sym in self.symbols[:2]:
            try:
                result = BacktestEngine(strategy=strat, symbol=sym).run(holdout_start, holdout_end)
                sharpes.append(result.metrics.sharpe_ratio)
            except Exception as e:
                logger.debug(f"  holdout {sym}: {e}")

        return sum(sharpes) / len(sharpes) if sharpes else 0.0

    def run(self) -> dict[str, Any]:
        """对所有策略在训练期内网格搜索，再在 holdout 期验证，返回最优参数字典。"""
        # ── 训练 / holdout 分割 ────────────────────────────────────────────────
        end_date   = date.fromisoformat(self.end)
        start_date = date.fromisoformat(self.start)
        total_months = max((end_date - start_date).days // 30, 0)

        holdout_days = _HOLDOUT_MONTHS * 30
        use_holdout  = total_months >= _MIN_TRAIN_MONTHS + _HOLDOUT_MONTHS

        if use_holdout:
            train_end     = str(end_date - timedelta(days=holdout_days))
            holdout_start = str(end_date - timedelta(days=holdout_days - 1))
            logger.info(f"训练期: {self.start}~{train_end}  "
                        f"样本外验证: {holdout_start}~{self.end}")
        else:
            train_end     = self.end
            holdout_start = None
            logger.warning(
                f"数据仅 {total_months} 个月，跳过样本外验证"
                f"（建议至少 {_MIN_TRAIN_MONTHS + _HOLDOUT_MONTHS} 个月）"
            )

        best: dict[str, Any] = {}
        scores: dict[str, float] = {}

        for name, cfg in _PARAM_GRIDS.items():
            logger.info(f"优化策略: {name}  数据: {self.start}~{train_end}")
            strategy_scores = []

            for sym in self.symbols:
                try:
                    opt = ParameterOptimizer(
                        strategy_class=cfg["class"],
                        param_grid=cfg["grid"],
                        symbol=sym,
                        start=self.start,
                        end=train_end,          # ← 只用训练期，不碰 holdout
                        metric=self.metric,
                        max_workers=2,
                    )
                    df = opt.run()
                    if df.empty:
                        continue
                    top = df.iloc[0]
                    strategy_scores.append({
                        "symbol": sym,
                        "params": top["params"],
                        "score": float(top[self.metric]),
                        "total_return": float(top["total_return"]),
                        "win_rate": float(top["win_rate"]),
                    })
                except Exception as e:
                    logger.warning(f"  {name}/{sym} 优化失败: {e}")

            if not strategy_scores:
                continue

            best_for_strategy = max(strategy_scores, key=lambda x: x["score"])
            avg_score = sum(s["score"] for s in strategy_scores) / len(strategy_scores)

            # ── 样本外验证 ──────────────────────────────────────────────────────
            oos_sharpe: float | None = None
            if use_holdout:
                oos_sharpe = self._validate_holdout(
                    cfg["class"], best_for_strategy["params"],
                    holdout_start, self.end,
                )
                flag = "✓" if oos_sharpe > 0 else "✗"
                logger.info(
                    f"  {name} 最优: {best_for_strategy['params']}"
                    f"  IS={avg_score:.4f}  OOS={oos_sharpe:.4f} {flag}"
                )
                if oos_sharpe < 0:
                    logger.warning(
                        f"  {name} 样本外 Sharpe 为负 ({oos_sharpe:.4f})，"
                        f"参数仍保存但 composite 权重将为 0（过拟合风险高）"
                    )
                # 权重依据样本外 Sharpe 排序，避免用样本内过拟合结果
                scores[name] = oos_sharpe
            else:
                logger.info(f"  {name} 最优: {best_for_strategy['params']}  IS={avg_score:.4f}")
                scores[name] = avg_score

            best[name] = {
                "params":     best_for_strategy["params"],
                "avg_score":  round(avg_score, 4),
                "oos_sharpe": round(oos_sharpe, 4) if oos_sharpe is not None else None,
                "details":    strategy_scores,
            }

        # 根据得分计算composite权重（得分高的策略权重大）
        weights = _calc_weights(scores)
        result = {
            "updated_at": str(date.today()),
            "lookback_months": self.lookback_months,
            "metric": self.metric,
            "strategies": best,
            "composite_weights": weights,
        }

        _BEST_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_BEST_PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"最优参数已保存: {_BEST_PARAMS_FILE}")
        return result


def _calc_weights(scores: dict[str, float]) -> dict[str, float]:
    """得分转权重：正得分按比例分配；若全为负则退回等权。"""
    if not scores:
        return {}
    n = len(scores)
    # 若所有策略 OOS 得分均为负，等权是更保守的选择（避免挑出"最不坏"造成过拟合）
    if all(v <= 0 for v in scores.values()):
        return {k: round(1.0 / n, 4) for k in scores}
    # 负分截断为 0，只给正分策略分配权重（shrinkage toward zero）
    clipped = {k: max(v, 0.0) for k, v in scores.items()}
    total = sum(clipped.values())
    return {k: round(v / total, 4) for k, v in clipped.items()}


def load_best_params() -> dict[str, Any] | None:
    """加载上次优化结果，没有则返回None。"""
    if not _BEST_PARAMS_FILE.exists():
        return None
    with open(_BEST_PARAMS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_optimized_composite(use_tv: bool = False):
    """用最优参数构建composite策略，没有优化记录则用默认参数。"""
    from strategy.composite import CompositeStrategy
    from strategy import MACrossStrategy, RSIStrategy, BollingerStrategy, MACDStrategy, StochasticStrategy

    data = load_best_params()
    if not data:
        return CompositeStrategy(use_tv=use_tv)

    def _make(name, cls):
        entry = data["strategies"].get(name, {})
        params = entry.get("params", {})
        try:
            return cls(**params)
        except Exception:
            return cls()

    weights = data.get("composite_weights", {})
    strategies_data = data.get("strategies", {})
    all_strats = [
        ("ma",   MACrossStrategy,  weights.get("ma",   0.20)),
        ("rsi",  RSIStrategy,       weights.get("rsi",  0.20)),
        ("bb",   BollingerStrategy, weights.get("bb",   0.20)),
        ("macd", MACDStrategy,      weights.get("macd", 0.20)),
        ("stoch", StochasticStrategy, weights.get("stoch", 0.20)),
    ]
    # 排除 OOS Sharpe 为负的策略（有 OOS 数据时用 OOS，否则退回 IS avg_score）
    def _should_include(info: dict) -> bool:
        oos = info.get("oos_sharpe")
        if oos is not None:
            return oos >= 0
        return info.get("avg_score", 0) >= 0

    strats = [
        (_make(name, cls), w)
        for name, cls, w in all_strats
        if _should_include(strategies_data.get(name, {}))
    ]
    if not strats:
        strats = [(_make(name, cls), w) for name, cls, w in all_strats]

    composite = CompositeStrategy(use_tv=use_tv)
    composite._strategies = strats
    return composite
