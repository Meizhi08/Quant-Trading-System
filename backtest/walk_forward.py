"""
Walk-forward validation — the only honest way to test strategy optimization.

For each rolling window:
  1. TRAIN: optimize params on in-sample period
  2. TEST:  run backtest with those params on the next out-of-sample period
  3. Report only the out-of-sample metrics

This prevents look-ahead bias: params chosen using only data available at that point in time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from loguru import logger

from .auto_optimizer import AutoOptimizer, build_optimized_composite
from .engine import BacktestEngine
from .metrics import BacktestMetrics


@dataclass
class WindowResult:
    window: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: dict
    metrics: BacktestMetrics          # out-of-sample only


@dataclass
class WalkForwardResult:
    symbol: str
    windows: list[WindowResult] = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    @property
    def avg_sharpe(self) -> float:
        if not self.windows:
            return 0.0
        return sum(w.metrics.sharpe_ratio for w in self.windows) / len(self.windows)

    @property
    def avg_return(self) -> float:
        if not self.windows:
            return 0.0
        return sum(w.metrics.total_return for w in self.windows) / len(self.windows)

    @property
    def pct_profitable(self) -> float:
        if not self.windows:
            return 0.0
        profitable = sum(1 for w in self.windows if w.metrics.total_return > 0)
        return profitable / len(self.windows)

    @property
    def avg_max_drawdown(self) -> float:
        if not self.windows:
            return 0.0
        return sum(w.metrics.max_drawdown for w in self.windows) / len(self.windows)

    @property
    def avg_win_rate(self) -> float:
        if not self.windows:
            return 0.0
        return sum(w.metrics.win_rate for w in self.windows) / len(self.windows)


class WalkForwardValidator:
    """
    Rolls a train/test window across the full backtest period.

    Two modes:
    - strategy=None (composite mode): optimize composite sub-params on each train window,
      test the optimized composite out-of-sample. Best for composite strategy.
    - strategy=<instance> (validation mode): use the given strategy as-is on each OOS
      window without re-optimizing. Honest OOS stability check for any strategy.

    Example with train_months=12, test_months=3, full period 2022~2024 (36 months):
      Window 1: train 2022-01~2022-12  →  test 2023-01~2023-03
      Window 2: train 2022-04~2023-03  →  test 2023-04~2023-06
      Window 3: train 2022-07~2023-06  →  test 2023-07~2023-09
      ...
    """

    def __init__(
        self,
        symbol: str,
        start: str,
        end: str,
        train_months: int = 12,
        test_months: int = 3,
        stop_loss_pct: float = 0.08,
        trailing_stop_pct: float = 0.12,
        metric: str = "sharpe_ratio",
        strategy=None,   # None = optimize composite each window; instance = validate as-is
    ):
        self.symbol = symbol
        self.full_start = date.fromisoformat(start)
        self.full_end = date.fromisoformat(end)
        self.train_months = train_months
        self.test_months = test_months
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.metric = metric
        self.strategy = strategy   # fixed strategy for validation mode

    @property
    def _optimize_mode(self) -> bool:
        return self.strategy is None

    def run(self) -> WalkForwardResult:
        result = WalkForwardResult(symbol=self.symbol)
        windows = self._build_windows()

        if not windows:
            logger.warning("数据期太短，无法做walk-forward（至少需要 train+test 个月）")
            return result

        mode_desc = "参数优化+样本外验证" if self._optimize_mode else "固定参数样本外验证"
        logger.info(f"Walk-forward ({mode_desc}): {self.symbol}  {len(windows)} 个窗口  "
                    f"训练{self.train_months}月/测试{self.test_months}月")

        for i, (train_s, train_e, test_s, test_e) in enumerate(windows, 1):
            logger.info(f"  窗口 {i}/{len(windows)}: "
                        f"训练 {train_s}~{train_e}  测试 {test_s}~{test_e}")

            if self._optimize_mode:
                # ── Composite mode: optimize sub-params on train, test OOS ──
                try:
                    optimizer = AutoOptimizer(
                        symbols=[self.symbol],
                        metric=self.metric,
                        start=train_s,
                        end=train_e,
                    )
                    optimizer.run()
                    strat = build_optimized_composite()
                    best_params = self._extract_params()
                except Exception as e:
                    logger.warning(f"  窗口 {i} 优化失败: {e}，使用默认策略")
                    from strategy.composite import CompositeStrategy
                    strat = CompositeStrategy()
                    best_params = {}
            else:
                # ── Validation mode: use fixed strategy as-is ──────────────
                strat = self.strategy
                best_params = getattr(strat, "params", {})

            # Test out-of-sample
            try:
                engine = BacktestEngine(
                    strategy=strat,
                    symbol=self.symbol,
                    stop_loss_pct=self.stop_loss_pct,
                    trailing_stop_pct=self.trailing_stop_pct,
                    take_profit_pct=0.0,
                )
                bt_result = engine.run(test_s, test_e)
                metrics = bt_result.metrics
            except Exception as e:
                logger.warning(f"  窗口 {i} 测试失败: {e}")
                continue

            result.windows.append(WindowResult(
                window=i,
                train_start=train_s, train_end=train_e,
                test_start=test_s,   test_end=test_e,
                best_params=best_params,
                metrics=metrics,
            ))
            logger.info(f"  窗口 {i} OOS: 收益={metrics.total_return:.1%}  "
                        f"Sharpe={metrics.sharpe_ratio:.2f}  胜率={metrics.win_rate:.0%}")

        return result

    def _build_windows(self) -> list[tuple[str, str, str, str]]:
        windows = []
        train_delta = timedelta(days=self.train_months * 30)
        test_delta  = timedelta(days=self.test_months  * 30)
        step_delta  = test_delta  # roll by one test period each time

        test_start = self.full_start + train_delta
        while test_start + test_delta <= self.full_end + timedelta(days=1):
            train_start = test_start - train_delta
            train_end   = test_start - timedelta(days=1)
            test_end    = min(test_start + test_delta - timedelta(days=1), self.full_end)
            windows.append((
                str(train_start), str(train_end),
                str(test_start),  str(test_end),
            ))
            test_start += step_delta

        return windows

    def _extract_params(self) -> dict:
        """Pull best params from the saved JSON for display."""
        import json
        from pathlib import Path
        p = Path(__file__).parent.parent / "data" / "best_params.json"
        if not p.exists():
            return {}
        data = json.loads(p.read_text())
        return {
            name: info.get("params", {})
            for name, info in data.get("strategies", {}).items()
        }
