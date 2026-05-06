"""
参数优化器 — 网格搜索最优策略参数。

示例:
    opt = ParameterOptimizer(
        strategy_class=MACrossStrategy,
        param_grid={"fast": [5, 10], "slow": [20, 30, 60]},
        symbol="AAPL",
        start="2020-01-01",
        end="2023-12-31",
        metric="sharpe_ratio",
    )
    best = opt.run()
"""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Type

import pandas as pd
from loguru import logger
from rich.progress import track

from strategy import BaseStrategy
from .engine import BacktestEngine, BacktestResult


_MIN_TRADES = 5  # 交易次数低于此值视为统计不可靠，Sharpe 强制归零


def _run_one(args: tuple) -> dict:
    strategy_class, params, symbol, start, end = args
    strategy = strategy_class(**params)
    engine = BacktestEngine(strategy=strategy, symbol=symbol)
    result = engine.run(start, end)
    m = result.metrics
    # 样本太少时 Sharpe / Calmar 不可信，强制归零避免被网格搜索选中
    reliable = m.total_trades >= _MIN_TRADES
    return {
        "params": params,
        "total_return": m.total_return,
        "annual_return": m.annual_return,
        "sharpe_ratio": m.sharpe_ratio if reliable else 0.0,
        "max_drawdown": m.max_drawdown,
        "win_rate": m.win_rate,
        "calmar_ratio": m.calmar_ratio if reliable else 0.0,
        "total_trades": m.total_trades,
    }


class ParameterOptimizer:
    def __init__(
        self,
        strategy_class: Type[BaseStrategy],
        param_grid: dict[str, list],
        symbol: str,
        start: str,
        end: str,
        metric: str = "sharpe_ratio",
        max_workers: int = 4,
    ):
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.symbol = symbol
        self.start = start
        self.end = end
        self.metric = metric
        self.max_workers = max_workers

    def _grid(self) -> list[dict[str, Any]]:
        keys = list(self.param_grid.keys())
        combos = list(itertools.product(*self.param_grid.values()))
        return [dict(zip(keys, c)) for c in combos]

    def run(self) -> pd.DataFrame:
        grid = self._grid()
        logger.info(f"Optimizer: {len(grid)} combinations for {self.symbol}")

        args = [
            (self.strategy_class, params, self.symbol, self.start, self.end)
            for params in grid
        ]

        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_run_one, a): a for a in args}
            for future in track(as_completed(futures), total=len(args),
                                description="优化中..."):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.warning(f"Optimization run failed: {e}")

        df = pd.DataFrame(results)
        df.sort_values(self.metric, ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
