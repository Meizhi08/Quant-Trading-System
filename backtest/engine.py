"""
BacktestEngine — 封装 backtrader，提供简洁的 run() 接口。

用法:
    engine = BacktestEngine(strategy=MACrossStrategy(), symbol="AAPL")
    result = engine.run("2020-01-01", "2023-12-31")
    print(result.metrics)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import backtrader as bt
import pandas as pd
from loguru import logger

from config import settings
from data import DataFetcher
from strategy import BaseStrategy
from .bt_strategy import BtStrategyAdapter
from .metrics import BacktestMetrics


class _QuestradeCommission(bt.CommInfoBase):
    """Questrade: $0.01/share, min $4.95, max $9.95 per side."""

    params = (
        ("per_share", settings.commission_per_share),
        ("min_comm",  settings.commission_min),
        ("max_comm",  settings.commission_max),
        ("stocklike", True),
        ("commtype",  bt.CommInfoBase.COMM_FIXED),
    )

    def getcommission(self, size, price):  # noqa: ARG002
        qty = abs(size)
        return max(self.params.min_comm,
                   min(qty * self.params.per_share, self.params.max_comm))


@dataclass
class BacktestResult:
    symbol: str
    strategy_name: str
    equity_curve: pd.Series
    trade_log: list[dict]
    metrics: BacktestMetrics
    params: dict = field(default_factory=dict)


class BacktestEngine:
    def __init__(
        self,
        strategy: BaseStrategy,
        symbol: str,
        initial_cash: float = settings.initial_cash,
        stop_loss_pct: float = 0.08,
        trailing_stop_pct: float = 0.12,
        take_profit_pct: float = 0.0,   # disabled — use trailing stop instead
        cooldown_bars: int = 5,
        stop_loss_cooldown_bars: int = 30,
    ):
        self.strategy = strategy
        self.symbol = symbol
        self.initial_cash = initial_cash
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.take_profit_pct = take_profit_pct
        self.cooldown_bars = cooldown_bars
        self.stop_loss_cooldown_bars = stop_loss_cooldown_bars
        self._fetcher = DataFetcher()

    def run(self, start: str, end: str, warmup_days: int = 150) -> BacktestResult:
        logger.info(f"Backtest: {self.symbol} {start}~{end} [{self.strategy.name}]")
        from datetime import datetime, timedelta
        warmup_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=warmup_days)).strftime("%Y-%m-%d")
        df_full = self._fetcher.get_kline(self.symbol, warmup_start, end)
        # Pass the full df (with warmup) but restrict the backtrader feed to start date
        # so indicators are pre-warmed but trading only happens from `start`
        result = self.run_with_df(df_full, self.symbol, trade_start=start)
        return result

    def run_with_df(self, df: pd.DataFrame, symbol: str, trade_start: str | None = None) -> BacktestResult:

        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.addcommissioninfo(_QuestradeCommission())
        cerebro.broker.set_slippage_perc(settings.slippage)

        # Feed full df (including warmup); fromdate restricts trading to trade_start
        feed_kwargs: dict = dict(
            dataname=df,
            datetime=None,
            open="open", high="high", low="low", close="close", volume="volume",
        )
        if trade_start:
            import datetime as _dt
            feed_kwargs["fromdate"] = _dt.datetime.strptime(trade_start, "%Y-%m-%d")

        data_feed = bt.feeds.PandasData(**feed_kwargs)
        data_feed._name = symbol
        cerebro.adddata(data_feed)

        cerebro.addstrategy(
            BtStrategyAdapter,
            quant_strategy=self.strategy,
            df=df,
            stop_loss_pct=self.stop_loss_pct,
            trailing_stop_pct=self.trailing_stop_pct,
            take_profit_pct=self.take_profit_pct,
            cooldown_bars=self.cooldown_bars,
            stop_loss_cooldown_bars=self.stop_loss_cooldown_bars,
        )

        # 记录资金曲线
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

        results = cerebro.run()
        strat = results[0]

        # 重建资金曲线
        time_return = strat.analyzers.time_return.get_analysis()
        equity = pd.Series(time_return).sort_index()
        equity = (1 + equity).cumprod() * self.initial_cash

        trade_log = strat.trade_log
        metrics = BacktestMetrics.from_equity_curve(equity, trade_log)

        logger.info(f"Backtest done. Total return: {metrics.total_return:.2%}")
        return BacktestResult(
            symbol=symbol,
            strategy_name=self.strategy.name,
            equity_curve=equity,
            trade_log=trade_log,
            metrics=metrics,
            params=self.strategy.params,
        )
