"""
Backtrader 策略适配器 — 将我们的 BaseStrategy 信号桥接进 backtrader 事件循环。
"""

from __future__ import annotations

import backtrader as bt
import pandas as pd

from strategy import BaseStrategy, SignalType


class BtStrategyAdapter(bt.Strategy):
    params = (
        ("quant_strategy", None),
        ("df", None),
        ("risk_manager", None),
        ("stop_loss_pct", 0.08),        # fallback fixed stop if ATR unavailable
        ("trailing_stop_pct", 0.12),    # trailing stop: fixed % below peak
        ("take_profit_pct", 0.0),       # disabled by default; use trailing stop instead
        ("cooldown_bars", 5),           # cooldown after signal-based exit
        ("stop_loss_cooldown_bars", 30),# longer cooldown after stop-loss
        ("min_hold_bars", 10),          # minimum bars before a signal-based SELL is honoured
        ("score_exit_threshold", 0.0),  # exit when score stays below this for N consecutive bars
        ("score_exit_confirm_bars", 3), # consecutive bars below threshold before exiting
        ("atr_stop_multiplier", 2.5),   # stop = entry_price - N × ATR14; 0 = use fixed stop_loss_pct
        ("atr_trail_multiplier", 3.5),  # trailing stop = peak - N × ATR14; 0 = use fixed trailing_stop_pct
    )

    def __init__(self):
        self._strategy: BaseStrategy = self.params.quant_strategy
        self._df: pd.DataFrame = self.params.df
        self._df = self._strategy.compute_indicators(self._df.copy())
        self.order = None
        self.buy_price: float | None = None
        self.peak_price: float | None = None
        self.entry_score: float | None = None
        self.entry_atr: float | None = None   # ATR at entry, for dynamic stop/trail
        self.trade_log: list[dict] = []
        self._cooldown: int = 0
        self._hold_bars: int = 0
        self._below_threshold_bars: int = 0   # consecutive bars score < threshold

    def log(self, msg: str, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"  [{dt}] {msg}")

    def notify_order(self, order: bt.Order):
        if order.status in (order.Completed,):
            side = "BUY " if order.isbuy() else "SELL"
            self.log(f"{side} {order.executed.price:.2f}  size={order.executed.size}")
            if order.isbuy():
                self.buy_price = order.executed.price
                self.peak_price = order.executed.price
            self.trade_log.append({
                "date": str(self.datas[0].datetime.date(0)),
                "side": side.strip(),
                "price": order.executed.price,
                "size": order.executed.size,
                "value": order.executed.value,
                "commission": order.executed.comm,
            })
        self.order = None

    def next(self):
        if self.order:
            return

        if self._cooldown > 0:
            self._cooldown -= 1
            return

        current_dt = self.datas[0].datetime.date(0)
        try:
            idx = self._df.index.get_loc(str(current_dt))
        except KeyError:
            return

        slice_df = self._df.iloc[: idx + 1]
        if len(slice_df) < 30:
            return

        symbol = self.datas[0]._name
        try:
            signal = self._strategy.generate_signal(slice_df, symbol)
        except Exception as e:
            from loguru import logger
            logger.warning(f"generate_signal 异常 [{current_dt}] {symbol}: {e}")
            return

        position = self.getposition()
        close = self.datas[0].close[0]

        if position:
            self._hold_bars += 1
            # update peak for trailing stop
            if self.peak_price is None or close > self.peak_price:
                self.peak_price = close
        else:
            self._hold_bars = 0
            self.peak_price = None

        if position and self.buy_price:
            chg = (close - self.buy_price) / self.buy_price

            # ── ATR-adaptive hard stop ────────────────────────────────────────
            # Use ATR at entry to set stop distance; wider for volatile stocks,
            # tighter for stable ones. Falls back to fixed % if ATR unavailable.
            if self.params.atr_stop_multiplier > 0 and self.entry_atr:
                stop_price = self.buy_price - self.params.atr_stop_multiplier * self.entry_atr
                hit_stop = close <= stop_price
            else:
                hit_stop = chg <= -self.params.stop_loss_pct

            if hit_stop:
                self.log(f"止损 {chg:.1%}")
                self.order = self.close()
                self._cooldown = self.params.stop_loss_cooldown_bars
                self._hold_bars = 0
                self._below_threshold_bars = 0
                self.entry_score = None
                self.entry_atr = None
                return

            # ── ATR-adaptive trailing stop ────────────────────────────────────
            if self.peak_price:
                if self.params.atr_trail_multiplier > 0 and self.entry_atr:
                    trail_floor = self.peak_price - self.params.atr_trail_multiplier * self.entry_atr
                    hit_trail = close <= trail_floor
                else:
                    trail_chg = (close - self.peak_price) / self.peak_price
                    hit_trail = trail_chg <= -self.params.trailing_stop_pct

                if hit_trail:
                    trail_pct = (close - self.peak_price) / self.peak_price
                    self.log(f"追踪止损 (峰值{self.peak_price:.2f} 回落{abs(trail_pct):.1%})")
                    self.order = self.close()
                    self._cooldown = self.params.stop_loss_cooldown_bars
                    self._hold_bars = 0
                    self._below_threshold_bars = 0
                    self.entry_score = None
                    self.entry_atr = None
                    return

            # fixed take-profit (optional, disabled by default)
            if self.params.take_profit_pct > 0 and chg >= self.params.take_profit_pct:
                self.log(f"止盈 {chg:.1%}")
                self.order = self.close()
                self._cooldown = self.params.cooldown_bars
                self._hold_bars = 0
                self._below_threshold_bars = 0
                self.entry_score = None
                self.entry_atr = None
                return

            # ── Signal-deterioration exit (requires N consecutive bars) ───────
            # Single-bar score dip is noise; require sustained deterioration.
            # Only triggers when in profit — losses handled by stop.
            if self._hold_bars >= self.params.min_hold_bars and self.params.score_exit_threshold is not None:
                cur_score = signal.metadata.get("score") if signal.metadata else None
                if cur_score is not None and cur_score < self.params.score_exit_threshold and chg > 0:
                    self._below_threshold_bars += 1
                else:
                    self._below_threshold_bars = 0

                if self._below_threshold_bars >= self.params.score_exit_confirm_bars:
                    self.log(f"信号消退 score={cur_score:+.3f} (入场{self.entry_score:+.3f}) chg={chg:.1%} ×{self._below_threshold_bars}天")
                    self.order = self.close()
                    self._cooldown = self.params.cooldown_bars
                    self._hold_bars = 0
                    self._below_threshold_bars = 0
                    self.entry_score = None
                    self.entry_atr = None
                    return

        if signal.signal == SignalType.BUY and not position:
            size = self._calc_size(close)
            if size > 0:
                self.order = self.buy(size=size)
                self.entry_score = signal.metadata.get("score") if signal.metadata else None
                self.entry_atr = self._calc_atr(slice_df)
        elif signal.signal == SignalType.SELL and position:
            if self._hold_bars < self.params.min_hold_bars:
                return
            self.order = self.sell(size=position.size)
            self._cooldown = self.params.cooldown_bars
            self._hold_bars = 0
            self.entry_score = None
            self.entry_atr = None

    def _calc_size(self, price: float) -> int:
        rm = self.params.risk_manager
        cash = self.broker.getcash()
        if rm:
            pct = rm.kelly_position_size(win_rate=0.55, avg_win=0.10, avg_loss=0.05)
            capital = cash * pct
        else:
            capital = cash * 0.95
        return max(1, int(capital / price))

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> float | None:
        """14-bar ATR using True Range. Returns None if insufficient data."""
        if len(df) < period + 1:
            return None
        high  = df["high"].values[-period - 1:]
        low   = df["low"].values[-period - 1:]
        close = df["close"].values[-period - 1:]
        tr = [max(high[i] - low[i],
                  abs(high[i] - close[i - 1]),
                  abs(low[i]  - close[i - 1]))
              for i in range(1, len(high))]
        return float(sum(tr) / len(tr))
