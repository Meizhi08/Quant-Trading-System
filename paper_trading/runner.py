"""
Paper trading executor for North American markets (Questrade-ready).

run()      — execute today's signals, save state, return daily report
backfill() — replay historical data day by day
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
from loguru import logger

from config import settings
from data import DataFetcher
from strategy import BaseStrategy, SignalType
from risk import RiskManager
from .state import PaperState


def _commission(qty: int) -> float:
    """Questrade: $0.01/share, min $4.95, max $9.95 per side."""
    return max(settings.commission_min,
               min(abs(qty) * settings.commission_per_share, settings.commission_max))


class PaperRunner:
    def __init__(
        self,
        symbols: list[str],
        strategy: BaseStrategy,
        initial_cash: float = 10_000.0,     # CAD
        stop_loss_pct: float = 0.05,
        trailing_stop_pct: float = 0.07,
        take_profit_pct: float = 0.15,
        max_positions: int = 5,
        max_single_pct: float = 0.30,
        max_daily_loss_pct: float = 0.03,
        max_drawdown_pct: float = 0.20,
        max_consec_losses: int = 3,
    ):
        self.symbols = symbols
        self.strategy = strategy
        self.state = PaperState(initial_cash)
        self.fetcher = DataFetcher(use_cache=False)
        self.risk = RiskManager(initial_capital=initial_cash,
                                max_drawdown_pct=max_drawdown_pct)
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.take_profit_pct = take_profit_pct
        self.max_positions = max_positions
        self.max_single_pct = max_single_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_consec_losses = max_consec_losses
        self._consec_losses = 0
        self._price_highs: dict[str, float] = {}

    def _market_allows_buy(self, start: str, end: str) -> bool:
        """Market filter: S&P500 must be above MA60 to allow full-size buys."""
        try:
            idx = self.fetcher.get_index_kline(start=start, end=end)
            if idx.empty or len(idx) < 60:
                return True
            ma60 = idx["close"].rolling(60).mean().iloc[-1]
            current = float(idx["close"].iloc[-1])
            if current < ma60:
                logger.warning(f"[Market filter] {settings.market_index} "
                               f"({current:.0f}) < MA60 ({ma60:.0f}) — reducing position size")
                return False
        except Exception as e:
            logger.warning(f"[Market filter] index fetch failed, skipping: {e}")
        return True

    def run(self) -> dict:
        """Execute today's paper trades. Returns daily report dict."""
        today = str(date.today())
        start = str(date.today() - timedelta(days=300))

        executed = []
        latest_prices: dict[str, float] = {}
        market_ok = self._market_allows_buy(start, today)

        for sym in self.symbols:
            try:
                df = self.fetcher.get_kline(sym, start, today)
                if df.empty or len(df) < 30:
                    logger.warning(f"{sym}: insufficient data, skipping")
                    continue

                price = float(df["close"].iloc[-1])
                latest_prices[sym] = price
                sig = self.strategy.run(df, sym)
                pos = self.state.get_position(sym)

                if pos:
                    high = self._price_highs.get(sym, pos["avg_cost"])
                    if price > high:
                        self._price_highs[sym] = price
                        high = price
                    chg = (price - pos["avg_cost"]) / pos["avg_cost"]
                    trail_chg = (price - high) / high

                    if chg <= -self.stop_loss_pct:
                        t = self._sell(sym, pos["qty"], price, f"stop-loss {chg:.1%}")
                        if t:
                            executed.append(t)
                            self._price_highs.pop(sym, None)
                        continue
                    if trail_chg <= -self.trailing_stop_pct:
                        t = self._sell(sym, pos["qty"], price,
                                       f"trailing-stop (peak-{abs(trail_chg):.1%})")
                        if t:
                            executed.append(t)
                            self._price_highs.pop(sym, None)
                        continue
                    if chg >= self.take_profit_pct:
                        t = self._sell(sym, pos["qty"], price, f"take-profit {chg:.1%}")
                        if t:
                            executed.append(t)
                            self._price_highs.pop(sym, None)
                        continue

                if sig.signal == SignalType.BUY and not pos:
                    total_equity = self.state.snapshot_equity(latest_prices)
                    if self.risk.is_drawdown_breached(total_equity):
                        logger.warning(f"Max drawdown breached — skip buy {sym}")
                        continue
                    win_rate, avg_win, avg_loss = self.risk.calc_kelly_params(
                        self.state.all_trades())
                    pct = self.risk.kelly_position_size(win_rate, avg_win, avg_loss)
                    if not market_ok:
                        pct *= 0.5
                    qty = max(1, int(total_equity * pct / price))
                    t = self._buy(sym, qty, price, reason=sig.reason)
                    if t:
                        self._price_highs[sym] = price
                        executed.append(t)

                elif sig.signal == SignalType.SELL and pos:
                    t = self._sell(sym, pos["qty"], price, reason=sig.reason)
                    if t:
                        executed.append(t)

            except Exception as e:
                logger.error(f"{sym} error: {e}")

        total_equity = self.state.snapshot_equity(latest_prices)
        self.risk.update_peak(total_equity)
        self.state.save()

        report = {
            "date": today,
            "executed_trades": executed,
            "cash": self.state.cash,
            "total_equity": total_equity,
            "total_return": self.state.total_return(),
            "positions": self.state.all_positions(),
        }
        self._save_daily_snapshot(report)
        return report

    # ── Risk checks ──────────────────────────────────────────────────────────

    def _check_risk(self, sym: str, qty: int, price: float) -> str | None:
        positions = self.state.all_positions()
        if sym not in positions and len(positions) >= self.max_positions:
            return f"max positions reached ({self.max_positions})"
        equity = self.state.snapshot_equity({sym: price})
        if qty * price > equity * self.max_single_pct:
            return f"single position > {self.max_single_pct:.0%}"
        if self._consec_losses >= self.max_consec_losses:
            return f"paused after {self._consec_losses} consecutive losses"
        return None

    def _update_loss_streak(self, sell_price: float, avg_cost: float) -> None:
        if sell_price < avg_cost:
            self._consec_losses += 1
        else:
            self._consec_losses = 0

    # ── Order execution ──────────────────────────────────────────────────────

    def _buy(self, sym: str, qty: int, price: float, reason: str,
             day: str | None = None) -> dict | None:
        reject = self._check_risk(sym, qty, price)
        if reject:
            logger.warning(f"Buy {sym} blocked by risk: {reject}")
            return None
        comm = _commission(qty)
        cost = qty * price + comm
        if cost > self.state.cash:
            logger.warning(f"Buy {sym} failed: insufficient cash")
            return None
        self.state.cash -= cost
        pos = self.state.get_position(sym)
        if pos:
            new_qty  = pos["qty"] + qty
            new_cost = (pos["avg_cost"] * pos["qty"] + price * qty) / new_qty
            self.state.set_position(sym, new_qty, new_cost)
        else:
            self.state.set_position(sym, qty, price)
        trade = {"date": day or str(date.today()), "symbol": sym, "side": "BUY",
                 "price": price, "qty": qty, "commission": round(comm, 2),
                 "reason": reason}
        self.state.add_trade(trade)
        logger.info(f"Paper BUY  {sym} x{qty} @ {price:.2f}  comm=${comm:.2f}  {reason}")
        return trade

    def _sell(self, sym: str, qty: int, price: float, reason: str,
              day: str | None = None) -> dict | None:
        pos = self.state.get_position(sym)
        if not pos or pos["qty"] < qty:
            return None
        avg_cost = pos["avg_cost"]
        self._update_loss_streak(price, avg_cost)
        comm = _commission(qty)
        self.state.cash += qty * price - comm
        self.state.set_position(sym, pos["qty"] - qty, avg_cost)
        trade = {"date": day or str(date.today()), "symbol": sym, "side": "SELL",
                 "price": price, "qty": qty, "commission": round(comm, 2),
                 "reason": reason, "avg_cost": avg_cost}
        self.state.add_trade(trade)
        logger.info(f"Paper SELL {sym} x{qty} @ {price:.2f}  comm=${comm:.2f}  {reason}")
        return trade

    # ── Daily snapshot logging ───────────────────────────────────────────────

    def _save_daily_snapshot(self, report: dict) -> None:
        """Append one row to data/paper_trading_log.csv after each run()."""
        import csv
        from pathlib import Path

        spy_close = None
        try:
            spy_df = self.fetcher.get_kline(
                "SPY",
                str(date.today() - timedelta(days=10)),
                str(date.today()),
            )
            if not spy_df.empty:
                spy_close = round(float(spy_df["close"].iloc[-1]), 2)
        except Exception:
            pass

        positions_str = "|".join(
            f"{sym}:{p['qty']}@{p['avg_cost']:.2f}"
            for sym, p in report["positions"].items()
        ) or "none"

        row = {
            "date":             report["date"],
            "total_equity":     round(report["total_equity"], 2),
            "cash":             round(report["cash"], 2),
            "total_return_pct": round(report["total_return"] * 100, 4),
            "spy_close":        spy_close,
            "positions":        positions_str,
            "trades_today":     len(report["executed_trades"]),
        }

        log_path = Path("data/paper_trading_log.csv")
        write_header = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        logger.info(f"Snapshot saved → {log_path}  equity={row['total_equity']}")

    # ── Historical backfill ──────────────────────────────────────────────────

    def backfill(self, start: str, end: str) -> list[dict]:
        """Replay historical data day by day. Returns list of daily snapshots."""
        lookback_start = str(
            datetime.strptime(start, "%Y-%m-%d").date() - timedelta(days=300)
        )

        all_df: dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            try:
                df = self.fetcher.get_kline(sym, lookback_start, end)
                if not df.empty:
                    all_df[sym] = df
            except Exception as e:
                logger.warning(f"Data fetch failed for {sym}: {e}")

        index_df: pd.DataFrame = pd.DataFrame()
        try:
            index_df = self.fetcher.get_index_kline(start=lookback_start, end=end)
            if not index_df.empty:
                index_df["ma60"] = index_df["close"].rolling(60).mean()
        except Exception as e:
            logger.warning(f"Index data fetch failed: {e}")

        if not all_df:
            logger.error("All symbol data fetch failed")
            return []

        trading_days = sorted(set(
            str(d.date()) for df in all_df.values()
            for d in df.loc[start:end].index
        ))

        daily_reports = []
        for day in trading_days:
            latest_prices: dict[str, float] = {}
            day_trades = []

            market_ok = True
            if not index_df.empty:
                try:
                    slice_ = index_df.loc[:day]
                    if not slice_.empty:
                        row = slice_.iloc[-1]
                        if not pd.isna(row["ma60"]) and float(row["close"]) < float(row["ma60"]):
                            market_ok = False
                except Exception:
                    pass

            for sym, full_df in all_df.items():
                slice_df = full_df.loc[:day]
                if len(slice_df) < 30:
                    continue
                price = float(slice_df["close"].iloc[-1])
                latest_prices[sym] = price

                try:
                    sig = self.strategy.run(slice_df, sym)
                except Exception:
                    continue

                pos = self.state.get_position(sym)

                if pos:
                    high = self._price_highs.get(sym, pos["avg_cost"])
                    if price > high:
                        self._price_highs[sym] = price
                        high = price
                    chg = (price - pos["avg_cost"]) / pos["avg_cost"]
                    trail_chg = (price - high) / high

                    if chg <= -self.stop_loss_pct:
                        t = self._sell(sym, pos["qty"], price, f"stop-loss {chg:.1%}", day)
                        if t:
                            day_trades.append(t)
                            self._price_highs.pop(sym, None)
                        continue
                    if trail_chg <= -self.trailing_stop_pct:
                        t = self._sell(sym, pos["qty"], price,
                                       f"trailing-stop (peak-{abs(trail_chg):.1%})", day)
                        if t:
                            day_trades.append(t)
                            self._price_highs.pop(sym, None)
                        continue
                    if chg >= self.take_profit_pct:
                        t = self._sell(sym, pos["qty"], price, f"take-profit {chg:.1%}", day)
                        if t:
                            day_trades.append(t)
                            self._price_highs.pop(sym, None)
                        continue

                if sig.signal == SignalType.BUY and not pos:
                    total_eq = self.state.snapshot_equity(latest_prices)
                    if self.risk.is_drawdown_breached(total_eq):
                        continue
                    win_rate, avg_win, avg_loss = self.risk.calc_kelly_params(
                        self.state.all_trades())
                    pct = self.risk.kelly_position_size(win_rate, avg_win, avg_loss)
                    if not market_ok:
                        pct *= 0.5
                    qty = max(1, int(total_eq * pct / price))
                    t = self._buy(sym, qty, price, sig.reason[:40], day)
                    if t:
                        self._price_highs[sym] = price
                        day_trades.append(t)

                elif sig.signal == SignalType.SELL and pos:
                    t = self._sell(sym, pos["qty"], price, sig.reason[:40], day)
                    if t:
                        day_trades.append(t)

            equity = self.state.snapshot_equity(latest_prices)
            self.risk.update_peak(equity)
            ret = (equity - self.state.initial_cash) / self.state.initial_cash
            daily_reports.append({
                "date": day, "equity": equity,
                "return": ret, "trades": day_trades,
            })

        self.state.save()
        return daily_reports
