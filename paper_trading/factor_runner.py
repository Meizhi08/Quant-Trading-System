"""
Factor-based paper trading runner.

Mirrors factor-backtest strategy in real-time:
  every `rebalance_days` calendar days → score full universe with FactorEngine
  → pick top N → rebalance to equal weight → record daily NAV vs SPY.
"""

from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from data import DataFetcher
from factor import FactorEngine

_STATE_PATH = Path("data/factor_paper_state.json")
_LOG_PATH   = Path("data/factor_paper_log.csv")
_LOOKBACK   = 180  # calendar days of price history for factor computation


class FactorPaperRunner:
    """
    Monthly factor-selection paper trader.

    Each run():
      - If rebalance is due: score universe, sell dropped stocks, buy new top N
      - Always: mark portfolio to market and append a row to factor_paper_log.csv
    """

    def __init__(
        self,
        universe: str = "sp500",
        top_n: int = 10,
        rebalance_days: int = 30,
        initial_cash: float = 100_000.0,
        transaction_cost_pct: float = 0.001,
    ):
        self.universe             = universe
        self.top_n                = top_n
        self.rebalance_days       = rebalance_days
        self.initial_cash         = initial_cash
        self.transaction_cost_pct = transaction_cost_pct
        self.fetcher              = DataFetcher(use_cache=True)
        self.engine               = FactorEngine()
        self._state               = self._load_state()

    # ── State persistence ─────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if _STATE_PATH.exists():
            try:
                return json.loads(_STATE_PATH.read_text())
            except Exception:
                pass
        return {
            "cash":            self.initial_cash,
            "initial_cash":    self.initial_cash,
            "holdings":        {},   # sym → {"qty": float, "avg_cost": float}
            "last_rebalance":  None,
            "trade_log":       [],
        }

    def _save_state(self) -> None:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps(self._state, indent=2, default=str))

    # ── Price helpers ─────────────────────────────────────────────────────────

    def _get_prices(self, symbols: list[str]) -> dict[str, float]:
        today = str(date.today())
        start = str(date.today() - timedelta(days=10))
        prices: dict[str, float] = {}
        for sym in symbols:
            try:
                df = self.fetcher.get_kline(sym, start, today)
                if not df.empty:
                    prices[sym] = float(df["close"].iloc[-1])
            except Exception:
                pass
        return prices

    def _portfolio_value(self, prices: dict[str, float]) -> float:
        val = self._state["cash"]
        for sym, pos in self._state["holdings"].items():
            val += pos["qty"] * prices.get(sym, pos["avg_cost"])
        return val

    # ── Rebalance ─────────────────────────────────────────────────────────────

    def _get_universe(self) -> list[str]:
        from data.stock_selector import get_sp500_symbols, get_tsx60_symbols
        if self.universe == "sp500":
            return get_sp500_symbols()
        if self.universe == "tsx60":
            return get_tsx60_symbols()
        return [s.strip() for s in self.universe.split(",")]

    def _score_universe(self) -> list[tuple[str, float]]:
        """Score all stocks in the universe with FactorEngine."""
        today_str = str(date.today())
        start     = str(date.today() - timedelta(days=_LOOKBACK))
        symbols   = self._get_universe()
        logger.info(f"Scoring {len(symbols)} stocks in {self.universe}...")

        scores: list[tuple[str, float]] = []
        failed = 0
        for i, sym in enumerate(symbols):
            try:
                df = self.fetcher.get_kline(sym, start, today_str)
                if df.empty or len(df) < 60:
                    continue
                fund = self.fetcher.get_fundamentals(sym)
                fs = self.engine.compute(df, sym, fundamentals=fund)
                scores.append((sym, fs.total_score))
            except Exception:
                failed += 1
            if (i + 1) % 50 == 0:
                logger.info(f"  {i+1}/{len(symbols)} scored, valid={len(scores)}")

        logger.info(f"Scoring done: {len(scores)} valid, {failed} failed")
        scores.sort(key=lambda x: -x[1])
        return scores

    def _rebalance(self) -> list[dict]:
        today_str = str(date.today())
        trades: list[dict] = []

        scores      = self._score_universe()
        new_symbols = [sym for sym, _ in scores[: self.top_n]]
        logger.info(f"New top {self.top_n}: {new_symbols}")

        all_needed = list(set(new_symbols) | set(self._state["holdings"].keys()))
        prices     = self._get_prices(all_needed)

        # Sell holdings not in new selection
        for sym in list(self._state["holdings"].keys()):
            if sym not in new_symbols:
                pos      = self._state["holdings"][sym]
                price    = prices.get(sym, pos["avg_cost"])
                proceeds = pos["qty"] * price * (1 - self.transaction_cost_pct)
                self._state["cash"] += proceeds
                trade = {
                    "date": today_str, "symbol": sym, "side": "SELL",
                    "qty": round(pos["qty"], 4), "price": round(price, 4),
                    "cost": round(pos["qty"] * price * self.transaction_cost_pct, 2),
                }
                trades.append(trade)
                self._state["trade_log"].append(trade)
                del self._state["holdings"][sym]
                logger.info(f"SELL {sym} x{pos['qty']:.2f} @ {price:.2f}")

        # Buy / rebalance to equal weight
        portfolio_val    = self._portfolio_value(prices)
        target_per_stock = portfolio_val / self.top_n

        for sym in new_symbols:
            price = prices.get(sym)
            if not price:
                logger.warning(f"No price for {sym}, skipping")
                continue

            current_qty = self._state["holdings"].get(sym, {}).get("qty", 0.0)
            current_val = current_qty * price
            diff_val    = target_per_stock - current_val

            if diff_val < price:  # already at or above target, skip
                continue

            qty  = diff_val / price / (1 + self.transaction_cost_pct)
            cost = qty * price * (1 + self.transaction_cost_pct)
            if cost > self._state["cash"]:
                qty  = self._state["cash"] / price / (1 + self.transaction_cost_pct)
                cost = self._state["cash"]
            if qty < 0.01:
                continue

            self._state["cash"] -= cost
            existing  = self._state["holdings"].get(sym, {"qty": 0.0, "avg_cost": price})
            new_qty   = existing["qty"] + qty
            new_cost  = (existing["qty"] * existing["avg_cost"] + qty * price) / new_qty
            self._state["holdings"][sym] = {
                "qty":      round(new_qty,  4),
                "avg_cost": round(new_cost, 4),
            }
            trade = {
                "date": today_str, "symbol": sym, "side": "BUY",
                "qty": round(qty, 4), "price": round(price, 4),
                "cost": round(cost * self.transaction_cost_pct, 2),
            }
            trades.append(trade)
            self._state["trade_log"].append(trade)
            logger.info(f"BUY  {sym} x{qty:.2f} @ {price:.2f}")

        self._state["last_rebalance"] = today_str
        return trades

    # ── Main entry ────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Execute one day of factor paper trading. Call once per trading day."""
        today_str  = str(date.today())
        last_rb    = self._state.get("last_rebalance")
        days_since = (
            (date.today() - date.fromisoformat(last_rb)).days
            if last_rb else self.rebalance_days
        )
        rebalance_due = days_since >= self.rebalance_days

        if rebalance_due:
            trades = self._rebalance()
        else:
            trades = []
            logger.info(f"Next rebalance in {self.rebalance_days - days_since} days")

        prices       = self._get_prices(list(self._state["holdings"].keys()))
        total_equity = self._portfolio_value(prices)
        total_return = total_equity / self._state["initial_cash"] - 1

        self._save_state()

        report = {
            "date":          today_str,
            "total_equity":  round(total_equity, 2),
            "cash":          round(self._state["cash"], 2),
            "total_return":  total_return,
            "holdings":      dict(self._state["holdings"]),
            "trades":        trades,
            "rebalanced":    rebalance_due,
        }
        self._save_snapshot(report)
        return report

    # ── CSV logging ───────────────────────────────────────────────────────────

    def _save_snapshot(self, report: dict) -> None:
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

        holdings_str = "|".join(
            f"{sym}:{round(p['qty'], 1)}@{p['avg_cost']:.2f}"
            for sym, p in report["holdings"].items()
        ) or "none"

        row = {
            "date":             report["date"],
            "total_equity":     report["total_equity"],
            "cash":             report["cash"],
            "total_return_pct": round(report["total_return"] * 100, 4),
            "spy_close":        spy_close,
            "holdings":         holdings_str,
            "rebalanced":       int(report["rebalanced"]),
            "trades_today":     len(report["trades"]),
        }

        write_header = not _LOG_PATH.exists()
        with open(_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        logger.info(
            f"Snapshot → {_LOG_PATH}  equity={row['total_equity']}  "
            f"return={row['total_return_pct']:.2f}%"
        )
