"""
Alpaca Paper Trading runner.

Flow (runs once per trading day after market close):
  Rebalance day  : score full universe → apply constraints → score-weighted rebalance
  Non-rebalance  : load cached scores, fetch only held stocks → ATR stop-loss check
  Every day      : log SPY closing price + write last-run timestamp
"""

from __future__ import annotations

import csv
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

from data import DataFetcher
from factor import FactorEngine

_LOG_PATH      = Path("data/alpaca_paper_log.csv")
_SCORES_CACHE  = Path("data/alpaca_paper_scores.json")
_LAST_RUN_PATH = Path("data/alpaca_last_run.txt")
_LOOKBACK      = 180
_ATR_LOOKBACK  = 60
_ET            = ZoneInfo("America/New_York")

# Share-class pairs: key → canonical ticker (keep highest scorer within pair)
_SHARE_CLASS_GROUPS: dict[str, str] = {
    "GOOGL": "GOOG",
    "BRK-A": "BRK-B",
    "NWS":   "NWSA",
}


def _get_client():
    from alpaca.trading.client import TradingClient
    api_key = os.getenv("ALPACA_API_KEY")
    secret  = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in .env")
    return TradingClient(api_key, secret, paper=True)


def _compute_atr(df: pd.DataFrame, period: int = 20) -> float:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


class AlpacaPaperRunner:
    """
    Factor-based paper trader backed by Alpaca Paper Trading API.

    Rebalance day  : score-weighted rebalance to constrained top N.
    Non-rebalance  : ATR stop-loss check on held stocks only (fast).
    """

    def __init__(
        self,
        universe: str         = "sp500",
        top_n: int            = 20,
        rebalance_days: int   = 30,
        stop_loss_pct: float  = 0.15,
        atr_multiplier: float = 2.5,
        max_sector_pct: float = 0.25,
    ):
        self.universe        = universe
        self.top_n           = top_n
        self.rebalance_days  = rebalance_days
        self.stop_loss_pct   = stop_loss_pct
        self.atr_multiplier  = atr_multiplier
        self.max_sector_pct  = max_sector_pct
        self.fetcher         = DataFetcher(use_cache=True)
        self.engine          = FactorEngine()
        self.client          = _get_client()

    # ── Alpaca helpers ────────────────────────────────────────────────────────

    def _account_equity(self) -> float:
        return float(self.client.get_account().equity)

    def _current_positions(self) -> dict[str, float]:
        return {p.symbol: float(p.market_value) for p in self.client.get_all_positions()}

    def _current_positions_detail(self) -> dict[str, dict]:
        return {
            p.symbol: {
                "market_value":    float(p.market_value),
                "avg_entry_price": float(p.avg_entry_price),
                "unrealized_plpc": float(p.unrealized_plpc),
            }
            for p in self.client.get_all_positions()
        }

    def _last_rebalance_date(self) -> date | None:
        if not _LOG_PATH.exists():
            return None
        with open(_LOG_PATH) as f:
            rows = list(csv.DictReader(f))
        rebalanced = [r for r in rows if r.get("rebalanced") == "1"]
        if not rebalanced:
            return None
        return date.fromisoformat(rebalanced[-1]["date"])

    def _submit_order(self, symbol: str, notional: float, side: str) -> bool:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        try:
            req = MarketOrderRequest(
                symbol=symbol,
                notional=round(notional, 2),
                side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            self.client.submit_order(req)
            logger.info(f"{side} {symbol} ${notional:.0f}")
            return True
        except Exception as e:
            logger.warning(f"Order failed {side} {symbol}: {e}")
            return False

    def _close_position(self, symbol: str) -> bool:
        try:
            self.client.close_position(symbol)
            logger.info(f"CLOSE {symbol}")
            return True
        except Exception as e:
            logger.warning(f"Close failed {symbol}: {e}")
            return False

    # ── SPY helpers ───────────────────────────────────────────────────────────

    def _get_spy_data(self) -> tuple[float | None, bool]:
        """Return (spy_close_price, is_above_ma200). Fails gracefully."""
        try:
            end    = str(date.today())
            start  = str(date.today() - timedelta(days=300))
            spy_df = self.fetcher.get_kline("SPY", start, end)
            if spy_df.empty or len(spy_df) < 10:
                return None, True
            close    = float(spy_df["close"].iloc[-1])
            ma200    = spy_df["close"].rolling(200, min_periods=150).mean().iloc[-1]
            above_ma = bool(close > ma200) if pd.notna(ma200) else True
            return close, above_ma
        except Exception:
            return None, True

    # ── Scores cache ──────────────────────────────────────────────────────────

    def _save_scores_cache(
        self, scores: list[tuple[str, float]], sector_map: dict[str, str]
    ) -> None:
        _SCORES_CACHE.parent.mkdir(exist_ok=True)
        with open(_SCORES_CACHE, "w") as f:
            json.dump({
                "date":    str(date.today()),
                "scores":  scores,
                "sectors": sector_map,
            }, f)

    def _load_scores_cache(self) -> tuple[list[tuple[str, float]], dict[str, str]]:
        """Return (scores, sector_map). Returns empty if missing or expired."""
        if not _SCORES_CACHE.exists():
            return [], {}
        with open(_SCORES_CACHE) as f:
            data = json.load(f)
        cache_date = date.fromisoformat(data.get("date", "1970-01-01"))
        if (date.today() - cache_date).days > self.rebalance_days:
            logger.warning(
                f"Scores cache expired ({cache_date}), will trigger full re-score"
            )
            return [], {}
        scores     = [(s, v) for s, v in data.get("scores", [])]
        sector_map = data.get("sectors", {})
        return scores, sector_map

    # ── Factor scoring (full universe — rebalance day only) ───────────────────

    def _score_universe(self) -> tuple[list[tuple[str, float]], dict[str, str]]:
        """
        Score all symbols. Returns (sorted_scores, sector_map) and saves cache.
        sector_map is built from cached fundamentals — no extra API calls.
        """
        from data.stock_selector import get_sp500_symbols, get_tsx60_symbols
        symbols = get_sp500_symbols() if self.universe == "sp500" else get_tsx60_symbols()

        end   = str(date.today())
        start = str(date.today() - timedelta(days=_LOOKBACK))

        scores: list[tuple[str, float]] = []
        sector_map: dict[str, str]      = {}

        for sym in sorted(symbols):
            try:
                df = self.fetcher.get_kline(sym, start, end)
                if df.empty or len(df) < 60:
                    continue
                fund = self.fetcher.get_fundamentals(sym)
                sector_map[sym] = fund.get("sector") or "Unknown"
                fs = self.engine.compute(df, sym, fundamentals=fund)
                scores.append((sym, fs.total_score))
            except Exception:
                pass

        scores.sort(key=lambda x: x[1], reverse=True)
        self._save_scores_cache(scores, sector_map)
        logger.info(f"Full universe scored: {len(scores)} stocks")
        return scores, sector_map

    # ── Constraints: dedup share classes + sector cap ─────────────────────────

    def _apply_constraints(
        self,
        scores: list[tuple[str, float]],
        sector_map: dict[str, str],
    ) -> list[tuple[str, float]]:
        """
        1. Share-class dedup: GOOG/GOOGL, BRK-A/BRK-B, etc. — keep highest scorer.
        2. Sector cap: no sector exceeds max_sector_pct of top_n positions.
        Returns top_n constrained selections, still sorted by score.
        """
        # Step 1: dedup
        groups: dict[str, tuple[str, float]] = {}
        for sym, score in scores:
            key = _SHARE_CLASS_GROUPS.get(sym, sym)
            if key not in groups or score > groups[key][1]:
                groups[key] = (sym, score)
        deduped = sorted(groups.values(), key=lambda x: x[1], reverse=True)

        # Step 2: sector cap
        max_per_sector = max(1, int(self.top_n * self.max_sector_pct))
        sector_counts: dict[str, int] = {}
        selected: list[tuple[str, float]] = []

        for sym, score in deduped:
            if len(selected) >= self.top_n:
                break
            sector = sector_map.get(sym, "Unknown")
            if sector_counts.get(sector, 0) < max_per_sector:
                selected.append((sym, score))
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

        # Fallback: relax constraint if not enough candidates
        if len(selected) < self.top_n:
            held    = {s for s, _ in selected}
            extras  = [(s, sc) for s, sc in deduped if s not in held]
            needed  = self.top_n - len(selected)
            selected.extend(extras[:needed])
            logger.warning(
                f"Sector constraint relaxed: added {min(needed, len(extras))} unconstrained stocks"
            )

        logger.info(
            "Sector breakdown: "
            + ", ".join(
                f"{sec}×{cnt}"
                for sec, cnt in sorted(sector_counts.items(), key=lambda x: -x[1])
            )
        )
        return selected

    # ── Score-weighted position sizing ────────────────────────────────────────

    def _score_weights(self, top_scores: list[tuple[str, float]]) -> dict[str, float]:
        """Allocation weights proportional to score, sum to 1. Min weight = ~epsilon."""
        syms    = [s for s, _ in top_scores]
        vals    = [v for _, v in top_scores]
        min_v   = min(vals)
        shifted = [v - min_v + 0.1 for v in vals]
        total   = sum(shifted)
        return {sym: w / total for sym, w in zip(syms, shifted)}

    # ── ATR stop-loss check (non-rebalance day) ───────────────────────────────

    def _check_and_swap_stop_loss(
        self, scores: list[tuple[str, float]], equity: float
    ) -> list[dict]:
        """
        For each held position compute ATR stop price.
        Trigger if current_price < avg_entry_price - atr_multiplier * ATR.
        Falls back to fixed stop_loss_pct if data unavailable.
        Replace with highest-scored unconstrained candidate.
        """
        positions  = self._current_positions_detail()
        held       = set(positions.keys())
        candidates = [s for s, _ in scores if s not in held]
        trades: list[dict] = []

        end   = str(date.today())
        start = str(date.today() - timedelta(days=_ATR_LOOKBACK))

        for sym, detail in list(positions.items()):
            entry         = detail["avg_entry_price"]
            current_price = entry * (1 + detail["unrealized_plpc"])
            triggered     = False
            stop_desc     = ""

            try:
                df        = self.fetcher.get_kline(sym, start, end)
                atr       = _compute_atr(df)
                stop_px   = entry - self.atr_multiplier * atr
                triggered = current_price < stop_px
                stop_desc = (
                    f"ATR stop ${stop_px:.2f}"
                    f" (entry ${entry:.2f} - {self.atr_multiplier}×ATR ${atr:.2f})"
                )
            except Exception:
                triggered = detail["unrealized_plpc"] < -self.stop_loss_pct
                stop_desc = f"fixed stop -{self.stop_loss_pct:.0%}"

            if triggered:
                logger.warning(f"Stop-loss: {sym} @ ${current_price:.2f} | {stop_desc}")
                if self._close_position(sym):
                    trades.append({
                        "symbol": sym, "side": "SELL",
                        "notional": detail["market_value"], "reason": "stop_loss",
                    })
                    held.discard(sym)

                    if candidates:
                        replacement = candidates.pop(0)
                        notional    = round(equity / self.top_n, 2)
                        if self._submit_order(replacement, notional, "BUY"):
                            trades.append({
                                "symbol": replacement, "side": "BUY",
                                "notional": notional, "reason": "stop_loss_replace",
                            })
                            held.add(replacement)
                            logger.info(f"Replaced {sym} → {replacement}")

        return trades

    # ── Full rebalance (constrained + score-weighted) ─────────────────────────

    def _rebalance(
        self,
        equity: float,
        scores: list[tuple[str, float]],
        sector_map: dict[str, str],
    ) -> list[dict]:
        # Cancel pending BUY orders so stale buys don't interfere.
        # Leave pending SELL orders intact — they free up capital we need.
        try:
            from alpaca.trading.enums import OrderSide, QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest
            open_orders = self.client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
            for order in open_orders:
                if order.side == OrderSide.BUY:
                    self.client.cancel_order_by_id(order.id)
            buy_count = sum(1 for o in open_orders if o.side == OrderSide.BUY)
            if buy_count:
                logger.info(f"Cancelled {buy_count} pending buy orders before rebalance")
        except Exception as e:
            logger.warning(f"Could not cancel pending buy orders: {e}")

        top_scores = self._apply_constraints(scores, sector_map)
        new_syms   = [s for s, _ in top_scores]
        weights    = self._score_weights(top_scores)
        logger.info(f"Top {self.top_n} (constrained): {new_syms}")

        current = self._current_positions()
        trades  = []

        for sym in list(current):
            if sym not in new_syms:
                if self._close_position(sym):
                    trades.append({"symbol": sym, "side": "SELL", "notional": current[sym]})

        for sym in new_syms:
            target = equity * weights[sym]
            held   = current.get(sym, 0.0)
            diff   = target - held
            if diff > 5:
                if self._submit_order(sym, diff, "BUY"):
                    trades.append({"symbol": sym, "side": "BUY", "notional": diff})

        return trades

    # ── Main entry ────────────────────────────────────────────────────────────

    def run(self, force_rebalance: bool = False) -> dict:
        today_str = str(date.today())

        last_rb = self._last_rebalance_date()
        if force_rebalance:
            rebalance_due = True
        elif last_rb == date.today():
            rebalance_due = False
        elif last_rb is None:
            rebalance_due = True
        else:
            rebalance_due = (date.today() - last_rb).days >= self.rebalance_days

        equity    = self._account_equity()
        trades: list[dict] = []
        rebalanced = False

        if rebalance_due:
            scores, sector_map = self._score_universe()
            spy_close, _       = self._get_spy_data()

            trades = self._rebalance(equity, scores, sector_map)
            rebalanced = True

        else:
            days_since = (date.today() - last_rb).days if last_rb else 0
            logger.info(
                f"Next rebalance in {self.rebalance_days - days_since} days"
                " — ATR stop-loss check"
            )
            scores, _ = self._load_scores_cache()
            if not scores:
                logger.warning("Cache expired mid-cycle, re-scoring universe (no rebalance)")
                scores, sector_map = self._score_universe()

            spy_close, _ = self._get_spy_data()
            sl_trades = self._check_and_swap_stop_loss(scores, equity)
            trades.extend(sl_trades)
            if sl_trades:
                logger.info(f"Stop-loss swaps today: {len(sl_trades) // 2}")

        equity    = self._account_equity()
        positions = self._current_positions()

        report = {
            "date":       today_str,
            "equity":     round(equity, 2),
            "rebalanced": rebalanced,
            "holdings":   len(positions),
            "trades":     len(trades),
            "spy_close":  round(spy_close, 2) if spy_close else "",
        }
        self._save_log(report, positions, trades)
        _LAST_RUN_PATH.write_text(datetime.now().isoformat())
        return report

    # ── Logging ───────────────────────────────────────────────────────────────

    def _save_log(self, report: dict, positions: dict, trades: list) -> None:
        _LOG_PATH.parent.mkdir(exist_ok=True)
        write_header = not _LOG_PATH.exists()
        with open(_LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "date", "equity", "spy_close", "rebalanced",
                    "n_holdings", "n_trades", "holdings",
                ])
            holdings_str = "|".join(
                f"{sym}:${val:.0f}" for sym, val in sorted(positions.items())
            )
            writer.writerow([
                report["date"],
                report["equity"],
                report["spy_close"],
                int(report["rebalanced"]),
                report["holdings"],
                report["trades"],
                holdings_str,
            ])
