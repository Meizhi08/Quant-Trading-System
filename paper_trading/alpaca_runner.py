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
import math
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

from config import settings
from data import DataFetcher
from factor import FactorEngine

_LOG_PATH          = Path("data/alpaca_paper_log.csv")
_SCORES_CACHE      = Path("data/alpaca_paper_scores.json")
_LAST_RUN_PATH     = Path("data/alpaca_last_run.txt")
_REBALANCE_MARKER  = Path("data/alpaca_rebalance_in_progress.marker")
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


def _today() -> date:
    """Trading-day boundary must use US/Eastern, not the host machine's local timezone."""
    return datetime.now(_ET).date()


# 15-minute buffer past the 4:00pm ET close. This runner is designed to score/trade
# using a full day's data ("runs once per trading day after market close" — see module
# docstring); if launchd catches up a missed wake at, say, 00:30 ET, _today() correctly
# advances to the new calendar date, but the market for that date hasn't opened yet, let
# alone closed — the SPY/kline data _score_universe() would fetch is still yesterday's,
# not "today's completed session".
_MARKET_CLOSE_ET = time(16, 15)


def _market_closed_for_today() -> bool:
    return datetime.now(_ET).time() >= _MARKET_CLOSE_ET


def _last_completed_session_cutoff() -> date:
    """
    The `end` date to use whenever fetching kline/fundamentals data for SCORING
    decisions (full rebalance, universe scan, ATR). Must never resolve to a still-in-
    progress trading day: if the market hasn't closed yet (e.g. an overdue rebalance is
    forced through before close, or the mid-cycle score-cache-expired fallback happens
    to run mid-day), this returns yesterday instead of today.

    "Yesterday" here is a plain CALENDAR-day subtraction, not the previous trading day
    (e.g. running before close on a Monday returns Sunday's date). That's intentional
    and sufficient — this value is only ever passed as yfinance's `end` bound, which is
    just an upper limit on the query range and does not need to be a trading day itself;
    yfinance returns whatever the most recent actual trading day at or before that date
    is regardless. This function is NEVER used for day-count/period arithmetic (e.g.
    days-since-last-rebalance, which uses _today() directly) — mixing calendar-day and
    trading-day semantics there would change what "N days overdue" means.
    """
    return _today() if _market_closed_for_today() else _today() - timedelta(days=1)


# How many multiples of rebalance_days a rebalance can go overdue before the
# market-close guard backs off and forces the full rebalance through anyway (a machine
# that only ever wakes launchd before close would otherwise defer indefinitely). 1.5x
# was chosen as "meaningfully late, not just a day slow" without waiting so long that a
# real staleness problem goes unnoticed; exposed as a constructor/CLI parameter for
# anyone who wants a different tradeoff.
_OVERDUE_MULTIPLIER = 1.5


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
        max_position_pct: float | None = None,
        dry_run: bool = False,
        overdue_multiplier: float = _OVERDUE_MULTIPLIER,
    ):
        self.universe           = universe
        self.top_n              = top_n
        self.rebalance_days     = rebalance_days
        self.stop_loss_pct      = stop_loss_pct
        self.atr_multiplier     = atr_multiplier
        self.max_sector_pct     = max_sector_pct
        self.max_position_pct   = max_position_pct if max_position_pct is not None else settings.max_position_pct
        self.dry_run            = dry_run
        self.overdue_multiplier = overdue_multiplier
        self.fetcher          = DataFetcher(use_cache=True)
        self.engine           = FactorEngine()
        self.client           = _get_client()
        self._notifier        = None
        if dry_run:
            logger.warning("DRY-RUN mode: no orders will be submitted to Alpaca")

    def _get_notifier(self):
        if self._notifier is None:
            from alert import Notifier
            self._notifier = Notifier()
        return self._notifier

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

    def _submit_order(self, symbol: str, notional: float, side: str) -> str | None:
        """Returns the broker order id on acceptance, None if the submit call itself failed."""
        if self.dry_run:
            logger.info(f"[DRY-RUN] would {side} {symbol} ${notional:.0f}")
            return "dry-run"
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        try:
            req = MarketOrderRequest(
                symbol=symbol,
                notional=round(notional, 2),
                side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = self.client.submit_order(req)
            logger.info(f"{side} {symbol} ${notional:.0f}")
            return str(order.id)
        except Exception as e:
            logger.warning(f"Order failed {side} {symbol}: {e}")
            return None

    def _close_position(self, symbol: str) -> str | None:
        """Returns the broker order id on acceptance, None if the close call itself failed."""
        if self.dry_run:
            logger.info(f"[DRY-RUN] would CLOSE {symbol}")
            return "dry-run"
        try:
            order = self.client.close_position(symbol)
            logger.info(f"CLOSE {symbol}")
            return str(getattr(order, "id", "")) or "n/a"
        except Exception as e:
            logger.warning(f"Close failed {symbol}: {e}")
            return None

    def _verify_orders(self, order_ids: list[str]) -> dict[str, int]:
        """
        Alpaca can accept an order and reject it asynchronously (PDT rule, halted symbol,
        wash-trade check) without raising at submit time, or fill only part of the
        requested notional (thin liquidity). Re-check each order's terminal status —
        checking the status string alone would silently count a partial fill as a full
        success. Returns {"rejected": n, "partial": n}.
        """
        counts = {"rejected": 0, "partial": 0}
        for oid in order_ids:
            if not oid or oid in ("n/a", "dry-run"):
                continue
            try:
                o = self.client.get_order_by_id(oid)
                status = str(getattr(o, "status", "")).lower()
                if any(s in status for s in ("rejected", "cancel", "expired")):
                    counts["rejected"] += 1
                    logger.warning(f"Order {oid} ({getattr(o, 'symbol', '?')}) ended up {status}")
                elif "partial" in status:
                    counts["partial"] += 1
                    logger.warning(
                        f"Order {oid} ({getattr(o, 'symbol', '?')}) only partially filled: "
                        f"{getattr(o, 'filled_qty', '?')}/{getattr(o, 'qty', '?')}"
                    )
                elif status == "filled":
                    filled_qty = getattr(o, "filled_qty", None)
                    req_qty    = getattr(o, "qty", None)
                    if filled_qty is not None and req_qty is not None and float(filled_qty) < float(req_qty) - 1e-6:
                        # Some brokers report "filled" once no more of the order will execute,
                        # even if the requested qty wasn't fully met — treat that as partial too.
                        counts["partial"] += 1
                        logger.warning(
                            f"Order {oid} ({getattr(o, 'symbol', '?')}) reported filled but "
                            f"filled_qty {filled_qty} < requested {req_qty}"
                        )
            except Exception:
                pass
        return counts

    # ── SPY helpers ───────────────────────────────────────────────────────────

    def _get_spy_data(self) -> tuple[float | None, bool]:
        """Return (spy_close_price, is_above_ma200). Fails gracefully."""
        try:
            end    = str(_last_completed_session_cutoff())
            start  = str(_today() - timedelta(days=300))
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
        tmp_path = _SCORES_CACHE.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump({
                "date":    str(_today()),
                "scores":  scores,
                "sectors": sector_map,
            }, f)
        os.replace(tmp_path, _SCORES_CACHE)

    def _load_scores_cache(self) -> tuple[list[tuple[str, float]], dict[str, str]]:
        """Return (scores, sector_map). Returns empty if missing or expired."""
        if not _SCORES_CACHE.exists():
            return [], {}
        with open(_SCORES_CACHE) as f:
            data = json.load(f)
        cache_date = date.fromisoformat(data.get("date", "1970-01-01"))
        if (_today() - cache_date).days >= self.rebalance_days:
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

        end   = str(_last_completed_session_cutoff())
        start = str(_today() - timedelta(days=_LOOKBACK))

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
        n: int | None = None,
    ) -> list[tuple[str, float]]:
        """
        1. Share-class dedup: GOOG/GOOGL, BRK-A/BRK-B, etc. — keep highest scorer.
        2. Sector cap: no sector exceeds max_sector_pct of n positions.
        Returns n constrained selections, still sorted by score.

        n defaults to self.top_n. Callers building a SMALLER portfolio than top_n (the
        bear-market half-size book) must pass that smaller n here rather than slicing
        the top_n-sized result afterward — max_per_sector below is computed relative to
        n, and simply taking the top half of an already-diversified top_n list can
        reconcentrate into far fewer sectors than that half-size book's own cap allows.
        """
        n = n if n is not None else self.top_n
        # Step 1: dedup
        groups: dict[str, tuple[str, float]] = {}
        for sym, score in scores:
            key = _SHARE_CLASS_GROUPS.get(sym, sym)
            if key not in groups or score > groups[key][1]:
                groups[key] = (sym, score)
        deduped = sorted(groups.values(), key=lambda x: x[1], reverse=True)

        # Step 2: sector cap, filling by score within the cap. If the universe doesn't
        # have enough sector-diverse candidates to reach n at the strict cap, relax the
        # per-sector cap by +1 and try again — repeatedly, only as far as necessary —
        # rather than falling back to an entirely unconstrained fill by raw score. The
        # old fallback took whatever scored highest regardless of sector once the
        # strict cap ran out of candidates, which could refill an already-at-cap sector
        # right back up (e.g. if that sector simply had the highest average scores),
        # defeating the diversification this function exists to enforce.
        max_per_sector = max(1, int(n * self.max_sector_pct))
        cap = max_per_sector
        relaxed = False
        sector_counts: dict[str, int] = {}
        selected: list[tuple[str, float]] = []
        held: set[str] = set()

        while len(selected) < n and len(selected) < len(deduped):
            progressed = False
            for sym, score in deduped:
                if len(selected) >= n:
                    break
                if sym in held:
                    continue
                sector = sector_map.get(sym, "Unknown")
                if sector_counts.get(sector, 0) < cap:
                    selected.append((sym, score))
                    held.add(sym)
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    progressed = True
            if len(selected) >= n:
                break
            if not progressed:
                cap += 1
                relaxed = True

        if relaxed:
            logger.warning(
                f"Sector cap relaxed from {max_per_sector} to {cap} per sector to fill "
                f"{n} positions — universe lacks enough sector-diverse candidates at "
                "the strict cap"
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
        if not top_scores:
            return {}
        syms    = [s for s, _ in top_scores]
        vals    = [v for _, v in top_scores]
        min_v   = min(vals)
        shifted = [v - min_v + 0.1 for v in vals]
        total   = sum(shifted)
        return {sym: w / total for sym, w in zip(syms, shifted)}

    @staticmethod
    def _cap_group(
        weights: dict[str, float], group_map: dict[str, str], max_pct: float
    ) -> dict[str, float]:
        """
        Enforce max_pct on the aggregate weight of each group (group_map maps symbol →
        group id; pass an identity map for a per-symbol cap). Excess weight from any
        over-cap group is redistributed proportionally across the remaining
        (not-yet-capped-in-this-call) positions. The iteration budget is derived from
        the number of distinct groups rather than a fixed magic number — each pass caps
        at least one more group, and there cannot be more newly-over groups per pass
        than there are groups, so `n_groups + 2` is always enough to reach convergence
        (the +2 covers the redistribution possibly pushing a previously-fine group over
        on a later pass).

        The cap is a hard risk limit — it must never be the thing that gives way. If
        every symbol is already capped and there is still excess left to place (the
        constraint is jointly infeasible for this weight set: e.g. too few groups for
        the cap to allow summing to 1.0), that excess is left UNALLOCATED rather than
        either forced onto already-capped symbols (breaches the cap) or silently
        dropped (untracked — looks like a bug). The caller must treat
        `sum(returned) < sum(input)` as intentional: the unplaceable fraction stays as
        cash instead of being deployed outside the risk limit.
        """
        weights = dict(weights)
        capped: set[str] = set()
        n_groups = len({group_map.get(s, s) for s in weights})
        for _ in range(n_groups + 2):
            group_totals: dict[str, float] = {}
            for sym, w in weights.items():
                g = group_map.get(sym, sym)
                group_totals[g] = group_totals.get(g, 0.0) + w
            over = {g: tot for g, tot in group_totals.items() if tot > max_pct + 1e-9}
            if not over:
                break
            excess = 0.0
            for g, tot in over.items():
                scale = max_pct / tot
                for sym in list(weights):
                    if sym in capped or group_map.get(sym, sym) != g:
                        continue
                    reduced = weights[sym] * (1 - scale)
                    excess += reduced
                    weights[sym] -= reduced
                    capped.add(sym)
            if excess <= 1e-12:
                break
            pool_syms = [s for s in weights if s not in capped]
            pool = sum(weights[s] for s in pool_syms)
            if pool > 1e-12:
                for sym in pool_syms:
                    weights[sym] += excess * (weights[sym] / pool)
            else:
                logger.warning(
                    f"_cap_group: max_pct={max_pct:.2%} is infeasible for {n_groups} "
                    f"groups — {excess:.1%} of equity left unallocated (held as cash) "
                    "rather than breaching the cap"
                )
                break
        return weights

    def _cap_sector_weights(
        self, weights: dict[str, float], sector_map: dict[str, str]
    ) -> dict[str, float]:
        """
        Enforce max_sector_pct on aggregate DOLLAR weight per sector, not just position
        count (_apply_constraints' count-based prefilter can still let a sector take
        >max_sector_pct of the book under score-weighted sizing). Also enforces
        max_position_pct per individual symbol as a defense-in-depth cap — sector-level
        redistribution alone can otherwise concentrate a large excess onto just one or
        two remaining uncapped names when the sector-diverse candidate pool is thin.
        Alternates the two passes since redistributing one can reopen the other.
        """
        identity_map = {sym: sym for sym in weights}
        for _ in range(3):
            weights = self._cap_group(weights, sector_map, self.max_sector_pct)
            weights = self._cap_group(weights, identity_map, self.max_position_pct)
        # Sector cap gets the final word: the per-symbol pass above can redistribute
        # excess back onto a sector that was already sitting right at its cap, and
        # without a closing sector pass that reopened violation would never get
        # re-checked. The reverse (this pass slightly reopening the symbol cap) is the
        # lesser risk given max_position_pct is normally the looser of the two limits.
        weights = self._cap_group(weights, sector_map, self.max_sector_pct)
        return weights

    # ── ATR stop-loss check (non-rebalance day) ───────────────────────────────

    def _check_and_swap_stop_loss(
        self, scores: list[tuple[str, float]], sector_map: dict[str, str], equity: float
    ) -> list[dict]:
        """
        For each held position compute ATR stop price.
        Trigger if current_price < avg_entry_price - atr_multiplier * ATR.
        Falls back to fixed stop_loss_pct if data unavailable.
        Replace with the highest-scored candidate whose sector is still under cap —
        a stop-loss swap must not push a sector past max_sector_pct and then sit there
        unconstrained for up to `rebalance_days` until the next full rebalance.
        """
        positions  = self._current_positions_detail()
        held       = set(positions.keys())
        candidates = [s for s, _ in scores if s not in held]
        trades: list[dict] = []

        max_per_sector = max(1, int(self.top_n * self.max_sector_pct))
        sector_counts: dict[str, int] = {}
        for sym in held:
            sec = sector_map.get(sym, "Unknown")
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

        # ATR is a slow-moving ~20-day rolling average, and the stop-loss trigger price
        # comparison uses Alpaca's own live position data (current_price below), not
        # this kline fetch — but the fetch itself must still never request a bar that
        # doesn't exist yet, now that this check can run before market close.
        end   = str(_last_completed_session_cutoff())
        start = str(_today() - timedelta(days=_ATR_LOOKBACK))

        for sym, detail in list(positions.items()):
            entry         = detail["avg_entry_price"]
            # detail comes from _current_positions_detail() -> self.client.get_all_positions()
            # (alpaca.trading.client.TradingClient — the real Alpaca paper-trading
            # positions endpoint, NOT broker/paper_broker.py's local avg_cost simulator,
            # which this runner never touches). unrealized_plpc is Alpaca's own
            # server-side mark against its latest quote, refreshed during market hours —
            # this is the "live-ish" current price, independent of the yfinance ATR
            # fetch below.
            current_price = entry * (1 + detail["unrealized_plpc"])
            if not math.isfinite(current_price) or current_price <= 0:
                # A NaN/inf/non-positive current_price (e.g. Alpaca returning a bad
                # unrealized_plpc) would make every comparison below silently evaluate
                # to False — the position would look "never triggered" instead of
                # erroring, which is the wrong failure mode for a stop-loss check.
                logger.warning(
                    f"{sym}: unusable current_price ({current_price}) from "
                    f"unrealized_plpc={detail['unrealized_plpc']!r} — skipping this "
                    "position's stop-loss check this cycle rather than silently "
                    "treating it as not-triggered"
                )
                continue
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
                order_id = self._close_position(sym)
                if order_id:
                    trades.append({
                        "symbol": sym, "side": "SELL", "order_id": order_id,
                        "notional": detail["market_value"], "reason": "stop_loss",
                    })
                    held.discard(sym)
                    sold_sector = sector_map.get(sym, "Unknown")
                    sector_counts[sold_sector] = max(0, sector_counts.get(sold_sector, 0) - 1)

                    replacement = None
                    for i, cand in enumerate(candidates):
                        cand_sector = sector_map.get(cand, "Unknown")
                        if sector_counts.get(cand_sector, 0) < max_per_sector:
                            replacement = candidates.pop(i)
                            break

                    if replacement:
                        notional = round(equity / self.top_n, 2)
                        buy_id   = self._submit_order(replacement, notional, "BUY")
                        if buy_id:
                            trades.append({
                                "symbol": replacement, "side": "BUY", "order_id": buy_id,
                                "notional": notional, "reason": "stop_loss_replace",
                            })
                            held.add(replacement)
                            sector_counts[sector_map.get(replacement, "Unknown")] = (
                                sector_counts.get(sector_map.get(replacement, "Unknown"), 0) + 1
                            )
                            logger.info(f"Replaced {sym} → {replacement}")
                    else:
                        logger.warning(
                            f"No replacement for {sym}: all remaining candidates would breach "
                            f"sector cap ({max_per_sector}/sector) — position left in cash"
                        )

        return trades

    # ── Full rebalance (constrained + score-weighted) ─────────────────────────

    def _rebalance(
        self,
        equity: float,
        scores: list[tuple[str, float]],
        sector_map: dict[str, str],
        bear_market: bool = False,
    ) -> list[dict]:
        # Cancel pending BUY orders so stale buys don't interfere.
        # Leave pending SELL orders intact — they free up capital we need.
        # This mutates real broker state, so it must not run in dry-run mode.
        if not self.dry_run:
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

        # The marker exists to detect a REAL rebalance that got killed mid-flight; a
        # dry run never touches real positions, so it must not leave one behind for the
        # next real run to trip over.
        if not self.dry_run:
            _REBALANCE_MARKER.parent.mkdir(exist_ok=True)
            _REBALANCE_MARKER.write_text(datetime.now().isoformat())

        # Bear-market defensive rule: SPY below its 200-day MA → hold only the top half
        # of positions and invest just 50% of equity, leaving the rest in cash.
        # Re-run _apply_constraints at the SMALLER n directly rather than slicing the
        # top_n-sized result — sector diversity is computed relative to the book size,
        # and blindly taking the top half of an already-diversified 20-name book can
        # land back on just 2-3 sectors, which the half-size book's own 25% cap can't
        # actually support.
        invest_fraction = 1.0
        if bear_market:
            half_n = max(1, self.top_n // 2)
            top_scores = self._apply_constraints(scores, sector_map, n=half_n)
            invest_fraction = 0.5
            logger.warning(
                f"Bear-market filter active (SPY < 200D MA): holding top {half_n} "
                f"positions at {invest_fraction:.0%} equity, rest in cash"
            )
        else:
            top_scores = self._apply_constraints(scores, sector_map)

        new_syms = [s for s, _ in top_scores]
        weights  = self._score_weights(top_scores)
        weights  = self._cap_sector_weights(weights, sector_map)
        weights  = {sym: w * invest_fraction for sym, w in weights.items()}
        logger.info(f"Top {len(new_syms)} (constrained): {new_syms}")

        current = self._current_positions()
        trades  = []

        for sym in list(current):
            if sym not in new_syms:
                order_id = self._close_position(sym)
                if order_id:
                    trades.append({
                        "symbol": sym, "side": "SELL", "order_id": order_id,
                        "notional": current[sym],
                    })

        for sym in new_syms:
            target = equity * weights[sym]
            held   = current.get(sym, 0.0)
            diff   = target - held
            if diff > 5:
                order_id = self._submit_order(sym, diff, "BUY")
                if order_id:
                    trades.append({"symbol": sym, "side": "BUY", "order_id": order_id, "notional": diff})
            elif diff < -5:
                # Overweight survivor from a prior cycle — trim back down to target so
                # position/sector weights don't drift indefinitely across rebalances.
                order_id = self._submit_order(sym, -diff, "SELL")
                if order_id:
                    trades.append({
                        "symbol": sym, "side": "SELL", "order_id": order_id,
                        "notional": -diff, "reason": "trim",
                    })

        _REBALANCE_MARKER.unlink(missing_ok=True)
        return trades

    # ── Main entry ────────────────────────────────────────────────────────────

    def run(self, force_rebalance: bool = False) -> dict:
        today_str = str(_today())
        last_rb = self._last_rebalance_date()
        days_since_rb = (_today() - last_rb).days if last_rb else None

        if force_rebalance:
            rebalance_due = True
        elif last_rb == _today():
            rebalance_due = False
        elif last_rb is None:
            rebalance_due = True
        else:
            rebalance_due = days_since_rb >= self.rebalance_days

        # A machine that only ever wakes launchd before market close (e.g. it sleeps
        # every evening and launchd's catch-up runs land mid-morning) would otherwise
        # defer the full rebalance forever. Once sufficiently overdue, force it through
        # anyway rather than risk indefinite postponement — being late is safer than
        # never rebalancing. (_score_universe/_get_spy_data use
        # _last_completed_session_cutoff() for their data `end`, so this never scores on
        # a still-in-progress trading day even when forced through before close.)
        overdue = days_since_rb is not None and days_since_rb >= self.rebalance_days * self.overdue_multiplier
        market_closed = _market_closed_for_today()

        # The market-close guard must only ever defer the FULL rebalance (which needs a
        # completed day's factor scores) — it must never also block the ATR stop-loss
        # check below, which only needs a live current price (from Alpaca, not
        # yfinance) plus a slow-moving ~20-day ATR. A held position blowing through its
        # stop doesn't stop being a risk just because the clock hasn't hit 16:15 ET yet.
        deferred_rebalance = rebalance_due and not force_rebalance and not market_closed and not overdue
        if deferred_rebalance:
            logger.warning(
                f"Rebalance is due but it's before ET market close + buffer "
                f"({_MARKET_CLOSE_ET.strftime('%H:%M')} ET) — deferring the FULL rebalance "
                "to a later run rather than scoring on an incomplete day's data. The ATR "
                "stop-loss check on current holdings below still runs this cycle "
                "regardless. No 'rebalanced' log row is written today, so "
                "_last_rebalance_date() is unaffected — the next scheduled invocation "
                "picks up correctly, delayed by one run rather than silently trading on "
                "partial data."
            )
            rebalance_due = False
        elif overdue and rebalance_due and not market_closed and not force_rebalance:
            logger.warning(
                f"Rebalance is {days_since_rb} days overdue (>= "
                f"{self.rebalance_days * self.overdue_multiplier:.0f}-day backstop "
                "threshold) despite running before ET market close + buffer — forcing "
                "the full rebalance through anyway to avoid indefinite postponement."
            )

        if _REBALANCE_MARKER.exists():
            logger.warning(
                "Found a stale rebalance-in-progress marker from a previous run that did "
                "not complete cleanly. Pending buys are cancelled and orders resized "
                f"against live positions each run, so this should self-heal: {_REBALANCE_MARKER.read_text()}"
            )

        equity    = self._account_equity()
        trades: list[dict] = []
        rebalanced = False

        if rebalance_due:
            scores, sector_map = self._score_universe()
            spy_close, above_ma = self._get_spy_data()

            trades = self._rebalance(equity, scores, sector_map, bear_market=not above_ma)
            rebalanced = True

        else:
            if deferred_rebalance:
                logger.info("Full rebalance deferred until after close — ATR stop-loss check")
            else:
                days_since = days_since_rb or 0
                logger.info(
                    f"Next rebalance in {self.rebalance_days - days_since} days"
                    " — ATR stop-loss check"
                )
            scores, sector_map = self._load_scores_cache()
            if not scores:
                logger.warning("Cache expired mid-cycle, re-scoring universe (no rebalance)")
                scores, sector_map = self._score_universe()

            spy_close, _ = self._get_spy_data()
            sl_trades = self._check_and_swap_stop_loss(scores, sector_map, equity)
            trades.extend(sl_trades)
            if sl_trades:
                logger.info(f"Stop-loss swaps today: {len(sl_trades) // 2}")

        equity    = self._account_equity()
        positions = self._current_positions()

        order_ids = [t["order_id"] for t in trades if t.get("order_id")]
        verify_counts = self._verify_orders(order_ids) if order_ids else {"rejected": 0, "partial": 0}
        failed_orders = verify_counts["rejected"] + verify_counts["partial"]
        if failed_orders:
            msg = (
                f"{verify_counts['rejected']} rejected/cancelled + {verify_counts['partial']} "
                f"partially filled out of {len(order_ids)} submitted orders — recorded trade "
                "count may not match actual fills, check Alpaca dashboard"
            )
            logger.warning(msg)
            try:
                self._get_notifier().send_risk_alert(f"[alpaca-paper] {msg}")
            except Exception as e:
                logger.warning(f"Could not send order-failure alert: {e}")

        report = {
            "date":          today_str,
            "equity":        round(equity, 2),
            "rebalanced":    rebalanced,
            "holdings":      len(positions),
            "trades":        len(trades),
            "spy_close":     round(spy_close, 2) if spy_close else "",
            "failed_orders": failed_orders,
            "deferred_rebalance": deferred_rebalance,
        }
        if self.dry_run:
            logger.info("[DRY-RUN] not writing to alpaca_paper_log.csv / alpaca_last_run.txt — "
                        "a dry run must not affect whether the next real run considers a "
                        "rebalance already done")
        else:
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
                    "n_holdings", "n_trades", "holdings", "failed_orders",
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
                report.get("failed_orders", 0),
            ])
