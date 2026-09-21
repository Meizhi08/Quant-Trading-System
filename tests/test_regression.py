"""
Regression tests for the bug fixes made to the trading-critical code paths.

Run with pytest if available:  pytest tests/test_regression.py -v
Or standalone (no pytest needed):  python tests/test_regression.py

These are not a full test suite for the project — they exist to pin down the
specific bugs found in review so a future change can't silently reintroduce them.

NOTE: while writing the TradingView-failure tests below, we discovered that
data/tv_signals.py — imported by strategy/composite.py, strategy/unified.py and
main.py's `signal`/`scan` commands — did not exist anywhere in this checkout. Every
`use_tv=True` call (the default for live trading) had therefore always hit the
except-branch in _tv_vote/_tv_score, silently as a fake "neutral" reading before this
review's fixes (now correctly flagged as ok=False instead). data/tv_signals.py has
since been implemented (tradingview-ta backed); the TV-failure tests below still use an
injected fake module rather than the real one, so they stay meaningful and fast
regardless of network access or future changes to that implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── factor/engine.py: Spearman tie-handling + RSI flat-case ────────────────────

def test_spearman_perfect_monotonic():
    from factor.engine import _spearman
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert abs(_spearman(x, y) - 1.0) < 1e-9
    assert abs(_spearman(x, y[::-1]) + 1.0) < 1e-9


def test_spearman_ties_bounded_and_finite():
    from factor.engine import _spearman
    x = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 3.0])
    y = np.array([5.0, 4.0, 6.0, 1.0, 2.0, 3.0])
    ic = _spearman(x, y)
    assert np.isfinite(ic) and -1.0 <= ic <= 1.0


def test_spearman_all_tied_x_returns_zero():
    from factor.engine import _spearman
    x = np.array([1.0, 1.0, 1.0, 1.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert _spearman(x, y) == 0.0


def test_rsi_flat_price_is_neutral_not_bearish():
    """Regression: a stock with zero price movement used to score -1.0 (max overbought)."""
    from factor.engine import FactorEngine
    eng = FactorEngine()
    flat = pd.DataFrame({
        "open": [100.0] * 70, "high": [100.0] * 70,
        "low": [100.0] * 70, "close": [100.0] * 70, "volume": [1000] * 70,
    })
    assert eng._rsi_score(flat) == 0.0


def test_rsi_genuine_uptrend_still_scores_overbought():
    """The flat-case fix must not affect the intended mean-reversion signal."""
    from factor.engine import FactorEngine
    eng = FactorEngine()
    up = pd.DataFrame({
        "open": [100.0 + i for i in range(20)], "high": [100.0 + i for i in range(20)],
        "low": [100.0 + i for i in range(20)], "close": [100.0 + i for i in range(20)],
        "volume": [1000] * 20,
    })
    assert eng._rsi_score(up) == -1.0


# ── backtest/metrics.py: Sortino ratio ──────────────────────────────────────────

def test_sortino_finite_on_mixed_returns():
    from backtest.metrics import BacktestMetrics
    rng = np.random.default_rng(1)
    rets = rng.normal(0.0008, 0.012, 756)
    equity = pd.Series(100000 * np.cumprod(1 + rets))
    m = BacktestMetrics.from_equity_curve(equity, trades=[])
    assert np.isfinite(m.sortino_ratio)


def test_sortino_zero_downside_deviation_does_not_crash():
    """Regression: every return above the tiny daily risk-free target -> true zero
    downside deviation must fall back to 0.0, not raise ZeroDivisionError / inf."""
    from backtest.metrics import BacktestMetrics
    equity = pd.Series(100000 * np.cumprod(1 + np.full(300, 0.01)))
    m = BacktestMetrics.from_equity_curve(equity, trades=[])
    assert m.sortino_ratio == 0.0


# ── paper_trading/alpaca_runner.py: sector / per-symbol weight caps ────────────

def test_cap_sector_weights_enforces_both_caps():
    from paper_trading.alpaca_runner import AlpacaPaperRunner
    runner = AlpacaPaperRunner.__new__(AlpacaPaperRunner)
    runner.top_n = 20
    runner.max_sector_pct = 0.25
    runner.max_position_pct = 0.30

    weights = {
        "T1": 0.15, "T2": 0.14, "T3": 0.13, "T4": 0.12, "T5": 0.11,
        "H1": 0.06, "H2": 0.05, "F1": 0.05, "F2": 0.04,
        "E1": 0.03, "E2": 0.03, "E3": 0.02, "E4": 0.02, "E5": 0.05,
    }
    sector_map = {
        **{f"T{i}": "Tech" for i in range(1, 6)},
        **{f"H{i}": "Health" for i in range(1, 3)},
        **{f"F{i}": "Fin" for i in range(1, 3)},
        **{f"E{i}": "Energy" for i in range(1, 6)},
    }
    out = runner._cap_sector_weights(weights, sector_map)

    sector_totals: dict[str, float] = {}
    for sym, w in out.items():
        sector_totals[sector_map[sym]] = sector_totals.get(sector_map[sym], 0.0) + w

    assert abs(sum(out.values()) - 1.0) < 1e-6
    for total in sector_totals.values():
        assert total <= runner.max_sector_pct + 1e-6
    for w in out.values():
        assert w <= runner.max_position_pct + 1e-6


def test_cap_sector_weights_thin_pool_holds_sector_cap():
    """
    Regression (round 2): with only 2 sectors present, capping Tech's excess used to
    dump the ENTIRE excess onto the single 'Other' stock — the per-symbol cap caught
    that specific case, but the sector cap itself (Tech ended at 70% vs a 25% limit)
    was still silently broken. A 25% cap can only ever be honored by >=4 sectors
    summing to 100%; with just 2 sectors present it is mathematically impossible to
    stay fully invested AND respect the cap, so the correct outcome is to leave the
    unplaceable half as cash — never to breach the limit that exists to bound risk.
    """
    from paper_trading.alpaca_runner import AlpacaPaperRunner
    runner = AlpacaPaperRunner.__new__(AlpacaPaperRunner)
    runner.top_n = 20
    runner.max_sector_pct = 0.25
    runner.max_position_pct = 0.30

    weights = {"T1": 0.30, "T2": 0.20, "T3": 0.15, "T4": 0.10, "T5": 0.10, "O1": 0.15}
    sector_map = {"T1": "Tech", "T2": "Tech", "T3": "Tech", "T4": "Tech", "T5": "Tech", "O1": "Other"}
    out = runner._cap_sector_weights(weights, sector_map)

    sector_totals: dict[str, float] = {}
    for sym, w in out.items():
        sector_totals[sector_map[sym]] = sector_totals.get(sector_map[sym], 0.0) + w

    # The cap must hold for BOTH sectors — this is the assertion the first version of
    # this test was missing, which let a 70%-in-one-sector result pass silently.
    for sec, total in sector_totals.items():
        assert total <= runner.max_sector_pct + 1e-6, f"{sec} sector cap breached: {total:.2%}"
    for w in out.values():
        assert w <= runner.max_position_pct + 1e-6
    # Capital must never be invented out of thin air; being < 1.0 here (held as cash)
    # is the correct, intentional outcome for a jointly-infeasible constraint.
    assert sum(out.values()) <= 1.0 + 1e-6


def test_cap_group_infeasible_constraint_holds_cap_leaves_shortfall_as_cash():
    """
    Regression (round 2): the round-1 fix made this conserve total weight by forcing
    the excess onto already-capped groups — which silently breached the cap it was
    supposed to enforce (exactly the bug caught in the thin-pool case above). The cap
    is the invariant that must hold; leaving the excess unallocated (implicit cash) is
    the correct behavior when 2 groups can't jointly satisfy a 25% cap while summing
    to 1.0. (The redistribution step only ever subtracts-then-redistributes what was
    taken, so "gaining" capital isn't a distinct failure mode this needs to separately
    guard against — the sum check below is just an ordinary sanity check, not the point
    of this test.)
    """
    from paper_trading.alpaca_runner import AlpacaPaperRunner
    weights = {"A": 0.5, "B": 0.5}
    group_map = {"A": "G1", "B": "G2"}
    out = AlpacaPaperRunner._cap_group(weights, group_map, max_pct=0.25)

    group_totals: dict[str, float] = {}
    for sym, w in out.items():
        group_totals[group_map[sym]] = group_totals.get(group_map[sym], 0.0) + w
    for g, total in group_totals.items():
        assert total <= 0.25 + 1e-6, f"{g} cap breached: {total:.2%}"
    assert sum(out.values()) <= sum(weights.values()) + 1e-9


def test_apply_constraints_respects_smaller_n_directly():
    """
    Regression (root cause of the thin-pool bug): bear-market mode used to build the
    normal top_n=20-sized diversified list, then slice its top half BY SCORE. If the
    highest scores all happen to cluster in one sector (plausible — that's exactly
    when a sector is "hot"), the sliced half can be far less diversified than a
    half-sized book's own cap allows. Calling _apply_constraints with n=half_n
    directly (instead of slicing afterward) must produce a list that is itself
    diversified relative to that smaller n.
    """
    from paper_trading.alpaca_runner import AlpacaPaperRunner
    runner = AlpacaPaperRunner.__new__(AlpacaPaperRunner)
    runner.top_n = 20
    runner.max_sector_pct = 0.25

    # Realistic proportions (closer to actual GICS sector sizes in the S&P 500): 8
    # sectors, 15-25 names each, Tech scoring highest across the board (a "hot sector"
    # scenario) but every sector has plenty of sector-diverse candidates available —
    # the universe itself is NOT the bottleneck here, unlike a truly thin universe.
    sectors = ["Tech", "Health", "Fin", "Energy", "Industrials", "Consumer", "Utilities", "Materials"]
    scores: list[tuple[str, float]] = []
    sector_map: dict[str, str] = {}
    for si, sec in enumerate(sectors):
        n_members = 20
        for i in range(n_members):
            sym = f"{sec[:2].upper()}{i}"
            # Tech scores highest overall; later sectors score progressively lower.
            score = 1.0 - si * 0.1 - i * 0.001
            scores.append((sym, score))
            sector_map[sym] = sec

    half_n = 10
    selected = runner._apply_constraints(scores, sector_map, n=half_n)
    sector_counts: dict[str, int] = {}
    for sym, _ in selected:
        sector_counts[sector_map[sym]] = sector_counts.get(sector_map[sym], 0) + 1

    max_per_sector = max(1, int(half_n * runner.max_sector_pct))  # = 2 at n=10
    assert len(selected) == half_n
    for sec, cnt in sector_counts.items():
        assert cnt <= max_per_sector, f"{sec} has {cnt} > {max_per_sector} allowed at n={half_n}"
    assert len(sector_counts) >= 4, (
        f"only {len(sector_counts)} distinct sectors in the n={half_n} selection — "
        "with 8 well-populated sectors available this should never need to concentrate"
    )
    # Contrast: naively taking scores.sort()[:10] (the old bear-market slicing) would
    # have put all 10 slots in Tech, since Tech's 20 names are exactly the top 20
    # scorers — this test fails against that old behavior and passes against the fix.


# ── paper_trading/alpaca_runner.py: order verification + dry-run ──────────────

def test_verify_orders_detects_rejected_and_partial_fills():
    from paper_trading.alpaca_runner import AlpacaPaperRunner

    class _FakeOrder:
        def __init__(self, status, symbol="AAPL", filled_qty=None, qty=None):
            self.status = status
            self.symbol = symbol
            self.filled_qty = filled_qty
            self.qty = qty

    class _FakeClient:
        def __init__(self, orders):
            self._orders = orders

        def get_order_by_id(self, oid):
            return self._orders[oid]

    orders = {
        "ok1":          _FakeOrder("filled", filled_qty="10", qty="10"),
        "rejected1":    _FakeOrder("rejected"),
        "cancelled1":   _FakeOrder("canceled"),
        "explicit_partial": _FakeOrder("partially_filled", filled_qty="4", qty="10"),
        # Regression: some brokers report "filled" once no more of the order will
        # execute, even when filled_qty < requested — a status-string-only check
        # (the round-1 implementation) would have silently counted this as success.
        "silent_underfill": _FakeOrder("filled", filled_qty="7", qty="10"),
    }
    runner = AlpacaPaperRunner.__new__(AlpacaPaperRunner)
    runner.client = _FakeClient(orders)
    counts = runner._verify_orders(list(orders.keys()))
    assert counts["rejected"] == 2       # rejected + cancelled
    assert counts["partial"] == 2        # explicit_partial + silent_underfill


def test_verify_orders_ignores_dry_run_and_missing_ids():
    from paper_trading.alpaca_runner import AlpacaPaperRunner

    class _BoomClient:
        def get_order_by_id(self, oid):
            raise AssertionError("must not look up dry-run/n-a/empty order ids")

    runner = AlpacaPaperRunner.__new__(AlpacaPaperRunner)
    runner.client = _BoomClient()
    counts = runner._verify_orders(["dry-run", "n/a", "", None])
    assert counts == {"rejected": 0, "partial": 0}


def test_run_defers_full_rebalance_but_still_runs_stop_loss_check():
    """
    Regression: the market-close guard originally returned early from run() entirely
    whenever a rebalance was due but the market hadn't closed yet — which also skipped
    the ATR stop-loss check on already-held positions. A held position blowing through
    its stop is a risk regardless of whether today's factor scores are complete, so the
    stop-loss check must run every cycle; only the FULL rebalance (which does need a
    completed day's data) gets deferred.
    """
    from datetime import date, timedelta
    from unittest.mock import patch
    from paper_trading.alpaca_runner import AlpacaPaperRunner

    runner = AlpacaPaperRunner.__new__(AlpacaPaperRunner)
    runner.rebalance_days = 30
    runner.overdue_multiplier = 1.5

    # 35 days >= rebalance_days(30) so a rebalance IS due, but 35 < 30*1.5=45 so it is
    # NOT yet overdue -> must defer the full rebalance, not force it through.
    with patch.object(AlpacaPaperRunner, "_last_rebalance_date", return_value=date.today() - timedelta(days=35)), \
         patch("paper_trading.alpaca_runner._market_closed_for_today", return_value=False), \
         patch.object(AlpacaPaperRunner, "_account_equity", return_value=100_000.0), \
         patch.object(AlpacaPaperRunner, "_current_positions", return_value={}), \
         patch.object(AlpacaPaperRunner, "_load_scores_cache", return_value=([("AAPL", 0.5)], {"AAPL": "Tech"})), \
         patch.object(AlpacaPaperRunner, "_get_spy_data", return_value=(500.0, True)), \
         patch.object(AlpacaPaperRunner, "_check_and_swap_stop_loss", return_value=[]) as mock_stop_loss, \
         patch.object(AlpacaPaperRunner, "_rebalance") as mock_rebalance, \
         patch.object(AlpacaPaperRunner, "_verify_orders", return_value={"rejected": 0, "partial": 0}), \
         patch.object(AlpacaPaperRunner, "_save_log"):
        report = runner.run()

    assert mock_stop_loss.called, "stop-loss check must still run even though the full rebalance is deferred"
    assert not mock_rebalance.called, "the full rebalance itself must be deferred, not forced through"
    assert report["deferred_rebalance"] is True
    assert report["rebalanced"] is False


def test_run_forces_full_rebalance_when_overdue_despite_market_still_open():
    """Contrast case: once past the overdue backstop, the full rebalance is forced
    through even though the market hasn't closed (data `end` cutoffs inside
    _score_universe/_get_spy_data are what actually keep this safe, tested
    separately via _last_completed_session_cutoff's usage)."""
    from datetime import date, timedelta
    from unittest.mock import patch
    from paper_trading.alpaca_runner import AlpacaPaperRunner

    runner = AlpacaPaperRunner.__new__(AlpacaPaperRunner)
    runner.rebalance_days = 30
    runner.overdue_multiplier = 1.5

    # 50 days >= 30*1.5=45 day backstop -> overdue, must force the rebalance through.
    with patch.object(AlpacaPaperRunner, "_last_rebalance_date", return_value=date.today() - timedelta(days=50)), \
         patch("paper_trading.alpaca_runner._market_closed_for_today", return_value=False), \
         patch.object(AlpacaPaperRunner, "_account_equity", return_value=100_000.0), \
         patch.object(AlpacaPaperRunner, "_current_positions", return_value={}), \
         patch.object(AlpacaPaperRunner, "_score_universe", return_value=([("AAPL", 0.5)], {"AAPL": "Tech"})), \
         patch.object(AlpacaPaperRunner, "_get_spy_data", return_value=(500.0, True)), \
         patch.object(AlpacaPaperRunner, "_rebalance", return_value=[]) as mock_rebalance, \
         patch.object(AlpacaPaperRunner, "_check_and_swap_stop_loss") as mock_stop_loss, \
         patch.object(AlpacaPaperRunner, "_verify_orders", return_value={"rejected": 0, "partial": 0}), \
         patch.object(AlpacaPaperRunner, "_save_log"):
        report = runner.run()

    assert mock_rebalance.called, "overdue rebalance must be forced through despite market not being closed"
    assert not mock_stop_loss.called
    assert report["rebalanced"] is True
    assert report["deferred_rebalance"] is False


def test_last_completed_session_cutoff_avoids_in_progress_trading_day():
    """The `end` date used for scoring must be yesterday when the market hasn't closed
    yet, and today once it has — this is what keeps an overdue-forced rebalance (which
    can run before close) from fetching a still-in-progress day's bar."""
    from datetime import timedelta
    from unittest.mock import patch
    import paper_trading.alpaca_runner as ar

    with patch.object(ar, "_market_closed_for_today", return_value=False):
        assert ar._last_completed_session_cutoff() == ar._today() - timedelta(days=1)
    with patch.object(ar, "_market_closed_for_today", return_value=True):
        assert ar._last_completed_session_cutoff() == ar._today()


def test_stop_loss_skips_symbol_with_unusable_current_price():
    """A NaN/invalid unrealized_plpc from Alpaca must not silently look like
    'not triggered' (NaN comparisons are always False in Python) — the position should
    be skipped and logged instead, and a genuinely triggered stop-loss on a DIFFERENT
    position in the same batch must still fire normally."""
    from unittest.mock import patch
    from paper_trading.alpaca_runner import AlpacaPaperRunner

    runner = AlpacaPaperRunner.__new__(AlpacaPaperRunner)
    runner.stop_loss_pct = 0.15
    runner.atr_multiplier = 2.5
    runner.top_n = 20
    runner.max_sector_pct = 0.25

    positions = {
        "BAD": {"market_value": 1000.0, "avg_entry_price": 100.0, "unrealized_plpc": float("nan")},
        "OK":  {"market_value": 1000.0, "avg_entry_price": 100.0, "unrealized_plpc": -0.30},
    }

    with patch.object(AlpacaPaperRunner, "_current_positions_detail", return_value=positions), \
         patch.object(AlpacaPaperRunner, "_close_position", return_value="order-id"), \
         patch.object(AlpacaPaperRunner, "_submit_order", return_value=None):
        trades = runner._check_and_swap_stop_loss(scores=[], sector_map={}, equity=100_000.0)

    closed_symbols = {t["symbol"] for t in trades if t["side"] == "SELL"}
    assert "BAD" not in closed_symbols, "NaN current_price must not be evaluated at all, not just fail to trigger"
    assert "OK" in closed_symbols, "a genuinely triggered stop-loss on another position must still fire"


def test_dry_run_never_touches_the_broker_client():
    from paper_trading.alpaca_runner import AlpacaPaperRunner

    class _BoomClient:
        def submit_order(self, *a, **k):
            raise AssertionError("submit_order must not be called in dry-run mode")

        def close_position(self, *a, **k):
            raise AssertionError("close_position must not be called in dry-run mode")

    runner = AlpacaPaperRunner.__new__(AlpacaPaperRunner)
    runner.dry_run = True
    runner.client = _BoomClient()
    assert runner._submit_order("AAPL", 1000.0, "BUY") == "dry-run"
    assert runner._close_position("AAPL") == "dry-run"


# ── strategy/{composite,unified}.py: TradingView failure must not look neutral ──

def _fake_tv_signals_module(raise_error: bool = True):
    """
    Build a fake data.tv_signals module and inject it via sys.modules, so the lazy
    `from data.tv_signals import get_tv_signal` inside _tv_vote/_tv_score resolves to it
    regardless of whether a real data/tv_signals.py exists on disk. Note: at the time
    this test suite was written, data/tv_signals.py does NOT exist in this checkout at
    all (see the note in the module docstring) — every real call currently hits this
    same except-branch for real. This helper keeps the test meaningful independent of
    whether that gets fixed later.
    """
    import sys
    import types

    mod = types.ModuleType("data.tv_signals")
    if raise_error:
        def _boom(symbol, interval="1d"):
            raise RuntimeError("simulated API outage")
        mod.get_tv_signal = _boom
    else:
        mod.get_tv_signal = lambda symbol, interval="1d": {
            "score": 0.0, "recommendation": "NEUTRAL", "buy": 0, "neutral": 1, "sell": 0,
        }
    return sys, mod


def test_composite_tv_vote_failure_is_flagged_not_silent_neutral():
    from unittest.mock import patch
    from strategy.composite import CompositeStrategy

    sys_mod, fake = _fake_tv_signals_module(raise_error=True)
    strat = CompositeStrategy(use_tv=True)
    with patch.dict(sys_mod.modules, {"data.tv_signals": fake}):
        sig, score, ok = strat._tv_vote("AAPL")
    assert ok is False, "a failed fetch must report ok=False, not masquerade as a real neutral read"


def test_unified_tv_score_failure_is_flagged_not_silent_neutral():
    from unittest.mock import patch
    from strategy.unified import UnifiedStrategy

    sys_mod, fake = _fake_tv_signals_module(raise_error=True)
    strat = UnifiedStrategy(use_tv=True)
    with patch.dict(sys_mod.modules, {"data.tv_signals": fake}):
        score, rec, ok = strat._tv_score("AAPL")
    assert ok is False


# ── strategy/factor_strategy.py: fundamentals fetch failure degrades gracefully ─

def test_factor_strategy_fundamentals_failure_degrades_to_empty_dict():
    from strategy.factor_strategy import FactorStrategy

    class _FailingFetcher:
        def get_fundamentals(self, symbol):
            raise RuntimeError("simulated API outage")

    strat = FactorStrategy()
    strat._fetcher = _FailingFetcher()
    assert strat._get_fundamentals("AAPL") == {}
    # Cached — a second call for the same symbol must not raise again either.
    assert strat._get_fundamentals("AAPL") == {}


# ── data/tv_signals.py: score derivation + exchange resolution ─────────────────
# Mocks tradingview_ta.TA_Handler so these stay fast/deterministic and don't depend on
# network access or TradingView actually having data for a given symbol right now.

def _clear_tv_cache(tv_signals, symbol: str, interval: str) -> None:
    """The module-level cache is disk-backed and persists across test runs — clear the
    specific key first so a test doesn't silently pass on a stale cached result from a
    previous run within the 5-minute TTL instead of actually exercising the mock."""
    path = tv_signals._cache._key("tv_signal", symbol=symbol, interval=interval)
    path.unlink(missing_ok=True)


def test_get_tv_signal_computes_score_from_vote_counts():
    from unittest.mock import patch, MagicMock
    import data.tv_signals as tv_signals

    _clear_tv_cache(tv_signals, "ZZZTEST1", "1d")
    fake_analysis = MagicMock()
    fake_analysis.summary = {"RECOMMENDATION": "BUY", "BUY": 15, "SELL": 2, "NEUTRAL": 9}

    with patch.object(tv_signals, "TA_Handler") as MockHandler:
        MockHandler.return_value.get_analysis.return_value = fake_analysis
        result = tv_signals.get_tv_signal("ZZZTEST1", "1d")

    assert result["recommendation"] == "BUY"
    assert result["buy"] == 15 and result["sell"] == 2 and result["neutral"] == 9
    # score = (buy - sell) / total = (15-2)/26
    assert abs(result["score"] - (15 - 2) / 26) < 1e-9


def test_get_tv_signal_falls_back_across_exchanges():
    """Regression-guard: the first exchange tried (NASDAQ) can raise for an NYSE-listed
    symbol — get_tv_signal must try the next exchange rather than propagate that."""
    from unittest.mock import patch, MagicMock
    import data.tv_signals as tv_signals

    tv_signals._exchange_cache.pop("ZZZTEST2", None)
    _clear_tv_cache(tv_signals, "ZZZTEST2", "1d")
    fake_analysis = MagicMock()
    fake_analysis.summary = {"RECOMMENDATION": "NEUTRAL", "BUY": 5, "SELL": 5, "NEUTRAL": 16}

    call_log = []

    def _handler_side_effect(*, symbol, screener, exchange, interval, timeout=None):
        call_log.append(exchange)
        h = MagicMock()
        if exchange == "NASDAQ":
            h.get_analysis.side_effect = Exception("Exchange or symbol not found.")
        else:
            h.get_analysis.return_value = fake_analysis
        return h

    with patch.object(tv_signals, "TA_Handler", side_effect=_handler_side_effect):
        result = tv_signals.get_tv_signal("ZZZTEST2", "1d")

    assert call_log[0] == "NASDAQ"          # tried first, as documented
    assert "NYSE" in call_log               # fell through to the next exchange
    assert result["recommendation"] == "NEUTRAL"
    assert tv_signals._exchange_cache.get("ZZZTEST2") == "NYSE"  # cached for next time


def test_get_tv_signal_cache_hit_skips_the_network_call():
    """A second call for the same (symbol, interval) within the 5-minute TTL must not
    hit TA_Handler again — this is what keeps a 60s-interval live scan loop from
    re-fetching the same symbol on every tick."""
    from unittest.mock import patch, MagicMock
    import data.tv_signals as tv_signals

    _clear_tv_cache(tv_signals, "ZZZTEST3", "1d")
    fake_analysis = MagicMock()
    fake_analysis.summary = {"RECOMMENDATION": "SELL", "BUY": 3, "SELL": 12, "NEUTRAL": 11}

    with patch.object(tv_signals, "TA_Handler") as MockHandler:
        MockHandler.return_value.get_analysis.return_value = fake_analysis
        first = tv_signals.get_tv_signal("ZZZTEST3", "1d")
        calls_after_first = MockHandler.call_count
        second = tv_signals.get_tv_signal("ZZZTEST3", "1d")
        calls_after_second = MockHandler.call_count

    assert first == second
    assert calls_after_first >= 1
    assert calls_after_second == calls_after_first, "second call should have hit the cache, not TA_Handler again"


def test_get_tv_signal_propagates_failure_when_every_exchange_fails():
    """If no exchange resolves, the caller (composite/unified's _tv_vote/_tv_score) must
    actually see an exception — that's the signal they rely on to set ok=False rather
    than silently treating a total failure as neutral."""
    import data.tv_signals as tv_signals
    from unittest.mock import patch, MagicMock

    _clear_tv_cache(tv_signals, "ZZZTEST4", "1d")
    tv_signals._exchange_cache.pop("ZZZTEST4", None)

    def _always_fails(**kwargs):
        h = MagicMock()
        h.get_analysis.side_effect = Exception("Exchange or symbol not found.")
        return h

    raised = False
    with patch.object(tv_signals, "TA_Handler", side_effect=_always_fails):
        try:
            tv_signals.get_tv_signal("ZZZTEST4", "1d")
        except Exception:
            raised = True
    assert raised, "get_tv_signal must raise, not silently return a fake reading, when every exchange fails"


def test_get_tv_signal_resolves_tsx_suffix_to_canada_screener():
    """'.TO' tickers (the TSX universe this system also supports) must route to the
    Canada screener/TSX exchange with the suffix stripped, not be tried as a US ticker."""
    from unittest.mock import patch, MagicMock
    import data.tv_signals as tv_signals

    _clear_tv_cache(tv_signals, "TD.TO", "1d")
    fake_analysis = MagicMock()
    fake_analysis.summary = {"RECOMMENDATION": "BUY", "BUY": 10, "SELL": 4, "NEUTRAL": 12}

    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        h = MagicMock()
        h.get_analysis.return_value = fake_analysis
        return h

    with patch.object(tv_signals, "TA_Handler", side_effect=_capture):
        result = tv_signals.get_tv_signal("TD.TO", "1d")

    assert captured["screener"] == "canada"
    assert captured["exchange"] == "TSX"
    assert captured["symbol"] == "TD"
    assert result["recommendation"] == "BUY"


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
