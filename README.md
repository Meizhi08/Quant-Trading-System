# Quantitative Factor Investing System

A fully automated equity selection system that scores all ~500 S&P 500 stocks using a 12-factor model, applies sector concentration limits, and rebalances to the top 30 holdings via the **Alpaca Paper Trading API** submitting real market orders automatically after each market close.

## Live Paper Trading Results (Apr 30 – May 6, 2026)

| Metric | Strategy | SPY Benchmark |
| :--- | :--- | :--- |
| Starting Capital | $100,000 | $100,000 |
| Current NAV | $107,154 | $100,570 (est.) |
| Total Return | **+7.15%** | +0.57% |
| Excess Return | **+6.58%** | |
| Rebalance Method | Score weighted top 30 | Buy and hold |

## How It Works

Every 30 calendar days the system rescores the full S&P 500 universe and rebalances. Between rebalances it checks each position **daily** for ATR-based stop losses and replaces triggered positions without waiting for the next cycle.

## Factor Model (12 Factors)

Weights are calibrated via rolling **Rank IC** — factors with higher recent predictive power receive more weight automatically.

| Factor | Weight | Category | What It Captures |
| :--- | :--- | :--- | :--- |
| Momentum 60d | 21% | Momentum | 3 months price trend |
| Momentum 20d | 13% | Momentum | 1 month price trend |
| ROE Score | 12% | Quality | Profitability vs 10% baseline |
| Growth Score | 10% | Quality | Earnings growth rate |
| MA Alignment | 9% | Structure | Bullish moving average stack |
| RSI Score | 7% | Mean revert | Oversold bounce signal |
| PB Score | 6% | Valuation | Low price to book preference |
| Debt Score | 6% | Quality | Low leverage preference |
| Vol Ratio | 5% | Volume | Recent volume surge |
| Price Position | 5% | Structure | 60 day price percentile |
| Vol Trend | 3% | Volume | Volume momentum |
| Momentum 5d | -3% | Mean revert | Short term reversal (contrarian) |

**Constraints:** sector concentration capped at 25% per GICS sector (and a per-symbol cap, default 30% of equity) · share class deduplication (e.g. GOOG/GOOGL) · score proportional position sizing

**When a cap can't be honored while staying fully invested** (e.g. the day's top scorers cluster too heavily in one sector, or a reduced bear-market book doesn't have enough sector-diverse candidates left over): the system holds the un-placeable fraction as cash rather than either breaching the cap or force-filling with lower-scored, less-diverse names. This is a deliberate choice — respecting the risk limit takes priority over staying fully deployed. It means the strategy can under-invest relative to a version that ignored the cap; a "reduce position count instead of holding cash" or "force diversity earlier in stock selection" variant would have different risk/return tradeoffs and hasn't been implemented.

**Unattended operation:** `alpaca-paper` is designed to run once per trading day after market close (see `setup_cron.sh`/launchd). If a rebalance is due but it's before 16:15 ET, it **defers only the full rebalance** (which needs a completed day's factor scores) and logs a warning — the ATR stop-loss check on current holdings still runs that same cycle regardless. `--force-rebalance` bypasses the deferral for manual runs. If a machine's wake pattern means it only ever catches up before close (e.g. it sleeps every evening), the guard backs off once a rebalance is more than `rebalance_days × overdue_multiplier` days overdue (`--overdue-multiplier`, default 1.5 — **a multiple of `rebalance_days`, not a fixed day count**: with `--rebalance-days 7` the threshold is 10.5 days, not 45) and forces the full rebalance through anyway rather than postponing indefinitely — universe/SPY data fetches always cut off at the most recently *completed* session in that case (see `_last_completed_session_cutoff()`), never a still-in-progress one.

**Two different "current price" sources, on purpose:** the ATR stop-loss check compares against Alpaca's own live position data (`client.get_all_positions()`'s `unrealized_plpc`, marked to Alpaca's latest quote during market hours) — it can react intraday. The factor scores driving a full rebalance are computed from `yfinance` daily bars as of the last **completed** session — they only update once a day, after close. This is why the two checks can be decoupled the way they are above: the stop-loss's price input is already close to real-time regardless of when in the day it runs; the rebalance's score input is not.

Order-rejection and risk alerts go out via `alert/notifier.py`, which is a **no-op** unless `ALERT_EMAIL_FROM/TO/PASSWORD` are set in `.env` — **the command still runs and still trades even if notifications aren't configured**; it just means failures only ever reach the log file, not a person. `alpaca-paper` prints a warning at startup if this is the case, but does not refuse to run — decide deliberately whether that's acceptable for your setup rather than relying on the printed warning to catch it every time.

`--dry-run` scores and sizes a full rebalance without submitting any orders, for testing changes safely.

**First-time deployment check:** before trusting this to launchd, verify the environment end-to-end once, manually — **using the same Python interpreter launchd will actually use** (check `ProgramArguments[0]` in the `.plist`; on macOS this is often the Xcode Command Line Tools' bundled Python, e.g. `/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3`, not whatever `python`/`python3` resolves to in an interactive shell — a conda `base` env or a separately-installed newer Python will silently diverge from what runs unattended):
```bash
# Recommended: a dedicated env matching launchd's interpreter version, so `python`
# resolves correctly without touching conda base or needing the full interpreter path
conda create -n quant-trading python=3.9 -y
conda activate quant-trading
pip install -r requirements.txt
# .env needs ALPACA_API_KEY / ALPACA_SECRET_KEY (and ALERT_EMAIL_* if you want real alerts)
python main.py alpaca-paper --dry-run --force-rebalance
```
This exercises the full scoring → constraint → sizing pipeline against your real environment and credentials without placing any orders. `tests/test_regression.py` (unit-level, no network) is not a substitute for this — it doesn't verify dependencies are installed correctly or that `.env` is readable. `--dry-run` also skips writing to `alpaca_paper_log.csv` / `alpaca_last_run.txt` — a dry run must never affect whether the next real run considers a rebalance already done.

## 10-Year Out-of-Sample Backtest (2015–2026, 497 stocks)

| Period | Strategy | SPY | Alpha | Sharpe | Max DD |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2015-01 to 2018-10 | +55.6% | +45.6% | +9.9% | 1.00 | -13.2% |
| 2018-10 to 2022-07 | -3.9% | +45.4% | -49.3% | 0.05 | -31.1% |
| 2022-07 to 2026-05 | +108.2% | +93.3% | +14.9% | 1.25 | -15.0% |
| **Total (compounded)** | +211.2% | +309.4% | -98.2% | | |

Rebalance: every 20 trading days SPY 200day MA bear-market filter Transaction cost: 0.1% per side

**Known limitation — survivorship bias:** the universe is built from the *current* S&P 500 constituent list applied across the entire 2015–2026 window (`data/stock_selector.py::get_sp500_symbols`). Stocks that were removed or delisted before today never appear in the backtest at any point in their history, and stocks that joined the index recently are backtested as if held since 2015. **This means the returns/Sharpe/alpha above are an optimistic upper bound: live trading should be expected to underperform this backtest, direction only — the exact magnitude has not been quantified** (doing so would require rerunning the backtest against a point-in-time historical constituent list, which we don't have a reliable free source for).

### Why Period 2 Underperformed and What Changed

The SPY 200 day MA filter correctly moved to cash during the 2018 and 2020 crashes, but V shaped recoveries in 2019 and 2020–2021 happened faster than the moving average could confirm, the strategy sat in cash through the strongest months of both rebounds. The 2019–2021 market was also dominated by loss making growth stocks that scored poorly on quality factors.

**Fix:** when SPY drops below its 200 day MA, the system now holds 50% of the portfolio (the highest scored 15 positions) instead of going fully to cash. This limits crash exposure while preserving participation in fast recoveries.

## Tech Stack

- **Data:** `yfinance`, `akshare` · **Broker API:** Alpaca Markets (paper trading)
- **Factor engine:** `pandas`, `numpy`, `scipy`
- **Scheduling:** macOS `launchd` daemon
- **Dashboard:** TradingView Lightweight Charts (self contained HTML)
- **Signal confirmation:** TradingView aggregated technical rating (`tradingview-ta`, `data/tv_signals.py`) — used by `composite`/`unified` strategies in live signal generation (`main.py live`/`signal`/`scan`), not by the factor scan/backtest paths below. It's an unofficial scraping-based API with no documented rate limit; keep watchlists passed to `scan --symbols` to a reasonable size (tens, not hundreds) rather than the full S&P 500 universe.
- **Risk report:** CAPM regression, Sharpe, Sortino, max drawdown

## Key Commands

Run from the project root with the `quant-trading` conda env active — the system/base python is missing dependencies (`loguru`, etc.), so bare `python main.py ...` only works after `conda activate quant-trading`:

```bash
cd Quant-Trading-System
conda activate quant-trading

# Daily rebalance / stop-loss check (needs ALPACA_API_KEY / ALPACA_SECRET_KEY in .env)
python main.py alpaca-paper

# Force full rebalance today / score & size without submitting orders
python main.py alpaca-paper --force-rebalance
python main.py alpaca-paper --dry-run

# Portfolio status & daemon health (local log only, no network, no credentials)
python main.py alpaca-status
python main.py alpaca-health

# Performance dashboard (needs ≥2 days of alpaca-paper logs; --no-open skips the browser)
python main.py alpaca-dashboard

# Risk report — CAPM alpha/beta, Sharpe, Sortino, drawdown (needs ≥5 days of logs)
python main.py risk-report

# 10-year backtest, full S&P 500 universe like the table above (~20-30 min);
# drop --no-sample for a faster 80-stock sampled run
python main.py factor-backtest --start 2015-01-01 --end 2026-01-01 --no-sample
```

