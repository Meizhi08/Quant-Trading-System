# Quantitative Factor Investing System

A fully automated equity selection system that scores all ~500 S&P 500 stocks using a 12-factor model, applies sector concentration limits, and rebalances to the top 30 holdings via the **Alpaca Paper Trading API** — submitting real market orders automatically after each market close.

## Live Paper Trading Results (Apr 30 – May 6, 2026)

| Metric | Strategy | SPY Benchmark |
| :--- | :--- | :--- |
| Starting Capital | $100,000 | $100,000 |
| Current NAV | $107,154 | $100,570 (est.) |
| Total Return | **+7.15%** | +0.57% |
| Excess Return | **+6.58%** | |
| Rebalance Method | Score-weighted top 30 | Buy-and-hold |

## How It Works

Every 30 calendar days the system rescores the full S&P 500 universe and rebalances. Between rebalances it checks each position **daily** for ATR-based stop-losses and replaces triggered positions without waiting for the next cycle.

The system runs entirely without manual intervention: a macOS `launchd` daemon fires at 4:47 PM ET on every trading day, logs all activity to CSV, and generates an interactive HTML performance dashboard.

## Factor Model (12 Factors)

Weights are calibrated via rolling **Rank-IC** — factors with higher recent predictive power receive more weight automatically.

| Factor | Weight | Category | What It Captures |
| :--- | :--- | :--- | :--- |
| Momentum 60d | 21% | Momentum | 3-month price trend |
| Momentum 20d | 13% | Momentum | 1-month price trend |
| ROE Score | 12% | Quality | Profitability vs 10% baseline |
| Growth Score | 10% | Quality | Earnings growth rate |
| MA Alignment | 9% | Structure | Bullish moving-average stack |
| RSI Score | 7% | Mean-revert | Oversold bounce signal |
| PB Score | 6% | Valuation | Low price-to-book preference |
| Debt Score | 6% | Quality | Low leverage preference |
| Vol Ratio | 5% | Volume | Recent volume surge |
| Price Position | 5% | Structure | 60-day price percentile |
| Vol Trend | 3% | Volume | Volume momentum |
| Momentum 5d | -3% | Mean-revert | Short-term reversal (contrarian) |

**Constraints:** sector concentration capped at 25% per GICS sector · share-class deduplication (e.g. GOOG/GOOGL) · score-proportional position sizing

## 10-Year Out-of-Sample Backtest (2015–2026, 497 stocks)

| Period | Strategy | SPY | Alpha | Sharpe | Max DD |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2015-01 to 2018-10 | +55.6% | +45.6% | +9.9% | 1.00 | -13.2% |
| 2018-10 to 2022-07 | -3.9% | +45.4% | -49.3% | 0.05 | -31.1% |
| 2022-07 to 2026-05 | +108.2% | +93.3% | +14.9% | 1.25 | -15.0% |
| **Total (compounded)** | +211.2% | +309.4% | -98.2% | | |

Rebalance: every 20 trading days · SPY 200-day MA bear-market filter · Transaction cost: 0.1% per side

### Why Period 2 Underperformed and What Changed

The SPY 200-day MA filter correctly moved to cash during the 2018 and 2020 crashes, but V-shaped recoveries in 2019 and 2020–2021 happened faster than the moving average could confirm — the strategy sat in cash through the strongest months of both rebounds. The 2019–2021 market was also dominated by loss-making growth stocks that scored poorly on quality factors.

**Fix:** when SPY drops below its 200-day MA, the system now holds 50% of the portfolio (the highest-scored 15 positions) instead of going fully to cash. This limits crash exposure while preserving participation in fast recoveries.

## Tech Stack

- **Data:** `yfinance`, `akshare` · **Broker API:** Alpaca Markets (paper trading)
- **Factor engine:** `pandas`, `numpy`, `scipy`
- **Scheduling:** macOS `launchd` daemon
- **Dashboard:** TradingView Lightweight Charts (self-contained HTML)
- **Risk report:** CAPM regression, Sharpe, Sortino, max drawdown

## Key Commands

```bash
# Run daily rebalance / stop-loss check
python main.py alpaca-paper

# Force full rebalance today
python main.py alpaca-paper --force-rebalance

# Generate performance dashboard
python main.py alpaca-dashboard

# Risk report (CAPM alpha/beta, Sharpe, Sortino, drawdown)
python main.py risk-report

# 10-year backtest
python main.py factor-backtest --start 2015-01-01 --end 2026-01-01
```

*Submitted to CUHK QFRM Admissions · May 2026*
