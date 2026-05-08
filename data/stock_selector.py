"""
Auto stock selection — scans S&P500 or TSX60 for the strongest current signals.

Flow:
1. Fetch stock universe list
2. Filter by price range and minimum volume
3. Run composite strategy + factor engine on each stock
4. Return top N by combined score
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
from loguru import logger

from .fetcher import DataFetcher


def get_sp500_symbols() -> list[str]:
    fetcher = DataFetcher(use_cache=True)
    df = fetcher.get_stock_list("sp500")
    return df["symbol"].tolist()


def get_tsx60_symbols() -> list[str]:
    fetcher = DataFetcher(use_cache=True)
    df = fetcher.get_stock_list("tsx60")
    return df["symbol"].tolist()


def get_russell2000_symbols() -> list[str]:
    fetcher = DataFetcher(use_cache=True)
    df = fetcher.get_stock_list("russell2000")
    return df["symbol"].tolist()


def select_stocks(
    universe: str = "sp500",
    top_n: int = 5,
    min_price: float = 10.0,
    max_price: float = 500.0,
    min_avg_volume: int = 500_000,
    lookback_days: int = 180,
) -> list[dict[str, Any]]:
    """
    Scan the given universe and return the top_n stocks by signal strength.
    """
    from strategy.composite import CompositeStrategy
    from backtest.auto_optimizer import build_optimized_composite

    fetcher = DataFetcher(use_cache=True)
    df_list = fetcher.get_stock_list(universe)
    symbols = df_list["symbol"].tolist() if not df_list.empty else []

    if not symbols:
        logger.error(f"Could not get symbol list for universe: {universe}")
        return []

    logger.info(f"{universe}: {len(symbols)} stocks, scanning...")

    end   = str(date.today())
    start = str(date.today() - timedelta(days=lookback_days))

    try:
        strategy = build_optimized_composite()
    except Exception:
        strategy = CompositeStrategy()

    results = []
    failed  = 0

    for i, sym in enumerate(symbols):
        try:
            df = fetcher.get_kline(sym, start, end)
            if df.empty or len(df) < 60:
                continue

            price = float(df["close"].iloc[-1])
            if not (min_price <= price <= max_price):
                continue

            avg_vol = float(df["volume"].tail(20).mean())
            if avg_vol < min_avg_volume:
                continue

            sig   = strategy.run(df, sym)
            score = float(sig.metadata.get("score", 0))
            results.append({
                "symbol": sym,
                "price":  price,
                "score":  score,
                "signal": sig.signal.value,
                "reason": sig.reason[:60],
            })

            if (i + 1) % 30 == 0:
                logger.info(f"  Scanned {i+1}/{len(symbols)}, valid: {len(results)}")

        except Exception:
            failed += 1

    logger.info(f"Scan done. Valid: {len(results)}, failed: {failed}")

    buy_results = [r for r in results if r["signal"] == "BUY"]
    buy_results.sort(key=lambda x: x["score"], reverse=True)

    if not buy_results:
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_n]

    return buy_results[:top_n]
