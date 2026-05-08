"""
DataFetcher — unified interface for yfinance (market data + fundamentals).

Symbols:
  US stocks  : "AAPL", "MSFT", "GOOGL"
  TSX stocks : "TD.TO", "RY.TO", "ENB.TO"
  Indices    : "^GSPC" (S&P500), "^GSPTSE" (TSX Composite)
  ETFs       : "XIU.TO", "SPY", "QQQ"

All public methods return pandas DataFrames indexed by date.
"""

from __future__ import annotations

import urllib.request
import json
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger

from config import settings
from .cache import DataCache


class DataFetcher:
    """Unified market data fetcher powered by yfinance."""

    def __init__(self, use_cache: bool = True):
        self.cache = DataCache(ttl_minutes=360) if use_cache else None  # 6h TTL for daily data

    # ── K-line / OHLCV ───────────────────────────────────────────────────────

    def get_kline(
        self,
        symbol: str,
        start: str,
        end: str,
        period: str = "1d",   # "1d" | "1wk" | "1mo"
        adjust: str = "auto", # "auto" = split+dividend adjusted (default)
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data via yfinance.
        Returns columns: open / high / low / close / volume
        Index: DatetimeIndex (UTC-naive, date only)
        """
        if self.cache:
            cached = self.cache.get("kline", symbol=symbol, start=start,
                                    end=end, period=period)
            if cached is not None:
                return cached

        logger.info(f"Fetching kline: {symbol} {start}~{end} {period}")
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                interval=period,
                auto_adjust=True,   # adjusts for splits and dividends
                progress=False,
            )
        except Exception as e:
            logger.error(f"yfinance download failed for {symbol}: {e}")
            return pd.DataFrame()

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        # Flatten MultiIndex columns if present (yfinance ≥0.2.x)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        df = df.sort_index()

        if self.cache:
            self.cache.set(df, "kline", symbol=symbol, start=start,
                           end=end, period=period)
        return df

    def get_realtime(self, symbols: list[str]) -> pd.DataFrame:
        """Get latest price snapshot for a list of symbols."""
        rows = []
        for sym in symbols:
            try:
                t = yf.Ticker(sym)
                info = t.fast_info
                rows.append({
                    "symbol": sym,
                    "price":  round(float(info.last_price), 4),
                    "volume": int(info.three_month_average_volume or 0),
                })
            except Exception as e:
                logger.warning(f"Realtime fetch failed for {sym}: {e}")
        return pd.DataFrame(rows)

    # ── Market index ─────────────────────────────────────────────────────────

    def get_index_kline(
        self,
        symbol: str | None = None,
        start: str = "2020-01-01",
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch index historical data. Defaults to S&P 500 (^GSPC)."""
        symbol = symbol or settings.market_index
        end = end or str(date.today())
        return self.get_kline(symbol, start, end)

    # ── Fundamentals ─────────────────────────────────────────────────────────

    def get_fundamentals(self, symbol: str) -> dict:
        """
        Fetch fundamental data via yfinance Ticker.info.
        Returns: roe, net_profit_growth, debt_ratio, pe_ttm, market_cap
        """
        if self.cache:
            cached = self.cache.get("fundamentals", symbol=symbol)
            if cached is not None and isinstance(cached, dict):
                return cached

        try:
            info = yf.Ticker(symbol).info
            result = {
                # ROE as percentage (yfinance returns decimal)
                "roe": round(info["returnOnEquity"] * 100, 2) if info.get("returnOnEquity") is not None else None,
                # Revenue/earnings growth as percentage
                "net_profit_growth": round((info.get("earningsGrowth") or
                                            info.get("earningsQuarterlyGrowth") or 0) * 100, 2),
                "revenue_growth":    round((info.get("revenueGrowth") or 0) * 100, 2),
                # Debt-to-equity → convert to debt ratio approx
                "debt_ratio": _de_to_debt_ratio(info.get("debtToEquity")),
                "pe_ttm":     round(info.get("trailingPE") or 0, 2),
                "pb":         round(info.get("priceToBook") or 0, 2),
                "market_cap": info.get("marketCap") or 0,
                "sector":     info.get("sector") or "",
            }
        except Exception as e:
            logger.warning(f"Fundamentals fetch failed for {symbol}: {e}")
            result = {}

        if self.cache and result:
            # Cache fundamentals for 24h
            self.cache.set(result, "fundamentals", symbol=symbol)  # type: ignore
        return result

    def get_historical_fundamentals(self, symbol: str) -> dict:
        """
        返回历史年报快照：{report_date_str: {roe, net_profit_growth, debt_ratio, pe_ttm, pb}}
        利用 yfinance 年度财务报表，通常能回溯 4-5 年。
        PE = 年末股价 / EPS，PB = 年末股价 / 每股净资产，均为历史值无前视偏差。
        """
        cache_key = f"hist_fund2_{symbol}"  # v2：含 PE/PB
        if self.cache:
            cached = self.cache.get(cache_key, symbol=symbol)
            if cached is not None and isinstance(cached, dict) and cached:
                return cached

        result: dict = {}
        try:
            t = yf.Ticker(symbol)
            inc = t.income_stmt
            bal = t.balance_sheet

            if inc is None or inc.empty or bal is None or bal.empty:
                return result

            ni_row     = next((r for r in inc.index if r == "Net Income"), None)
            eq_row     = next((r for r in bal.index if r in ("Common Stock Equity", "Stockholders Equity")), None)
            asset_row  = next((r for r in bal.index if r == "Total Assets"), None)
            debt_row   = next((r for r in bal.index if r == "Total Debt"), None)
            shares_row = next((r for r in bal.index if r in ("Ordinary Shares Number", "Share Issued")), None)

            dates = sorted(inc.columns)
            for i, col in enumerate(dates):
                try:
                    ni     = float(inc.loc[ni_row, col])    if ni_row    else None
                    eq     = float(bal.loc[eq_row, col])    if eq_row    and col in bal.columns else None
                    assets = float(bal.loc[asset_row, col]) if asset_row and col in bal.columns else None
                    debt   = float(bal.loc[debt_row, col])  if debt_row  and col in bal.columns else None

                    roe = round(ni / eq * 100, 2) if (ni and eq and eq != 0) else 0.0

                    growth = 0.0
                    if i > 0 and ni_row:
                        prev_ni = float(inc.loc[ni_row, dates[i - 1]])
                        if prev_ni and abs(prev_ni) > 1:
                            growth = round((ni - prev_ni) / abs(prev_ni) * 100, 2)

                    debt_ratio = 0.0
                    if debt is not None and assets and assets > 0:
                        debt_ratio = round(debt / assets * 100, 2)

                    # 历史 PE / PB：年报截止日前后 5 日收盘价 + 资产负债表中的流通股数
                    pe_ttm, pb = None, None
                    try:
                        shares = None
                        if shares_row and col in bal.columns:
                            v = bal.loc[shares_row, col]
                            if pd.notna(v) and float(v) > 0:
                                shares = float(v)

                        if shares:
                            start_p = col - pd.Timedelta(days=5)
                            end_p   = col + pd.Timedelta(days=5)
                            ph = t.history(start=start_p, end=end_p, auto_adjust=True)
                            if not ph.empty and "Close" in ph.columns:
                                price = float(ph["Close"].iloc[-1])
                                if ni and shares > 0:
                                    eps = ni / shares
                                    if eps > 0:
                                        pe_ttm = round(price / eps, 1)
                                if eq and shares > 0:
                                    bvps = eq / shares
                                    if bvps > 0:
                                        pb = round(price / bvps, 2)
                    except Exception:
                        pass

                    # Add 75-day delay: SEC mandates 10-K within 60 days (large filers)
                    # or 75 days (accelerated filers). 75 days covers both cases safely.
                    available = col + pd.Timedelta(days=75)
                    date_str = str(available.date()) if hasattr(available, "date") else str(available)[:10]
                    result[date_str] = {
                        "roe":               roe,
                        "net_profit_growth": growth,
                        "debt_ratio":        debt_ratio,
                        "pe_ttm":            pe_ttm,
                        "pb":                pb,
                    }
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Historical fundamentals failed for {symbol}: {e}")

        if self.cache and result:
            self.cache.set(result, cache_key, symbol=symbol)  # type: ignore
        return result

    # ── Stock universe ────────────────────────────────────────────────────────

    def get_stock_list(self, universe: str = "sp500") -> pd.DataFrame:
        """
        Return stock list for the given universe.
        universe: "sp500" | "tsx60" | "nasdaq100"
        """
        if self.cache:
            cached = self.cache.get("stock_list", universe=universe)
            if cached is not None:
                return cached

        if universe == "sp500":
            df = _fetch_sp500()
        elif universe == "tsx60":
            df = _tsx60_list()
        elif universe == "nasdaq100":
            df = _fetch_nasdaq100()
        elif universe == "russell2000":
            df = _fetch_russell2000()
        else:
            df = pd.DataFrame(columns=["symbol", "name"])

        if self.cache and not df.empty:
            self.cache.set(df, "stock_list", universe=universe)
        return df

    def search_symbol(self, keyword: str, universe: str = "sp500") -> pd.DataFrame:
        """Fuzzy-search by ticker or company name."""
        stocks = self.get_stock_list(universe)
        kw = keyword.upper()
        mask = (stocks["symbol"].str.upper().str.contains(kw) |
                stocks["name"].str.upper().str.contains(kw))
        return stocks[mask].reset_index(drop=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _de_to_debt_ratio(de: float | None) -> float:
    """Convert debt/equity ratio to approximate debt/(debt+equity) ratio."""
    if de is None or de <= 0:
        return 0.0
    return round(de / (de + 1) * 100, 2)  # de is decimal (1.2 = 120% D/E), convert to D/(D+E)%


def _fetch_sp500() -> pd.DataFrame:
    """Fetch S&P 500 components from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read()
        tables = pd.read_html(html)
        df = tables[0][["Symbol", "Security"]].rename(
            columns={"Symbol": "symbol", "Security": "name"}
        )
        df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)
        logger.info(f"S&P500 list fetched from Wikipedia: {len(df)} stocks")
        return df.reset_index(drop=True)
    except Exception as e:
        logger.warning(f"S&P500 list fetch failed: {e}, using fallback")
        return _sp500_fallback()


def _fetch_nasdaq100() -> pd.DataFrame:
    """Fetch NASDAQ-100 components from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read()
        tables = pd.read_html(html)
        for t in tables:
            if "Ticker" in t.columns:
                df = t[["Ticker", "Company"]].rename(
                    columns={"Ticker": "symbol", "Company": "name"}
                )
                logger.info(f"NASDAQ-100 list fetched from Wikipedia: {len(df)} stocks")
                return df.reset_index(drop=True)
    except Exception as e:
        logger.warning(f"NASDAQ-100 list fetch failed: {e}")
    return pd.DataFrame(columns=["symbol", "name"])


def _fetch_russell2000() -> pd.DataFrame:
    """Fetch Russell 2000 components from iShares IWM public holdings CSV."""
    url = (
        "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf"
        "/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        from io import StringIO
        lines = content.splitlines()
        # iShares CSV: first few rows are fund metadata, find the actual header row
        header_idx = next(
            (i for i, ln in enumerate(lines) if "Ticker" in ln or "ticker" in ln.lower()),
            2,
        )
        df = pd.read_csv(StringIO("\n".join(lines[header_idx:])))
        ticker_col = next((c for c in df.columns if c.strip().lower() == "ticker"), None)
        name_col   = next((c for c in df.columns if c.strip().lower() == "name"), df.columns[1])
        if ticker_col is None:
            raise ValueError(f"Ticker column not found. Columns: {list(df.columns)}")
        df = df[[ticker_col, name_col]].rename(columns={ticker_col: "symbol", name_col: "name"})
        df = df.dropna(subset=["symbol"])
        df["symbol"] = df["symbol"].astype(str).str.strip()
        df = df[df["symbol"].str.match(r"^[A-Z]{1,5}$", na=False)]
        logger.info(f"Russell 2000 list fetched from iShares: {len(df)} stocks")
        return df.reset_index(drop=True)
    except Exception as e:
        logger.warning(f"Russell 2000 fetch failed: {e}")
        return pd.DataFrame(columns=["symbol", "name"])


def _tsx60_list() -> pd.DataFrame:
    """TSX 60 components (static — updated periodically)."""
    syms = [
        ("AEM.TO", "Agnico Eagle Mines"), ("AGF-B.TO", "AGF Management"),
        ("ALA.TO", "AltaGas"), ("ATD.TO", "Alimentation Couche-Tard"),
        ("BAM.TO", "Brookfield Asset Management"), ("BCE.TO", "BCE Inc"),
        ("BIP-UN.TO", "Brookfield Infrastructure"), ("BMO.TO", "Bank of Montreal"),
        ("BNS.TO", "Bank of Nova Scotia"), ("CAE.TO", "CAE Inc"),
        ("CCO.TO", "Cameco"), ("CHP-UN.TO", "Choice Properties REIT"),
        ("CM.TO", "CIBC"), ("CNQ.TO", "Canadian Natural Resources"),
        ("CNR.TO", "Canadian National Railway"), ("CP.TO", "Canadian Pacific Kansas City"),
        ("CSU.TO", "Constellation Software"), ("CVE.TO", "Cenovus Energy"),
        ("DOL.TO", "Dollarama"), ("EMA.TO", "Emera"),
        ("ENB.TO", "Enbridge"), ("FFH.TO", "Fairfax Financial"),
        ("FM.TO", "First Quantum Minerals"), ("FNV.TO", "Franco-Nevada"),
        ("FTS.TO", "Fortis"), ("GWO.TO", "Great-West Lifeco"),
        ("H.TO", "Hydro One"), ("IAG.TO", "iA Financial Group"),
        ("IFC.TO", "Intact Financial"), ("IMO.TO", "Imperial Oil"),
        ("K.TO", "Kinross Gold"), ("KXS.TO", "Kinaxis"),
        ("L.TO", "Loblaw Companies"), ("LUN.TO", "Lundin Mining"),
        ("MFC.TO", "Manulife Financial"), ("MG.TO", "Magna International"),
        ("MRU.TO", "Metro Inc"), ("NA.TO", "National Bank"),
        ("NTR.TO", "Nutrien"), ("NPI.TO", "Northland Power"),
        ("OTEX.TO", "Open Text"), ("POW.TO", "Power Corporation"),
        ("PPL.TO", "Pembina Pipeline"), ("RCI-B.TO", "Rogers Communications"),
        ("RY.TO", "Royal Bank of Canada"), ("SAP.TO", "Saputo"),
        ("SLF.TO", "Sun Life Financial"), ("SNC.TO", "SNC-Lavalin"),
        ("SU.TO", "Suncor Energy"), ("T.TO", "TELUS"),
        ("TD.TO", "Toronto-Dominion Bank"), ("TECK-B.TO", "Teck Resources"),
        ("TRI.TO", "Thomson Reuters"), ("TRP.TO", "TC Energy"),
        ("WCN.TO", "Waste Connections"), ("WFG.TO", "West Fraser Timber"),
        ("WN.TO", "George Weston"), ("WSP.TO", "WSP Global"),
        ("X.TO", "TMX Group"), ("XTC.TO", "Exco Technologies"),
    ]
    return pd.DataFrame(syms, columns=["symbol", "name"])


def _sp500_fallback() -> pd.DataFrame:
    """Minimal S&P 500 fallback list (top 50 by market cap)."""
    syms = [
        ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "NVIDIA"),
        ("GOOGL", "Alphabet A"), ("AMZN", "Amazon"), ("META", "Meta"),
        ("TSLA", "Tesla"), ("BRK-B", "Berkshire Hathaway B"), ("LLY", "Eli Lilly"),
        ("JPM", "JPMorgan Chase"), ("V", "Visa"), ("XOM", "ExxonMobil"),
        ("UNH", "UnitedHealth"), ("MA", "Mastercard"), ("AVGO", "Broadcom"),
        ("JNJ", "Johnson & Johnson"), ("PG", "Procter & Gamble"), ("HD", "Home Depot"),
        ("MRK", "Merck"), ("COST", "Costco"), ("ABBV", "AbbVie"),
        ("CVX", "Chevron"), ("KO", "Coca-Cola"), ("WMT", "Walmart"),
        ("BAC", "Bank of America"), ("PEP", "PepsiCo"), ("ORCL", "Oracle"),
        ("CRM", "Salesforce"), ("AMD", "AMD"), ("TMO", "Thermo Fisher"),
        ("MCD", "McDonald's"), ("ACN", "Accenture"), ("CSCO", "Cisco"),
        ("ABT", "Abbott Labs"), ("GE", "GE Aerospace"), ("NOW", "ServiceNow"),
        ("DHR", "Danaher"), ("LIN", "Linde"), ("TXN", "Texas Instruments"),
        ("PM", "Philip Morris"), ("INTU", "Intuit"), ("AMGN", "Amgen"),
        ("CAT", "Caterpillar"), ("IBM", "IBM"), ("GS", "Goldman Sachs"),
        ("SPGI", "S&P Global"), ("BLK", "BlackRock"), ("ISRG", "Intuitive Surgical"),
        ("RTX", "RTX Corp"), ("QCOM", "Qualcomm"),
    ]
    return pd.DataFrame(syms, columns=["symbol", "name"])
