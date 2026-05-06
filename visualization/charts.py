"""
可视化模块。

K 线图使用 TradingView Lightweight Charts（专业级，开源免费）。
资金曲线、月度收益、交易记录继续使用 Plotly。
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from strategy import Signal, SignalType


# ── TradingView Lightweight Charts K 线 ──────────────────────────────────────

def _df_to_tv_candles(df: pd.DataFrame) -> list[dict]:
    rows = []
    for idx, row in df.iterrows():
        t = str(idx.date()) if hasattr(idx, "date") else str(idx)[:10]
        rows.append({
            "time":  t,
            "open":  round(float(row["open"]),  4),
            "high":  round(float(row["high"]),  4),
            "low":   round(float(row["low"]),   4),
            "close": round(float(row["close"]), 4),
        })
    return rows


def _df_to_tv_volume(df: pd.DataFrame) -> list[dict]:
    rows = []
    for idx, row in df.iterrows():
        t = str(idx.date()) if hasattr(idx, "date") else str(idx)[:10]
        up = float(row["close"]) >= float(row["open"])
        rows.append({
            "time":  t,
            "value": float(row["volume"]),
            "color": "rgba(239,35,42,0.5)" if up else "rgba(20,177,67,0.5)",
        })
    return rows


def _df_to_tv_line(df: pd.DataFrame, col: str) -> list[dict]:
    rows = []
    for idx, val in df[col].items():
        if pd.isna(val):
            continue
        t = str(idx.date()) if hasattr(idx, "date") else str(idx)[:10]
        rows.append({"time": t, "value": round(float(val), 4)})
    return rows


def _trade_log_to_markers(trade_log: list[dict]) -> list[dict]:
    markers = []
    for t in trade_log:
        is_buy = t["side"] == "BUY"
        markers.append({
            "time":     t["date"],
            "position": "belowBar" if is_buy else "aboveBar",
            "color":    "#ef232a" if is_buy else "#14b143",
            "shape":    "arrowUp" if is_buy else "arrowDown",
            "text":     f"B {t['price']:.2f}" if is_buy else f"S {t['price']:.2f}",
            "size":     1.5,
        })
    return sorted(markers, key=lambda x: x["time"])


def tv_kline_html(
    df: pd.DataFrame,
    symbol: str,
    trade_log: list[dict] | None = None,
    height: int = 520,
) -> str:
    """Generate a TradingView Lightweight Charts K-line HTML block (no <html> wrapper)."""

    for p in [5, 20, 60]:
        col = f"ma{p}"
        if col not in df.columns:
            df = df.copy()
            df[col] = df["close"].rolling(p).mean()

    candles = json.dumps(_df_to_tv_candles(df))
    volume  = json.dumps(_df_to_tv_volume(df))
    ma5     = json.dumps(_df_to_tv_line(df, "ma5"))
    ma20    = json.dumps(_df_to_tv_line(df, "ma20"))
    ma60    = json.dumps(_df_to_tv_line(df, "ma60"))
    markers = json.dumps(_trade_log_to_markers(trade_log or []))

    return f"""
<div id="tv_chart" style="width:100%;height:{height}px;"></div>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
(function() {{
  const el = document.getElementById('tv_chart');
  const chart = LightweightCharts.createChart(el, {{
    width: el.clientWidth,
    height: {height},
    layout: {{ background: {{ color: '#0d0d1a' }}, textColor: '#cccccc' }},
    grid: {{ vertLines: {{ color: '#1a1a2e' }}, horzLines: {{ color: '#1a1a2e' }} }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
    timeScale: {{ borderColor: '#333', timeVisible: true }},
    rightPriceScale: {{ borderColor: '#333' }},
  }});

  // Candlestick
  const candle = chart.addCandlestickSeries({{
    upColor: '#ef232a', downColor: '#14b143',
    borderUpColor: '#ef232a', borderDownColor: '#14b143',
    wickUpColor: '#ef232a', wickDownColor: '#14b143',
  }});
  candle.setData({candles});
  candle.setMarkers({markers});

  // Moving averages — same right price scale as candles
  chart.addLineSeries({{ color:'#ffa500', lineWidth:1, title:'MA5',  priceLineVisible:false, lastValueVisible:false }}).setData({ma5});
  chart.addLineSeries({{ color:'#00aaff', lineWidth:1, title:'MA20', priceLineVisible:false, lastValueVisible:false }}).setData({ma20});
  chart.addLineSeries({{ color:'#ff44ff', lineWidth:1, title:'MA60', priceLineVisible:false, lastValueVisible:false }}).setData({ma60});

  // Volume overlay — priceScaleId:'' keeps it on its own scale, not shared with price
  const volSeries = chart.addHistogramSeries({{
    priceFormat: {{ type: 'volume' }},
    priceScaleId: '',
  }});
  volSeries.priceScale().applyOptions({{ scaleMargins: {{ top: 0.8, bottom: 0 }} }});
  volSeries.setData({volume});

  chart.timeScale().fitContent();
  window.addEventListener('resize', () => chart.applyOptions({{ width: el.clientWidth }}));
}})();
</script>
<div style="color:#555;font-size:11px;margin-top:4px;text-align:right">
  Powered by <a href="https://tradingview.github.io/lightweight-charts/" target="_blank"
  style="color:#555">TradingView Lightweight Charts</a>
</div>"""


def tv_equity_html(
    port_data: list[dict],
    spy_data: list[dict],
    rebalance_dates: list[str],
    height: int = 440,
) -> str:
    """TradingView Lightweight Charts: portfolio NAV (area) vs SPY (line) + rebalance markers."""
    port_json = json.dumps(port_data)
    spy_json  = json.dumps(spy_data)
    markers   = json.dumps([{
        "time": d, "position": "aboveBar", "color": "#00aaff",
        "shape": "circle", "text": "↺", "size": 1,
    } for d in rebalance_dates])

    return f"""
<div id="tv_equity_chart" style="width:100%;height:{height}px;"></div>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
(function() {{
  const el = document.getElementById('tv_equity_chart');
  const chart = LightweightCharts.createChart(el, {{
    width: el.clientWidth, height: {height},
    layout: {{ background: {{ color: '#0d0d1a' }}, textColor: '#cccccc' }},
    grid:   {{ vertLines: {{ color: '#1a1a2e' }}, horzLines: {{ color: '#1a1a2e' }} }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
    timeScale: {{ borderColor: '#333', timeVisible: true }},
    rightPriceScale: {{ borderColor: '#333' }},
  }});

  const portSeries = chart.addAreaSeries({{
    topColor: 'rgba(255,165,0,0.35)', bottomColor: 'rgba(255,165,0,0.0)',
    lineColor: '#ffa500', lineWidth: 2, title: '因子组合',
    priceFormat: {{ type: 'price', precision: 3, minMove: 0.001 }},
  }});
  portSeries.setData({port_json});
  portSeries.setMarkers({markers});

  chart.addLineSeries({{
    color: '#aaaaaa', lineWidth: 1.5, lineStyle: 2,
    title: 'SPY', priceLineVisible: false, lastValueVisible: true,
    priceFormat: {{ type: 'price', precision: 3, minMove: 0.001 }},
  }}).setData({spy_json});

  chart.timeScale().fitContent();
  window.addEventListener('resize', () => chart.applyOptions({{ width: el.clientWidth }}));
}})();
</script>
<div style="color:#555;font-size:11px;margin-top:4px;text-align:right">
  Powered by <a href="https://tradingview.github.io/lightweight-charts/" target="_blank"
  style="color:#555">TradingView Lightweight Charts</a>
</div>"""


# ── Plotly helpers (equity curve, monthly bar, trade table) ──────────────────

class ChartBuilder:

    @staticmethod
    def equity_curve(
        equity: pd.Series,
        benchmark: pd.Series | None = None,
        title: str = "资金曲线",
    ) -> go.Figure:
        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                            shared_xaxes=True, vertical_spacing=0.03)
        norm_equity = equity / equity.iloc[0]
        fig.add_trace(go.Scatter(x=equity.index, y=norm_equity, name="策略净值",
                                 line=dict(color="#ffa500", width=2)), row=1, col=1)
        if benchmark is not None:
            norm_bench = (benchmark / benchmark.iloc[0]).reindex(equity.index).ffill()
            fig.add_trace(go.Scatter(x=equity.index, y=norm_bench, name="S&P 500",
                                     line=dict(color="#aaaaaa", width=1.5, dash="dash")),
                          row=1, col=1)
        roll_max = norm_equity.cummax()
        drawdown = (norm_equity - roll_max) / roll_max
        fig.add_trace(go.Scatter(x=equity.index, y=drawdown, name="回撤",
                                 fill="tozeroy", fillcolor="rgba(255,0,0,0.2)",
                                 line=dict(color="rgba(255,0,0,0.6)", width=1)),
                      row=2, col=1)
        fig.update_yaxes(title_text="净值", row=1, col=1)
        fig.update_yaxes(title_text="回撤", tickformat=".1%", row=2, col=1)
        fig.update_layout(title=title, template="plotly_dark", height=500)
        return fig

    @staticmethod
    def full_report(result, df: pd.DataFrame, benchmark: pd.Series | None = None) -> str:
        import plotly.io as pio

        # ── K 线（TradingView Lightweight Charts）────────────────────────────
        for p in [5, 20, 60]:
            col = f"ma{p}"
            if col not in df.columns:
                df[col] = df["close"].rolling(p).mean()

        kline_block = tv_kline_html(df, result.symbol, result.trade_log, height=540)

        # ── 资金曲线 + 回撤（Plotly）────────────────────────────────────────
        eq_fig = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35],
                               shared_xaxes=True, vertical_spacing=0.04)
        norm_equity = result.equity_curve / result.equity_curve.iloc[0]
        eq_fig.add_trace(go.Scatter(x=norm_equity.index, y=norm_equity,
                                    name="策略净值", line=dict(color="#ffa500", width=2)),
                         row=1, col=1)
        if benchmark is not None:
            nb = (benchmark / benchmark.iloc[0]).reindex(norm_equity.index).ffill()
            eq_fig.add_trace(go.Scatter(x=nb.index, y=nb, name="S&P 500",
                                        line=dict(color="#aaaaaa", width=1.5, dash="dash")),
                             row=1, col=1)
        eq_fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)
        roll_max  = norm_equity.cummax()
        drawdown  = (norm_equity - roll_max) / roll_max
        eq_fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name="回撤",
                                    fill="tozeroy", fillcolor="rgba(255,0,0,0.2)",
                                    line=dict(color="rgba(255,0,0,0.6)", width=1)),
                         row=2, col=1)
        eq_fig.update_yaxes(title_text="净值",  row=1, col=1)
        eq_fig.update_yaxes(title_text="回撤", tickformat=".1%", row=2, col=1)
        eq_fig.update_layout(title="资金曲线 vs 基准", template="plotly_dark",
                             height=460, legend=dict(orientation="h", y=1.05))

        # ── 月度收益柱状图 ────────────────────────────────────────────────
        monthly_ret = result.equity_curve.resample("M").last().pct_change().dropna()
        monthly_ret.index = monthly_ret.index.to_period("M").astype(str)
        mv = [round(v * 100, 2) for v in monthly_ret.values]
        monthly_fig = go.Figure(go.Bar(
            x=list(monthly_ret.index), y=mv,
            marker_color=["#ef232a" if v >= 0 else "#14b143" for v in mv],
            text=[f"{v:+.1f}%" for v in mv], textposition="outside",
        ))
        monthly_fig.update_layout(title="月度收益", template="plotly_dark",
                                  height=280, yaxis_ticksuffix="%",
                                  margin=dict(t=50, b=30))

        # ── 交易记录表格 ──────────────────────────────────────────────────
        paired: list[dict] = []
        buy_q: list[dict] = []
        for t in result.trade_log:
            if t["side"] == "BUY":
                buy_q.append(t)
            elif t["side"] == "SELL" and buy_q:
                b = buy_q.pop(0)
                pnl = (t["price"] - b["price"]) * abs(t["size"])
                pct = (t["price"] - b["price"]) / b["price"]
                paired.append({
                    "买入日期": b["date"], "买入价": f"{b['price']:.2f}",
                    "卖出日期": t["date"], "卖出价": f"{t['price']:.2f}",
                    "数量": abs(int(t["size"])),
                    "盈亏($)": f"{pnl:+.0f}",
                    "涨跌幅":  f"{pct:+.2%}",
                })
        cols = ["买入日期", "买入价", "卖出日期", "卖出价", "数量", "盈亏($)", "涨跌幅"]
        pnl_colors = ["#ef232a" if "+" in r["盈亏($)"] else "#14b143" for r in paired]
        table_fig = go.Figure(go.Table(
            header=dict(values=cols, fill_color="#1e1e2e",
                        font=dict(color="white", size=13), align="center"),
            cells=dict(
                values=[[r[c] for r in paired] for c in cols],
                fill_color=[["#1a1a2e"] * len(paired)] * (len(cols) - 1) + [pnl_colors],
                font=dict(color="white", size=12), align="center", height=28,
            ),
        ))
        table_fig.update_layout(title="交易记录", template="plotly_dark",
                                height=max(280, len(paired) * 30 + 80),
                                margin=dict(t=50, b=10))

        # ── 指标卡片 ──────────────────────────────────────────────────────
        m = result.metrics
        bench_ret = None
        if benchmark is not None:
            try:
                bench_ret = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1)
            except Exception:
                pass

        cards = [
            ("总收益率",     f"{m.total_return:+.2%}",    "#ef232a" if m.total_return > 0 else "#14b143"),
            ("年化收益率",   f"{m.annual_return:+.2%}",   "#ef232a" if m.annual_return > 0 else "#14b143"),
            ("S&P 500",      f"{bench_ret:+.2%}" if bench_ret is not None else "N/A",
             "#ef232a" if bench_ret and bench_ret > 0 else "#14b143"),
            ("超额收益",     f"{(m.total_return - bench_ret):+.2%}" if bench_ret is not None else "N/A",
             "#ef232a" if bench_ret is not None and m.total_return > bench_ret else "#14b143"),
            ("最大回撤",     f"{m.max_drawdown:.2%}",     "#ff6666"),
            ("Sharpe",       f"{m.sharpe_ratio:.3f}",     "#ffa500"),
            ("Calmar",       f"{m.calmar_ratio:.3f}",     "#ffa500"),
            ("胜率",         f"{m.win_rate:.2%}",         "#ffa500"),
            ("盈亏比",       f"{m.profit_factor:.3f}",    "#ffa500"),
            ("总交易次数",   str(m.total_trades),         "#aaaaaa"),
            ("平均持仓天数", f"{m.avg_holding_days:.1f}d","#aaaaaa"),
        ]
        card_html = "".join(f"""
            <div style="background:#1e1e2e;border-radius:8px;padding:14px 18px;
                        min-width:110px;text-align:center;flex:1 1 110px">
              <div style="color:#666;font-size:11px;margin-bottom:5px">{label}</div>
              <div style="color:{color};font-size:19px;font-weight:bold">{value}</div>
            </div>""" for label, value, color in cards)

        eq_html      = pio.to_html(eq_fig,      full_html=False, include_plotlyjs="cdn")
        monthly_html = pio.to_html(monthly_fig, full_html=False, include_plotlyjs=False)
        table_html   = pio.to_html(table_fig,   full_html=False, include_plotlyjs=False)

        return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{result.symbol} 回测报告</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ background:#0d0d1a; color:#eee; font-family:'Segoe UI',Arial,sans-serif;
          margin:0; padding:24px; }}
  h1 {{ text-align:center; color:#ffa500; margin-bottom:20px; font-size:22px; }}
  .cards {{ display:flex; flex-wrap:wrap; gap:10px; justify-content:center; margin-bottom:24px; }}
  .section {{ background:#0f0f20; border-radius:10px; padding:16px; margin-bottom:20px; }}
  .section-title {{ color:#888; font-size:13px; margin-bottom:10px; letter-spacing:1px; }}
</style>
</head>
<body>
<h1>{result.symbol} · {result.strategy_name} · Backtest Report</h1>
<div class="cards">{card_html}</div>

<div class="section">
  <div class="section-title">K LINE · MA5 · MA20 · MA60 · BUY/SELL SIGNALS</div>
  {kline_block}
</div>

<div class="section">
  <div class="section-title">EQUITY CURVE vs BENCHMARK</div>
  {eq_html}
</div>

<div class="section">
  <div class="section-title">MONTHLY RETURNS</div>
  {monthly_html}
</div>

<div class="section">
  <div class="section-title">TRADE LOG</div>
  {table_html}
</div>
</body>
</html>"""

    @staticmethod
    def kline_chart(
        df: pd.DataFrame,
        symbol: str,
        trade_log: list[dict] | None = None,
        height: int = 600,
    ) -> str:
        """Generate a standalone HTML page with K-line + MA + volume + buy/sell markers."""
        block = tv_kline_html(df, symbol, trade_log, height=height)
        return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{symbol} K Line</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ background:#0d0d1a; color:#eee;
          font-family:'Segoe UI',Arial,sans-serif; margin:0; padding:24px; }}
  h1 {{ text-align:center; color:#ffa500; font-size:20px; margin-bottom:16px; }}
</style>
</head>
<body>
<h1>{symbol} · K Line</h1>
{block}
</body>
</html>"""

    @staticmethod
    def factor_backtest_report(
        seg_results: list[dict],
        universe: str,
        top_n: int,
        rebalance_days: int,
        start: str,
        end: str,
        initial_cash: float,
    ) -> str:
        """Generate a full HTML backtest report for factor-based cross-sectional strategy."""
        import numpy as np
        import plotly.io as pio

        # ── Chain segments into a continuous equity curve ─────────────────
        # Each segment resets to initial_cash internally; chain them so
        # segment N starts where segment N-1 ended (compound growth).
        all_dates:  list[str]   = []
        chain_port: list[float] = []
        chain_spy:  list[float] = []
        all_rb_log: list[dict]  = []
        port_mult = 1.0  # running end-of-segment multiplier
        spy_mult  = 1.0

        for r in seg_results:
            dates  = r.get("dates",  [])
            c_port = r.get("c_port", [])
            c_spy  = r.get("c_spy",  [])
            if not dates:
                continue
            seg_p0 = c_port[0] if c_port else 1.0
            seg_s0 = c_spy[0]  if c_spy  else 1.0
            for d, pv, sv in zip(dates, c_port, c_spy):
                all_dates.append(d)
                chain_port.append(round(pv / seg_p0 * port_mult, 6))
                chain_spy.append( round(sv / seg_s0 * spy_mult,  6))
            if chain_port:
                port_mult = chain_port[-1]
                spy_mult  = chain_spy[-1]
            all_rb_log.extend(r.get("rb_log", []))

        # ── TradingView equity curve ──────────────────────────────────────
        port_nav = [{"time": d, "value": round(v, 4)} for d, v in zip(all_dates, chain_port)]
        spy_nav  = [{"time": d, "value": round(v, 4)} for d, v in zip(all_dates, chain_spy)]
        rb_dates = [rb["date"] for rb in all_rb_log]
        tv_block = tv_equity_html(port_nav, spy_nav, rb_dates, height=440)

        # ── Monthly returns (from chained curve) ──────────────────────────
        port_series = pd.Series(
            {d: v for d, v in zip(all_dates, chain_port)}, dtype=float
        )
        port_series.index = pd.to_datetime(port_series.index)
        monthly_ret = port_series.resample("M").last().pct_change().dropna()
        monthly_ret.index = monthly_ret.index.to_period("M").astype(str)
        mv = [round(v * 100, 2) for v in monthly_ret.values]
        monthly_fig = go.Figure(go.Bar(
            x=list(monthly_ret.index), y=mv,
            marker_color=["#ef232a" if v >= 0 else "#14b143" for v in mv],
            text=[f"{v:+.1f}%" for v in mv], textposition="outside",
        ))
        monthly_fig.update_layout(
            title="月度收益", template="plotly_dark",
            height=300, yaxis_ticksuffix="%", margin=dict(t=50, b=30),
        )

        # ── Rebalance log table ───────────────────────────────────────────
        cols_rb = ["日期", "持仓", "换手率", "最高分", "最低分"]
        rb_rows = [
            [rb["date"],
             " ".join(rb["holdings"][:6]) + ("…" if len(rb["holdings"]) > 6 else ""),
             f"{rb['turnover']}%", str(rb["top_score"]), str(rb["bottom_score"])]
            for rb in all_rb_log
        ]
        rb_table = go.Figure(go.Table(
            header=dict(values=cols_rb, fill_color="#1e1e2e",
                        font=dict(color="white", size=13), align="center"),
            cells=dict(
                values=list(zip(*rb_rows)) if rb_rows else [[] for _ in cols_rb],
                fill_color="#1a1a2e",
                font=dict(color="white", size=12), align="center", height=28,
            ),
        ))
        rb_table.update_layout(
            title="调仓记录", template="plotly_dark",
            height=max(300, len(all_rb_log) * 30 + 80), margin=dict(t=50, b=10),
        )

        # ── Combined metrics (compound across all segments) ───────────────
        port_mult_total = 1.0
        spy_mult_total  = 1.0
        total_rebalances = 0
        for r in seg_results:
            port_mult_total *= (1 + r["port_total"] / 100)
            spy_mult_total  *= (1 + r["spy_total"]  / 100)
            total_rebalances += r["n_rebalance"]

        port_total_pct = (port_mult_total - 1) * 100
        spy_total_pct  = (spy_mult_total  - 1) * 100
        alpha_pct      = port_total_pct - spy_total_pct

        pa = np.array(chain_port)
        pr = np.diff(pa) / pa[:-1]
        n_days = len(pa)
        ann_ret = port_mult_total ** (252 / max(n_days, 1)) - 1 if n_days > 1 else 0.0

        def _sharpe(r: np.ndarray) -> float:
            return float(np.mean(r) / np.std(r) * np.sqrt(252)) if np.std(r) > 1e-9 else 0.0
        def _max_dd(vals: np.ndarray) -> float:
            peak = np.maximum.accumulate(vals)
            return float(((vals - peak) / peak).min()) * 100

        combined_sharpe = _sharpe(pr)
        combined_mdd    = _max_dd(pa)
        calmar = ann_ret / abs(combined_mdd / 100) if combined_mdd < -0.001 else 0.0

        cards = [
            ("总收益率",   f"{port_total_pct:+.1f}%", "#ef232a" if port_total_pct > 0 else "#14b143"),
            ("年化收益率", f"{ann_ret:+.1%}",          "#ef232a" if ann_ret > 0 else "#14b143"),
            ("SPY 收益",   f"{spy_total_pct:+.1f}%",  "#aaaaaa"),
            ("超额收益",   f"{alpha_pct:+.1f}%",       "#ef232a" if alpha_pct > 0 else "#14b143"),
            ("Sharpe",     f"{combined_sharpe:.3f}",   "#ffa500"),
            ("最大回撤",   f"{combined_mdd:.1f}%",     "#ff6666"),
            ("Calmar",     f"{calmar:.3f}",            "#ffa500"),
            ("调仓次数",   str(total_rebalances),      "#aaaaaa"),
        ]
        card_html = "".join(f"""
            <div style="background:#1e1e2e;border-radius:8px;padding:14px 18px;
                        min-width:110px;text-align:center;flex:1 1 110px">
              <div style="color:#666;font-size:11px;margin-bottom:5px">{label}</div>
              <div style="color:{color};font-size:19px;font-weight:bold">{value}</div>
            </div>""" for label, value, color in cards)

        monthly_html = pio.to_html(monthly_fig, full_html=False, include_plotlyjs="cdn")
        rb_html      = pio.to_html(rb_table,    full_html=False, include_plotlyjs=False)

        return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>因子选股回测报告</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ background:#0d0d1a; color:#eee; font-family:'Segoe UI',Arial,sans-serif;
          margin:0; padding:24px; }}
  h1 {{ text-align:center; color:#ffa500; margin-bottom:20px; font-size:22px; }}
  .cards {{ display:flex; flex-wrap:wrap; gap:10px; justify-content:center; margin-bottom:24px; }}
  .section {{ background:#0f0f20; border-radius:10px; padding:16px; margin-bottom:20px; }}
  .section-title {{ color:#888; font-size:13px; margin-bottom:10px; letter-spacing:1px; }}
</style>
</head>
<body>
<h1>因子选股回测 · {universe.upper()} Top{top_n} · {start} → {end}</h1>
<div class="cards">{card_html}</div>

<div class="section">
  <div class="section-title">PORTFOLIO NAV vs SPY · ↺ = 调仓日 · TradingView Lightweight Charts</div>
  {tv_block}
</div>

<div class="section">
  <div class="section-title">MONTHLY RETURNS</div>
  {monthly_html}
</div>

<div class="section">
  <div class="section-title">REBALANCE LOG</div>
  {rb_html}
</div>
</body>
</html>"""

    @staticmethod
    def save(html_or_fig, path: str) -> None:
        if isinstance(html_or_fig, str):
            from pathlib import Path
            Path(path).write_text(html_or_fig, encoding="utf-8")
        else:
            html_or_fig.write_html(path)
