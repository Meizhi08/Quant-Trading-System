"""
North American Quantitative Trading System — Main Entry Point.

Commands:
  python main.py backtest   --symbol AAPL --start 2022-01-01 --end 2024-12-31
  python main.py optimize   --symbol AAPL --start 2022-01-01 --end 2024-12-31
  python main.py signal     --symbol NVDA
  python main.py live       --symbol AAPL,MSFT,NVDA
  python main.py chart      --symbol AAPL --start 2023-01-01
  python main.py scan       --symbols AAPL,MSFT,NVDA,GOOGL,META
  python main.py select-stocks
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

# 把项目根目录加入 path，方便相对导入
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from data import DataFetcher
from strategy import (
    MACrossStrategy, RSIStrategy, BollingerStrategy,
    AISignalStrategy, CompositeStrategy, FactorStrategy, UnifiedStrategy,
)
from backtest import BacktestEngine, ParameterOptimizer
from risk import RiskManager
from visualization import ChartBuilder
from alert import Notifier

app = typer.Typer(help="North American Quantitative Trading System")


def _comp_score_to_rec(score: float) -> str:
    if score >= 0.5:  return "STRONG_BUY"
    if score >= 0.2:  return "BUY"
    if score <= -0.5: return "STRONG_SELL"
    if score <= -0.2: return "SELL"
    return "NEUTRAL"
console = Console()

# 日志配置
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add("logs/quant_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days")


_VALID_STRATEGIES = ("ma", "rsi", "bb", "macd", "kdj", "ai", "composite", "factor", "unified")


def _get_strategy(name: str, start: str | None = None, end: str | None = None, live: bool = False) -> object:
    if name not in _VALID_STRATEGIES:
        typer.echo(f"未知策略: {name}. 可选: {list(_VALID_STRATEGIES)}")
        raise typer.Exit(1)

    if name == "ma":        return MACrossStrategy()
    if name == "rsi":       return RSIStrategy()
    if name == "bb":        return BollingerStrategy()
    if name == "macd":
        from strategy import MACDStrategy
        return MACDStrategy()
    if name == "kdj":
        from strategy import KDJStrategy
        return KDJStrategy()
    if name == "ai":        return AISignalStrategy()
    if name == "composite":
        from backtest.auto_optimizer import build_optimized_composite
        return build_optimized_composite(use_tv=live)

    # "factor" or "unified" — load market data for regime filter
    strat = FactorStrategy() if name == "factor" else UnifiedStrategy(use_tv=live)
    if start and end:
        try:
            market_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
            market_df = DataFetcher().get_index_kline(settings.market_index, market_start, end)
            strat.set_market_data(market_df["close"])
            logger.info("已加载S&P 500大盘过滤数据")
        except Exception as e:
            logger.warning(f"大盘数据加载失败，跳过市场过滤: {e}")
    return strat


# ── 回测命令 ──────────────────────────────────────────────────────────────────

@app.command()
def backtest(
    symbol: str = typer.Option("AAPL", help="ticker symbol, e.g. AAPL, TD.TO"),
    start: str = typer.Option("2020-01-01", help="开始日期 YYYY-MM-DD"),
    end: str = typer.Option(str(date.today()), help="结束日期 YYYY-MM-DD"),
    strategy: str = typer.Option("unified", help="策略: ma/rsi/bb/ai/composite/factor/unified"),
    stop_loss: float = typer.Option(0.08, help="止损比例"),
    trailing_stop: float = typer.Option(0.12, help="追踪止损（从峰值回落多少出场）"),
    cooldown: int = typer.Option(5, help="止损后冷静天数"),
    save_chart: bool = typer.Option(False, help="保存 HTML 图表"),
    optimize: bool = typer.Option(False, "--optimize", help="回测后自动优化策略参数并对比"),
):
    """对单只股票运行历史回测。加 --optimize 可回测完自动滚动验证（walk-forward），避免过拟合。"""
    from backtest.walk_forward import WalkForwardValidator
    from rich.table import Table

    # 普通回测
    strat = _get_strategy(strategy, start, end)
    engine = BacktestEngine(
        strategy=strat, symbol=symbol,
        stop_loss_pct=stop_loss, trailing_stop_pct=trailing_stop,
        take_profit_pct=0.0, cooldown_bars=cooldown,
    )
    result = engine.run(start, end)
    console.print(f"\n[bold]── 回测结果（全样本，仅供参考）──[/bold]")
    console.print(result.metrics)

    if not optimize:
        return

    # Walk-forward 验证
    is_composite = (strategy == "composite")
    if is_composite:
        console.print(f"\n[cyan]开始 Walk-Forward 验证 — composite 模式（每窗口优化参数 → 样本外测试）...[/cyan]")
        console.print(f"[dim]每个窗口：用过去12个月网格搜索最优参数，只看接下来3个月的样本外结果[/dim]\n")
        wf_strategy = None   # composite 模式：每窗口内部优化
    else:
        console.print(f"\n[cyan]开始 Walk-Forward 验证 — {strategy} 策略（固定参数，滚动样本外验证）...[/cyan]")
        console.print(f"[dim]用当前策略参数在多个3个月窗口上依次测试，看稳定性而非最优值[/dim]\n")
        wf_strategy = strat  # 直接用刚才回测的那个策略实例

    wf = WalkForwardValidator(
        symbol=symbol, start=start, end=end,
        train_months=12, test_months=3,
        stop_loss_pct=stop_loss, trailing_stop_pct=trailing_stop,
        metric="sharpe_ratio",
        strategy=wf_strategy,
    )
    wf_result = wf.run()

    if not wf_result.windows:
        console.print("[red]数据期太短，无法做 walk-forward（至少需要15个月）[/red]")
        return

    # 每个窗口明细
    tbl = Table(title=f"{symbol} Walk-Forward 样本外表现")
    tbl.add_column("窗口", style="dim")
    tbl.add_column("训练期")
    tbl.add_column("测试期（样本外）", style="cyan")
    tbl.add_column("收益", justify="right")
    tbl.add_column("Sharpe", justify="right")
    tbl.add_column("胜率", justify="right")
    tbl.add_column("最大回撤", justify="right")

    for w in wf_result.windows:
        m = w.metrics
        ret_color = "green" if m.total_return > 0 else "red"
        tbl.add_row(
            str(w.window),
            f"{w.train_start} ~ {w.train_end}",
            f"{w.test_start} ~ {w.test_end}",
            f"[{ret_color}]{m.total_return:+.1%}[/{ret_color}]",
            f"{m.sharpe_ratio:.2f}",
            f"{m.win_rate:.0%}",
            f"{m.max_drawdown:.1%}",
        )

    console.print(tbl)

    # 汇总
    avg_sharpe = wf_result.avg_sharpe
    avg_ret    = wf_result.avg_return
    pct_win    = wf_result.pct_profitable

    grade_color = "green" if avg_sharpe >= 0.5 and pct_win >= 0.6 else \
                  "yellow" if avg_sharpe >= 0.0 else "red"

    console.print(f"\n[bold]── Walk-Forward 汇总（这才是真实策略能力）──[/bold]")
    console.print(f"  样本外平均收益  : [{grade_color}]{avg_ret:+.1%}[/{grade_color}]  每个测试窗口的均值")
    console.print(f"  样本外平均Sharpe: [{grade_color}]{avg_sharpe:.2f}[/{grade_color}]")
    console.print(f"  盈利窗口占比    : [{grade_color}]{pct_win:.0%}[/{grade_color}]  ({sum(1 for w in wf_result.windows if w.metrics.total_return>0)}/{wf_result.n_windows} 个)")
    console.print(f"  平均最大回撤    : {wf_result.avg_max_drawdown:.1%}")

    if avg_sharpe >= 0.5 and pct_win >= 0.6:
        console.print(f"\n[green]✓ 策略通过 walk-forward 检验，样本外表现稳健，参数已保存[/green]")
    elif avg_sharpe >= 0.0:
        console.print(f"\n[yellow]⚠ 策略勉强通过，样本外表现不稳定，实盘需谨慎[/yellow]")
    else:
        console.print(f"\n[red]✗ 策略未通过 walk-forward 检验，样本外平均亏损，不建议实盘[/red]")

    if save_chart:
        fetcher = DataFetcher()
        df = fetcher.get_kline(symbol, start, end)
        fig = ChartBuilder.equity_curve(result.equity_curve, title=f"{symbol} 回测资金曲线")
        out = Path(f"logs/{symbol}_{strategy}_equity.html")
        ChartBuilder.save(fig, str(out))
        console.print(f"[green]图表已保存: {out}[/green]")


# ── 参数优化命令 ──────────────────────────────────────────────────────────────

@app.command()
def optimize(
    symbol: str = typer.Option("AAPL"),
    start: str = typer.Option("2020-01-01"),
    end: str = typer.Option(str(date.today())),
    metric: str = typer.Option("sharpe_ratio", help="优化目标指标"),
):
    """网格搜索 MACross 最优参数。"""
    opt = ParameterOptimizer(
        strategy_class=MACrossStrategy,
        param_grid={"fast": [5, 10, 15], "slow": [20, 30, 60]},
        symbol=symbol, start=start, end=end,
        metric=metric, max_workers=3,
    )
    df = opt.run()
    table = Table(title=f"{symbol} 参数优化结果（TOP 10）")
    for col in df.columns:
        table.add_column(str(col))
    for _, row in df.head(10).iterrows():
        table.add_row(*[str(v) for v in row])
    console.print(table)


# ── 实时信号命令 ──────────────────────────────────────────────────────────────

@app.command()
def signal(
    symbol: str = typer.Option(..., help="股票代码"),
    strategy: str = typer.Option("composite"),
    lookback: int = typer.Option(60, help="历史K线天数"),
):
    """对单只股票生成当前交易信号，同时显示 TradingView 技术面评级。"""
    from data.tv_signals import get_tv_signal
    fetcher = DataFetcher()
    end = str(date.today())
    start = str(date.today() - timedelta(days=lookback * 2))
    df = fetcher.get_kline(symbol, start, end)

    strat = _get_strategy(strategy, start, end, live=True)
    sig = strat.run(df, symbol)

    console.print(f"\n[bold]{sig}[/bold]")

    # TradingView 信号
    tv = get_tv_signal(symbol)
    rec = tv["recommendation"]
    rec_color = {"STRONG_BUY": "green", "BUY": "green",
                 "NEUTRAL": "yellow", "SELL": "red", "STRONG_SELL": "red"}.get(rec, "white")
    console.print(f"TradingView : [{rec_color}]{rec}[/{rec_color}]  "
                  f"(Buy={tv['buy']} Neutral={tv['neutral']} Sell={tv['sell']})\n")


# ── 完整回测报告 ──────────────────────────────────────────────────────────────

@app.command()
def report(
    symbol: str = typer.Option("AAPL", help="Ticker symbol"),
    start: str = typer.Option("2020-01-01", help="开始日期"),
    end: str = typer.Option(str(date.today()), help="结束日期"),
    strategy: str = typer.Option("ma", help="策略: ma/rsi/bb/composite"),
    stop_loss: float = typer.Option(0.08, help="止损比例"),
    trailing_stop: float = typer.Option(0.12, help="追踪止损（从峰值回落多少出场）"),
    output: str = typer.Option("", help="输出路径（空=自动命名）"),
):
    """生成完整回测报告（K线+买卖点+资金曲线+指标+交易记录）。"""
    strat = _get_strategy(strategy, start, end)
    engine = BacktestEngine(
        strategy=strat, symbol=symbol,
        stop_loss_pct=stop_loss, trailing_stop_pct=trailing_stop, take_profit_pct=0.0,
    )
    result = engine.run(start, end)

    fetcher = DataFetcher()
    df = fetcher.get_kline(symbol, start, end)
    df = strat.compute_indicators(df)

    # 拉S&P 500作为基准
    benchmark = None
    try:
        bench_df = fetcher.get_index_kline(settings.market_index, start, end)
        benchmark = bench_df["close"]
    except Exception:
        pass

    html = ChartBuilder.full_report(result, df, benchmark=benchmark)
    out = output or f"logs/{symbol}_{strategy}_report.html"
    Path(out).write_text(html, encoding="utf-8")
    console.print(f"[green]报告已保存: {out}[/green]")
    subprocess.run(["open", out])


# ── K 线图命令 ────────────────────────────────────────────────────────────────

@app.command()
def chart(
    symbol: str = typer.Option("AAPL"),
    start: str = typer.Option(str(date.today() - timedelta(days=180))),
    end: str = typer.Option(str(date.today())),
    strategy: str = typer.Option("composite"),
    output: str = typer.Option("", help="输出 HTML 路径（空=自动命名）"),
):
    """生成 K 线交互图表并保存为 HTML。"""
    fetcher = DataFetcher()
    df = fetcher.get_kline(symbol, start, end)

    strat = _get_strategy(strategy, start, end)
    df = strat.compute_indicators(df)

    # 生成历史信号并转成 trade_log 格式用于标注
    trade_log = []
    for i in range(30, len(df)):
        slice_df = df.iloc[:i + 1]
        s = strat.generate_signal(slice_df, symbol)
        if s.signal.value != "HOLD":
            trade_log.append({
                "date": str(slice_df.index[-1].date()),
                "side": s.signal.value,
                "price": float(slice_df["close"].iloc[-1]),
                "size": 0,
            })

    html = ChartBuilder.kline_chart(df, symbol, trade_log=trade_log)
    out = output or f"logs/{symbol}_kline.html"
    ChartBuilder.save(html, out)
    console.print(f"[green]K线图已保存: {out}[/green]")
    subprocess.run(["open", out])


# ── 定时扫描（实盘/模拟） ─────────────────────────────────────────────────────

@app.command()
def live(
    symbols: str = typer.Option("AAPL,MSFT", help="逗号分隔的股票代码"),
    strategy: str = typer.Option("composite"),
    scan_interval: int = typer.Option(60, help="扫描间隔（秒）"),
):
    """实盘/模拟盘信号扫描（每隔 N 秒扫描一次）。"""
    import time

    from broker import PaperBroker
    broker = PaperBroker()

    if not broker.connect():
        console.print("[red]券商连接失败[/red]")
        raise typer.Exit(1)

    fetcher = DataFetcher(use_cache=False)
    notifier = Notifier()
    risk = RiskManager()
    strat = _get_strategy(strategy, live=True)
    symbol_list = [s.strip() for s in symbols.split(",")]

    console.print(f"[green]实盘扫描启动: {symbol_list} 策略={strategy}[/green]")

    try:
        while True:
            now = datetime.now()
            # 仅在交易时间运行
            if now.weekday() < 5 and ((9, 30) <= (now.hour, now.minute) <= (15, 0)):
                for sym in symbol_list:
                    try:
                        end = str(date.today())
                        start = str(date.today() - timedelta(days=120))
                        df = fetcher.get_kline(sym, start, end)
                        sig = strat.run(df, sym)
                        console.print(sig)

                        if sig.signal.value == "BUY":
                            balance = broker.get_balance()
                            total_equity = balance["total_equity"]
                            pos_pct = risk.kelly_position_size(0.55, 0.10, 0.05)
                            buy_value = total_equity * pos_pct
                            price = float(df["close"].iloc[-1])
                            qty = int(buy_value / price / 100) * 100
                            if qty > 0 and risk.can_open_position(sym, buy_value, total_equity):
                                order = broker.buy(sym, qty, price)
                                console.print(f"[red]买单: {order}[/red]")
                                notifier.send_signal(sig)

                        elif sig.signal.value == "SELL":
                            positions = broker.get_positions()
                            for p in positions:
                                if p["symbol"] == sym and p["quantity"] > 0:
                                    price = float(df["close"].iloc[-1])
                                    order = broker.sell(sym, p["quantity"], price)
                                    console.print(f"[green]卖单: {order}[/green]")
                                    notifier.send_signal(sig)
                                    break

                        if risk.is_daily_loss_breached(broker.get_balance()["total_equity"]):
                            notifier.send_risk_alert("单日亏损触发熔断，暂停交易！")
                            break

                    except Exception as e:
                        logger.error(f"扫描 {sym} 失败: {e}")

            time.sleep(scan_interval)

    except KeyboardInterrupt:
        console.print("\n[yellow]已停止扫描[/yellow]")
        broker.disconnect()


@app.command(name="factor-paper")
def factor_paper(
    universe: str         = typer.Option("sp500",    help="股票池: sp500 | tsx60"),
    top_n: int            = typer.Option(10,          help="每期持有股票数"),
    rebalance_days: int   = typer.Option(30,          help="调仓周期（日历日）"),
    initial_cash: float   = typer.Option(100_000.0,  help="初始资金（首次运行有效）"),
    transaction_cost: float = typer.Option(0.001,    help="单边交易成本（0.1%）"),
):
    """
    因子选股实盘模拟（路线A）：每月用 FactorEngine 从全市场选 Top N，等权持有，对标 SPY。
    与 factor-backtest 逻辑完全一致，每次运行自动追加记录到 data/factor_paper_log.csv。
    """
    from paper_trading.factor_runner import FactorPaperRunner

    runner = FactorPaperRunner(
        universe=universe,
        top_n=top_n,
        rebalance_days=rebalance_days,
        initial_cash=initial_cash,
        transaction_cost_pct=transaction_cost,
    )
    report = runner.run()

    console.print(f"\n[bold green]── 因子选股日报 {report['date']} ──[/bold green]")
    console.print(f"  总资产: {report['total_equity']:,.2f}  现金: {report['cash']:,.2f}")
    ret_color = "green" if report["total_return"] >= 0 else "red"
    console.print(f"  累计收益: [{ret_color}]{report['total_return']:.2%}[/]")

    if report["rebalanced"]:
        console.print("\n  [bold cyan]本次调仓成交:[/bold cyan]")
        for t in report["trades"]:
            color = "red" if t["side"] == "BUY" else "green"
            console.print(f"    [{color}]{t['side']}[/] {t['symbol']} x{t['qty']:.1f} @ {t['price']:.2f}")
    else:
        last_rb = runner._state.get("last_rebalance", "未知")
        console.print(f"\n  [dim]距上次调仓: {last_rb}，下次调仓还需 "
                      f"{rebalance_days - (date.today() - date.fromisoformat(last_rb)).days} 天[/dim]")

    if report["holdings"]:
        console.print("\n  [bold]当前持仓:[/bold]")
        for sym, p in report["holdings"].items():
            console.print(f"    {sym}  {p['qty']:.1f}股  成本:{p['avg_cost']:.2f}")
    console.print(f"\n  [dim]日志 → data/factor_paper_log.csv[/dim]")


@app.command(name="alpaca-paper")
def alpaca_paper(
    universe: str         = typer.Option("sp500", help="股票池: sp500 | tsx60"),
    top_n: int            = typer.Option(20,       help="每期持有股票数"),
    rebalance_days: int   = typer.Option(30,       help="调仓周期（日历日）"),
    stop_loss_pct: float  = typer.Option(0.15,     help="ATR不可用时的固定止损比例"),
    atr_multiplier: float = typer.Option(2.5,      help="ATR止损倍数，止损价=买入价-N×ATR"),
    max_sector_pct: float = typer.Option(0.25,     help="行业集中度上限，如0.25表示每个行业不超过25%仓位"),
    force_rebalance: bool = typer.Option(False, "--force-rebalance/--no-force-rebalance", help="忽略日期检查，立即执行全量换仓"),
):
    """
    因子选股实盘 —— 通过 Alpaca Paper Trading API 执行真实（模拟）订单。
    需要在 .env 中设置 ALPACA_API_KEY 和 ALPACA_SECRET_KEY。
    """
    from paper_trading.alpaca_runner import AlpacaPaperRunner
    from dotenv import load_dotenv
    load_dotenv()

    runner = AlpacaPaperRunner(
        universe=universe,
        top_n=top_n,
        rebalance_days=rebalance_days,
        stop_loss_pct=stop_loss_pct,
        atr_multiplier=atr_multiplier,
        max_sector_pct=max_sector_pct,
    )
    report = runner.run(force_rebalance=force_rebalance)

    spy_str = f"  SPY 收盘:  ${report['spy_close']}\n" if report.get("spy_close") else ""
    console.print(f"\n[bold green]Alpaca Paper Trading — {report['date']}[/bold green]")
    console.print(f"  账户净值:  [bold]${report['equity']:,.2f}[/bold]")
    if spy_str:
        console.print(spy_str.rstrip())
    console.print(f"  当前持仓:  {report['holdings']} 只")
    console.print(f"  今日订单:  {report['trades']} 笔")
    if report["rebalanced"]:
        console.print("  [cyan]已调仓（评分加权）[/cyan]")
    else:
        console.print("  [dim]未到调仓日（ATR止损检查完成）[/dim]")
    console.print(f"\n  [dim]日志 → data/alpaca_paper_log.csv[/dim]")


@app.command(name="alpaca-status")
def alpaca_status():
    """显示 Alpaca 模拟盘当前净值、收益 vs SPY、距下次换仓天数。"""
    import csv as _csv
    import pandas as pd
    from dotenv import load_dotenv
    load_dotenv()

    log_file = Path("data/alpaca_paper_log.csv")
    if not log_file.exists():
        console.print("[red]没有日志文件，请先运行 alpaca-paper[/red]")
        return

    # Read manually to handle mixed 6-column (old) and 7-column (new) rows
    rows = []
    with open(log_file, newline="") as f:
        reader = _csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) == 7:
                rows.append(dict(zip(
                    ["date","equity","spy_close","rebalanced","n_holdings","n_trades","holdings"],
                    row
                )))
            elif len(row) == 6:
                rows.append(dict(zip(
                    ["date","equity","rebalanced","n_holdings","n_trades","holdings"],
                    row
                )))
    if not rows:
        console.print("[red]日志为空[/red]")
        return
    df = pd.DataFrame(rows)
    df["equity"]     = pd.to_numeric(df["equity"])
    df["rebalanced"] = pd.to_numeric(df["rebalanced"])
    df["n_holdings"] = pd.to_numeric(df["n_holdings"])
    if "spy_close" in df.columns:
        df["spy_close"] = pd.to_numeric(df["spy_close"], errors="coerce")
    if df.empty:
        console.print("[red]日志为空[/red]")
        return

    # 起点：第一次全量换仓那天
    first_rb = df[df["rebalanced"] == 1].iloc[0] if any(df["rebalanced"] == 1) else df.iloc[0]
    last_rb  = df[df["rebalanced"] == 1].iloc[-1] if any(df["rebalanced"] == 1) else None
    latest   = df.iloc[-1]

    equity_start = float(first_rb["equity"])
    equity_now   = float(latest["equity"])
    total_ret    = (equity_now - equity_start) / equity_start * 100

    # SPY 对比（只用有 spy_close 的行）
    spy_rows = df[pd.to_numeric(df.get("spy_close", pd.Series(dtype=float)), errors="coerce").notna()]
    spy_ret_str = "—"
    if len(spy_rows) >= 2:
        spy_start = float(spy_rows.iloc[0]["spy_close"])
        spy_now   = float(spy_rows.iloc[-1]["spy_close"])
        if spy_start > 0:
            spy_ret     = (spy_now - spy_start) / spy_start * 100
            alpha       = total_ret - spy_ret
            spy_ret_str = f"{spy_ret:+.2f}%  (超额 {alpha:+.2f}%)"

    # 距下次换仓
    if last_rb is not None:
        last_rb_date  = date.fromisoformat(str(last_rb["date"]))
        days_since    = (date.today() - last_rb_date).days
        days_to_next  = max(0, 30 - days_since)
        rebalance_str = f"{days_to_next} 天后（上次 {last_rb_date}）"
    else:
        rebalance_str = "未知"

    # 最近 5 个交易日收益
    recent = df.tail(5)
    if len(recent) >= 2:
        r5 = (float(recent.iloc[-1]["equity"]) / float(recent.iloc[0]["equity"]) - 1) * 100
        recent_str = f"{r5:+.2f}%"
    else:
        recent_str = "—"

    console.print("\n[bold cyan]Alpaca 模拟盘状态[/bold cyan]")
    console.print(f"  当前净值:      [bold]${equity_now:,.2f}[/bold]")
    console.print(f"  建仓以来收益:  [bold green]{total_ret:+.2f}%[/bold green]")
    console.print(f"  同期 SPY:      {spy_ret_str}")
    console.print(f"  近 5 日收益:   {recent_str}")
    console.print(f"  下次全量换仓:  {rebalance_str}")
    console.print(f"  当前持仓:      {int(latest['n_holdings'])} 只\n")


@app.command(name="alpaca-health")
def alpaca_health():
    """检查策略是否正常运行（超过 48 小时未跑则警告）。"""
    last_run_file = Path("data/alpaca_last_run.txt")
    if not last_run_file.exists():
        console.print("[red]从未运行过 alpaca-paper，或版本太旧（重新跑一次即可）[/red]")
        return
    last_run = datetime.fromisoformat(last_run_file.read_text().strip())
    diff     = datetime.now() - last_run
    hours    = diff.total_seconds() / 3600
    if hours > 48:
        console.print(
            f"[red]警告：策略已 {int(hours)} 小时未运行！"
            f"（上次: {last_run.strftime('%Y-%m-%d %H:%M')}）[/red]"
        )
    else:
        console.print(
            f"[green]正常：上次运行于 {last_run.strftime('%Y-%m-%d %H:%M')}"
            f"（{int(hours)} 小时前）[/green]"
        )


# ── 自动策略优化 ──────────────────────────────────────────────────────────────

@app.command(name="auto-optimize")
def auto_optimize(
    symbols: str = typer.Option("AAPL,MSFT,GOOGL", help="用哪些股票做优化基准"),
    lookback: int = typer.Option(12, help="用最近几个月数据优化（最后3个月自动留作样本外验证）"),
    metric: str = typer.Option("sharpe_ratio", help="优化目标: sharpe_ratio/total_return/calmar_ratio"),
):
    """自动优化所有策略参数，含样本外验证，结果保存后composite策略自动使用最优配置。"""
    from backtest.auto_optimizer import AutoOptimizer
    symbol_list = [s.strip() for s in symbols.split(",")]

    train_months = max(lookback - 3, 9)
    console.print(f"[green]开始自动优化: {symbol_list}  总数据{lookback}个月  目标={metric}[/green]")
    console.print(f"[dim]训练期: 前{train_months}个月  样本外验证: 最后3个月（防止过拟合）[/dim]")
    console.print("[yellow]这需要几分钟，请耐心等待...[/yellow]\n")

    opt = AutoOptimizer(symbol_list, lookback_months=lookback, metric=metric)
    result = opt.run()

    console.print(f"\n[bold green]── 优化完成 ──[/bold green]")
    console.print(f"  更新时间: {result['updated_at']}")

    weights = result.get("composite_weights", {})
    has_oos = any(info.get("oos_sharpe") is not None for info in result["strategies"].values())

    console.print(f"\n  [bold]各策略得分及权重:[/bold]")
    cols = ["策略", "最优参数", f"样本内{metric}"]
    if has_oos:
        cols.append("样本外Sharpe")
    cols.append("权重")
    table = Table(*cols)

    for name, info in result["strategies"].items():
        oos = info.get("oos_sharpe")
        oos_color = "green" if oos is not None and oos > 0 else "red"
        oos_str = f"[{oos_color}]{oos:.4f}[/{oos_color}]" if oos is not None else "[dim]N/A[/dim]"
        row = [name, str(info["params"]), f"{info['avg_score']:.4f}"]
        if has_oos:
            row.append(oos_str)
        row.append(f"{weights.get(name, 0):.1%}")
        table.add_row(*row)

    console.print(table)

    if has_oos:
        oos_vals = [info["oos_sharpe"] for info in result["strategies"].values()
                    if info.get("oos_sharpe") is not None]
        avg_oos = sum(oos_vals) / len(oos_vals) if oos_vals else 0.0
        n_positive = sum(1 for v in oos_vals if v > 0)
        grade_color = "green" if avg_oos > 0.3 and n_positive >= len(oos_vals) * 0.6 else \
                      "yellow" if avg_oos > 0 else "red"
        console.print(f"\n  样本外平均Sharpe: [{grade_color}]{avg_oos:.4f}[/{grade_color}]"
                      f"  ({n_positive}/{len(oos_vals)} 个策略样本外为正)")

    console.print(f"\n[green]最优参数已保存，下次运行 composite 策略自动使用。[/green]")


# ── 多因子选股扫描 ────────────────────────────────────────────────────────────

@app.command()
def scan(
    symbols: str = typer.Option(
        "AAPL,MSFT,GOOGL,AMZN,META,NVDA,JPM,V,JNJ,HD",
        help="逗号分隔的股票代码",
    ),
    lookback: int = typer.Option(120, help="拉取历史K线天数"),
    top: int = typer.Option(5, help="显示评分最高的N只股票"),
    use_ic: bool = typer.Option(False, "--use-ic", help="使用 ic-calibrate 保存的权重（data/factor_weights.json）"),
):
    """多因子评分选股：对一批股票打分，按综合得分排名（含基本面因子）。"""
    from factor import FactorEngine

    symbol_list = [s.strip() for s in symbols.split(",")]
    fetcher = DataFetcher()

    # FactorEngine 启动时自动加载 data/factor_weights.json（若存在）
    # --use-ic 保留向后兼容，现在等同于默认行为
    engine = FactorEngine()
    if Path("data/factor_weights.json").exists():
        console.print(f"[cyan]已自动加载IC权重[/cyan]")

    end = str(date.today())
    start = str(date.today() - timedelta(days=lookback * 2))

    from data.tv_signals import get_tv_signal

    scores = []
    tv_cache: dict[str, dict] = {}
    composite = CompositeStrategy(use_tv=True)

    with console.status("[bold green]正在计算因子得分 + TradingView 信号（含周线确认）...[/bold green]"):
        for sym in symbol_list:
            try:
                df = fetcher.get_kline(sym, start, end)
                fund = fetcher.get_fundamentals(sym)
                fs = engine.compute(df, sym, fundamentals=fund)

                # Use composite strategy score (TV already included with weekly confirmation)
                df_ind = composite.compute_indicators(df.copy())
                comp_sig = composite.generate_signal(df_ind, sym)
                comp_score = float(comp_sig.metadata.get("score", 0.0))

                # Final score: factor score (60%) + composite/TV score (40%)
                fs.total_score = round(fs.total_score * 0.6 + comp_score * 0.4, 4)

                # Keep TV cache for display
                tv_cache[sym] = {
                    "recommendation": _comp_score_to_rec(comp_score),
                    "weekly": comp_sig.metadata.get("tv_weekly", "HOLD"),
                    "blocked": "[weekly bearish — BUY blocked]" in comp_sig.reason,
                }
                scores.append(fs)
            except Exception as e:
                logger.warning(f"{sym} 获取失败: {e}")

    if not scores:
        console.print("[red]所有股票获取失败[/red]")
        raise typer.Exit(1)

    scores.sort(key=lambda x: x.total_score, reverse=True)

    table = Table(title=f"多因子 + TV（日线+周线）评分排行（共{len(scores)}只）", show_lines=True)
    table.add_column("排名", style="bold", width=4)
    table.add_column("代码", width=8)
    table.add_column("评级", width=4)
    table.add_column("综合得分", width=8)
    table.add_column("动量20", width=8)
    table.add_column("均线排列", width=8)
    table.add_column("RSI", width=6)
    table.add_column("TV日线", width=12)
    table.add_column("周线", width=8)
    table.add_column("主要驱动", style="dim")

    for rank, fs in enumerate(scores, 1):
        grade_color = {"A": "bold green", "B": "green", "C": "white",
                       "D": "red", "E": "bold red"}.get(fs.grade, "white")
        score_color = "green" if fs.total_score > 0 else "red"
        tv = tv_cache.get(fs.symbol, {})
        rec = tv.get("recommendation", "N/A")
        weekly = tv.get("weekly", "HOLD")
        blocked = tv.get("blocked", False)
        tv_color = {"STRONG_BUY": "green", "BUY": "green",
                    "NEUTRAL": "yellow", "SELL": "red", "STRONG_SELL": "red"}.get(rec, "white")
        weekly_color = "red" if weekly == "SELL" else "green" if weekly == "BUY" else "yellow"
        table.add_row(
            str(rank),
            fs.symbol,
            f"[{grade_color}]{fs.grade}[/]",
            f"[{score_color}]{fs.total_score:+.3f}[/]",
            f"{fs.factors.get('momentum_20', 0):+.3f}",
            f"{fs.factors.get('ma_alignment', 0):+.3f}",
            f"{fs.factors.get('rsi_score', 0):+.3f}",
            f"[{tv_color}]{rec}[/{tv_color}]{'⛔' if blocked else ''}",
            f"[{weekly_color}]{weekly}[/{weekly_color}]",
            fs.reason,
        )

    console.print(table)
    console.print(f"\n[bold green]推荐关注 TOP {min(top, len(scores))}：[/bold green] "
                  + "  ".join(f"[green]{s.symbol}[/green]({s.grade})"
                              for s in scores[:top]))


# ── 自动选股（S&P 500）─────────────────────────────────────────────────────────

@app.command(name="select-stocks")
def select_stocks_cmd(
    top: int = typer.Option(5, help="选出最强的N只股票"),
    min_price: float = typer.Option(5.0, help="最低股价过滤"),
    max_price: float = typer.Option(150.0, help="最高股价过滤"),
    min_grade: str = typer.Option("C", help="因子评级门槛 A/B/C/D/E，只保留达标股票"),
):
    """从S&P 500成分股中自动选出当前信号最强的股票（策略信号 + 因子评分双重过滤）。"""
    import json
    from data.stock_selector import select_stocks
    from factor import FactorEngine

    grade_order = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "N/A": 0}
    min_grade_val = grade_order.get(min_grade.upper(), 3)

    console.print("[bold green]开始从S&P 500自动选股...[/bold green]")
    console.print(f"  价格范围: {min_price}-{max_price}元  因子门槛: {min_grade}级以上  选前{top}只\n")

    # FactorEngine 自动加载 data/factor_weights.json
    engine = FactorEngine()
    if Path("data/factor_weights.json").exists():
        console.print(f"[cyan]已自动加载IC权重[/cyan]")
    fetcher = DataFetcher()
    end = str(date.today())
    start = str(date.today() - timedelta(days=240))

    with console.status("[bold]扫描中，大约需要3-5分钟...[/bold]"):
        candidates = select_stocks(top_n=top * 4, min_price=min_price, max_price=max_price)

    if not candidates:
        console.print("[red]未找到符合条件的股票[/red]")
        raise typer.Exit(1)

    # 因子评分二次过滤
    enriched = []
    with console.status("[bold]计算因子评分...[/bold]"):
        for r in candidates:
            sym = r["symbol"]
            try:
                df = fetcher.get_kline(sym, start, end)
                fund = fetcher.get_fundamentals(sym)
                fs = engine.compute(df, sym, fundamentals=fund)
                r["factor_grade"] = fs.grade
                r["factor_score"] = fs.total_score
                r["factor_reason"] = fs.reason
                if grade_order.get(fs.grade, 0) >= min_grade_val:
                    enriched.append(r)
            except Exception:
                r["factor_grade"] = "N/A"
                r["factor_score"] = 0.0
                r["factor_reason"] = ""
                if min_grade_val <= 0:
                    enriched.append(r)

    # 综合排序：策略得分 × 0.5 + 因子得分 × 0.5
    for r in enriched:
        r["combined_score"] = r["score"] * 0.5 + r["factor_score"] * 0.5
    enriched.sort(key=lambda x: x["combined_score"], reverse=True)
    results = enriched[:top]

    if not results:
        console.print(f"[yellow]策略找到{len(candidates)}只候选，但因子评级均低于{min_grade}级[/yellow]")
        console.print("[dim]可用 --min-grade E 放宽门槛[/dim]")
        raise typer.Exit(0)

    table = Table(title=f"自动选股结果 TOP {len(results)}（策略+因子双重过滤）",
                  show_lines=True, expand=False)
    table.add_column("代码",   style="cyan",  min_width=8)
    table.add_column("现价",   justify="right", min_width=8)
    table.add_column("信号",   justify="center", min_width=6)
    table.add_column("策略分", justify="right", min_width=8)
    table.add_column("因子级", justify="center", min_width=6)
    table.add_column("因子分", justify="right", min_width=8)
    table.add_column("主要驱动", style="dim",  min_width=20)

    for r in results:
        sig_color = "green" if r["signal"] == "BUY" else "red" if r["signal"] == "SELL" else "yellow"
        grade = r.get("factor_grade", "N/A")
        grade_color = {"A": "bold green", "B": "green", "C": "white",
                       "D": "red", "E": "bold red"}.get(grade, "dim")
        table.add_row(
            r["symbol"],
            f"{r['price']:.2f}",
            f"[{sig_color}]{r['signal']}[/]",
            f"{r['score']:+.3f}",
            f"[{grade_color}]{grade}[/]",
            f"{r.get('factor_score', 0):+.3f}",
            r.get("factor_reason", r["reason"])[:40],
        )

    console.print(table)


@app.command(name="factor-backtest")
def factor_backtest(
    universe: str = typer.Option("sp500", help="股票池: sp500 | tsx60 | russell2000 | 逗号分隔代码"),
    top: int = typer.Option(10, help="每期持有股票数"),
    rebalance_days: int = typer.Option(20, help="调仓周期（交易日）"),
    start: str = typer.Option("2020-01-01", help="回测开始日期"),
    end: str = typer.Option(str(date.today()), help="回测结束日期"),
    max_stocks: int = typer.Option(80, help="抽样数量（--no-sample 时忽略）"),
    no_sample: bool = typer.Option(False, "--no-sample", help="用全量股票池（~500 只，与 factor-paper 一致，需 20-30 分钟）"),
    initial_cash: float = typer.Option(100_000, help="初始资金"),
    transaction_cost: float = typer.Option(0.001, help="单边交易成本（默认 0.1%）"),
    n_splits: int = typer.Option(1, help="滚动测试段数（1=不分段，3=三段滚动）"),
    save_report: bool = typer.Option(True, "--save-report/--no-save-report", help="保存 CSV 报告到 data/"),
    no_fundamentals: bool = typer.Option(False, "--no-fundamentals", help="禁用基本面因子，纯价量/动量策略（用于对比测试）"),
    max_sector_pct: float = typer.Option(0.25, "--max-sector-pct", help="行业集中度上限（默认0.25）"),
):
    """
    截面因子选股回测：每隔 rebalance_days 个交易日用 FactorEngine 打分，
    选 Top N 只等权持有，对比 SPY 买入持有基准。

    基本面使用 yfinance 历史年报快照（可回溯约 4-5 年），每次调仓查调仓日之前最近一期。
    2020-2021 年初若无历史年报，对应调仓退化为纯价量因子（无前视偏差）。
    """
    import random
    import pandas as pd
    from factor import FactorEngine
    from data.stock_selector import get_sp500_symbols

    fetcher = DataFetcher(use_cache=True)
    engine = FactorEngine()
    if Path("data/factor_weights.json").exists():
        console.print("[cyan]已加载 IC 校准权重（data/factor_weights.json）[/cyan]")

    # 确定股票池
    if universe.lower() in ("sp500", "nasdaq100"):
        all_syms = get_sp500_symbols()
        if not all_syms:
            console.print("[red]获取股票列表失败[/red]")
            raise typer.Exit(1)
        all_syms = sorted(all_syms)
        if no_sample:
            symbols = all_syms
            console.print(f"[cyan]全量模式：使用全部 {len(symbols)} 只股票（与 factor-paper 一致）[/cyan]")
        else:
            random.seed(42)
            symbols = random.sample(all_syms, min(max_stocks, len(all_syms)))
    elif universe.lower() == "tsx60":
        from data.stock_selector import get_tsx60_symbols
        symbols = get_tsx60_symbols()
    elif universe.lower() == "russell2000":
        from data.stock_selector import get_russell2000_symbols
        all_syms = get_russell2000_symbols()
        if not all_syms:
            console.print("[red]Russell 2000 列表获取失败[/red]")
            raise typer.Exit(1)
        all_syms = sorted(all_syms)
        if no_sample:
            symbols = all_syms
            console.print(f"[cyan]全量模式：Russell 2000 全部 {len(symbols)} 只[/cyan]")
        else:
            random.seed(42)
            symbols = random.sample(all_syms, min(max_stocks, len(all_syms)))
    else:
        symbols = [s.strip() for s in universe.split(",")]
    console.print(f"[bold]股票池：{len(symbols)} 只  调仓周期：{rebalance_days}日  每期持有：{top} 只[/bold]")

    # 拉取 OHLCV
    stock_data: dict[str, pd.DataFrame] = {}
    console.print(f"拉取历史数据（{start} ~ {end}）...")
    for i, sym in enumerate(symbols):
        try:
            df = fetcher.get_kline(sym, start, end)
            if not df.empty and len(df) >= 130:
                stock_data[sym] = df
        except Exception:
            pass
        if (i + 1) % 20 == 0:
            console.print(f"  {i+1}/{len(symbols)} 只，有效 {len(stock_data)} 只")

    # 拉取历史基本面（yfinance 年报）
    hist_fund: dict[str, dict[str, dict]] = {}  # sym → {date_str → fund_dict}

    if no_fundamentals:
        console.print("[yellow]--no-fundamentals：跳过基本面，纯价量/动量策略[/yellow]")
    else:
        missing = list(stock_data.keys())
        if missing:
            console.print(f"拉取历史年报数据（yfinance，{len(missing)} 只）...")
            yf_ok = 0
            for sym in missing:
                try:
                    hf = fetcher.get_historical_fundamentals(sym)
                    if hf:
                        hist_fund[sym] = hf
                        yf_ok += 1
                except Exception:
                    pass
            console.print(f"  yfinance 年报：{yf_ok}/{len(missing)} 只有效")

    console.print(f"[green]就绪：{len(stock_data)} 只股票，{len(hist_fund)} 只有历史基本面[/green]")

    if len(stock_data) < top * 2:
        console.print(f"[red]有效股票太少（{len(stock_data)}），至少需要 {top * 2} 只[/red]")
        raise typer.Exit(1)

    # 行业映射（用于约束，从已缓存的 fundamentals 读取，不产生额外网络请求）
    sector_map: dict[str, str] = {}
    for sym in stock_data:
        try:
            sector_map[sym] = fetcher.get_fundamentals(sym).get("sector") or "Unknown"
        except Exception:
            sector_map[sym] = "Unknown"

    # 与 alpaca_runner 相同的约束逻辑（share-class 去重 + 行业上限）
    _BT_SHARE_CLASS = {"GOOGL": "GOOG", "BRK-A": "BRK-B", "NWS": "NWSA"}

    def _bt_apply_constraints(
        scores: list[tuple[str, float]], top_n: int, max_sector_pct: float = 0.25
    ) -> list[tuple[str, float]]:
        groups: dict[str, tuple[str, float]] = {}
        for sym, score in scores:
            key = _BT_SHARE_CLASS.get(sym, sym)
            if key not in groups or score > groups[key][1]:
                groups[key] = (sym, score)
        deduped = sorted(groups.values(), key=lambda x: x[1], reverse=True)

        max_per = max(1, int(top_n * max_sector_pct))
        sec_cnt: dict[str, int] = {}
        selected: list[tuple[str, float]] = []
        for sym, score in deduped:
            if len(selected) >= top_n:
                break
            sec = sector_map.get(sym, "Unknown")
            if sec_cnt.get(sec, 0) < max_per:
                selected.append((sym, score))
                sec_cnt[sec] = sec_cnt.get(sec, 0) + 1
        if len(selected) < top_n:
            held = {s for s, _ in selected}
            extras = [(s, sc) for s, sc in deduped if s not in held]
            selected.extend(extras[: top_n - len(selected)])
        return selected

    def _bt_score_weights(top_scores: list[tuple[str, float]]) -> dict[str, float]:
        syms  = [s for s, _ in top_scores]
        vals  = [v for _, v in top_scores]
        min_v = min(vals)
        shifted = [v - min_v + 0.1 for v in vals]
        total   = sum(shifted)
        return {sym: w / total for sym, w in zip(syms, shifted)}

    # 取公共交易日轴
    all_dates = sorted(set().union(*[set(df.index) for df in stock_data.values()]))

    # SPY 基准
    spy_df = fetcher.get_kline("SPY", start, end)
    spy_ret = spy_df["close"].pct_change().fillna(0)
    spy_index = {d: i for i, d in enumerate(spy_df.index)}

    window = 120  # 因子计算所需历史窗口

    def _get_fund_at_date(sym: str, today) -> dict:
        """返回 today 之前最近一期年报基本面，没有则返回空 dict（退化为纯价量）。"""
        snapshots = hist_fund.get(sym)
        if not snapshots:
            return {}
        today_str = str(today.date()) if hasattr(today, "date") else str(today)[:10]
        valid = [d for d in snapshots if d <= today_str]
        return snapshots[max(valid)] if valid else {}

    def sharpe(r: np.ndarray) -> float:
        return float(np.mean(r) / np.std(r) * np.sqrt(252)) if np.std(r) > 1e-9 else 0.0

    def max_dd(vals: np.ndarray) -> float:
        peak = np.maximum.accumulate(vals)
        return float(((vals - peak) / peak).min()) * 100

    def _run_segment(seg_dates: list, seg_label: str) -> dict:
        """在给定日期列表上跑一段回测，返回汇总指标。"""
        port_val  = initial_cash
        spy_v     = initial_cash
        holdings: list[str] = []
        weights:  dict[str, float] = {}
        last_rb   = -rebalance_days
        rb_log: list[dict] = []
        c_port: list[float] = []
        c_spy:  list[float] = []

        for di, today in enumerate(seg_dates):
            if di < window:
                c_port.append(port_val)
                c_spy.append(spy_v)
                continue

            if today in spy_index:
                si = spy_index[today]
                if si > 0:
                    spy_v *= (1 + float(spy_ret.iloc[si]))

            # 调仓
            if di - last_rb >= rebalance_days:
                scores: list[tuple[str, float]] = []
                for sym, full_df in stock_data.items():
                    loc = full_df.index.get_indexer([today], method="ffill")[0]
                    if loc < window:
                        continue
                    slice_df = full_df.iloc[loc - window: loc]
                    if len(slice_df) < 60:
                        continue
                    try:
                        fs = engine.compute(slice_df, sym,
                                            fundamentals=_get_fund_at_date(sym, today))
                        scores.append((sym, fs.total_score))
                    except Exception:
                        pass

                if len(scores) >= top:
                    scores.sort(key=lambda x: x[1], reverse=True)
                    constrained = _bt_apply_constraints(scores, top, max_sector_pct)
                    new_h   = [s for s, _ in constrained]
                    weights = _bt_score_weights(constrained)

                    old_set  = set(holdings)
                    new_set  = set(new_h)
                    turnover = len(new_set - old_set) / top
                    port_val *= (1 - turnover * transaction_cost)

                    date_str = str(today.date()) if hasattr(today, "date") else str(today)[:10]
                    rb_log.append({
                        "date": date_str,
                        "holdings": new_h,
                        "top_score": round(scores[0][1], 3),
                        "bottom_score": round(constrained[-1][1], 3),
                        "turnover": round(turnover * 100, 1),
                    })
                    holdings = new_h
                    last_rb  = di

            # 持仓收益（评分加权）
            if holdings:
                day_ret = 0.0
                eq_w = 1.0 / len(holdings)  # fallback if weights missing
                for sym in holdings:
                    df_sym = stock_data.get(sym)
                    if df_sym is None:
                        continue
                    loc = df_sym.index.get_indexer([today], method="ffill")[0]
                    if loc > 0:
                        ret = float(df_sym["close"].iloc[loc]) / float(df_sym["close"].iloc[loc - 1]) - 1
                        day_ret += ret * weights.get(sym, eq_w)
                port_val *= (1 + day_ret)

            c_port.append(round(port_val, 2))
            c_spy.append(round(spy_v, 2))

        if len(c_port) < 2:
            return {}

        pa = np.array(c_port)
        sa = np.array(c_spy)
        pr = np.diff(pa) / pa[:-1]
        sr = np.diff(sa) / sa[:-1]
        pt = (port_val / initial_cash - 1) * 100
        st = (spy_v   / initial_cash - 1) * 100

        dates_str = [str(d.date()) if hasattr(d, "date") else str(d)[:10] for d in seg_dates]
        return {
            "label":      seg_label,
            "port_total": pt,
            "spy_total":  st,
            "alpha":      pt - st,
            "port_sharpe": sharpe(pr),
            "spy_sharpe":  sharpe(sr),
            "port_mdd":   max_dd(pa),
            "spy_mdd":    max_dd(sa),
            "n_rebalance": len(rb_log),
            "rb_log":     rb_log,
            "dates":      dates_str,
            "c_port":     c_port,
            "c_spy":      c_spy,
        }

    # ── 分段回测 ─────────────────────────────────────────────────────────────────
    all_dates = sorted(set().union(*[set(df.index) for df in stock_data.values()]))

    segments: list[list] = []
    if n_splits <= 1:
        segments = [all_dates]
    else:
        sz = len(all_dates) // n_splits
        for i in range(n_splits):
            seg = all_dates[i * sz: (i + 1) * sz if i < n_splits - 1 else len(all_dates)]
            segments.append(seg)

    seg_results: list[dict] = []
    for i, seg in enumerate(segments):
        label = f"全段" if n_splits <= 1 else f"第{i+1}段 {str(seg[0])[:10]}~{str(seg[-1])[:10]}"
        console.print(f"[bold]回测中：{label}...[/bold]")
        r = _run_segment(seg, label)
        if r:
            seg_results.append(r)

    if not seg_results:
        console.print("[red]回测无有效结果[/red]")
        raise typer.Exit(1)

    # ── 输出结果 ─────────────────────────────────────────────────────────────────
    console.print()
    result_table = Table(title="截面因子选股回测结果", show_lines=True)
    result_table.add_column("段", style="bold")
    result_table.add_column("因子收益", justify="right")
    result_table.add_column("SPY收益", justify="right")
    result_table.add_column("超额", justify="right")
    result_table.add_column("因子Sharpe", justify="right")
    result_table.add_column("SPY Sharpe", justify="right")
    result_table.add_column("最大回撤", justify="right")
    result_table.add_column("调仓次数", justify="right")

    for r in seg_results:
        alpha_color = "green" if r["alpha"] > 0 else "red"
        result_table.add_row(
            r["label"],
            f"{r['port_total']:+.1f}%",
            f"{r['spy_total']:+.1f}%",
            f"[{alpha_color}]{r['alpha']:+.1f}%[/{alpha_color}]",
            f"{r['port_sharpe']:.2f}",
            f"{r['spy_sharpe']:.2f}",
            f"{r['port_mdd']:.1f}%",
            str(r["n_rebalance"]),
        )

    # 合计行（各段复利相乘）— 与 HTML 报告一致
    if len(seg_results) > 1:
        p_mult = 1.0
        s_mult = 1.0
        total_rb = 0
        for r in seg_results:
            p_mult *= (1 + r["port_total"] / 100)
            s_mult *= (1 + r["spy_total"]  / 100)
            total_rb += r["n_rebalance"]
        p_tot = (p_mult - 1) * 100
        s_tot = (s_mult - 1) * 100
        a_tot = p_tot - s_tot
        a_col = "green" if a_tot > 0 else "red"
        result_table.add_row(
            "[bold]合计（复利）[/bold]",
            f"[bold]{p_tot:+.1f}%[/bold]",
            f"[bold]{s_tot:+.1f}%[/bold]",
            f"[bold][{a_col}]{a_tot:+.1f}%[/{a_col}][/bold]",
            "[dim]—[/dim]", "[dim]—[/dim]", "[dim]—[/dim]",
            f"[bold]{total_rb}[/bold]",
        )

    console.print(result_table)

    # 显示最近几次调仓（取最后一段）
    last_rb_log = seg_results[-1].get("rb_log", [])
    if last_rb_log:
        rb_table = Table(title="最近调仓记录", show_lines=True)
        rb_table.add_column("日期")
        rb_table.add_column("持仓")
        rb_table.add_column("换手率", justify="right")
        rb_table.add_column("最高分", justify="right")
        for rb in last_rb_log[-5:]:
            rb_table.add_row(
                rb["date"],
                " ".join(rb["holdings"][:5]) + ("…" if len(rb["holdings"]) > 5 else ""),
                f"{rb['turnover']}%",
                str(rb["top_score"]),
            )
        console.print(rb_table)

    # 3段一致性评价
    if len(seg_results) >= 2:
        wins = sum(1 for r in seg_results if r["alpha"] > 0)
        console.print(
            f"\n[bold]跨段一致性：{wins}/{len(seg_results)} 段超额为正[/bold] "
            + ("[green]稳定[/green]" if wins == len(seg_results) else
               "[yellow]部分有效[/yellow]" if wins > len(seg_results) // 2 else
               "[red]不稳定[/red]")
        )

    console.print(
        f"\n[dim]股票池 {len(stock_data)} 只  调仓周期 {rebalance_days}日  "
        f"每期持 {top} 只  单边交易成本 {transaction_cost*100:.2f}%[/dim]"
    )

    if save_report:
        import csv as _csv
        import subprocess
        report_dir = Path("data")
        logs_dir   = Path("logs")
        report_dir.mkdir(exist_ok=True)
        logs_dir.mkdir(exist_ok=True)

        # 每日净值曲线 CSV
        equity_path = report_dir / "factor_backtest_equity.csv"
        with open(equity_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["date", "portfolio", "spy", "port_return_pct", "spy_return_pct"])
            for r in seg_results:
                init     = r["c_port"][0] if r["c_port"] else initial_cash
                spy_init = r["c_spy"][0]  if r["c_spy"]  else initial_cash
                for date_s, pv, sv in zip(r["dates"], r["c_port"], r["c_spy"]):
                    w.writerow([
                        date_s, round(pv, 2), round(sv, 2),
                        round((pv / init - 1) * 100, 4),
                        round((sv / spy_init - 1) * 100, 4),
                    ])

        # 调仓记录 CSV
        trades_path = report_dir / "factor_backtest_trades.csv"
        with open(trades_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["date", "holdings", "turnover_pct", "top_score", "bottom_score"])
            for r in seg_results:
                for rb in r["rb_log"]:
                    w.writerow([rb["date"], "|".join(rb["holdings"]),
                                rb["turnover"], rb["top_score"], rb["bottom_score"]])

        # HTML 报告（TradingView + Plotly）
        html = ChartBuilder.factor_backtest_report(
            seg_results=seg_results,
            universe=universe,
            top_n=top,
            rebalance_days=rebalance_days,
            start=start,
            end=end,
            initial_cash=initial_cash,
        )
        html_path = logs_dir / f"factor_backtest_{universe}_top{top}_report.html"
        html_path.write_text(html, encoding="utf-8")

        console.print(f"\n[green]报告已保存 →[/green]")
        console.print(f"  [cyan]{html_path}[/cyan]  （HTML 报告，自动打开）")
        console.print(f"  [dim]{equity_path}[/dim]  （净值曲线 CSV）")
        console.print(f"  [dim]{trades_path}[/dim]  （调仓记录 CSV）")
        subprocess.run(["open", str(html_path)])


@app.command(name="rotation-backtest")
def rotation_backtest(
    assets: str = typer.Option("SPY,QQQ,TLT,GLD", help="轮动资产池（逗号分隔）"),
    lookback: int = typer.Option(63, help="动量回看窗口（交易日，63≈3个月）"),
    rebalance_days: int = typer.Option(21, help="调仓周期（约一个月）"),
    start: str = typer.Option("2015-01-01", help="回测开始日期"),
    end: str = typer.Option(str(date.today()), help="回测结束日期"),
    initial_cash: float = typer.Option(100_000, help="初始资金"),
    transaction_cost: float = typer.Option(0.001, help="单边交易成本"),
    cash_filter: bool = typer.Option(True, "--cash-filter/--no-cash-filter",
                                      help="所有资产动量为负时退出为现金"),
    benchmark: str = typer.Option("SPY", help="对比基准"),
):
    """
    绝对动量轮动策略：每月选动量最强的单一资产全仓持有。

    默认轮动池：SPY（美股）、QQQ（科技）、TLT（长债）、GLD（黄金）。
    加现金过滤：若最强资产的动量仍为负，持现金，避免熊市裸跑。
    """
    import pandas as pd

    fetcher = DataFetcher(use_cache=True)
    asset_list = [a.strip().upper() for a in assets.split(",")]

    # 拉取所有资产价格
    price_data: dict[str, pd.Series] = {}
    console.print(f"拉取数据：{' '.join(asset_list)} + {benchmark}...")
    for sym in set(asset_list + [benchmark]):
        try:
            df = fetcher.get_kline(sym, start, end)
            if not df.empty:
                price_data[sym] = df["close"].rename(sym)
        except Exception as e:
            console.print(f"  [red]{sym} 失败：{e}[/red]")

    missing = [a for a in asset_list if a not in price_data]
    if missing:
        console.print(f"[red]缺少数据：{missing}[/red]")
        raise typer.Exit(1)

    # 对齐到公共日期索引
    combined = pd.concat([price_data[a] for a in asset_list], axis=1).dropna()
    combined.columns = asset_list

    bench_series = price_data.get(benchmark, combined.iloc[:, 0])
    bench_aligned = bench_series.reindex(combined.index).ffill()

    # 回测
    port_val = initial_cash
    bench_val = initial_cash
    holding = None            # 当前持有的资产代码
    last_rb = -rebalance_days
    rb_log: list[dict] = []
    c_port: list[float] = []
    c_bench: list[float] = []

    def sharpe(r: np.ndarray) -> float:
        return float(np.mean(r) / np.std(r) * np.sqrt(252)) if np.std(r) > 1e-9 else 0.0

    def max_dd(vals: np.ndarray) -> float:
        peak = np.maximum.accumulate(vals)
        return float(((vals - peak) / peak).min()) * 100

    for di in range(len(combined)):
        today = combined.index[di]

        # 基准每日更新
        if di > 0:
            bench_ret = float(bench_aligned.iloc[di]) / float(bench_aligned.iloc[di - 1]) - 1
            bench_val *= (1 + bench_ret)

        # 调仓：每隔 rebalance_days 且有足够历史数据
        if di - last_rb >= rebalance_days and di >= lookback:
            # 计算各资产过去 lookback 日的总收益作为动量信号
            momentum: dict[str, float] = {}
            for a in asset_list:
                p_now  = float(combined[a].iloc[di])
                p_past = float(combined[a].iloc[di - lookback])
                momentum[a] = p_now / p_past - 1.0

            best_asset = max(momentum, key=lambda a: momentum[a])
            best_mom   = momentum[best_asset]

            # 现金过滤：最强动量仍为负 → 持现金
            if cash_filter and best_mom < 0:
                new_holding = "CASH"
            else:
                new_holding = best_asset

            # 交易成本（换仓才扣）
            if new_holding != holding:
                port_val *= (1 - transaction_cost)  # 卖出
                if new_holding != "CASH":
                    port_val *= (1 - transaction_cost)  # 买入

            date_str = str(today.date()) if hasattr(today, "date") else str(today)[:10]
            rb_log.append({
                "date": date_str,
                "holding": new_holding,
                "momentum": {a: round(v * 100, 1) for a, v in momentum.items()},
            })
            holding = new_holding
            last_rb = di

        # 当日收益
        if di > 0 and holding and holding != "CASH":
            day_ret = (float(combined[holding].iloc[di])
                       / float(combined[holding].iloc[di - 1]) - 1)
            port_val *= (1 + day_ret)

        c_port.append(round(port_val, 2))
        c_bench.append(round(bench_val, 2))

    pa = np.array(c_port)
    ba = np.array(c_bench)
    pr = np.diff(pa) / pa[:-1]
    br = np.diff(ba) / ba[:-1]

    port_total = (port_val / initial_cash - 1) * 100
    bench_total = (bench_val / initial_cash - 1) * 100

    # 结果表
    result_table = Table(title=f"动量轮动回测  {start} ~ {end}", show_lines=True)
    result_table.add_column("指标", style="bold")
    result_table.add_column("轮动策略", justify="right")
    result_table.add_column(benchmark, justify="right")

    def fmt(v: float, pct: bool = True) -> str:
        s = f"{v:+.1f}%" if pct else f"{v:.2f}"
        color = "green" if v > 0 else "red"
        return f"[{color}]{s}[/{color}]"

    result_table.add_row("总收益",    fmt(port_total),         fmt(bench_total))
    result_table.add_row("Sharpe",    f"{sharpe(pr):.2f}",    f"{sharpe(br):.2f}")
    result_table.add_row("最大回撤",  f"{max_dd(pa):.1f}%",   f"{max_dd(ba):.1f}%")
    result_table.add_row("超额收益",  fmt(port_total - bench_total), "—")
    console.print(result_table)

    # 最近调仓记录
    rb_table = Table(title="最近调仓记录", show_lines=True)
    rb_table.add_column("日期")
    rb_table.add_column("持仓")
    rb_table.add_column("各资产动量（3M）")
    for rb in rb_log[-8:]:
        mom_str = "  ".join(f"{a}:{v:+.0f}%" for a, v in rb["momentum"].items())
        rb_table.add_row(rb["date"], rb["holding"], mom_str)
    console.print(rb_table)

    # 各资产持仓比例统计
    holding_counts: dict[str, int] = {}
    for rb in rb_log:
        h = rb["holding"]
        holding_counts[h] = holding_counts.get(h, 0) + 1
    total_rb = len(rb_log) or 1
    console.print("\n[bold]各资产持仓月数占比：[/bold]")
    for h, cnt in sorted(holding_counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(cnt / total_rb * 30)
        console.print(f"  {h:6s}  {bar}  {cnt}/{total_rb} ({cnt/total_rb:.0%})")

    console.print(
        f"\n[dim]轮动池：{assets}  动量窗口：{lookback}日  "
        f"调仓周期：{rebalance_days}日  现金过滤：{'开' if cash_filter else '关'}[/dim]"
    )


@app.command(name="alpaca-dashboard")
def alpaca_dashboard(
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser after generating"),
    out: str = typer.Option("logs/alpaca_dashboard.html", help="Output HTML path"),
):
    """
    Read data/alpaca_paper_log.csv and generate an equity vs SPY dashboard (HTML).
    Run after each trading day to review latest performance.
    """
    import webbrowser
    import json as _json
    import csv as _csv
    import numpy as np
    import pandas as pd

    log_path = Path("data/alpaca_paper_log.csv")
    if not log_path.exists():
        console.print("[red]data/alpaca_paper_log.csv not found — run alpaca-paper first[/red]")
        raise typer.Exit(1)

    # ── Read log (handles both old 6-col and new 7-col format) ───────────────
    records = []
    with open(log_path, newline="") as _f:
        for _i, _row in enumerate(_csv.reader(_f)):
            if _i == 0:
                continue
            if len(_row) == 6:
                _date, _eq, _rb, _nh, _nt, _h = _row
                _spy = ""
            elif len(_row) == 7:
                _date, _eq, _spy, _rb, _nh, _nt, _h = _row
            else:
                continue
            records.append({
                "date": _date, "equity": float(_eq),
                "spy_close": float(_spy) if _spy else None,
                "rebalanced": int(_rb), "n_holdings": int(_nh),
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").groupby("date").last().reset_index()

    if len(df) < 2:
        console.print("[yellow]Not enough data — need at least 2 days of records[/yellow]")
        raise typer.Exit(1)

    start_dt = df["date"].iloc[0]
    end_dt   = df["date"].iloc[-1]

    # ── Fetch SPY prices ──────────────────────────────────────────────────────
    spy_prices: dict = {}
    try:
        import yfinance as yf
        spy_raw   = yf.download("SPY", start=start_dt.strftime("%Y-%m-%d"),
                                 end=(end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                                 progress=False)
        close_col = spy_raw["Close"].squeeze()
        for d, px in close_col.items():
            spy_prices[pd.Timestamp(d).normalize()] = float(px)
    except Exception as e:
        console.print(f"[yellow]SPY data unavailable ({e}) — showing equity curve only[/yellow]")

    # ── Normalize to 100 ──────────────────────────────────────────────────────
    base_equity = df["equity"].iloc[0]
    dates_str   = [d.strftime("%Y-%m-%d") for d in df["date"]]

    port_data = [{"time": t, "value": round(float(v) / base_equity * 100, 4)}
                 for t, v in zip(dates_str, df["equity"])]

    spy_base = None
    spy_data = []
    for d, ts in zip(df["date"], dates_str):
        px = spy_prices.get(pd.Timestamp(d).normalize())
        if px is not None:
            if spy_base is None:
                spy_base = px
            spy_data.append({"time": ts, "value": round(px / spy_base * 100, 4)})

    # ── Drawdown ──────────────────────────────────────────────────────────────
    eq   = np.array(df["equity"].values, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak * 100
    dd_data = [{"time": t, "value": round(float(v), 4), "color": "rgba(244,67,54,0.6)"}
               for t, v in zip(dates_str, dd)]

    # ── Stats ─────────────────────────────────────────────────────────────────
    total_ret = (df["equity"].iloc[-1] / base_equity - 1) * 100
    max_dd    = float(dd.min())
    days_held = (end_dt - start_dt).days
    ann_ret   = ((1 + total_ret / 100) ** (365 / max(days_held, 1)) - 1) * 100
    spy_label = f" &nbsp;|&nbsp; SPY {spy_data[-1]['value'] - 100:+.1f}%" if spy_data else ""

    rb_dates = [d.strftime("%Y-%m-%d") for d in df[df["rebalanced"] == 1]["date"]]

    # ── Build HTML with TradingView Lightweight Charts ────────────────────────
    port_json = _json.dumps(port_data)
    spy_json  = _json.dumps(spy_data)
    dd_json   = _json.dumps(dd_data)
    rb_markers = _json.dumps([{
        "time": d, "position": "aboveBar", "color": "#26a69a",
        "shape": "circle", "text": "R", "size": 0.8,
    } for d in rb_dates])

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Alpaca Paper Trading Dashboard</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  body {{ margin:0; background:#0d0d1a; color:#ccc; font-family:sans-serif; }}
  #header {{ padding:16px 24px 8px; border-bottom:1px solid #1a1a2e; }}
  h2 {{ margin:0 0 6px; font-size:18px; color:#eee; }}
  .stats {{ display:flex; gap:28px; font-size:13px; color:#aaa; }}
  .stat-val {{ font-size:16px; font-weight:bold; color:#eee; }}
  .pos {{ color:#26a69a; }} .neg {{ color:#ef5350; }}
  #chart1, #chart2 {{ width:100%; }}
  .chart-label {{ padding:8px 24px 2px; font-size:12px; color:#666; }}
  footer {{ text-align:right; padding:6px 24px; font-size:11px; color:#444; }}
  footer a {{ color:#444; }}
</style>
</head>
<body>
<div id="header">
  <h2>Alpaca Paper Trading Dashboard</h2>
  <div class="stats">
    <div>Total Return<br><span class="stat-val {'pos' if total_ret>=0 else 'neg'}">{total_ret:+.2f}%</span>{spy_label}</div>
    <div>Annualized<br><span class="stat-val {'pos' if ann_ret>=0 else 'neg'}">{ann_ret:+.1f}%</span></div>
    <div>Max Drawdown<br><span class="stat-val neg">{max_dd:.1f}%</span></div>
    <div>Days Running<br><span class="stat-val">{days_held}</span></div>
    <div>Holdings<br><span class="stat-val">{int(df['n_holdings'].iloc[-1])}</span></div>
  </div>
</div>
<div class="chart-label">Equity vs SPY — normalized to 100 &nbsp; <span style="color:#26a69a">● R = rebalance day</span></div>
<div id="chart1"></div>
<div class="chart-label">Drawdown (%)</div>
<div id="chart2"></div>
<footer>Powered by <a href="https://tradingview.github.io/lightweight-charts/" target="_blank">TradingView Lightweight Charts</a></footer>
<script>
(function() {{
  const BG = '#0d0d1a', GRID = '#1a1a2e', TEXT = '#cccccc';
  const baseOpts = {{
    layout: {{ background: {{ color: BG }}, textColor: TEXT }},
    grid:   {{ vertLines: {{ color: GRID }}, horzLines: {{ color: GRID }} }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
    timeScale: {{ borderColor: '#333', timeVisible: true }},
    rightPriceScale: {{ borderColor: '#333' }},
    handleScroll: true, handleScale: true,
  }};

  // ── Chart 1: Equity vs SPY ────────────────────────────────────────────────
  const el1   = document.getElementById('chart1');
  el1.style.height = Math.round(window.innerHeight * 0.55) + 'px';
  const chart1 = LightweightCharts.createChart(el1, {{ ...baseOpts, width: el1.clientWidth, height: el1.clientHeight }});

  const portSeries = chart1.addAreaSeries({{
    topColor: 'rgba(255,165,0,0.3)', bottomColor: 'rgba(255,165,0,0.0)',
    lineColor: '#ffa500', lineWidth: 2, title: 'Strategy',
    priceFormat: {{ type: 'price', precision: 2, minMove: 0.01 }},
  }});
  portSeries.setData({port_json});
  portSeries.setMarkers({rb_markers});

  const spySeries = chart1.addLineSeries({{
    color: '#aaaaaa', lineWidth: 1.5, lineStyle: 2,
    title: 'SPY', priceLineVisible: false,
    priceFormat: {{ type: 'price', precision: 2, minMove: 0.01 }},
  }});
  spySeries.setData({spy_json});

  // ── Chart 2: Drawdown ─────────────────────────────────────────────────────
  const el2   = document.getElementById('chart2');
  el2.style.height = Math.round(window.innerHeight * 0.25) + 'px';
  const chart2 = LightweightCharts.createChart(el2, {{ ...baseOpts, width: el2.clientWidth, height: el2.clientHeight }});

  const ddSeries = chart2.addHistogramSeries({{
    color: 'rgba(244,67,54,0.6)',
    priceFormat: {{ type: 'price', precision: 2, minMove: 0.01 }},
  }});
  ddSeries.setData({dd_json});

  // Sync crosshair & scroll between charts
  function syncCrosshair(src, dst) {{
    src.subscribeCrosshairMove(p => {{
      if (p.time) dst.setCrosshairPosition(p.seriesData.values().next().value?.value, p.time, dst.series);
      else dst.clearCrosshairPosition();
    }});
  }}
  chart1.timeScale().subscribeVisibleLogicalRangeChange(r => chart2.timeScale().setVisibleLogicalRange(r));
  chart2.timeScale().subscribeVisibleLogicalRangeChange(r => chart1.timeScale().setVisibleLogicalRange(r));

  chart1.timeScale().fitContent();
  chart2.timeScale().fitContent();

  window.addEventListener('resize', () => {{
    chart1.applyOptions({{ width: el1.clientWidth }});
    chart2.applyOptions({{ width: el2.clientWidth }});
  }});
}})();
</script>
</body>
</html>"""

    out_path = Path(out)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    console.print(f"[green]Dashboard saved → {out_path}[/green]")
    console.print(
        f"  Total: [bold]{total_ret:+.1f}%[/bold]{spy_label.replace('&nbsp;', ' ').replace('|', '|')}  "
        f"Annualized: {ann_ret:+.1f}%  Max Drawdown: {max_dd:.1f}%  Days: {days_held}"
    )

    if open_browser:
        webbrowser.open(out_path.resolve().as_uri())


@app.command(name="risk-report")
def risk_report():
    """
    CAPM-based performance attribution: Alpha, Beta, R², Sharpe, Sortino,
    Max Drawdown, and recovery days — answers "am I beating SPY or just riding it?"
    """
    import csv as _csv
    import numpy as np
    import pandas as pd
    from scipy import stats

    log_path = Path("data/alpaca_paper_log.csv")
    if not log_path.exists():
        console.print("[red]data/alpaca_paper_log.csv not found — run alpaca-paper first[/red]")
        raise typer.Exit(1)

    # ── Load log ──────────────────────────────────────────────────────────────
    records = []
    with open(log_path, newline="") as f:
        for i, row in enumerate(_csv.reader(f)):
            if i == 0:
                continue
            if len(row) == 6:
                d, eq, rb, nh, nt, h = row
            elif len(row) == 7:
                d, eq, _, rb, nh, nt, h = row
            else:
                continue
            records.append({"date": d, "equity": float(eq)})

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").groupby("date").last().reset_index()

    if len(df) < 5:
        console.print("[yellow]Need at least 5 days of data[/yellow]")
        raise typer.Exit(1)

    start_dt = df["date"].iloc[0]
    end_dt   = df["date"].iloc[-1]
    days     = (end_dt - start_dt).days or 1

    # ── Fetch SPY ─────────────────────────────────────────────────────────────
    try:
        import yfinance as yf
        spy_raw = yf.download(
            "SPY",
            start=start_dt.strftime("%Y-%m-%d"),
            end=(end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
        )
        spy_close = spy_raw["Close"].squeeze()
        spy_close.index = pd.to_datetime(spy_close.index).normalize()
    except Exception as e:
        console.print(f"[red]SPY data unavailable: {e}[/red]")
        raise typer.Exit(1)

    # ── Align and compute daily returns ───────────────────────────────────────
    df = df.set_index("date")
    df.index = df.index.normalize()
    merged = df.join(spy_close.rename("spy"), how="inner")
    merged["strat_ret"] = merged["equity"].pct_change()
    merged["spy_ret"]   = merged["spy"].pct_change()
    merged = merged.dropna()

    if len(merged) < 2:
        console.print("[yellow]Not enough overlapping trading days yet[/yellow]")
        raise typer.Exit(1)
    if len(merged) < 20:
        console.print(f"[yellow]⚠  Only {len(merged)} trading days — CAPM estimates unreliable until ~20+ days[/yellow]")

    r_p = merged["strat_ret"].values
    r_m = merged["spy_ret"].values

    # Risk-free rate: ~5% annual → daily
    rf_daily = 0.05 / 252
    excess_p = r_p - rf_daily
    excess_m = r_m - rf_daily

    # ── CAPM regression ───────────────────────────────────────────────────────
    slope, intercept, r_val, _, _ = stats.linregress(excess_m, excess_p)
    beta     = slope
    alpha_d  = intercept
    alpha_ann = alpha_d * 252
    r2       = r_val ** 2

    # ── Sharpe & Sortino ──────────────────────────────────────────────────────
    sharpe = float(np.mean(excess_p) / np.std(excess_p) * np.sqrt(252)) if np.std(excess_p) > 0 else 0.0
    down   = excess_p[excess_p < 0]
    if len(down) < 2:
        sortino = float("inf")
    else:
        sortino = float(np.mean(excess_p) * 252 / (np.std(down) * np.sqrt(252)))

    # ── Drawdown ──────────────────────────────────────────────────────────────
    eq_arr  = merged["equity"].values
    peak    = np.maximum.accumulate(eq_arr)
    dd_arr  = (eq_arr - peak) / peak * 100
    max_dd  = float(dd_arr.min())
    cur_dd  = float(dd_arr[-1])

    # Recovery days: how many days since the trough until new high (or still recovering)
    trough_idx = int(np.argmin(dd_arr))
    recovery_days: int | str = "still recovering"
    for i in range(trough_idx + 1, len(eq_arr)):
        if eq_arr[i] >= peak[trough_idx]:
            recovery_days = i - trough_idx
            break

    # ── Summary stats ─────────────────────────────────────────────────────────
    total_ret     = (eq_arr[-1] / eq_arr[0] - 1) * 100
    spy_total_ret = (merged["spy"].iloc[-1] / merged["spy"].iloc[0] - 1) * 100
    ann_ret       = ((1 + total_ret / 100) ** (365 / days) - 1) * 100
    ann_spy       = ((1 + spy_total_ret / 100) ** (365 / days) - 1) * 100

    # ── Verdict ───────────────────────────────────────────────────────────────
    if alpha_ann > 0.02 and beta < 1.1:
        verdict = "[bold green]Generating real alpha — strategy is adding value beyond SPY[/bold green]"
    elif alpha_ann > 0 and r2 > 0.85:
        verdict = "[yellow]Positive alpha but high R² — mostly tracking SPY with slight edge[/yellow]"
    elif alpha_ann <= 0:
        verdict = "[red]Negative alpha — consider just holding SPY[/red]"
    else:
        verdict = "[cyan]Too early to conclude — need more data[/cyan]"

    # ── Print ─────────────────────────────────────────────────────────────────
    console.print()
    console.print("[bold white]─── Strategy Risk Report ───────────────────────────────────[/bold white]")
    console.print()

    col_w = 32
    def row(label, value, note=""):
        note_str = f"  [dim]{note}[/dim]" if note else ""
        console.print(f"  {label:<{col_w}}{value}{note_str}")

    row("Period",          f"{start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')}  ({days}d, {len(merged)} trading days)")
    console.print()

    strat_color = "green" if total_ret >= 0 else "red"
    alpha_color = "green" if alpha_ann >= 0 else "red"
    dd_color    = "red" if cur_dd < -5 else "yellow" if cur_dd < -2 else "green"

    console.print("  [bold]Returns[/bold]")
    row("  Strategy total",  f"[{strat_color}]{total_ret:+.2f}%[/{strat_color}]")
    row("  SPY total",       f"{spy_total_ret:+.2f}%")
    row("  Strategy annual", f"{ann_ret:+.1f}%")
    row("  SPY annual",      f"{ann_spy:+.1f}%")
    console.print()

    console.print("  [bold]CAPM Attribution[/bold]")
    row("  Alpha (annualized)", f"[{alpha_color}]{alpha_ann*100:+.2f}%[/{alpha_color}]", "pure stock-picking contribution")
    row("  Beta",               f"{beta:.3f}",                                            "1.0 = moves with SPY")
    row("  R²",                 f"{r2:.3f}",                                              "1.0 = pure SPY clone")
    console.print()

    console.print("  [bold]Risk-Adjusted[/bold]")
    row("  Sharpe Ratio",  f"{sharpe:.2f}",  ">1.0 good, >2.0 excellent")
    sortino_str = "∞ (no down days yet)" if sortino == float("inf") else f"{sortino:.2f}"
    row("  Sortino Ratio", sortino_str, "Sharpe using downside vol only")
    console.print()

    console.print("  [bold]Drawdown[/bold]")
    row("  Current drawdown", f"[{dd_color}]{cur_dd:.2f}%[/{dd_color}]")
    row("  Max drawdown",     f"[red]{max_dd:.2f}%[/red]")
    row("  Recovery",         f"{recovery_days}" + (" days to new high" if isinstance(recovery_days, int) else ""))
    console.print()

    console.print(f"  [bold]Verdict[/bold]")
    console.print(f"  {verdict}")
    console.print()
    console.print("[bold white]────────────────────────────────────────────────────────────[/bold white]")
    console.print()


@app.command(name="ic-calibrate")
def ic_calibrate(
    start: str  = typer.Option("2023-01-01", help="回测开始日期（用于计算截面IC）"),
    end: str    = typer.Option(str(date.today()), help="回测结束日期"),
    lookback: int = typer.Option(6, help="用最近几期IC来平均（每期=rebalance_every天）"),
    blend: float  = typer.Option(0.5, help="IC权重与默认权重的混合比例（0=纯默认，1=纯IC）"),
    forward: int  = typer.Option(20, help="IC前瞻天数（与换仓周期对齐）"),
    no_sample: bool = typer.Option(False, "--no-sample", help="用全量S&P500（慢）"),
):
    """
    截面 Rank-IC 因子权重校准。

    在历史数据上，每个换仓日对全量股票计算各因子排名 vs 未来收益排名的 Spearman 相关系数，
    用平均 IC 重新校准因子权重，保存到 data/factor_weights.json。
    下次运行任何选股命令时自动加载。
    """
    import random
    import pandas as pd
    from factor.engine import DEFAULT_WEIGHTS
    from factor.ic_calibrator import ICCalibrator
    from data.stock_selector import get_sp500_symbols

    fetcher = DataFetcher(use_cache=True)

    # ── 股票池 ─────────────────────────────────────────────────────────────────
    console.print("[cyan]加载股票池...[/cyan]")
    try:
        all_syms = get_sp500_symbols()
    except Exception:
        all_syms = []
    if not all_syms:
        console.print("[red]获取 S&P500 列表失败[/red]")
        raise typer.Exit(1)

    if no_sample:
        syms = all_syms
    else:
        syms = random.sample(all_syms, min(120, len(all_syms)))
    console.print(f"[cyan]使用 {len(syms)} 只股票 ({start} → {end})[/cyan]")

    # ── 价格数据 ───────────────────────────────────────────────────────────────
    console.print("[cyan]下载价格数据（使用缓存）...[/cyan]")
    stock_data: dict = {}
    for i, sym in enumerate(syms):
        try:
            df = fetcher.get_kline(sym, start, end)
            if df is not None and len(df) >= 80:
                stock_data[sym] = df
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            console.print(f"  {i+1}/{len(syms)} 完成...")

    console.print(f"[cyan]有效股票: {len(stock_data)} 只[/cyan]")
    if len(stock_data) < 30:
        console.print("[red]有效股票不足30只，中止[/red]")
        raise typer.Exit(1)

    # ── 基本面数据 ─────────────────────────────────────────────────────────────
    console.print("[cyan]下载基本面数据（使用缓存）...[/cyan]")
    fund_map: dict = {}
    for sym in stock_data:
        try:
            fund_map[sym] = fetcher.get_fundamentals(sym) or {}
        except Exception:
            fund_map[sym] = {}

    # ── 截面 IC 校准 ───────────────────────────────────────────────────────────
    console.print("[bold cyan]计算截面 Rank-IC...[/bold cyan]")
    cal = ICCalibrator(stock_data, fund_map, forward_days=forward, rebalance_every=forward)
    mean_ic = cal.calibrate(lookback_periods=lookback)
    weights  = ICCalibrator.ic_to_weights(mean_ic, blend=blend)
    ICCalibrator.save(weights, mean_ic)

    # ── 打印结果 ───────────────────────────────────────────────────────────────
    from rich.table import Table as RichTable
    from factor.engine import DEFAULT_WEIGHTS

    t = RichTable(title="截面 Rank-IC 校准结果", show_header=True, header_style="bold cyan")
    t.add_column("Factor",     style="white",  width=18)
    t.add_column("Mean IC",    justify="right", width=10)
    t.add_column("Default W",  justify="right", width=10)
    t.add_column("New Weight", justify="right", width=10)
    t.add_column("Change",     justify="right", width=10)

    for fname in DEFAULT_WEIGHTS:
        ic_val  = mean_ic.get(fname, 0.0)
        dw      = DEFAULT_WEIGHTS[fname]
        nw      = weights.get(fname, dw)
        delta   = nw - dw
        ic_str  = f"[green]{ic_val:+.3f}[/green]" if ic_val > 0.02 else (
                  f"[red]{ic_val:+.3f}[/red]"     if ic_val < -0.02 else f"{ic_val:+.3f}")
        dw_str  = f"{dw:+.3f}"
        nw_str  = f"[bold]{nw:+.3f}[/bold]"
        dl_str  = f"[green]{delta:+.3f}[/green]" if delta > 0.001 else (
                  f"[red]{delta:+.3f}[/red]"      if delta < -0.001 else f"{delta:+.3f}")
        t.add_row(fname, ic_str, dw_str, nw_str, dl_str)

    console.print(t)
    console.print(f"\n[bold green]✓ 权重已保存 → data/factor_weights.json[/bold green]")
    console.print("[dim]下次运行 alpaca-paper / factor-backtest 时自动加载新权重[/dim]")


if __name__ == "__main__":
    app()
