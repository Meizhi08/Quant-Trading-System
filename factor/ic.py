"""
IC（信息系数）分析 — 验证因子是否真的有预测价值。

IC = 某天各股因子值 与 未来N日收益率 的 Spearman 相关系数。
  IC > 0.05 且稳定：因子有效
  IC < 0.02 或不稳定：因子无效

用法：
  analyzer = ICAnalyzer(engine)
  result = analyzer.run(df, symbol, forward_days=5)
  result.print_summary()
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .engine import FactorEngine


@dataclass
class ICResult:
    symbol: str
    forward_days: int
    ic_mean: dict[str, float] = field(default_factory=dict)   # 各因子平均IC
    ic_std:  dict[str, float] = field(default_factory=dict)   # IC标准差
    ic_ir:   dict[str, float] = field(default_factory=dict)   # IR = mean/std（越高越稳）
    ic_series: dict[str, list[float]] = field(default_factory=dict)  # IC时间序列
    n_periods: int = 0

    def summary_df(self) -> pd.DataFrame:
        rows = []
        for factor in self.ic_mean:
            mean = self.ic_mean[factor]
            std  = self.ic_std.get(factor, 0)
            ir   = self.ic_ir.get(factor, 0)
            win_rate = (
                sum(1 for v in self.ic_series.get(factor, []) if v > 0)
                / max(len(self.ic_series.get(factor, [1])), 1)
            )
            if abs(mean) >= 0.05 and abs(ir) >= 0.5:
                verdict = "★ 有效"
            elif abs(mean) >= 0.03:
                verdict = "△ 弱有效"
            else:
                verdict = "✗ 无效"
            rows.append({
                "因子":     factor,
                "IC均值":   round(mean, 4),
                "IC标准差": round(std, 4),
                "IR":       round(ir, 4),
                "胜率":     f"{win_rate:.1%}",
                "评价":     verdict,
            })
        return pd.DataFrame(rows).sort_values("IR", key=abs, ascending=False)

    def best_factors(self, min_ic: float = 0.03, min_ir: float = 0.3) -> list[str]:
        return [
            f for f in self.ic_mean
            if abs(self.ic_mean[f]) >= min_ic
            and abs(self.ic_ir.get(f, 0)) >= min_ir
        ]

    def suggest_weights(self) -> dict[str, float]:
        """
        根据IC结果自动计算因子权重。
        权重 = IC均值（保留正负方向），再归一化到绝对值之和=1。
        IC无效的因子权重趋近0。
        """
        raw = {}
        for f, mean in self.ic_mean.items():
            ir = abs(self.ic_ir.get(f, 0))
            # IC绝对值越大、IR越高 → 权重越大，方向跟着IC走
            raw[f] = mean * min(ir, 2.0)  # IR上限2，防止单因子过度主导

        total = sum(abs(v) for v in raw.values())
        if total < 1e-8:
            return {}
        return {k: round(v / total, 4) for k, v in raw.items()}


class CrossSectionalICAnalyzer:
    """
    截面IC分析器 — 正确的因子有效性验证方式。

    每隔 step 天，对股票池里所有股票同时计算因子值，
    与各股票 forward_days 后的收益做 Spearman 截面相关。
    每个时间点产生一个IC值，再对时序取均值/标准差/IR。

    这比单股时序IC统计意义强得多：样本量 = 股票数 × 时间点数。
    """

    def __init__(self, engine: FactorEngine | None = None, window: int = 120):
        self.engine = engine or FactorEngine()
        self.window = window

    @staticmethod
    def _lookup_fund(
        hist: dict[str, dict] | None,
        static: dict | None,
        today_str: str,
    ) -> dict:
        """
        优先从历史快照（hist）里查 today_str 之前最近一期，
        没有则用 static 快照，再没有则返回空 dict。
        hist 格式：{date_str: fund_dict}
        """
        if hist:
            valid = [d for d in hist if d <= today_str]
            if valid:
                return hist[max(valid)]
        return static or {}

    def run(
        self,
        stock_data: dict[str, pd.DataFrame],
        forward_days: int = 5,
        step: int = 5,
        fundamentals: dict[str, dict] | None = None,
        hist_fundamentals: dict[str, dict[str, dict]] | None = None,
    ) -> ICResult:
        """
        fundamentals      : 静态快照 {sym: fund_dict}（兜底，yfinance 当前值）
        hist_fundamentals : 历史快照 {sym: {date_str: fund_dict}}
        """
        factor_names = list(self.engine.weights.keys())
        ic_series: dict[str, list[float]] = {f: [] for f in factor_names}
        fund_static  = fundamentals or {}
        fund_hist    = hist_fundamentals or {}

        # 取所有股票共有交易日的并集（用最宽泛的日期序列）
        all_dates: list = sorted(
            set().union(*[set(df.index) for df in stock_data.values()])
        )
        if len(all_dates) < self.window + forward_days + 10:
            return ICResult(symbol="cross_section", forward_days=forward_days)

        min_pos = self.window
        max_pos = len(all_dates) - forward_days - 1
        time_points = list(range(min_pos, max_pos, step))

        for ti in time_points:
            date_t   = all_dates[ti]
            date_fwd = all_dates[ti + forward_days]
            today_str = str(date_t.date()) if hasattr(date_t, "date") else str(date_t)[:10]

            cross_factors: dict[str, list[float]] = {f: [] for f in factor_names}
            cross_fwd: list[float] = []

            for sym, full_df in stock_data.items():
                # 取截止 date_t 的 window 条数据
                loc_t = full_df.index.get_indexer([date_t], method="ffill")[0]
                if loc_t < self.window:
                    continue
                slice_df = full_df.iloc[loc_t - self.window: loc_t + 1]
                if len(slice_df) < 60:
                    continue

                fund_snap = self._lookup_fund(
                    fund_hist.get(sym), fund_static.get(sym), today_str
                )
                score = self.engine.compute(slice_df, sym, fundamentals=fund_snap)
                if not score.factors:
                    continue

                # 找 date_fwd 对应的价格
                loc_fwd = full_df.index.get_indexer([date_fwd], method="ffill")[0]
                if loc_fwd <= loc_t or loc_fwd >= len(full_df):
                    continue
                fwd_ret = (
                    full_df["close"].iloc[loc_fwd]
                    / full_df["close"].iloc[loc_t] - 1
                )
                cross_fwd.append(float(fwd_ret))
                for f in factor_names:
                    cross_factors[f].append(score.factors.get(f, 0.0))

            if len(cross_fwd) < 10:
                continue

            fwd_arr = np.array(cross_fwd)
            for f in factor_names:
                fv = np.array(cross_factors[f])
                if len(fv) != len(fwd_arr) or fv.std() < 1e-8:
                    ic_series[f].append(0.0)
                    continue
                corr, _ = stats.spearmanr(fv, fwd_arr)
                ic_series[f].append(float(corr) if not np.isnan(corr) else 0.0)

        result = ICResult(symbol="cross_section", forward_days=forward_days)
        result.n_periods = len(next(iter(ic_series.values()), []))
        result.ic_series = ic_series
        result.ic_mean = {f: float(np.mean(v)) if v else 0.0 for f, v in ic_series.items()}
        result.ic_std  = {f: float(np.std(v))  if v else 0.0 for f, v in ic_series.items()}
        result.ic_ir   = {
            f: result.ic_mean[f] / result.ic_std[f]
            if result.ic_std[f] > 1e-8 else 0.0
            for f in factor_names
        }
        return result

    def rolling_stability(
        self,
        stock_data: dict[str, pd.DataFrame],
        forward_days: int = 5,
        step: int = 5,
        n_splits: int = 3,
        fundamentals: dict[str, dict] | None = None,
        hist_fundamentals: dict[str, dict[str, dict]] | None = None,
    ) -> list[ICResult]:
        """
        把时间轴等分成 n_splits 段，分别跑截面IC。
        用于检验因子有效性是否稳定（不同时期 IC 方向一致才可信）。
        """
        all_dates = sorted(set().union(*[set(df.index) for df in stock_data.values()]))
        split_size = len(all_dates) // n_splits
        results = []
        for i in range(n_splits):
            start = all_dates[i * split_size]
            end   = all_dates[min((i + 1) * split_size, len(all_dates) - 1)]
            sliced = {
                sym: df.loc[start:end]
                for sym, df in stock_data.items()
                if len(df.loc[start:end]) >= 60
            }
            if len(sliced) < 5:
                continue
            r = self.run(sliced, forward_days=forward_days, step=step,
                         fundamentals=fundamentals, hist_fundamentals=hist_fundamentals)
            r.symbol = f"period_{i+1}"
            results.append(r)
        return results


class ICAnalyzer:
    """
    对单只股票的历史数据做滚动IC分析。

    每隔 step 天计算一次因子值，与 forward_days 后的收益做相关性。
    """

    def __init__(self, engine: FactorEngine | None = None, window: int = 120):
        self.engine = engine or FactorEngine()
        self.window = window   # 每次计算因子用多少天历史数据

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        forward_days: int = 5,
        step: int = 5,
    ) -> ICResult:
        result = ICResult(symbol=symbol, forward_days=forward_days)
        factor_names = list(self.engine.weights.keys())
        ic_series: dict[str, list[float]] = {f: [] for f in factor_names}

        min_idx = self.window
        max_idx = len(df) - forward_days - 1

        if max_idx <= min_idx:
            return result

        indices = list(range(min_idx, max_idx, step))

        # 针对每个时间点，计算因子值和对应的未来收益
        factor_vals: dict[str, list[float]] = {f: [] for f in factor_names}
        fwd_returns: list[float] = []

        for idx in indices:
            slice_df = df.iloc[idx - self.window: idx + 1]
            score = self.engine.compute(slice_df, symbol)
            if not score.factors:
                continue
            fwd_ret = df["close"].iloc[idx + forward_days] / df["close"].iloc[idx] - 1
            fwd_returns.append(fwd_ret)
            for f in factor_names:
                factor_vals[f].append(score.factors.get(f, 0.0))

        if len(fwd_returns) < 10:
            return result

        fwd_arr = np.array(fwd_returns)
        result.n_periods = len(fwd_returns)

        for f in factor_names:
            fv = np.array(factor_vals[f])
            if fv.std() < 1e-8:
                ic_series[f] = [0.0]
                continue
            # 滚动IC（用整体做一次快速估算；逐期IC需要截面数据，单股用整体相关代替）
            corr, _ = stats.spearmanr(fv, fwd_arr)
            ic_val = float(corr) if not np.isnan(corr) else 0.0
            # 用bootstrap近似IC置信区间（每次有放回抽样）
            rng = np.random.default_rng(seed=hash(f) % (2**32))
            bootstrap_ics = []
            for _ in range(50):
                idx_b = rng.choice(len(fv), size=len(fv), replace=True)
                c, _ = stats.spearmanr(fv[idx_b], fwd_arr[idx_b])
                bootstrap_ics.append(float(c) if not np.isnan(c) else 0.0)
            ic_series[f] = bootstrap_ics

        result.ic_series = ic_series
        result.ic_mean = {
            f: float(np.mean(vals)) for f, vals in ic_series.items()
        }
        result.ic_std = {
            f: float(np.std(vals)) for f, vals in ic_series.items()
        }
        result.ic_ir = {
            f: (result.ic_mean[f] / result.ic_std[f]
                if result.ic_std[f] > 1e-8 else 0.0)
            for f in factor_names
        }

        return result
