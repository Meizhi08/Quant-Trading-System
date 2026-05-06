"""
Cross-sectional Rank-IC calibrator for factor weight estimation.

At each rebalance date, computes Spearman IC between factor ranks and
20-day forward return ranks across all stocks in the universe.
Averages IC over recent periods and blends into calibrated weights.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from factor.engine import DEFAULT_WEIGHTS, FactorEngine, _spearman

logger = logging.getLogger(__name__)

_IC_WEIGHTS_PATH = Path("data/factor_weights.json")

_FACTOR_NAMES = list(DEFAULT_WEIGHTS.keys())


class ICCalibrator:
    """
    Compute cross-sectional Rank-IC for all 12 factors over a historical window.

    Usage:
        cal = ICCalibrator(stock_data, fund_map)
        mean_ic = cal.calibrate(lookback_periods=6)
        weights = cal.ic_to_weights(mean_ic, blend=0.5)
        cal.save(weights, mean_ic)
    """

    def __init__(
        self,
        stock_data: dict[str, pd.DataFrame],
        fund_map: dict[str, dict],
        forward_days: int = 20,
        rebalance_every: int = 20,
    ):
        self.stock_data      = stock_data
        self.fund_map        = fund_map
        self.forward_days    = forward_days
        self.rebalance_every = rebalance_every
        # Engine with dynamic weights off — pure factor scores, no IC loop
        self._engine = FactorEngine(use_dynamic_weights=False)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _factor_values(
        self, sym: str, df: pd.DataFrame, end_idx: int
    ) -> dict[str, float] | None:
        """Compute all 12 factor scores for `sym` using data up to end_idx (exclusive)."""
        window = 120
        if end_idx < window:
            return None
        slice_df = df.iloc[end_idx - window: end_idx]
        if len(slice_df) < 60:
            return None
        fund = self.fund_map.get(sym, {})
        try:
            fs = self._engine.compute(slice_df, sym, fundamentals=fund)
            return fs.factors
        except Exception:
            return None

    # ── Main calibration ───────────────────────────────────────────────────────

    def calibrate(self, lookback_periods: int = 6) -> dict[str, float]:
        """
        Run cross-sectional IC at each rebalance date over available history.
        Returns mean IC per factor averaged over the last `lookback_periods`.
        """
        all_dates = sorted(
            set().union(*[set(df.index) for df in self.stock_data.values()])
        )
        n_dates = len(all_dates)
        date_pos = {d: i for i, d in enumerate(all_dates)}

        ic_history: list[dict[str, float]] = []

        for di, today in enumerate(all_dates):
            # Warmup + need forward window
            if di < 120:
                continue
            if di % self.rebalance_every != 0:
                continue
            fwd_di = di + self.forward_days
            if fwd_di >= n_dates:
                break
            future = all_dates[fwd_di]

            # Collect factor values and forward returns for all stocks
            factor_cols: dict[str, list[float]] = {f: [] for f in _FACTOR_NAMES}
            fwd_ret_list: list[float] = []

            for sym, df in self.stock_data.items():
                idx = df.index.get_indexer([today],  method="ffill")[0]
                fix = df.index.get_indexer([future], method="ffill")[0]
                if idx < 0 or fix <= idx:
                    continue

                fvals = self._factor_values(sym, df, idx)
                if fvals is None:
                    continue

                close = df["close"]
                fwd_r = float(close.iloc[fix]) / float(close.iloc[idx]) - 1
                if not np.isfinite(fwd_r):
                    continue

                for fname in _FACTOR_NAMES:
                    v = fvals.get(fname, np.nan)
                    factor_cols[fname].append(v if np.isfinite(v) else np.nan)
                fwd_ret_list.append(fwd_r)

            n_stocks = len(fwd_ret_list)
            if n_stocks < 30:
                logger.warning(f"{str(today)[:10]}: only {n_stocks} stocks, skipping IC")
                continue

            fwd_arr = np.array(fwd_ret_list)

            ic_date: dict[str, float] = {}
            for fname in _FACTOR_NAMES:
                f_arr = np.array(factor_cols[fname])
                mask  = np.isfinite(f_arr) & np.isfinite(fwd_arr)
                if mask.sum() < 20:
                    ic_date[fname] = 0.0
                else:
                    ic_date[fname] = _spearman(f_arr[mask], fwd_arr[mask])

            ic_history.append(ic_date)
            logger.info(
                f"{str(today)[:10]}  n={n_stocks}  "
                + "  ".join(f"{k[:6]}={v:+.3f}" for k, v in ic_date.items())
            )

        if not ic_history:
            logger.error("No IC observations computed — returning default weights IC=0")
            return {k: 0.0 for k in _FACTOR_NAMES}

        recent = ic_history[-lookback_periods:]
        mean_ic: dict[str, float] = {
            fname: float(np.mean([d.get(fname, 0.0) for d in recent]))
            for fname in _FACTOR_NAMES
        }
        logger.info(f"Mean IC over last {len(recent)} periods: {mean_ic}")
        return mean_ic

    # ── Weight derivation ──────────────────────────────────────────────────────

    @staticmethod
    def ic_to_weights(
        mean_ic: dict[str, float],
        blend: float = 0.5,
        ic_scale: float = 0.05,
    ) -> dict[str, float]:
        """
        Multiplicative IC adjustment (same logic as engine._blend_weights).

        new_weight = default * (1 + blend * clip(aligned_IC / ic_scale, -1, 1))

        aligned_IC = sign(default) * IC:
          > 0  IC confirms direction  → weight grows
          < 0  IC contradicts        → weight shrinks toward 0
          = 0  no signal             → weight unchanged

        ic_scale: IC value that gives full adjustment (default 0.05 = 5%).
        blend: max fractional adjustment (default 0.5 = ±50% of default weight).
        """
        default = DEFAULT_WEIGHTS

        blended: dict[str, float] = {}
        for k, w in default.items():
            aligned_ic = float(np.sign(w)) * mean_ic.get(k, 0.0)
            adjustment = blend * float(np.clip(aligned_ic / ic_scale, -1.0, 1.0))
            blended[k] = w * (1.0 + adjustment)

        # Re-normalize: keep positive-weight total equal to default
        def_pos = sum(w for w in default.values() if w > 0)
        bld_pos = sum(w for w in blended.values() if w > 0)
        if bld_pos > 1e-9 and def_pos > 1e-9:
            scale = def_pos / bld_pos
            blended = {k: v * scale for k, v in blended.items()}

        return blended

    # ── Persistence ────────────────────────────────────────────────────────────

    @staticmethod
    def save(weights: dict[str, float], mean_ic: dict[str, float]) -> None:
        _IC_WEIGHTS_PATH.parent.mkdir(exist_ok=True)
        payload = {
            "weights":      {k: round(v, 4) for k, v in weights.items()},
            "mean_ic":      {k: round(v, 4) for k, v in mean_ic.items()},
            "calibrated_at": str(pd.Timestamp.now().date()),
        }
        _IC_WEIGHTS_PATH.write_text(json.dumps(payload, indent=2))
        logger.info(f"Saved calibrated weights → {_IC_WEIGHTS_PATH}")
