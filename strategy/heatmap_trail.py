"""
Python port of Zeiierman's "Heatmap Trailing Stop with Breakouts".
Visualization (line colors, plotshape) is omitted — only signal logic is kept.

Returns a dict with:
  trend_dir : int    1 (bull) | -1 (bear)
  score     : float  1-10, heatmap strength at current bar
  sig_long  : bool   bullish breakout confirmed
  sig_short : bool   bearish breakout confirmed
  trail     : float  current trailing-stop level
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(
    df: pd.DataFrame,
    stop_atr_len: int = 28,
    stop_mult: float = 5.0,
    heat_atr: int = 50,
    levels: int = 3,
    heat_thresh: int = 3,
    score_val: int = 6,
    cooldown_bars: int = 20,
) -> dict:
    n = len(df)
    min_bars = max(stop_atr_len, heat_atr) + 10
    if n < min_bars:
        return {
            "trend_dir": 1,
            "score": 5.0,
            "sig_long": False,
            "sig_short": False,
            "trail": float(df["close"].iloc[-1]),
        }

    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    close = df["close"].values.astype(float)
    open_ = df["open"].values.astype(float)

    # ── ATR: EMA of True Range ────────────────────────────────────────────────
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i]  - close[i - 1]),
        )

    alpha = 2.0 / (stop_atr_len + 1)
    atr_ema = np.empty(n)
    atr_ema[0] = tr[0]
    for i in range(1, n):
        atr_ema[i] = alpha * tr[i] + (1 - alpha) * atr_ema[i - 1]

    stop_offset = stop_mult * atr_ema
    bull_stop   = high - stop_offset
    bear_stop   = low  + stop_offset

    # ── Trailing-stop state ───────────────────────────────────────────────────
    trend_up   = np.empty(n)
    trend_down = np.empty(n)
    trend_dir  = np.ones(n, dtype=int)
    ex         = np.empty(n)

    trend_up[0]   = bull_stop[0]
    trend_down[0] = bear_stop[0]
    ex[0]         = high[0]

    for i in range(1, n):
        trend_up[i] = (
            max(bull_stop[i], trend_up[i - 1])
            if close[i - 1] > trend_up[i - 1]
            else bull_stop[i]
        )
        trend_down[i] = (
            min(bear_stop[i], trend_down[i - 1])
            if close[i - 1] < trend_down[i - 1]
            else bear_stop[i]
        )

        if close[i] > trend_down[i - 1]:
            trend_dir[i] = 1
        elif close[i] < trend_up[i - 1]:
            trend_dir[i] = -1
        else:
            trend_dir[i] = trend_dir[i - 1]

        if trend_dir[i] != trend_dir[i - 1]:
            ex[i] = high[i] if trend_dir[i] == 1 else low[i]
        elif trend_dir[i] == 1:
            ex[i] = max(ex[i - 1], high[i])
        else:
            ex[i] = min(ex[i - 1], low[i])

    trail = np.where(trend_dir == 1, trend_up, trend_down)

    # ── Fibonacci levels from trailing-stop (last bar) ────────────────────────
    i    = n - 1
    fib1 = ex[i] + (trail[i] - ex[i]) * 0.618
    fib2 = ex[i] + (trail[i] - ex[i]) * 0.786
    fib3 = ex[i] + (trail[i] - ex[i]) * 0.886

    # ── Heatmap score (last bar) ──────────────────────────────────────────────
    start = max(0, i - heat_atr)
    lo_r  = float(np.min(low[start : i + 1]))
    hi_r  = float(np.max(high[start : i + 1]))
    rng   = hi_r - lo_r

    def _heat_score(val: float) -> int:
        if rng == 0:
            return 5
        step = rng / levels
        best_d, best_s = 1e10, 1
        for k in range(levels):
            lvl = lo_r + step * k
            cnt = int(np.sum(
                (high[start : i + 1] >= lvl) & (low[start : i + 1] <= lvl)
            ))
            raw = min(max((cnt - heat_thresh) / 10.0, 0.0), 1.0)
            s   = round(1 + raw * 9)
            d   = abs(val - lvl)
            if d < best_d:
                best_d, best_s = d, s
        return best_s

    # Pine script: avg(Score(trail), Score(fib1), Score(fib2), Score(fib3), Score(l100))
    # l100 = trail, so trail is counted twice (faithful to original)
    score = float(np.mean([
        _heat_score(trail[i]),
        _heat_score(fib1),
        _heat_score(fib2),
        _heat_score(fib3),
        _heat_score(trail[i]),
    ]))

    # ── Breakout signals ──────────────────────────────────────────────────────
    def _raw_long(j: int) -> bool:
        if j < 2:
            return False
        bull_candle = (close[j] > open_[j] and
                       (close[j] - open_[j]) > (high[j] - low[j]) * 0.5)
        mom_up      = close[j] > high[j - 1] and close[j] > high[j - 2]
        return bull_candle and mom_up and trend_dir[j] == 1

    def _raw_short(j: int) -> bool:
        if j < 2:
            return False
        bear_candle = (close[j] < open_[j] and
                       (open_[j] - close[j]) > (high[j] - low[j]) * 0.5)
        mom_dn      = close[j] < low[j - 1] and close[j] < low[j - 2]
        return bear_candle and mom_dn and trend_dir[j] == -1

    # Find last raw signals within cooldown window
    last_long  = i - cooldown_bars - 1
    last_short = i - cooldown_bars - 1
    for j in range(max(2, i - cooldown_bars), i):
        if _raw_long(j):
            last_long = j
        if _raw_short(j):
            last_short = j

    cd_long  = (i - last_long  > cooldown_bars) or (last_short > last_long)
    cd_short = (i - last_short > cooldown_bars) or (last_long  > last_short)

    sig_long  = _raw_long(i)  and cd_long  and score > score_val
    sig_short = _raw_short(i) and cd_short and score > score_val

    return {
        "trend_dir": int(trend_dir[i]),
        "score":     round(score, 2),
        "sig_long":  bool(sig_long),
        "sig_short": bool(sig_short),
        "trail":     round(float(trail[i]), 4),
    }
