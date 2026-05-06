"""
模拟盘持久化状态 — 所有数据保存在 data/paper_trading.json。
程序重启后状态不丢失。
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

_STATE_FILE = Path(__file__).parent.parent / "data" / "paper_trading.json"


def _default_state(initial_cash: float = 100_000.0) -> dict:
    return {
        "cash": initial_cash,
        "positions": {},        # symbol -> {qty, avg_cost}
        "trades": [],           # 所有成交记录
        "daily_equity": [],     # 每日总资产快照
        "created_at": str(date.today()),
    }


class PaperState:
    """读写模拟盘状态文件。"""

    def __init__(self, initial_cash: float = 100_000.0):
        self._initial_cash = initial_cash
        self._data = self._load()

    def _load(self) -> dict:
        if _STATE_FILE.exists():
            with open(_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return _default_state(self._initial_cash)

    def save(self) -> None:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ── 资金 ──────────────────────────────────────────────────────────────────

    @property
    def cash(self) -> float:
        return self._data["cash"]

    @cash.setter
    def cash(self, v: float) -> None:
        self._data["cash"] = round(v, 2)

    @property
    def initial_cash(self) -> float:
        return self._initial_cash

    # ── 持仓 ──────────────────────────────────────────────────────────────────

    def get_position(self, symbol: str) -> dict | None:
        return self._data["positions"].get(symbol)

    def set_position(self, symbol: str, qty: int, avg_cost: float) -> None:
        if qty <= 0:
            self._data["positions"].pop(symbol, None)
        else:
            self._data["positions"][symbol] = {"qty": qty, "avg_cost": round(avg_cost, 4)}

    def all_positions(self) -> dict[str, dict]:
        return self._data["positions"]

    # ── 交易记录 ──────────────────────────────────────────────────────────────

    def add_trade(self, trade: dict[str, Any]) -> None:
        self._data["trades"].append(trade)

    def all_trades(self) -> list[dict]:
        return self._data["trades"]

    # ── 每日资产快照 ──────────────────────────────────────────────────────────

    def snapshot_equity(self, prices: dict[str, float]) -> float:
        """计算当前总资产并保存快照，返回总资产金额。"""
        mv = sum(
            p["qty"] * prices.get(sym, p["avg_cost"])
            for sym, p in self._data["positions"].items()
        )
        total = round(self.cash + mv, 2)
        today = str(date.today())

        # 更新当天快照（同一天只保留最新）
        snaps = self._data["daily_equity"]
        if snaps and snaps[-1]["date"] == today:
            snaps[-1]["equity"] = total
        else:
            snaps.append({"date": today, "equity": total})
        return total

    def daily_equity(self) -> list[dict]:
        return self._data["daily_equity"]

    def total_return(self) -> float:
        snaps = self._data["daily_equity"]
        if not snaps:
            return 0.0
        start_equity = snaps[0]["equity"]
        if start_equity <= 0:
            return 0.0
        return round((snaps[-1]["equity"] - start_equity) / start_equity, 4)
