"""
纯内存模拟盘 — 不依赖任何外部系统，用于策略验证和单元测试。

按限价单撮合：如果下一个 bar 的 open 价格能满足限价，则成交。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from loguru import logger

from config import settings
from .base import BrokerBase, Order, OrderSide, OrderStatus


class PaperBroker(BrokerBase):
    """内存模拟盘。"""

    def __init__(self, initial_cash: float = settings.initial_cash):
        self._cash = initial_cash
        self._positions: dict[str, dict] = {}   # symbol -> {qty, avg_cost}
        self._orders: dict[str, Order] = {}
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        logger.info("[PaperBroker] 连接成功（模拟盘）")
        return True

    def disconnect(self) -> None:
        self._connected = False

    def buy(self, symbol: str, quantity: int, price: float = 0.0) -> Order:
        order = Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol, side=OrderSide.BUY,
            quantity=quantity, price=price,
        )
        cost = quantity * price * (1 + settings.commission_rate)
        if cost > self._cash:
            order.status = OrderStatus.REJECTED
            order.error_msg = "资金不足"
            logger.warning(f"[PaperBroker] 买单拒绝: {symbol} 资金不足")
        else:
            self._cash -= cost
            pos = self._positions.setdefault(symbol, {"qty": 0, "avg_cost": 0.0})
            total_cost = pos["avg_cost"] * pos["qty"] + price * quantity
            pos["qty"] += quantity
            pos["avg_cost"] = total_cost / pos["qty"]
            order.status = OrderStatus.FILLED
            order.filled_qty = quantity
            order.filled_price = price
            order.commission = quantity * price * settings.commission_rate
            logger.info(f"[PaperBroker] 买入 {symbol} x{quantity} @ {price:.2f}")
        self._orders[order.order_id] = order
        return order

    def sell(self, symbol: str, quantity: int, price: float = 0.0) -> Order:
        order = Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol, side=OrderSide.SELL,
            quantity=quantity, price=price,
        )
        pos = self._positions.get(symbol)
        if not pos or pos["qty"] < quantity:
            order.status = OrderStatus.REJECTED
            order.error_msg = "持仓不足"
            logger.warning(f"[PaperBroker] 卖单拒绝: {symbol} 持仓不足")
        else:
            revenue = quantity * price * (1 - settings.commission_rate - settings.stamp_tax)
            self._cash += revenue
            pos["qty"] -= quantity
            if pos["qty"] == 0:
                del self._positions[symbol]
            order.status = OrderStatus.FILLED
            order.filled_qty = quantity
            order.filled_price = price
            order.commission = quantity * price * (settings.commission_rate + settings.stamp_tax)
            logger.info(f"[PaperBroker] 卖出 {symbol} x{quantity} @ {price:.2f}")
        self._orders[order.order_id] = order
        return order

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELLED
            return True
        return False

    def get_order(self, order_id: str) -> Order:
        return self._orders[order_id]

    def get_positions(self) -> list[dict]:
        return [
            {"symbol": sym, "quantity": p["qty"], "avg_cost": p["avg_cost"]}
            for sym, p in self._positions.items()
        ]

    def get_balance(self) -> dict:
        total_mv = sum(p["qty"] * p["avg_cost"] for p in self._positions.values())
        return {
            "cash": self._cash,
            "market_value": total_mv,
            "total_equity": self._cash + total_mv,
        }

    def get_today_orders(self) -> list[Order]:
        today = datetime.now().date()
        return [o for o in self._orders.values()
                if o.created_at.date() == today]
