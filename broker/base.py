"""券商接口抽象基类 — 所有实盘/模拟接入均实现此接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float                            # 0 = 市价单
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    filled_price: float = 0.0
    commission: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    error_msg: str = ""


class BrokerBase(ABC):
    """所有券商适配器必须实现的统一接口。"""

    @abstractmethod
    def connect(self) -> bool:
        """连接/登录券商系统，返回是否成功。"""

    @abstractmethod
    def disconnect(self) -> None:
        """断开连接。"""

    @abstractmethod
    def buy(self, symbol: str, quantity: int, price: float = 0.0) -> Order:
        """发送买单。price=0 为市价单。"""

    @abstractmethod
    def sell(self, symbol: str, quantity: int, price: float = 0.0) -> Order:
        """发送卖单。"""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤单，返回是否成功。"""

    @abstractmethod
    def get_order(self, order_id: str) -> Order:
        """查询单笔订单状态。"""

    @abstractmethod
    def get_positions(self) -> list[dict]:
        """获取当前持仓。"""

    @abstractmethod
    def get_balance(self) -> dict:
        """获取账户资金信息（cash / market_value / total_equity）。"""

    @abstractmethod
    def get_today_orders(self) -> list[Order]:
        """获取当日委托列表。"""
