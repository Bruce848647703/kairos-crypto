"""加密货币量化的基础数据类型（交易所无关）。

本模块只定义「数据」，不含任何网络、鉴权或撮合逻辑：

- 行情：``Candle``（K 线）、``Ticker``（最新报价快照）
- 交易：``Side`` / ``OrderType`` / ``OrderStatus``、``Order``（委托）、``Trade``（成交回报）
- 账户：``Balance``（币种余额，含冻结）、``SpotPosition``（现货持仓与已实现盈亏）

设计意图：内置的 :class:`~kairos_crypto.exchange.PaperExchange`（纸面撮合）与
用户自行实现的真实交易所适配器共用同一套类型，因此策略与引擎代码
无需关心底层是模拟撮合还是真实下单。所有对象都是纯数据结构，可自由序列化。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import pandas as pd

SYMBOL_SEPARATOR = "/"


def parse_symbol(symbol: str, separator: str = SYMBOL_SEPARATOR) -> Tuple[str, str]:
    """把交易对拆成 (base, quote)，如 ``"BTC/USDT" -> ("BTC", "USDT")``。

    参数
    ----
    symbol:    交易对字符串，必须包含分隔符。
    separator: 分隔符，默认 ``"/"``（加密行业惯例）。

    返回
    ----
    (base_currency, quote_currency) 二元组。
    """
    if not isinstance(symbol, str) or separator not in symbol:
        raise ValueError(f"交易对格式应为 BASE{separator}QUOTE，收到: {symbol!r}")
    base, quote = symbol.split(separator, 1)
    base, quote = base.strip().upper(), quote.strip().upper()
    if not base or not quote:
        raise ValueError(f"交易对格式应为 BASE{separator}QUOTE，收到: {symbol!r}")
    return base, quote


def join_symbol(base: str, quote: str, separator: str = SYMBOL_SEPARATOR) -> str:
    """由基础币与计价币拼出交易对字符串。"""
    return f"{base.strip().upper()}{separator}{quote.strip().upper()}"


class Side(Enum):
    """买卖方向。值为小写字符串，便于与外部接口/日志对齐。"""

    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        """方向符号：买入 +1，卖出 -1（用于持仓与现金流计算）。"""
        return 1 if self is Side.BUY else -1

    @classmethod
    def parse(cls, value: Any) -> "Side":
        """宽松解析方向：接受 ``Side``、``"buy"/"b"/1``、``"sell"/"s"/-1``。"""
        if isinstance(value, cls):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return cls.BUY if value > 0 else cls.SELL
        if isinstance(value, str):
            key = value.strip().lower()
            if key in ("buy", "b", "bid", "long"):
                return cls.BUY
            if key in ("sell", "s", "ask", "short"):
                return cls.SELL
        raise ValueError(f"无法解析交易方向: {value!r}")


class OrderType(Enum):
    """委托类型：市价单与限价单（纸面交易只实现这两类最常用订单）。"""

    MARKET = "market"
    LIMIT = "limit"

    @classmethod
    def parse(cls, value: Any) -> "OrderType":
        """宽松解析委托类型：接受枚举本身或 ``"market"/"limit"`` 等字符串。"""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            key = value.strip().lower()
            if key in ("market", "mkt", "m"):
                return cls.MARKET
            if key in ("limit", "lmt", "l"):
                return cls.LIMIT
        raise ValueError(f"无法解析委托类型: {value!r}")


class OrderStatus(Enum):
    """委托生命周期状态。"""

    NEW = "new"                    # 已挂出、尚未成交
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"              # 全部成交
    CANCELED = "canceled"          # 已撤单
    REJECTED = "rejected"          # 被拒绝（余额不足 / 无行情价等）

    @property
    def is_active(self) -> bool:
        """是否仍挂在订单簿上（可被撤单或继续撮合）。"""
        return self in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)


@dataclass(frozen=True)
class Candle:
    """单根 K 线（OHLCV）。

    参数
    ----
    symbol: 交易对，如 ``"BTC/USDT"``。
    ts:     K 线**开盘**时间戳，统一用毫秒 epoch（加密交易所惯例）。
    open/high/low/close: 四价，均为计价币（quote）报价。
    volume: 成交量，单位为**基础币**（base，如 BTC 数量）。

    说明
    ----
    对象不可变（``frozen=True``），可安全地在策略之间传递与缓存。
    """

    symbol: str
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def datetime(self) -> pd.Timestamp:
        """毫秒时间戳转 UTC ``pandas.Timestamp``，便于索引与打印。"""
        return pd.to_datetime(int(self.ts), unit="ms", utc=True)

    @property
    def mid(self) -> float:
        """最高价与最低价的中点。"""
        return 0.5 * (self.high + self.low)

    @property
    def typical_price(self) -> float:
        """典型价 (H+L+C)/3，常用于成交量加权指标。"""
        return (self.high + self.low + self.close) / 3.0

    @property
    def body(self) -> float:
        """实体幅度（收盘 - 开盘，带符号）。"""
        return self.close - self.open

    @property
    def is_bullish(self) -> bool:
        """是否阳线（收盘不低于开盘）。"""
        return self.close >= self.open

    @property
    def quote_volume(self) -> float:
        """以典型价估算的成交额（quote 计价）。"""
        return self.volume * self.typical_price

    def touched(self, price: float) -> bool:
        """本根 K 线的价格区间是否触及 ``price``（限价撮合判定用）。"""
        return self.low <= price <= self.high

    def crossed_below(self, price: float) -> bool:
        """本根 K 线是否下探到 ``price`` 及以下（限价买单可成交）。"""
        return self.low <= price

    def crossed_above(self, price: float) -> bool:
        """本根 K 线是否上冲到 ``price`` 及以上（限价卖单可成交）。"""
        return self.high >= price

    def is_consistent(self, tol: float = 1e-9) -> bool:
        """自检 OHLC 关系：``low <= min(open, close) <= max(open, close) <= high`` 且非负。"""
        prices = (self.open, self.high, self.low, self.close)
        if any(p != p for p in prices):            # NaN 检查
            return False
        if min(prices) < -tol:
            return False
        if self.low > min(self.open, self.close) + tol:
            return False
        if self.high < max(self.open, self.close) - tol:
            return False
        if self.high < self.low - tol:
            return False
        return self.volume >= -tol

    def to_dict(self) -> Dict[str, Any]:
        """转 dict，便于写入 DataFrame 或 CSV。"""
        return {
            "symbol": self.symbol, "ts": int(self.ts), "open": float(self.open),
            "high": float(self.high), "low": float(self.low),
            "close": float(self.close), "volume": float(self.volume),
        }


@dataclass(frozen=True)
class Ticker:
    """最新报价快照（纸面交易所由最近一根 K 线推导，无买卖价差）。

    参数
    ----
    last:        最新成交价。
    bid/ask:     买一/卖一价；纸面环境下等于最新价（无点差）。
    base_volume: 滚动成交量（基础币）。
    quote_volume: 滚动成交额（计价币）。
    """

    symbol: str
    ts: int
    last: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    base_volume: float = 0.0
    quote_volume: float = 0.0

    @property
    def mid(self) -> float:
        """买卖中间价；缺失时退化为最新价。"""
        if self.bid is None or self.ask is None:
            return self.last
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        """买卖价差；缺失时为 0。"""
        if self.bid is None or self.ask is None:
            return 0.0
        return self.ask - self.bid

    @classmethod
    def from_candle(cls, candle: Candle) -> "Ticker":
        """由一根 K 线构造快照（收盘价作为最新价）。"""
        return cls(symbol=candle.symbol, ts=candle.ts, last=candle.close,
                   bid=candle.close, ask=candle.close,
                   base_volume=candle.volume, quote_volume=candle.quote_volume)


@dataclass
class Order:
    """委托单。

    参数
    ----
    symbol:    交易对。
    side:      买卖方向。
    quantity:  委托数量（基础币，正数）；余额不足时可能被交易所下调为可成交量。
    order_type: 市价 / 限价。
    price:     限价单委托价；市价单为 ``None``。
    order_id:  交易所（或纸面撮合器）分配的委托号。
    client_id: 客户端自定义标识，便于策略跟踪自己挂出的单。
    status:    委托状态。
    filled_quantity:   已成交数量。
    average_fill_price: 成交均价（不含手续费）。
    fee_paid:  该委托累计支付的手续费（计价币）。
    created_ts/updated_ts: 创建/最后更新的毫秒时间戳。
    note:      补充说明（如「余额不足，数量已下调」「无行情价，等待撮合」）。
    """

    symbol: str
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    order_id: str = ""
    client_id: str = ""
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    fee_paid: float = 0.0
    created_ts: int = 0
    updated_ts: int = 0
    note: str = ""

    @property
    def remaining_quantity(self) -> float:
        """剩余未成交数量。"""
        return max(self.quantity - self.filled_quantity, 0.0)

    @property
    def is_limit(self) -> bool:
        return self.order_type is OrderType.LIMIT

    @property
    def is_market(self) -> bool:
        return self.order_type is OrderType.MARKET

    @property
    def is_active(self) -> bool:
        """是否仍挂在订单簿上。"""
        return self.status.is_active

    @property
    def base_currency(self) -> str:
        return parse_symbol(self.symbol)[0]

    @property
    def quote_currency(self) -> str:
        return parse_symbol(self.symbol)[1]

    def to_dict(self) -> Dict[str, Any]:
        """转 dict（枚举转为字符串值）。"""
        return {
            "order_id": self.order_id, "client_id": self.client_id, "symbol": self.symbol,
            "side": self.side.value, "order_type": self.order_type.value,
            "price": self.price, "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "status": self.status.value, "fee_paid": self.fee_paid,
            "created_ts": int(self.created_ts), "updated_ts": int(self.updated_ts),
            "note": self.note,
        }


@dataclass(frozen=True)
class Trade:
    """成交回报。

    参数
    ----
    price:    成交价（**不含**手续费）；市价单可能已含滑点。
    quantity: 成交数量（基础币）。
    fee:      手续费（计价币），= ``price * quantity * fee_rate``。
    is_maker: 是否为挂单方成交（限价单成交视为 maker，费率可更低）。
    """

    trade_id: str
    order_id: str
    symbol: str
    side: Side
    price: float
    quantity: float
    fee: float = 0.0
    ts: int = 0
    is_maker: bool = False

    @property
    def gross_value(self) -> float:
        """成交额（不含手续费），计价币。"""
        return self.price * self.quantity

    @property
    def cash_flow(self) -> float:
        """该笔成交对计价币现金的影响：买入为负、卖出为正，均已扣手续费。"""
        return -self.side.sign * self.gross_value - self.fee

    @property
    def effective_price(self) -> float:
        """含手续费的等效成交价 = ``价格 × (1 ± 费率)``（买入抬价、卖出压价）。"""
        if self.quantity <= 0:
            return self.price
        return (self.gross_value + (self.fee if self.side is Side.BUY else -self.fee)) / self.quantity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id, "order_id": self.order_id, "symbol": self.symbol,
            "side": self.side.value, "price": self.price, "quantity": self.quantity,
            "fee": self.fee, "gross_value": self.gross_value, "cash_flow": self.cash_flow,
            "ts": int(self.ts), "is_maker": self.is_maker,
        }


@dataclass
class Balance:
    """单币种余额（现货账户）。

    参数
    ----
    free:   可用余额。
    locked: 冻结余额（挂限价单预占的部分）。

    ``total = free + locked``，与真实交易所的账户视图一致。
    """

    currency: str
    free: float = 0.0
    locked: float = 0.0

    @property
    def total(self) -> float:
        return self.free + self.locked

    def reserve(self, amount: float) -> None:
        """从可用余额冻结 ``amount``（挂单预占）；余额不足抛 ``ValueError``。"""
        amount = float(amount)
        if amount < 0:
            raise ValueError("冻结金额不能为负")
        if amount > self.free + 1e-12:
            raise ValueError(f"{self.currency} 可用余额不足: 需要 {amount}, 仅有 {self.free}")
        self.free -= amount
        self.locked += amount

    def release(self, amount: float) -> None:
        """解冻 ``amount``（撤单/成交后退还多冻结部分）。"""
        amount = min(float(amount), self.locked)
        self.locked -= amount
        self.free += amount

    def settle_locked(self, amount: float) -> None:
        """用冻结余额支付 ``amount``（限价单成交时消耗预占资金）。"""
        amount = float(amount)
        if amount > self.locked + 1e-9:
            raise ValueError(f"{self.currency} 冻结余额不足: 需要 {amount}, 仅有 {self.locked}")
        self.locked -= amount

    def credit(self, amount: float) -> None:
        """增加可用余额（成交入账）。"""
        self.free += float(amount)

    def to_dict(self) -> Dict[str, Any]:
        return {"currency": self.currency, "free": self.free,
                "locked": self.locked, "total": self.total}


@dataclass
class SpotPosition:
    """现货持仓状态（按交易对聚合的成交结果）。

    只做会计，不做撮合：每次成交调用 :meth:`apply_fill` 更新数量、均价与已实现盈亏。
    均价采用「买入摊薄、卖出不改均价」的经典口径，且**不含手续费**，
    手续费单独累计在 :attr:`fee_paid`，便于净值与盈亏拆分核对。
    """

    symbol: str
    base_currency: str = ""
    quote_currency: str = ""
    quantity: float = 0.0
    average_price: float = 0.0
    realized_pnl: float = 0.0
    fee_paid: float = 0.0
    traded_quote: float = 0.0     # 累计成交额（计价币，不含手续费）
    buy_fills: int = 0
    sell_fills: int = 0
    highest_price: float = 0.0    # 持仓期间见过的最高价（移动止损用）
    lowest_price: float = 0.0     # 持仓期间见过的最低价

    def market_value(self, price: float) -> float:
        """按 ``price`` 估值的持仓市值（计价币）。"""
        return self.quantity * float(price)

    def unrealized_pnl(self, price: float) -> float:
        """浮动盈亏 = (现价 - 持仓均价) × 数量。"""
        return (float(price) - self.average_price) * self.quantity

    def return_pct(self, price: float) -> float:
        """持仓收益率（相对均价，不含手续费）。"""
        if self.quantity == 0 or self.average_price <= 0:
            return 0.0
        return float(price) / self.average_price - 1.0

    def apply_fill(self, side: Side, quantity: float, price: float, fee: float = 0.0) -> None:
        """把一笔成交计入持仓。

        参数
        ----
        side:     买卖方向。
        quantity: 成交数量（正数）。
        price:    成交价（不含手续费）。
        fee:      手续费（计价币）。
        """
        quantity = abs(float(quantity))
        price = float(price)
        signed = side.sign * quantity
        new_qty = self.quantity + signed
        if side is Side.BUY:
            if new_qty > 0:
                self.average_price = (self.quantity * self.average_price + quantity * price) / new_qty
            self.buy_fills += 1
        else:
            self.realized_pnl += (price - self.average_price) * quantity
            if abs(new_qty) < 1e-12:
                self.average_price = 0.0
            self.sell_fills += 1
        self.quantity = new_qty if abs(new_qty) > 1e-12 else 0.0
        if self.quantity == 0.0:
            self.highest_price = 0.0
            self.lowest_price = 0.0
        else:
            self.highest_price = price if self.highest_price <= 0 else max(self.highest_price, price)
            self.lowest_price = price if self.lowest_price <= 0 else min(self.lowest_price, price)
        self.fee_paid += float(fee)
        self.traded_quote += quantity * price

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol, "quantity": self.quantity,
            "average_price": self.average_price, "realized_pnl": self.realized_pnl,
            "fee_paid": self.fee_paid, "traded_quote": self.traded_quote,
            "buy_fills": self.buy_fills, "sell_fills": self.sell_fills,
        }
