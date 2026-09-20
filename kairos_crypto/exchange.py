"""交易所抽象接口 + 内置纸面（模拟）交易所。

两层设计：

1. :class:`Exchange`：**交易所无关**的抽象基类，只声明行情/下单/账户六类接口。
   用户若要对接真实交易所，只需继承它并实现这些方法（本仓库不提供、也不执行
   任何真实网络调用）。
2. :class:`PaperExchange`：**自研内存模拟实现**，用于纸面交易与研究验证。
   它维护余额与冻结、撮合市价单与限价单、按费率收取手续费、生成成交回报
   :class:`~kairos_crypto.types.Trade` 并更新持仓，全程无任何网络 I/O。

纸面撮合规则（确定性、可复现，且不使用未来数据）：

- 市价单：以「最近一次观测到的价格」立即成交，可叠加滑点（``slippage_bps``），
  手续费 = 成交额 × 费率，等价于按 ``价格 × (1 ± 费率)`` 的等效价格成交。
- 限价单：挂出后进入订单簿；只有当后续 K 线的价格区间**触及**委托价才成交
  （买单要求 ``low <= 限价``，卖单要求 ``high >= 限价``），成交价取委托价，
  若 K 线开盘即穿越委托价，则以更优的开盘价成交。
- 挂单会预占（冻结）资金：限价买单冻结 ``限价 × 数量 × (1 + 费率)``，
  限价卖单冻结基础币数量；成交后多冻结的部分自动退回，撤单则全额解冻。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .types import (
    Balance,
    Candle,
    Order,
    OrderStatus,
    OrderType,
    Side,
    SpotPosition,
    Ticker,
    Trade,
    join_symbol,
    parse_symbol,
)

DEFAULT_QUOTE = "USDT"
_EPS = 1e-12


class Exchange(ABC):
    """交易所无关的抽象接口。

    真实交易所适配器由使用者自行实现（REST/WebSocket 细节不在本仓库范围内）。
    基类只提供与具体交易所无关的便捷方法（``market_buy`` 等），
    以及一个默认不可用的 :meth:`equity`（模拟实现会覆盖它）。
    """

    #: 交易所名称，子类覆盖
    name: str = "abstract"
    #: 是否为纸面/模拟实现
    is_paper: bool = False

    # ---- 行情 ----
    @abstractmethod
    def fetch_candles(self, symbol: str, interval: str = "1h",
                      limit: int = 500, since: Optional[int] = None) -> List[Candle]:
        """获取 K 线历史（按时间升序返回）。

        参数
        ----
        symbol:   交易对，如 ``"BTC/USDT"``。
        interval: K 线周期，如 ``"1m"/"5m"/"1h"/"1d"``。
        limit:    最多返回根数。
        since:    起始毫秒时间戳（含），``None`` 表示不限。
        """

    @abstractmethod
    def fetch_ticker(self, symbol: str) -> Ticker:
        """获取单个交易对的最新报价快照。"""

    # ---- 交易 ----
    @abstractmethod
    def create_order(self, symbol: str, side: Side, quantity: float,
                     order_type: OrderType = OrderType.MARKET,
                     price: Optional[float] = None,
                     client_id: Optional[str] = None) -> Order:
        """提交委托，返回带状态的 :class:`Order`（模拟环境可能已直接成交）。"""

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """撤单；成功返回 ``True``，委托不存在或已终态返回 ``False``。"""

    # ---- 账户 ----
    @abstractmethod
    def fetch_balance(self, currency: Optional[str] = None) -> Dict[str, Balance]:
        """查询余额；``currency`` 为 ``None`` 时返回全部币种。"""

    @abstractmethod
    def fetch_orders(self, symbol: Optional[str] = None,
                     status: Optional[OrderStatus] = None) -> List[Order]:
        """查询委托列表，可按交易对与状态过滤。"""

    # ---- 便捷方法（基于上述抽象接口实现，子类无需重写）----
    def market_buy(self, symbol: str, quantity: float,
                   client_id: Optional[str] = None) -> Order:
        """市价买入。"""
        return self.create_order(symbol, Side.BUY, quantity, OrderType.MARKET, None, client_id)

    def market_sell(self, symbol: str, quantity: float,
                    client_id: Optional[str] = None) -> Order:
        """市价卖出。"""
        return self.create_order(symbol, Side.SELL, quantity, OrderType.MARKET, None, client_id)

    def limit_buy(self, symbol: str, price: float, quantity: float,
                  client_id: Optional[str] = None) -> Order:
        """限价买入。"""
        return self.create_order(symbol, Side.BUY, quantity, OrderType.LIMIT, price, client_id)

    def limit_sell(self, symbol: str, price: float, quantity: float,
                   client_id: Optional[str] = None) -> Order:
        """限价卖出。"""
        return self.create_order(symbol, Side.SELL, quantity, OrderType.LIMIT, price, client_id)

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """查询仍在订单簿上的委托（NEW / PARTIALLY_FILLED）。"""
        orders = [o for o in self.fetch_orders(symbol) if o.status.is_active]
        return orders

    def last_price(self, symbol: str) -> float:
        """最新价（由 :meth:`fetch_ticker` 推导）。"""
        return float(self.fetch_ticker(symbol).last)

    def equity(self, prices: Optional[Dict[str, float]] = None) -> float:
        """账户总权益（以计价币计）。抽象接口默认不支持，需子类实现。"""
        raise NotImplementedError(f"{type(self).__name__} 未实现 equity()")


class PaperExchange(Exchange):
    """自研内存纸面交易所：余额 + 订单簿 + 撮合 + 成交回报，零网络。

    参数
    ----
    quote:           计价币（记账货币），默认 ``"USDT"``；:meth:`equity` 以它为单位。
    balances:        初始余额字典，如 ``{"USDT": 10000, "BTC": 0.5}``。
    initial_quote:   未显式给 ``balances`` 时，计价币的初始金额。
    fee_rate:        吃单（taker）手续费率，如 ``0.001`` 表示千一。
    maker_fee_rate:  挂单（maker）手续费率；``None`` 表示与 ``fee_rate`` 相同。
    slippage_bps:    市价单滑点基点（1bp = 0.01%），买入抬价、卖出压价。
    allow_short:     是否允许卖出超过持有的基础币（默认否，现货语义）。
    clamp_to_balance: 余额不足时是否把**市价单**数量下调为「可成交量」（``False`` 则拒单）。
                     限价单始终要求能全额预占资金/币，否则拒单（与现货交易所一致）。

    典型用法::

        ex = PaperExchange(quote="USDT", initial_quote=10_000, fee_rate=0.001)
        for candle in candles:          # 逐根喂行情
            ex.on_candle(candle)        # 更新最新价 + 撮合挂单
            ex.market_buy("BTC/USDT", 0.01)
    """

    name = "paper"
    is_paper = True

    def __init__(self,
                 quote: str = DEFAULT_QUOTE,
                 balances: Optional[Dict[str, float]] = None,
                 initial_quote: float = 10_000.0,
                 fee_rate: float = 0.001,
                 maker_fee_rate: Optional[float] = None,
                 slippage_bps: float = 0.0,
                 allow_short: bool = False,
                 clamp_to_balance: bool = True):
        self.quote = quote.strip().upper()
        self.fee_rate = float(fee_rate)
        self.maker_fee_rate = self.fee_rate if maker_fee_rate is None else float(maker_fee_rate)
        self.slippage_bps = float(slippage_bps)
        self.allow_short = bool(allow_short)
        self.clamp_to_balance = bool(clamp_to_balance)

        seed: Dict[str, float] = {self.quote: float(initial_quote)}
        for cur, amount in (balances or {}).items():
            seed[cur.strip().upper()] = float(amount)
        self._initial_balances: Dict[str, float] = dict(seed)

        self._balances: Dict[str, Balance] = {c: Balance(c, a, 0.0) for c, a in seed.items()}
        self._orders: Dict[str, Order] = {}
        self._trades: List[Trade] = []
        self._positions: Dict[str, SpotPosition] = {}
        self._reservations: Dict[str, Tuple[float, float]] = {}   # order_id -> (冻结计价币, 冻结基础币)
        self._last_price: Dict[str, float] = {}
        self._candles: Dict[str, List[Candle]] = {}
        self._clock: int = 0
        self._order_seq: int = 0
        self._trade_seq: int = 0

    # ------------------------------------------------------------------ 状态
    def reset(self) -> None:
        """清空全部订单/成交/行情，并把余额恢复到初始状态。"""
        self._balances = {c: Balance(c, a, 0.0) for c, a in self._initial_balances.items()}
        self._orders.clear()
        self._trades.clear()
        self._positions.clear()
        self._reservations.clear()
        self._last_price.clear()
        self._candles.clear()
        self._clock = 0
        self._order_seq = 0
        self._trade_seq = 0

    @property
    def clock(self) -> int:
        """当前模拟时钟（最近一次观测到的毫秒时间戳）。"""
        return self._clock

    @property
    def trades(self) -> List[Trade]:
        """全部成交回报（按时间顺序）。"""
        return list(self._trades)

    @property
    def cash(self) -> float:
        """计价币可用余额。"""
        return self.free(self.quote)

    def balance(self, currency: str) -> Balance:
        """返回某币种的**活动**余额对象（内部记账用，勿在策略里直接修改）。"""
        return self._balance(currency)

    def balances(self) -> Dict[str, Balance]:
        """全部币种余额（副本）。"""
        return {c: replace(b) for c, b in self._balances.items()}

    def free(self, currency: str) -> float:
        return self._balance(currency).free

    def locked(self, currency: str) -> float:
        return self._balance(currency).locked

    def total(self, currency: str) -> float:
        return self._balance(currency).total

    def position(self, symbol: str) -> SpotPosition:
        """某交易对的持仓状态（数量/均价/已实现盈亏/手续费）。"""
        if symbol not in self._positions:
            base, quote = parse_symbol(symbol)
            self._positions[symbol] = SpotPosition(symbol=symbol, base_currency=base,
                                                   quote_currency=quote)
        return self._positions[symbol]

    def positions(self) -> Dict[str, SpotPosition]:
        """全部持仓（只返回数量非零的）。"""
        return {s: p for s, p in self._positions.items() if p.quantity != 0}

    def holdings_value(self, prices: Optional[Dict[str, float]] = None) -> float:
        """非计价币资产的市值合计（计价币计）。"""
        value = 0.0
        for cur, bal in self._balances.items():
            if cur == self.quote or bal.total == 0:
                continue
            value += bal.total * self._price_in_quote(cur, prices)
        return value

    def equity(self, prices: Optional[Dict[str, float]] = None) -> float:
        """账户总权益 = 计价币余额（含冻结） + 其它币种市值。"""
        return self.total(self.quote) + self.holdings_value(prices)

    def realized_pnl(self, symbol: Optional[str] = None) -> float:
        """已实现盈亏合计（不含手续费）。"""
        items = self._positions.values() if symbol is None else [self.position(symbol)]
        return float(sum(p.realized_pnl for p in items))

    def fee_paid(self, symbol: Optional[str] = None) -> float:
        """累计手续费。"""
        if symbol is None:
            return float(sum(t.fee for t in self._trades))
        return float(sum(t.fee for t in self._trades if t.symbol == symbol))

    def unrealized_pnl(self, prices: Optional[Dict[str, float]] = None) -> float:
        """浮动盈亏合计（按最新价对持仓均价）。"""
        total = 0.0
        for sym, pos in self._positions.items():
            if pos.quantity == 0:
                continue
            total += pos.unrealized_pnl(self._price_of(sym, prices))
        return float(total)

    # ------------------------------------------------------------- 行情输入
    def on_candle(self, candle: Candle) -> List[Trade]:
        """喂入一根新 K 线：更新最新价、保存历史，并撮合挂单。

        返回本根 K 线产生的成交列表（可能为空）。挂单只会与**调用本方法时**
        传入的 K 线撮合，因此「当根下单、当根按最低价成交」这类未来函数不会发生。
        """
        self._clock = max(self._clock, int(candle.ts))
        self._candles.setdefault(candle.symbol, []).append(candle)
        self._last_price[candle.symbol] = float(candle.close)
        return self._match_candle(candle)

    def on_candles(self, candles: Iterable[Candle]) -> List[Trade]:
        """批量喂入 K 线，返回全部成交。"""
        fills: List[Trade] = []
        for candle in candles:
            fills.extend(self.on_candle(candle))
        return fills

    def on_ticker(self, ticker: Ticker) -> List[Trade]:
        """喂入一次报价快照：更新最新价并按最新价撮合挂单。"""
        self._clock = max(self._clock, int(ticker.ts))
        last = float(ticker.last)
        self._last_price[ticker.symbol] = last
        fills: List[Trade] = []
        for order in self._active_orders(ticker.symbol):
            price = self._tick_price(order, last)
            if price is None:
                continue
            trade = self._fill(order, price, order.remaining_quantity,
                               int(ticker.ts), is_maker=order.is_limit)
            if trade is not None:
                fills.append(trade)
        return fills

    def has_price(self, symbol: str) -> bool:
        """是否已观测到该交易对的行情价。"""
        return symbol in self._last_price

    def mark(self, symbol: str, price: float, ts: Optional[int] = None) -> None:
        """手工更新最新价（不撮合），用于外部估值。"""
        self._last_price[symbol] = float(price)
        if ts is not None:
            self._clock = max(self._clock, int(ts))

    # --------------------------------------------------------------- 下单
    def create_order(self, symbol: str, side: Side, quantity: float,
                     order_type: OrderType = OrderType.MARKET,
                     price: Optional[float] = None,
                     client_id: Optional[str] = None) -> Order:
        """提交委托。

        市价单立即以最新价（含滑点与手续费）成交；限价单若当前价已可成交则
        立即成交，否则挂入订单簿等待后续 K 线触及。余额不足时：市价单在
        ``clamp_to_balance=True`` 下把数量下调为可成交量，限价单一律返回 ``REJECTED``。
        """
        side = Side.parse(side)
        order_type = OrderType.parse(order_type)
        base, quote = parse_symbol(symbol)
        quantity = float(quantity)
        if quantity <= 0:
            raise ValueError(f"委托数量必须为正数，收到: {quantity}")
        limit_price: Optional[float] = None
        if order_type is OrderType.LIMIT:
            if price is None or float(price) <= 0:
                raise ValueError("限价单必须提供正的委托价 price")
            limit_price = float(price)

        order = Order(symbol=symbol, side=side, quantity=quantity, order_type=order_type,
                      price=limit_price, order_id=self._next_order_id(),
                      client_id=client_id or "", status=OrderStatus.NEW,
                      created_ts=self._clock, updated_ts=self._clock)
        self._orders[order.order_id] = order

        if not self._admit(order, base, quote):
            return order                      # 已被拒单

        if order.is_market:
            ref = self._last_price.get(symbol)
            if ref is None:
                order.status = OrderStatus.REJECTED
                order.note = "尚未观测到行情价，市价单无法成交"
                order.updated_ts = self._clock
                return order
            exec_price = self._apply_slippage(ref, side)
            self._fill(order, exec_price, order.remaining_quantity,
                       self._clock, is_maker=False)
            return order

        ref = self._last_price.get(symbol)
        if ref is not None and self._is_crossing(order):
            # 可立即成交的限价单：以最新价（对委托方更优）成交，按吃单费率计费
            self._fill(order, float(ref), order.remaining_quantity,
                       self._clock, is_maker=False)
            return order
        order.note = "已挂出，等待价格触及限价"
        return order

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """撤单并解冻预占资金；委托不存在或已终态返回 ``False``。"""
        order = self._orders.get(order_id)
        if order is None or not order.is_active:
            return False
        if symbol is not None and order.symbol != symbol:
            return False
        self._release(order)
        order.status = OrderStatus.CANCELED
        order.updated_ts = self._clock
        order.note = "已撤单"
        return True

    def cancel_all(self, symbol: Optional[str] = None) -> int:
        """撤销全部（或某交易对的）挂单，返回撤单笔数。"""
        count = 0
        for order in self._active_orders(symbol):
            if self.cancel_order(order.order_id):
                count += 1
        return count

    def open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """当前挂单（活动状态），按创建顺序。"""
        return [replace(o) for o in self._active_orders(symbol)]

    # --------------------------------------------------------------- 查询
    def fetch_candles(self, symbol: str, interval: str = "1h",
                      limit: int = 500, since: Optional[int] = None) -> List[Candle]:
        """返回纸面交易所**已观测到**的 K 线（不联网，历史来自 :meth:`on_candle`）。"""
        hist = [c for c in self._candles.get(symbol, [])
                if since is None or int(c.ts) >= int(since)]
        if limit is not None and limit > 0:
            hist = hist[-int(limit):]
        return hist

    def fetch_ticker(self, symbol: str) -> Ticker:
        """由最近一根 K 线/最新价构造报价快照（纸面环境无买卖价差）。"""
        price = self._last_price.get(symbol)
        if price is None:
            raise ValueError(f"尚未观测到 {symbol} 的行情，请先调用 on_candle()/on_ticker()")
        hist = self._candles.get(symbol) or []
        candle = hist[-1] if hist else Candle(symbol=symbol, ts=self._clock, open=price,
                                              high=price, low=price, close=price)
        ticker = Ticker.from_candle(candle)
        if ticker.last != price:                # mark() 手工改价时以最新价为准
            ticker = Ticker(symbol=symbol, ts=self._clock, last=price, bid=price, ask=price,
                            base_volume=ticker.base_volume, quote_volume=ticker.quote_volume)
        return ticker

    def fetch_balance(self, currency: Optional[str] = None) -> Dict[str, Balance]:
        """查询余额（返回副本，避免外部误改内部记账）。"""
        if currency is not None:
            cur = currency.strip().upper()
            return {cur: replace(self._balance(cur))}
        return self.balances()

    def fetch_orders(self, symbol: Optional[str] = None,
                     status: Optional[OrderStatus] = None) -> List[Order]:
        """查询委托列表（返回副本），可按交易对与状态过滤。"""
        want: Optional[OrderStatus] = None
        if status is not None:
            want = status if isinstance(status, OrderStatus) else OrderStatus(status)
        out: List[Order] = []
        for order in self._orders.values():
            if symbol is not None and order.symbol != symbol:
                continue
            if want is not None and order.status is not want:
                continue
            out.append(replace(order))
        return out

    # ------------------------------------------------------------- 内部实现
    def _balance(self, currency: str) -> Balance:
        cur = currency.strip().upper()
        if cur not in self._balances:
            self._balances[cur] = Balance(cur, 0.0, 0.0)
        return self._balances[cur]

    def _price_in_quote(self, currency: str, prices: Optional[Dict[str, float]] = None) -> float:
        """把某币种折算成计价币的价格；无法定价时返回 0（并建议外部传入 ``prices``）。"""
        cur = currency.strip().upper()
        if cur == self.quote:
            return 1.0
        if prices and cur in prices:
            return float(prices[cur])
        symbol = join_symbol(cur, self.quote)
        if symbol in self._last_price:
            return float(self._last_price[symbol])
        pos = self._positions.get(symbol)
        if pos is not None and pos.average_price > 0:
            return float(pos.average_price)
        return 0.0

    def _price_of(self, symbol: str, prices: Optional[Dict[str, float]] = None) -> float:
        if symbol in self._last_price:
            return float(self._last_price[symbol])
        base, _ = parse_symbol(symbol)
        return self._price_in_quote(base, prices)

    def _next_order_id(self) -> str:
        self._order_seq += 1
        return f"P{self._order_seq:06d}"

    def _next_trade_id(self) -> str:
        self._trade_seq += 1
        return f"T{self._trade_seq:06d}"

    def _apply_slippage(self, price: float, side: Side) -> float:
        """市价单滑点：买入抬价、卖出压价。"""
        return float(price) * (1.0 + side.sign * self.slippage_bps / 10_000.0)

    def _fee_rate_for(self, is_maker: bool) -> float:
        return self.maker_fee_rate if is_maker else self.fee_rate

    def _active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        return [o for o in self._orders.values()
                if o.is_active and (symbol is None or o.symbol == symbol)]

    def _admit(self, order: Order, base: str, quote: str) -> bool:
        """下单前置校验：数量裁剪与资金/持仓冻结。

        返回 ``True`` 表示可以进入撮合流程（或挂单等待），
        ``False`` 表示已置为 ``REJECTED``（余额不足）。
        """
        if self._lacks_price(order):
            return True                     # 无行情价：由 create_order 统一拒单
        immediate = self._is_crossing(order)
        is_maker = order.is_limit and not immediate
        ref_price = self._reference_price(order, immediate)
        fee_rate = self._fee_rate_for(is_maker)

        # 限价单必须能全额预占，否则拒单（与现货交易所一致）；
        # 市价单在 clamp_to_balance=True 时可下调数量，便于「一把梭」全仓买入。
        clampable = order.is_market and self.clamp_to_balance
        if order.side is Side.BUY:
            unit_cost = ref_price * (1.0 + fee_rate)
            affordable = self.free(quote) / unit_cost if unit_cost > 0 else 0.0
            if order.quantity > affordable + _EPS:
                if not clampable or affordable <= _EPS:
                    order.status = OrderStatus.REJECTED
                    order.note = (f"{quote} 可用余额不足: 需要约 "
                                  f"{order.quantity * unit_cost:.8f}, 仅有 {self.free(quote):.8f}")
                    order.updated_ts = self._clock
                    return False
                order.quantity = max(affordable * (1.0 - 1e-9), 0.0)
                order.note = f"余额不足，数量已下调为可成交量 {order.quantity:.8f}"
        else:
            available = self.free(base)
            if not self.allow_short and order.quantity > available + _EPS:
                if not clampable or available <= _EPS:
                    order.status = OrderStatus.REJECTED
                    order.note = (f"{base} 可用余额不足: 需要 {order.quantity:.8f}, "
                                  f"仅有 {available:.8f}")
                    order.updated_ts = self._clock
                    return False
                order.quantity = max(available * (1.0 - 1e-9), 0.0)
                order.note = f"持仓不足，数量已下调为可成交量 {order.quantity:.8f}"

        if order.is_limit and not immediate:
            # 挂单预占资金/币，成交后按实际成交额结算并退回差额
            reserve_quote = 0.0
            reserve_base = 0.0
            try:
                if order.side is Side.BUY:
                    reserve_quote = float(order.price) * order.quantity * (1.0 + fee_rate)
                    self._balance(quote).reserve(reserve_quote)
                else:
                    reserve_base = order.quantity
                    self._balance(base).reserve(reserve_base)
            except ValueError as exc:
                order.status = OrderStatus.REJECTED
                order.note = str(exc)
                order.updated_ts = self._clock
                return False
            self._reservations[order.order_id] = (reserve_quote, reserve_base)
        return True

    def _lacks_price(self, order: Order) -> bool:
        """该委托是否缺少可用于定价/校验的最新价。"""
        return order.symbol not in self._last_price

    def _reference_price(self, order: Order, immediate: bool = False) -> float:
        """资金校验用的参考价格。

        - 市价单：含滑点的最新价；
        - 可立即成交的限价单：最新价（吃单，无滑点）；
        - 普通挂单：委托价（最坏情况下的成本）。
        """
        if order.is_limit and not immediate:
            return float(order.price)
        last = float(self._last_price.get(order.symbol, 0.0))
        if order.is_market:
            return self._apply_slippage(last, order.side)
        return last

    def _is_crossing(self, order: Order) -> bool:
        """限价单是否已可立即成交（最新价已优于/等于委托价）。"""
        if not order.is_limit:
            return False
        last = self._last_price.get(order.symbol)
        if last is None:
            return False
        if order.side is Side.BUY:
            return last <= float(order.price)
        return last >= float(order.price)

    def _release(self, order: Order) -> None:
        """解冻某委托的预占资金。"""
        reserve_quote, reserve_base = self._reservations.pop(order.order_id, (0.0, 0.0))
        if reserve_quote:
            _, quote = parse_symbol(order.symbol)
            self._balance(quote).release(reserve_quote)
        if reserve_base:
            base, _ = parse_symbol(order.symbol)
            self._balance(base).release(reserve_base)

    def _match_candle(self, candle: Candle) -> List[Trade]:
        fills: List[Trade] = []
        for order in self._active_orders(candle.symbol):
            price = self._touch_price(order, candle)
            if price is None:
                continue
            trade = self._fill(order, price, order.remaining_quantity,
                               int(candle.ts), is_maker=order.is_limit)
            if trade is not None:
                fills.append(trade)
        return fills

    @staticmethod
    def _touch_price(order: Order, candle: Candle) -> Optional[float]:
        """限价单是否被这根 K 线触及；触及则返回成交价，否则 ``None``。"""
        if order.is_market:                     # 遗留市价单（下单时无行情）：开盘价成交
            return float(candle.open)
        limit = float(order.price)
        if order.side is Side.BUY:
            if candle.open <= limit:            # 开盘已穿越 → 以更优的开盘价成交
                return float(candle.open)
            if candle.low <= limit:
                return limit
            return None
        if candle.open >= limit:
            return float(candle.open)
        if candle.high >= limit:
            return limit
        return None

    @staticmethod
    def _tick_price(order: Order, last: float) -> Optional[float]:
        """按最新价判定挂单是否可成交（tick 级撮合）。"""
        if order.is_market:
            return float(last)
        limit = float(order.price)
        if order.side is Side.BUY:
            return float(last) if last <= limit else None
        return float(last) if last >= limit else None

    def _fill(self, order: Order, price: float, quantity: float, ts: int,
              is_maker: bool) -> Optional[Trade]:
        """执行成交：更新余额、持仓、委托状态，返回成交回报（非法输入返回 ``None``）。"""
        quantity = float(quantity)
        price = float(price)
        if quantity <= 0 or price <= 0:
            order.status = OrderStatus.REJECTED
            order.note = order.note or "成交数量或价格非法"
            return None

        base, quote = parse_symbol(order.symbol)
        fee_rate = self._fee_rate_for(is_maker)
        gross = price * quantity
        fee = gross * fee_rate

        quote_bal = self._balance(quote)
        base_bal = self._balance(base)
        reserve_quote, reserve_base = self._reservations.pop(order.order_id, (0.0, 0.0))

        if order.side is Side.BUY:
            cost = gross + fee
            if reserve_quote > 0:
                quote_bal.settle_locked(reserve_quote)
                quote_bal.credit(reserve_quote - cost)     # 退回多冻结部分（限价低于委托价时）
            else:
                quote_bal.free -= cost
            base_bal.credit(quantity)
        else:
            proceeds = gross - fee
            if reserve_base > 0:
                base_bal.settle_locked(reserve_base)
            else:
                base_bal.free -= quantity
            quote_bal.credit(proceeds)

        position = self.position(order.symbol)
        position.apply_fill(order.side, quantity, price, fee)

        filled_before = order.filled_quantity
        order.filled_quantity = filled_before + quantity
        order.average_fill_price = (
            (order.average_fill_price * filled_before + price * quantity) / order.filled_quantity
        )
        order.fee_paid += fee
        order.status = (OrderStatus.FILLED if order.remaining_quantity <= _EPS
                        else OrderStatus.PARTIALLY_FILLED)
        order.updated_ts = ts

        trade = Trade(trade_id=self._next_trade_id(), order_id=order.order_id,
                      symbol=order.symbol, side=order.side, price=price, quantity=quantity,
                      fee=fee, ts=ts, is_maker=is_maker)
        self._trades.append(trade)
        return trade

    def last_trade(self) -> Optional[Trade]:
        """最近一笔成交（便于策略即时核对）。"""
        return self._trades[-1] if self._trades else None

    # ------------------------------------------------------------- 调试输出
    def snapshot(self, prices: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """账户快照，用于日志与断言。"""
        return {
            "equity": self.equity(prices),
            "cash": self.total(self.quote),
            "free_cash": self.free(self.quote),
            "locked_cash": self.locked(self.quote),
            "holdings_value": self.holdings_value(prices),
            "realized_pnl": self.realized_pnl(),
            "unrealized_pnl": self.unrealized_pnl(prices),
            "fee_paid": self.fee_paid(),
            "trades": float(len(self._trades)),
            "open_orders": float(len(self._active_orders())),
        }

    def trades_frame(self) -> pd.DataFrame:
        """成交回报转 DataFrame。"""
        return pd.DataFrame([t.to_dict() for t in self._trades])

    def orders_frame(self) -> pd.DataFrame:
        """全部委托转 DataFrame。"""
        return pd.DataFrame([o.to_dict() for o in self._orders.values()])

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return (f"PaperExchange(quote={self.quote!r}, fee_rate={self.fee_rate}, "
                f"equity={self.equity():.4f}, open_orders={len(self._active_orders())}, "
                f"trades={len(self._trades)})")
