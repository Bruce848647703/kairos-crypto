"""策略基类、运行时上下文与三个自研示例策略。

- :class:`Context`：引擎在每根 K 线更新一次的「策略视角世界」，
  提供**只含历史数据**的行情访问（``closes/history``）与统一下单接口
  （``buy/sell/limit_buy/limit_sell/order_target_percent``），
  策略因此与具体交易所实现解耦。
- :class:`Strategy`：策略基类，子类实现 :meth:`~Strategy.on_candle`。
- 示例策略（全部自研，仅用 numpy/pandas）：
  :class:`MomentumStrategy`（均线 + 动量突破）、
  :class:`GridStrategy`（等差网格挂限价单）、
  :class:`DcaStrategy`（定期定额买入）。

防未来函数：策略在 ``on_candle(candle, ctx)`` 中只能拿到截至当根 K 线的数据；
市价单以当根收盘价成交，限价单要等**后续** K 线触及才成交。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from . import risk
from .data import Replayer
from .exchange import Exchange
from .types import (
    Balance,
    Candle,
    Order,
    OrderStatus,
    Side,
    SpotPosition,
    Trade,
    parse_symbol,
)

DEFAULT_QUOTE = "USDT"


@dataclass
class Context:
    """策略运行时上下文（由引擎维护，策略通过它读行情、查账户、下单）。

    参数
    ----
    exchange:   交易所实例（内置为 :class:`~kairos_crypto.exchange.PaperExchange`）。
    replayer:   K 线回放器，提供「截至当前」的历史。
    symbol:     当前 K 线所属交易对。
    index:      当前 K 线序号（0 起）。
    candle:     当前 K 线。
    new_trades: 本根 K 线撮合阶段产生的成交（策略可据此补挂/记账）。
    quote:      计价币；``None`` 时自动取交易所的计价币。
    """

    exchange: Exchange
    replayer: Replayer
    symbol: str = ""
    index: int = 0
    candle: Optional[Candle] = None
    new_trades: List[Trade] = field(default_factory=list)
    quote: Optional[str] = None

    def __post_init__(self) -> None:
        if self.quote is None:
            self.quote = str(getattr(self.exchange, "quote", DEFAULT_QUOTE)).upper()

    # ------------------------------------------------------------- 行情访问
    def _resolve(self, symbol: Optional[str] = None) -> str:
        target = symbol or self.symbol
        if not target:
            raise ValueError("未指定交易对：请传入 symbol 或等待引擎设置 ctx.symbol")
        return target

    @property
    def bar(self) -> Candle:
        """当前 K 线。"""
        if self.candle is None:
            raise RuntimeError("尚未开始回放，当前无 K 线")
        return self.candle

    def history(self, n: Optional[int] = None, symbol: Optional[str] = None) -> List[Candle]:
        """已回放（含当根）的最近 ``n`` 根 K 线。"""
        return self.replayer.history(self._resolve(symbol), n)

    def closes(self, n: Optional[int] = None, symbol: Optional[str] = None) -> np.ndarray:
        """已回放 K 线的收盘价数组（无未来数据）。"""
        return self.replayer.closes(self._resolve(symbol), n)

    def returns(self, n: Optional[int] = None, symbol: Optional[str] = None) -> np.ndarray:
        """已回放 K 线的简单收益率数组（首元素为 0）。"""
        return self.replayer.returns(self._resolve(symbol), n)

    def last_price(self, symbol: Optional[str] = None) -> float:
        """最新价：优先取当前 K 线收盘价，其次取交易所记录的最近价格。"""
        sym = self._resolve(symbol)
        if self.candle is not None and self.candle.symbol == sym:
            return float(self.candle.close)
        getter = getattr(self.exchange, "has_price", None)
        if callable(getter) and getter(sym):
            return float(self.exchange.last_price(sym))     # type: ignore[attr-defined]
        if self.candle is not None:
            return float(self.candle.close)
        raise ValueError(f"尚无 {sym} 的行情价")

    def base_currency(self, symbol: Optional[str] = None) -> str:
        return parse_symbol(self._resolve(symbol))[0]

    def quote_currency(self, symbol: Optional[str] = None) -> str:
        return parse_symbol(self._resolve(symbol))[1]

    # ------------------------------------------------------------- 账户访问
    def equity(self) -> float:
        """账户总权益（计价币）。真实适配器若未实现 equity，则由余额与最新价估算。"""
        fn = getattr(self.exchange, "equity", None)
        if callable(fn):
            try:
                return float(fn())
            except NotImplementedError:
                pass
        total = 0.0
        for currency, balance in self.exchange.fetch_balance().items():
            if currency == self.quote:
                total += balance.total
            else:
                sym = f"{currency}/{self.quote}"
                try:
                    total += balance.total * self.last_price(sym)
                except (ValueError, KeyError):
                    total += 0.0
        return total

    def _balance_of(self, currency: Optional[str] = None):
        """取某币种余额对象（默认计价币）；交易所未记录时返回零余额。"""
        cur = (currency or self.quote or DEFAULT_QUOTE).upper()
        balances = self.exchange.fetch_balance(cur)
        return balances.get(cur) or Balance(cur, 0.0, 0.0)

    def free(self, currency: Optional[str] = None) -> float:
        """某币种可用余额（默认计价币）。"""
        return float(self._balance_of(currency).free)

    def locked(self, currency: Optional[str] = None) -> float:
        """某币种冻结余额（默认计价币）。"""
        return float(self._balance_of(currency).locked)

    def total(self, currency: Optional[str] = None) -> float:
        """某币种总余额（可用 + 冻结）。"""
        return float(self._balance_of(currency).total)

    def position(self, symbol: Optional[str] = None) -> SpotPosition:
        """持仓状态（数量/均价/已实现盈亏）。"""
        sym = self._resolve(symbol)
        getter = getattr(self.exchange, "position", None)
        if callable(getter):
            return getter(sym)
        base = parse_symbol(sym)[0]
        balance = self.exchange.fetch_balance(base)[base]
        return SpotPosition(symbol=sym, base_currency=base, quantity=balance.total)

    def holding(self, symbol: Optional[str] = None) -> float:
        """当前持有的基础币数量。"""
        return float(self.position(symbol).quantity)

    def open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """当前挂单。"""
        return list(self.exchange.fetch_open_orders(self._resolve(symbol)))

    def trades(self) -> List[Trade]:
        """全部历史成交（交易所未记录时返回空列表）。"""
        trades = getattr(self.exchange, "trades", None)
        return list(trades) if trades else []

    # ---------------------------------------------------------------- 下单
    @property
    def fee_rate(self) -> float:
        """吃单费率（交易所未暴露时按 0 处理）。"""
        return float(getattr(self.exchange, "fee_rate", 0.0))

    def buy(self, quantity: Optional[float] = None, symbol: Optional[str] = None,
            quote: Optional[float] = None, client_id: Optional[str] = None) -> Optional[Order]:
        """市价买入：给定基础币 ``quantity``，或给定计价币金额 ``quote``。"""
        sym = self._resolve(symbol)
        if quantity is None and quote is None:
            raise ValueError("buy() 需要 quantity 或 quote 之一")
        if quantity is None:
            quantity = risk.quote_to_quantity(float(quote), self.last_price(sym), self.fee_rate)
        quantity = float(quantity)
        if quantity <= 0:
            return None
        return self.exchange.market_buy(sym, quantity, client_id)

    def sell(self, quantity: Optional[float] = None, symbol: Optional[str] = None,
             quote: Optional[float] = None, client_id: Optional[str] = None) -> Optional[Order]:
        """市价卖出：给定基础币 ``quantity``，或给定「卖出价值」``quote``。"""
        sym = self._resolve(symbol)
        if quantity is None and quote is None:
            raise ValueError("sell() 需要 quantity 或 quote 之一")
        if quantity is None:
            quantity = float(quote) / self.last_price(sym)
        quantity = float(quantity)
        if quantity <= 0:
            return None
        return self.exchange.market_sell(sym, quantity, client_id)

    def flatten(self, symbol: Optional[str] = None,
                client_id: Optional[str] = None) -> Optional[Order]:
        """市价平掉全部持仓（现货语义下即卖出全部基础币）。"""
        sym = self._resolve(symbol)
        quantity = self.holding(sym)
        if quantity <= 0:
            return None
        return self.sell(quantity=quantity, symbol=sym, client_id=client_id)

    def limit_buy(self, price: float, quantity: float, symbol: Optional[str] = None,
                  client_id: Optional[str] = None) -> Order:
        """限价买入挂单。"""
        return self.exchange.limit_buy(self._resolve(symbol), float(price), float(quantity),
                                       client_id)

    def limit_sell(self, price: float, quantity: float, symbol: Optional[str] = None,
                   client_id: Optional[str] = None) -> Order:
        """限价卖出挂单。"""
        return self.exchange.limit_sell(self._resolve(symbol), float(price), float(quantity),
                                        client_id)

    def order_target_value(self, value: float, symbol: Optional[str] = None,
                           client_id: Optional[str] = None) -> Optional[Order]:
        """把某交易对的持仓市值调整到 ``value``（计价币），差额用市价单成交。"""
        sym = self._resolve(symbol)
        price = self.last_price(sym)
        if price <= 0:
            return None
        current = self.holding(sym) * price
        delta = float(value) - current
        if delta > 0:
            return self.buy(quantity=delta / price, symbol=sym, client_id=client_id)
        if delta < 0:
            return self.sell(quantity=min(-delta / price, self.holding(sym)), symbol=sym,
                             client_id=client_id)
        return None

    def order_target_percent(self, percent: float, symbol: Optional[str] = None,
                             client_id: Optional[str] = None) -> Optional[Order]:
        """把某交易对持仓调整到净值的 ``percent`` 比例（如 1.0 = 满仓）。"""
        return self.order_target_value(float(percent) * self.equity(), symbol, client_id)

    def cancel(self, order_id: str) -> bool:
        """撤单。"""
        return bool(self.exchange.cancel_order(order_id))

    def cancel_all(self, symbol: Optional[str] = None) -> int:
        """撤销某交易对（默认当前交易对）的全部挂单，返回撤单笔数。"""
        sym = self._resolve(symbol)
        count = 0
        for order in self.exchange.fetch_open_orders(sym):
            if self.exchange.cancel_order(order.order_id):
                count += 1
        return count


class Strategy:
    """策略基类。

    子类至少要实现 :meth:`on_candle`；可选实现 :meth:`on_start` / :meth:`on_stop`
    做初始化与收尾（如网格建底仓、结束时平仓）。

    ``symbol=None`` 表示「跟随引擎当前 K 线的交易对」，便于同一策略跑多个标的。
    """

    name: str = "strategy"

    def __init__(self, symbol: Optional[str] = None):
        self.symbol = symbol
        self.started = False

    def resolve_symbol(self, ctx: Context) -> str:
        """确定本策略作用的交易对（首次调用时固定下来）。"""
        if self.symbol is None:
            self.symbol = ctx.symbol or (ctx.replayer.symbols[0] if len(ctx.replayer) else "")
        return self.symbol

    def on_start(self, ctx: Context) -> None:
        """回放开始前的钩子（此时还没有行情价）。"""

    def on_candle(self, candle: Candle, ctx: Context) -> None:
        """每根 K 线调用一次，策略在此决策下单。"""
        raise NotImplementedError

    def on_stop(self, ctx: Context) -> None:
        """回放结束后的钩子。"""

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"{type(self).__name__}(symbol={self.symbol!r})"


class MomentumStrategy(Strategy):
    """均线 + 区间动量确认的趋势跟随策略（自研示例，现货只做多）。

    信号（只用截至当根 K 线的收盘价，无未来数据）::

        ma_fast = mean(closes[-fast:])
        ma_slow = mean(closes[-slow:])
        roc     = closes[-1] / closes[-slow-1] - 1        # 慢周期区间收益率

        开多：ma_fast >= ma_slow × (1 + entry_band) 且 roc >= 0
        平多：ma_fast <= ma_slow × (1 - entry_band) 或 roc <= -exit_momentum
        其余：维持当前仓位（滞回带，避免频繁翻转与手续费损耗）

    参数
    ----
    fast/slow:        快慢均线窗口（根数）。
    entry_band:       开仓所需的均线偏离带（如 0.002 = 0.2%）。
    exit_momentum:    动量转弱的平仓阈值（如 0.01 = -1%）。
    exposure:         目标仓位占净值比例上限（1.0 = 满仓，不加杠杆）。
    vol_target:       目标年化波动率；给定后按波动率目标缩放仓位（波动大→仓位小）。
    vol_window:       估计波动率使用的收益率样本长度。
    periods_per_year: 年化期数（1h=8760，1d=365）。
    stop_pct/take_pct/trail_pct: 止损 / 止盈 / 移动止损（``None`` 表示不启用）。
    rebalance_band:   调仓阈值（占净值比例），差额小于它则不下单。
    min_trade_value:  单笔最小成交额（计价币），过滤碎单。
    """

    name = "momentum"

    def __init__(self, symbol: Optional[str] = None, fast: int = 12, slow: int = 48,
                 entry_band: float = 0.0, exit_momentum: float = 0.0,
                 exposure: float = 1.0, vol_target: Optional[float] = None,
                 vol_window: int = 48, periods_per_year: float = 8760.0,
                 stop_pct: Optional[float] = None, take_pct: Optional[float] = None,
                 trail_pct: Optional[float] = None, rebalance_band: float = 0.02,
                 min_trade_value: float = 1.0):
        super().__init__(symbol)
        if fast <= 0 or slow <= fast:
            raise ValueError("需要 0 < fast < slow")
        self.fast = int(fast)
        self.slow = int(slow)
        self.entry_band = float(entry_band)
        self.exit_momentum = float(exit_momentum)
        self.exposure = float(exposure)
        self.vol_target = None if vol_target is None else float(vol_target)
        self.vol_window = int(vol_window)
        self.periods_per_year = float(periods_per_year)
        self.stop_pct = stop_pct
        self.take_pct = take_pct
        self.trail_pct = trail_pct
        self.rebalance_band = float(rebalance_band)
        self.min_trade_value = float(min_trade_value)
        self.state: str = "flat"              # flat / long
        self.exits: List[Dict[str, object]] = []
        self.orders: List[Order] = []

    # ------------------------------------------------------------- 内部计算
    def target_exposure(self, ctx: Context, symbol: str) -> float:
        """目标仓位比例：``exposure`` 与波动率目标缩放后的较小值。"""
        exposure = max(float(self.exposure), 0.0)
        if self.vol_target is None:
            return exposure
        equity = ctx.equity()
        if equity <= 0:
            return 0.0
        rets = ctx.returns(self.vol_window, symbol)
        notional = risk.volatility_target_notional(equity, rets, self.vol_target,
                                                   self.periods_per_year,
                                                   max_leverage=exposure)
        return float(min(exposure, notional / equity))

    def signal(self, ctx: Context, symbol: str) -> Optional[str]:
        """当前信号：``"long"`` / ``"exit"`` / ``None``（数据不足或维持现状）。"""
        closes = ctx.closes(self.slow + 1, symbol)
        if closes.size < self.slow:
            return None
        ma_fast = float(np.mean(closes[-self.fast:]))
        ma_slow = float(np.mean(closes[-self.slow:]))
        base = closes[-self.slow - 1] if closes.size > self.slow else closes[0]
        roc = float(closes[-1] / base - 1.0) if base > 0 else 0.0
        if ma_fast <= ma_slow * (1.0 - self.entry_band) or roc <= -self.exit_momentum:
            return "exit"
        if ma_fast >= ma_slow * (1.0 + self.entry_band) and roc >= 0.0:
            return "long"
        return None

    # ---------------------------------------------------------------- 主逻辑
    def on_candle(self, candle: Candle, ctx: Context) -> None:
        symbol = self.resolve_symbol(ctx)
        price = float(candle.close)
        position = ctx.position(symbol)

        # 1) 风控优先：触发止损/止盈/移动止损则立即平仓
        if position.quantity > 0 and (self.stop_pct or self.take_pct or self.trail_pct):
            reason = risk.check_exits(position.average_price, price, self.stop_pct,
                                      self.take_pct, position.highest_price, self.trail_pct)
            if reason is not None:
                order = ctx.flatten(symbol, client_id=f"{self.name}-exit")
                if order is not None:
                    self.orders.append(order)
                self.state = "flat"
                self.exits.append({"index": ctx.index, "ts": candle.ts, "price": price,
                                   "reason": reason})
                return

        # 2) 趋势信号
        sig = self.signal(ctx, symbol)
        if sig is None:
            return
        target = self.target_exposure(ctx, symbol) if sig == "long" else 0.0
        self.state = "long" if sig == "long" else "flat"

        # 3) 调仓（小于阈值不动，避免手续费磨损）
        equity = ctx.equity()
        current_value = position.quantity * price
        target_value = target * equity
        threshold = max(self.min_trade_value, self.rebalance_band * equity)
        if abs(target_value - current_value) < threshold:
            return
        order = ctx.order_target_percent(target, symbol, client_id=f"{self.name}-{sig}")
        if order is not None:
            self.orders.append(order)


@dataclass
class _GridLine:
    """网格中的一档委托（内部记账用）。

    ``level`` 为档位标签：初始网格用 ``±1..±levels``（负号表示买档），
    成交后补挂的档位用 ``"R1"/"R2"...``。
    """

    level: Union[int, str]
    price: float
    side: Side
    quantity: float
    order_id: Optional[str] = None
    client_id: str = ""
    status: OrderStatus = OrderStatus.NEW


class GridStrategy(Strategy):
    """等差网格策略（自研示例）：围绕中心价上下按固定步长挂限价单，成交后自动补挂。

    规则
    ----
    - 步长 ``step = center × spacing``（``spacing=0.02`` 即 2%），也可用 ``step`` 直接指定绝对步长。
    - 买入档 ``center - k·step``（k=1..levels），卖出档 ``center + k·step``，价格**等距**。
    - 每档数量固定为 ``qty_per_grid``（基础币）。
    - 某档买单成交后，在成交价上方 ``step`` 处补挂卖单；卖单成交后在下方 ``step`` 处补挂买单，
      从而在震荡行情中反复「低买高卖」赚取网格价差。
    - ``build_inventory=True`` 时，建仓阶段先市价买入卖出档所需的底仓
      （否则卖单会因现货余额不足被拒）。

    参数
    ----
    levels:        中心价单侧的档位数。
    spacing:       相对步长（占中心价比例）。
    step:          绝对步长（给定则覆盖 spacing）。
    qty_per_grid:  每档委托数量（基础币）。
    center:        网格中心价；``None`` 表示用首根 K 线收盘价。
    side:          ``"both"`` / ``"buy"`` / ``"sell"``，只挂某一侧。
    rearm:         成交后是否自动补挂反向单。
    build_inventory: 是否为卖出档预先建立底仓。
    """

    name = "grid"

    def __init__(self, symbol: Optional[str] = None, levels: int = 4, spacing: float = 0.02,
                 qty_per_grid: float = 0.01, center: Optional[float] = None,
                 step: Optional[float] = None, side: str = "both", rearm: bool = True,
                 build_inventory: bool = True):
        super().__init__(symbol)
        if levels <= 0:
            raise ValueError("levels 必须为正整数")
        if qty_per_grid <= 0:
            raise ValueError("qty_per_grid 必须为正数")
        if step is None and spacing <= 0:
            raise ValueError("spacing 或 step 至少给一个正数")
        self.levels = int(levels)
        self.spacing = float(spacing)
        self.qty_per_grid = float(qty_per_grid)
        self.center = None if center is None else float(center)
        self.step_override = None if step is None else float(step)
        self.side = str(side).lower()
        self.rearm = bool(rearm)
        self.build_inventory = bool(build_inventory)

        self.step: float = 0.0
        self.lines: List[_GridLine] = []           # 当前挂着的档位
        self.closed_lines: List[_GridLine] = []    # 已成交/被拒的档位（审计用）
        self._line_by_order: Dict[str, _GridLine] = {}
        self._rearm_queue: List[_GridLine] = []
        self._rearm_seq: int = 0
        self.initialized: bool = False
        self.inventory_order: Optional[Order] = None
        self.filled_buys: int = 0
        self.filled_sells: int = 0

    @property
    def grid_cycles(self) -> int:
        """完成的网格轮次（卖档成交次数，即「低买高卖」落袋的次数）。"""
        return self.filled_sells

    # ------------------------------------------------------------- 网格构造
    def _plan(self, center: float) -> List[_GridLine]:
        """生成初始网格档位：买档在中心价下方、卖档在上方，价格**等差**。"""
        self.step = (self.step_override if self.step_override is not None
                     else center * self.spacing)
        lines: List[_GridLine] = []
        if self.side in ("both", "buy"):
            for level in range(1, self.levels + 1):
                price = center - level * self.step
                if price > 0:
                    lines.append(_GridLine(level=-level, price=price, side=Side.BUY,
                                           quantity=self.qty_per_grid))
        if self.side in ("both", "sell"):
            for level in range(1, self.levels + 1):
                lines.append(_GridLine(level=level, price=center + level * self.step,
                                       side=Side.SELL, quantity=self.qty_per_grid))
        lines.sort(key=lambda line: line.price)
        return lines

    def _place(self, ctx: Context, line: _GridLine, symbol: str) -> Optional[Order]:
        """挂出一档网格；被拒（余额不足等）则记入 closed_lines 并返回该委托。"""
        line.client_id = f"grid-L{line.level}-i{ctx.index}"
        order = (ctx.limit_buy(line.price, line.quantity, symbol, line.client_id)
                 if line.side is Side.BUY
                 else ctx.limit_sell(line.price, line.quantity, symbol, line.client_id))
        line.order_id = order.order_id
        if order.status is OrderStatus.REJECTED:
            line.status = OrderStatus.REJECTED
            self.closed_lines.append(line)
            return order
        self._line_by_order[order.order_id] = line
        if order.status is OrderStatus.FILLED:      # 挂出即成交（价格已穿越委托价）
            self._mark_filled(line, order.average_fill_price)
        else:
            line.status = OrderStatus.NEW
            self.lines.append(line)
        return order

    def _mark_filled(self, line: _GridLine, price: float) -> None:
        """登记一档成交，并按需排队补挂反向档位。"""
        line.status = OrderStatus.FILLED
        if line in self.lines:
            self.lines.remove(line)
        self.closed_lines.append(line)
        if line.order_id is not None:
            self._line_by_order.pop(line.order_id, None)
        if line.side is Side.BUY:
            self.filled_buys += 1
            new_side, new_price = Side.SELL, price + self.step
        else:
            self.filled_sells += 1
            new_side, new_price = Side.BUY, price - self.step
        if self.rearm and new_price > 0 and self.step > 0:
            self._rearm_seq += 1
            self._rearm_queue.append(_GridLine(level=f"R{self._rearm_seq}", price=new_price,
                                               side=new_side, quantity=self.qty_per_grid))

    def _drain_rearm(self, ctx: Context, symbol: str, max_rounds: int = 16) -> None:
        """把补挂队列清空（补挂本身可能立即成交并再次触发补挂，故循环处理）。"""
        rounds = 0
        while self._rearm_queue and rounds < max_rounds:
            batch, self._rearm_queue = self._rearm_queue, []
            for line in batch:
                self._place(ctx, line, symbol)
            rounds += 1

    def _build_inventory(self, ctx: Context, symbol: str, plan: List[_GridLine]) -> None:
        """为卖出档预先建立底仓（现货网格的必要步骤，否则卖单会因余额不足被拒）。"""
        need = sum(line.quantity for line in plan if line.side is Side.SELL)
        shortfall = need - ctx.holding(symbol)
        if shortfall > 0:
            self.inventory_order = ctx.buy(quantity=shortfall, symbol=symbol,
                                           client_id="grid-inventory")

    def _consume_trades(self, ctx: Context, symbol: str) -> None:
        """把本根 K 线撮合出的成交映射回网格档位。"""
        for trade in ctx.new_trades:
            line = self._line_by_order.get(trade.order_id)
            if line is None or trade.symbol != symbol:
                continue
            self._mark_filled(line, trade.price)

    # ---------------------------------------------------------------- 主逻辑
    def on_candle(self, candle: Candle, ctx: Context) -> None:
        symbol = self.resolve_symbol(ctx)
        if not self.initialized:
            self.center = self.center if self.center is not None else float(candle.close)
            plan = self._plan(self.center)
            if self.build_inventory and self.side in ("both", "sell"):
                self._build_inventory(ctx, symbol, plan)   # 先建底仓，再挂卖档
            for line in plan:
                self._place(ctx, line, symbol)
            self._drain_rearm(ctx, symbol)
            self.initialized = True
            return
        self._consume_trades(ctx, symbol)
        self._drain_rearm(ctx, symbol)

    # ---------------------------------------------------------------- 查看
    def grid_frame(self) -> pd.DataFrame:
        """全部网格档位表（含已成交/被拒），便于检查阶梯结构与状态。"""
        rows = [{"level": line.level, "price": line.price, "side": line.side.value,
                 "quantity": line.quantity, "order_id": line.order_id,
                 "status": line.status.value}
                for line in self.lines + self.closed_lines]
        frame = pd.DataFrame(rows, columns=["level", "price", "side", "quantity",
                                            "order_id", "status"])
        if frame.empty:
            return frame
        return frame.sort_values("price").reset_index(drop=True)

    def buy_prices(self) -> List[float]:
        """当前仍挂着的买入档价格（升序）。"""
        return sorted(line.price for line in self.lines if line.side is Side.BUY)

    def sell_prices(self) -> List[float]:
        """当前仍挂着的卖出档价格（升序）。"""
        return sorted(line.price for line in self.lines if line.side is Side.SELL)

    def on_stop(self, ctx: Context) -> None:
        """回放结束保留挂单，便于观察网格状态（如需清盘可调用 ``ctx.cancel_all()``）。"""


class DcaStrategy(Strategy):
    """定期定额买入策略（Dollar-Cost Averaging，自研示例）。

    每隔 ``every_n`` 根 K 线，用 ``quote_per_buy`` 的计价币市价买入，
    不预测价格、只靠纪律摊薄成本；可选 ``stop_pct`` 风控（触发后停止定投并平仓）。

    参数
    ----
    quote_per_buy: 每次投入的计价币金额（如 100 USDT）。
    every_n:       定投周期（K 线根数），如 1h K 线下 ``24`` = 每天一次。
    start_index:   从第几根 K 线开始定投（0 起）。
    max_buys:      最多买入次数；``None`` 表示不限。
    stop_pct:      组合级止损（相对持仓均价的回撤比例），触发后平仓并停止。
    keep_reserve:  预留的计价币余额（不参与定投），避免手续费导致余额不足。
    """

    name = "dca"

    def __init__(self, symbol: Optional[str] = None, quote_per_buy: float = 100.0,
                 every_n: int = 24, start_index: int = 0, max_buys: Optional[int] = None,
                 stop_pct: Optional[float] = None, keep_reserve: float = 0.0):
        super().__init__(symbol)
        if quote_per_buy <= 0:
            raise ValueError("quote_per_buy 必须为正数")
        if every_n <= 0:
            raise ValueError("every_n 必须为正整数")
        self.quote_per_buy = float(quote_per_buy)
        self.every_n = int(every_n)
        self.start_index = int(start_index)
        self.max_buys = max_buys
        self.stop_pct = stop_pct
        self.keep_reserve = float(keep_reserve)
        self.buys: int = 0
        self.invested: float = 0.0
        self.stopped: bool = False
        self.schedule: List[Dict[str, float]] = []

    @property
    def average_cost(self) -> float:
        """定投均价 = 累计投入（含手续费）/ 累计买入数量。"""
        quantity = sum(item["quantity"] for item in self.schedule)
        return self.invested / quantity if quantity > 0 else 0.0

    @property
    def average_fill_price(self) -> float:
        """成交均价（不含手续费），按数量加权。"""
        quantity = sum(item["quantity"] for item in self.schedule)
        if quantity <= 0:
            return 0.0
        return sum(item["price"] * item["quantity"] for item in self.schedule) / quantity

    def due(self, index: int) -> bool:
        """第 ``index`` 根 K 线是否为定投日。"""
        if self.stopped or index < self.start_index:
            return False
        if self.max_buys is not None and self.buys >= self.max_buys:
            return False
        return (index - self.start_index) % self.every_n == 0

    def on_candle(self, candle: Candle, ctx: Context) -> None:
        symbol = self.resolve_symbol(ctx)
        price = float(candle.close)

        if self.stop_pct is not None and not self.stopped:
            position = ctx.position(symbol)
            if position.quantity > 0 and risk.stop_loss_triggered(
                    position.average_price, price, self.stop_pct):
                ctx.flatten(symbol, client_id="dca-stop")
                self.stopped = True
                return

        if not self.due(ctx.index):
            return

        quote_currency = ctx.quote_currency(symbol)
        available = ctx.free(quote_currency) - self.keep_reserve
        amount = min(self.quote_per_buy, available)
        if amount <= 0:
            self.stopped = True                 # 资金耗尽，停止定投
            return
        order = ctx.buy(quote=amount, symbol=symbol, client_id=f"dca-{self.buys}")
        if order is None or order.status is OrderStatus.REJECTED:
            self.stopped = True
            return
        filled = order.filled_quantity
        self.buys += 1
        self.invested += filled * order.average_fill_price + order.fee_paid
        self.schedule.append({"index": float(ctx.index), "ts": float(candle.ts),
                              "price": float(order.average_fill_price), "quantity": filled,
                              "quote": float(amount), "fee": float(order.fee_paid)})

    def schedule_frame(self) -> pd.DataFrame:
        """定投明细表。"""
        return pd.DataFrame(self.schedule, columns=["index", "ts", "price", "quantity",
                                                    "quote", "fee"])
