"""纸面交易引擎：把「行情回放 + 纸面撮合 + 策略」串成一条可复现的流水线。

主循环（每根 K 线一次，严格不使用未来数据）::

    for i, candle in enumerate(replayer):
        1) exchange.on_candle(candle)   # 更新最新价，并撮合**此前**挂出的限价单
        2) strategy.on_candle(candle, ctx)   # 策略决策：市价单当根收盘价成交，
                                        #    限价单挂出后从下一根 K 线开始撮合
        3) 记录净值快照（现金 + 持仓市值 - 已发生手续费）

因此：限价单永远不会在「挂出的那一根」按当根最低价成交，杜绝了常见的回测未来函数。

引擎同时负责会计核对：期末净值变动 = 毛盈亏（现金流 + 持仓市值变化） - 手续费，
见 :meth:`PaperTradingResult.check_accounting`。

本模块自带一组轻量绩效函数（总收益/夏普/最大回撤等），**不依赖任何外部项目**，
且它们直接作用于「净值曲线」（而非收益率序列），与加密 24/7 的年化口径一致。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .data import Replayer
from .exchange import DEFAULT_QUOTE, Exchange, PaperExchange
from .strategy import Context, Strategy
from .types import Candle, Order, Trade

CandlesLike = Union[Replayer, Sequence[Candle], Iterable[Candle]]


# --------------------------------------------------------------------- 绩效
def equity_returns(equity: pd.Series) -> pd.Series:
    """由净值曲线求每期收益率（首期为 0）。"""
    if equity is None or len(equity) == 0:
        return pd.Series(dtype="float64")
    return equity.astype("float64").pct_change().fillna(0.0)


def total_return(equity: Sequence[float]) -> float:
    """累计收益率 = 期末净值 / 期初净值 - 1。"""
    values = np.asarray(list(equity), dtype="float64")
    if values.size < 2 or values[0] == 0:
        return 0.0
    return float(values[-1] / values[0] - 1.0)


def cagr(equity: Sequence[float], periods_per_year: float = 365.0) -> float:
    """年化复合增长率（按 24/7 的年化期数折算）。"""
    values = np.asarray(list(equity), dtype="float64")
    if values.size < 2 or values[0] <= 0:
        return 0.0
    growth = float(values[-1] / values[0])
    if growth <= 0:
        return -1.0
    periods = values.size - 1
    years = periods / float(periods_per_year)
    if years <= 0:
        return 0.0
    return float(growth ** (1.0 / years) - 1.0)


def annualized_volatility(returns: Sequence[float], periods_per_year: float = 365.0) -> float:
    """年化波动率。"""
    arr = np.asarray(list(returns), dtype="float64")
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: Sequence[float], risk_free: float = 0.0,
                 periods_per_year: float = 365.0) -> float:
    """夏普比率；波动为 0 时约定返回 0（避免除零）。"""
    arr = np.asarray(list(returns), dtype="float64")
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return 0.0
    excess = arr - float(risk_free) / float(periods_per_year)
    sd = float(np.std(excess, ddof=1))
    if sd < 1e-12 or not np.isfinite(sd):
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns: Sequence[float], risk_free: float = 0.0,
                  periods_per_year: float = 365.0) -> float:
    """索提诺比率（只计下行波动）。"""
    arr = np.asarray(list(returns), dtype="float64")
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return 0.0
    excess = arr - float(risk_free) / float(periods_per_year)
    downside = excess[excess < 0]
    if downside.size == 0:
        return float("inf") if float(np.mean(excess)) > 0 else 0.0
    dvol = float(np.sqrt(np.sum(downside ** 2) / max(arr.size - 1, 1)))
    if dvol < 1e-12:
        return 0.0
    return float(np.mean(excess) / dvol * np.sqrt(periods_per_year))


def drawdown_series(equity: Sequence[float]) -> pd.Series:
    """净值回撤序列（<=0）。"""
    curve = pd.Series(np.asarray(list(equity), dtype="float64"))
    if curve.empty:
        return curve
    return curve / curve.cummax() - 1.0


def max_drawdown(equity: Sequence[float]) -> float:
    """最大回撤（正数表示回撤幅度，如 0.25 表示 -25%）。"""
    dd = drawdown_series(equity)
    if dd.empty:
        return 0.0
    return float(-dd.min())


def performance_summary(equity: Sequence[float], periods_per_year: float = 365.0,
                        risk_free: float = 0.0) -> Dict[str, float]:
    """常用绩效指标一览（输入为净值曲线）。"""
    curve = np.asarray(list(equity), dtype="float64")
    returns = equity_returns(pd.Series(curve))
    return {
        "periods": float(curve.size),
        "initial_equity": float(curve[0]) if curve.size else 0.0,
        "final_equity": float(curve[-1]) if curve.size else 0.0,
        "total_return": total_return(curve),
        "cagr": cagr(curve, periods_per_year),
        "volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, risk_free, periods_per_year),
        "sortino": sortino_ratio(returns, risk_free, periods_per_year),
        "max_drawdown": max_drawdown(curve),
    }


def infer_periods_per_year(candles: Sequence[Candle], fallback: float = 365.0) -> float:
    """由相邻 K 线时间戳推断年化期数（24/7 口径）。"""
    stamps = sorted({int(c.ts) for c in candles})
    if len(stamps) < 2:
        return fallback
    diffs = np.diff(np.asarray(stamps, dtype="int64"))
    step_seconds = float(np.median(diffs)) / 1000.0
    if step_seconds <= 0:
        return fallback
    return (365.0 * 24 * 3600) / step_seconds


# ------------------------------------------------------------------ 结果对象
@dataclass
class PaperTradingResult:
    """纸面交易结果。

    参数
    ----
    equity:    净值曲线（index 为 K 线时间，单位为计价币）。
    history:   每根 K 线的明细快照（净值/现金/持仓/手续费/已实现与浮动盈亏）。
    trades:    全部成交回报；orders: 全部委托（含被拒与已撤）。
    metrics:   绩效与会计指标字典。
    gross_pnl: 毛盈亏 = 成交现金流（不含手续费） + 持仓市值变化。
    total_fees: 累计手续费。
    net_pnl:   净盈亏 = 期末净值 - 期初净值 = ``gross_pnl - total_fees``。
    """

    initial_equity: float
    final_equity: float
    equity: pd.Series
    history: pd.DataFrame
    trades: List[Trade]
    orders: List[Order]
    metrics: Dict[str, float]
    periods_per_year: float = 365.0
    quote: str = DEFAULT_QUOTE
    symbol: str = ""
    gross_pnl: float = 0.0
    total_fees: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    initial_holdings_value: float = 0.0
    final_holdings_value: float = 0.0
    final_position: float = 0.0

    @property
    def net_pnl(self) -> float:
        """净盈亏（期末 - 期初）。"""
        return self.final_equity - self.initial_equity

    @property
    def total_return(self) -> float:
        """累计收益率（以期初真实权益为基准）。"""
        if self.initial_equity == 0:
            return 0.0
        return self.final_equity / self.initial_equity - 1.0

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    def trades_frame(self) -> pd.DataFrame:
        """成交回报表。"""
        if not self.trades:
            return pd.DataFrame(columns=["trade_id", "order_id", "symbol", "side", "price",
                                         "quantity", "fee", "gross_value", "cash_flow",
                                         "ts", "is_maker"])
        return pd.DataFrame([t.to_dict() for t in self.trades])

    def returns(self) -> pd.Series:
        """每期收益率（首期以期初真实权益为基准，长度与净值曲线一致）。"""
        values = np.concatenate(([float(self.initial_equity)],
                                 self.equity.to_numpy(dtype="float64")))
        if values.size < 2:
            return pd.Series(dtype="float64", name="returns")
        rets = np.diff(values) / values[:-1]
        return pd.Series(rets, index=self.equity.index, name="returns")

    def orders_frame(self) -> pd.DataFrame:
        """委托表。"""
        if not self.orders:
            return pd.DataFrame(columns=["order_id", "symbol", "side", "order_type", "price",
                                         "quantity", "filled_quantity", "status", "fee_paid"])
        return pd.DataFrame([o.to_dict() for o in self.orders])

    def summary_frame(self) -> pd.DataFrame:
        """绩效 + 会计指标的两列表，便于打印。"""
        data = dict(self.metrics)
        data.update({
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "total_fees": self.total_fees,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "trade_count": float(self.trade_count),
        })
        return pd.DataFrame({"metric": list(data.keys()),
                             "value": [float(v) for v in data.values()]})

    def check_accounting(self, tol: float = 1e-6) -> bool:
        """会计自洽校验：净盈亏 == 毛盈亏 - 手续费（相对误差 < ``tol``）。"""
        scale = max(abs(self.gross_pnl), abs(self.total_fees), abs(self.net_pnl), 1.0)
        return abs(self.net_pnl - (self.gross_pnl - self.total_fees)) <= tol * scale


# -------------------------------------------------------------------- 引擎
class PaperTradingEngine:
    """纸面交易引擎：逐根 K 线推进「撮合 → 策略 → 记账」。

    参数
    ----
    candles:  K 线序列或 :class:`~kairos_crypto.data.Replayer`。
    strategy: 策略实例（:class:`~kairos_crypto.strategy.Strategy` 子类）。
    exchange: 交易所实例；``None`` 时自动创建 :class:`PaperExchange`。
    quote:    计价币（自建交易所时使用）。
    initial_quote: 初始计价币资金。
    fee_rate / slippage_bps: 自建交易所的手续费率与市价单滑点。
    periods_per_year: 年化期数；``None`` 时按 K 线间隔自动推断（24/7）。

    用法::

        engine = PaperTradingEngine(SyntheticCandles(n=1000, seed=7), MomentumStrategy(),
                                    initial_quote=10_000, fee_rate=0.001)
        result = engine.run()
        print(result.summary_frame())
    """

    def __init__(self, candles: CandlesLike, strategy: Strategy,
                 exchange: Optional[Exchange] = None, *,
                 quote: str = DEFAULT_QUOTE,
                 initial_quote: float = 10_000.0,
                 fee_rate: float = 0.001,
                 slippage_bps: float = 0.0,
                 periods_per_year: Optional[float] = None):
        if strategy is None:
            raise ValueError("必须提供 strategy")
        self.replayer = candles if isinstance(candles, Replayer) else Replayer(candles)
        if len(self.replayer) == 0:
            raise ValueError("行情为空，无法运行纸面交易")
        self.strategy = strategy
        self.exchange: Exchange = exchange if exchange is not None else PaperExchange(
            quote=quote, initial_quote=initial_quote, fee_rate=fee_rate,
            slippage_bps=slippage_bps)
        self.quote = str(getattr(self.exchange, "quote", quote) or quote).upper()
        self.periods_per_year = (float(periods_per_year) if periods_per_year
                                 else infer_periods_per_year(self.replayer.candles))
        self.context = Context(exchange=self.exchange, replayer=self.replayer,
                               quote=self.quote)
        self.result: Optional[PaperTradingResult] = None
        self._rows: List[Dict[str, float]] = []
        self._initial_equity: float = 0.0
        self._initial_holdings: float = 0.0

    # ------------------------------------------------------------- 内部工具
    def _call(self, name: str, *args, default=None):
        """调用交易所的可选扩展方法（真实适配器可能没有）。"""
        fn = getattr(self.exchange, name, None)
        if not callable(fn):
            return default
        try:
            return fn(*args)
        except NotImplementedError:
            return default

    def _equity(self) -> float:
        value = self._call("equity")
        return float(value) if value is not None else float(self.context.equity())

    def _holdings(self) -> float:
        value = self._call("holdings_value")
        if value is None:
            value = self._equity() - float(self.context.total(self.quote))
        return float(value)

    def _all_trades(self) -> List[Trade]:
        trades = getattr(self.exchange, "trades", None)
        return list(trades) if trades else []

    def _prime_prices(self) -> None:
        """运行前用各交易对**首根 K 线的收盘价**给交易所打底价格。

        目的：若账户初始就持有币（例如预置了 0.1 BTC），期初权益与期初持仓市值
        才能被一致地估出来，会计恒等式（净盈亏 = 毛盈亏 - 手续费）才成立。
        真实适配器若没有 ``mark()``，此处自动跳过。
        """
        marker = getattr(self.exchange, "mark", None)
        if not callable(marker):
            return
        has_price = getattr(self.exchange, "has_price", None)
        first_seen: Dict[str, Candle] = {}
        for candle in self.replayer.candles:
            first_seen.setdefault(candle.symbol, candle)
        for symbol, candle in first_seen.items():
            if callable(has_price) and has_price(symbol):
                continue
            marker(symbol, float(candle.close))

    def _snapshot(self, candle: Candle, index: int) -> Dict[str, object]:
        trades = self._all_trades()
        realized = self._call("realized_pnl", default=0.0)
        unrealized = self._call("unrealized_pnl", default=0.0)
        return {
            "index": index,
            "ts": int(candle.ts),
            "time": candle.datetime,
            "symbol": candle.symbol,
            "price": float(candle.close),
            "equity": self._equity(),
            "cash": float(self.context.total(self.quote)),
            "position": float(self.context.holding(candle.symbol)),
            "fee_paid": float(sum(t.fee for t in trades)),
            "realized_pnl": float(realized or 0.0),
            "unrealized_pnl": float(unrealized or 0.0),
            "open_orders": float(len(self.exchange.fetch_open_orders(candle.symbol))),
        }

    # --------------------------------------------------------------- 主循环
    def run(self) -> PaperTradingResult:
        """跑完全部 K 线，返回 :class:`PaperTradingResult`。"""
        self.replayer.reset()
        self._rows = []
        self._prime_prices()
        self._initial_equity = self._equity()
        self._initial_holdings = self._holdings()
        match_hook = getattr(self.exchange, "on_candle", None)
        ctx = self.context

        self.strategy.on_start(ctx)
        for index, candle in enumerate(self.replayer):
            ctx.index = index
            ctx.symbol = candle.symbol
            ctx.candle = candle
            if callable(match_hook):
                ctx.new_trades = list(match_hook(candle))
            else:                                   # 真实适配器：自行维护行情与撮合
                ctx.new_trades = []
            self.strategy.on_candle(candle, ctx)
            self._rows.append(self._snapshot(candle, index))
        self.strategy.on_stop(ctx)
        ctx.candle = None
        ctx.new_trades = []

        return self._build_result()

    def _build_result(self) -> PaperTradingResult:
        history = pd.DataFrame(self._rows)
        equity = pd.Series(history["equity"].to_numpy(dtype="float64"),
                           index=pd.DatetimeIndex(history["time"]), name="equity")
        trades = self._all_trades()
        orders = list(self.exchange.fetch_orders())
        final_equity = self._equity()
        final_holdings = self._holdings()

        total_fees = float(sum(t.fee for t in trades))
        # 成交现金流（不含手续费）：买入为负、卖出为正
        trade_cash_flow = float(sum(-t.side.sign * t.price * t.quantity for t in trades))
        gross_pnl = trade_cash_flow + (final_holdings - self._initial_holdings)
        realized = float(self._call("realized_pnl", default=0.0) or 0.0)
        unrealized = float(self._call("unrealized_pnl", default=0.0) or 0.0)

        closes = history["price"].to_numpy(dtype="float64")
        benchmark = float(closes[-1] / closes[0] - 1.0) if closes.size and closes[0] else 0.0
        # 绩效以期初真实权益为基准（把 initial_equity 作为曲线的第 0 点）
        base_curve = np.concatenate(([self._initial_equity],
                                     equity.to_numpy(dtype="float64")))
        metrics = performance_summary(base_curve, self.periods_per_year)
        metrics["periods"] = float(len(equity))
        metrics.update({
            "net_pnl": final_equity - self._initial_equity,
            "gross_pnl": gross_pnl,
            "total_fees": total_fees,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "trade_cash_flow": trade_cash_flow,
            "trade_count": float(len(trades)),
            "order_count": float(len(orders)),
            "final_position": float(history["position"].iloc[-1]) if len(history) else 0.0,
            "final_holdings_value": final_holdings,
            "benchmark_return": benchmark,
            "periods_per_year": float(self.periods_per_year),
        })

        symbol = self.replayer.symbols[0] if self.replayer.symbols else ""
        self.result = PaperTradingResult(
            initial_equity=self._initial_equity, final_equity=final_equity,
            equity=equity, history=history, trades=trades, orders=orders,
            metrics=metrics, periods_per_year=self.periods_per_year, quote=self.quote,
            symbol=symbol, gross_pnl=gross_pnl, total_fees=total_fees,
            realized_pnl=realized, unrealized_pnl=unrealized,
            initial_holdings_value=self._initial_holdings,
            final_holdings_value=final_holdings,
            final_position=float(history["position"].iloc[-1]) if len(history) else 0.0,
        )
        return self.result

    # --------------------------------------------------------------- 便捷访问
    def metrics(self) -> Dict[str, float]:
        """绩效与会计指标（需先 :meth:`run`）。"""
        if self.result is None:
            raise RuntimeError("请先调用 run()")
        return dict(self.result.metrics)

    def equity_curve(self) -> pd.Series:
        """净值曲线（需先 :meth:`run`）。"""
        if self.result is None:
            raise RuntimeError("请先调用 run()")
        return self.result.equity

    def trades_frame(self) -> pd.DataFrame:
        """成交回报表（需先 :meth:`run`）。"""
        if self.result is None:
            raise RuntimeError("请先调用 run()")
        return self.result.trades_frame()

    def returns(self) -> pd.Series:
        """每期收益率序列（需先 :meth:`run`）。"""
        if self.result is None:
            raise RuntimeError("请先调用 run()")
        return self.result.returns()

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return (f"PaperTradingEngine(strategy={type(self.strategy).__name__}, "
                f"candles={len(self.replayer)}, quote={self.quote!r}, "
                f"ppy={self.periods_per_year:.1f})")
