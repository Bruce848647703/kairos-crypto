"""纸面交易引擎测试：净值序列、会计自洽、防未来函数、绩效函数。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos_crypto import (
    DcaStrategy,
    MomentumStrategy,
    PaperExchange,
    PaperTradingEngine,
    Strategy,
    SyntheticCandles,
    annualized_volatility,
    cagr,
    drawdown_series,
    equity_returns,
    infer_periods_per_year,
    max_drawdown,
    performance_summary,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)
from helpers import SYMBOL, flat_candles, make_candles, make_exchange, ohlc_candles

CANDLES = SyntheticCandles(n=400, seed=3, interval="1h").candles


class BuyAndHold(Strategy):
    """第 1 根 K 线全仓买入并一直持有。"""

    def __init__(self, symbol=SYMBOL, percent=1.0):
        super().__init__(symbol)
        self.percent = percent

    def on_candle(self, candle, ctx):
        if ctx.index == 1:
            ctx.order_target_percent(self.percent, self.symbol)


class BuyOnce(Strategy):
    """在指定 K 线序号市价买入固定数量（用于验证成交价与时点）。"""

    def __init__(self, index=2, quantity=1.0):
        super().__init__(SYMBOL)
        self.index = index
        self.quantity = quantity

    def on_candle(self, candle, ctx):
        if ctx.index == self.index:
            ctx.buy(quantity=self.quantity)


class LimitBelowBar(Strategy):
    """在第 0 根挂一个低于该根最低价的限价买单（用于验证不会当根成交）。"""

    def __init__(self):
        super().__init__(SYMBOL)
        self.trades_right_after_order = None
        self.open_orders_after_placement = None

    def on_candle(self, candle, ctx):
        if ctx.index == 0:
            ctx.limit_buy(candle.low * 0.99, 0.01)
            self.trades_right_after_order = len(ctx.trades())
            self.open_orders_after_placement = len(ctx.open_orders())


class Snapshot(Strategy):
    """记录每根 K 线上「策略视角」的净值/持仓（策略在快照之前被调用）。"""

    def __init__(self, buy_index=0, quantity=1.0):
        super().__init__(SYMBOL)
        self.buy_index = buy_index
        self.quantity = quantity
        self.seen = []

    def on_candle(self, candle, ctx):
        self.seen.append((ctx.index, ctx.equity(), ctx.holding()))
        if ctx.index == self.buy_index:
            ctx.buy(quantity=self.quantity)


# ------------------------------------------------------------------ 净值序列
def test_equity_length_equals_candle_count():
    result = PaperTradingEngine(CANDLES, BuyAndHold(), make_exchange()).run()
    assert len(result.equity) == len(CANDLES)
    assert len(result.history) == len(CANDLES)
    assert list(result.equity.index) == [c.datetime for c in CANDLES]
    assert result.equity.notna().all()
    assert (result.equity > 0).all()


def test_equity_length_matches_for_short_runs_and_multiple_strategies():
    candles = flat_candles([100.0] * 7)
    for strategy in (BuyAndHold(), BuyOnce(index=3, quantity=0.1),
                     DcaStrategy(quote_per_buy=10.0, every_n=2)):
        result = PaperTradingEngine(candles, strategy, make_exchange()).run()
        assert len(result.equity) == 7
        assert result.metrics["periods"] == 7.0


def test_buy_and_hold_equity_tracks_price():
    closes = np.linspace(100.0, 200.0, 60)
    candles = make_candles(closes, wick=0.001)
    ex = make_exchange(quote=10_000.0, fee_rate=0.0)
    result = PaperTradingEngine(candles, BuyAndHold(), ex).run()
    holdings = result.history["position"] * result.history["price"]
    np.testing.assert_allclose(result.equity.to_numpy(),
                               result.history["cash"].to_numpy() + holdings.to_numpy(),
                               rtol=1e-12)
    # 无手续费时，全仓买入的净值涨幅应等于价格涨幅
    assert result.metrics["total_return"] == pytest.approx(closes[-1] / closes[1] - 1.0,
                                                           rel=1e-9)
    assert result.final_position == pytest.approx(10_000.0 / closes[1])


# ------------------------------------------------------------------ 会计自洽
def test_accounting_identity_net_equals_gross_minus_fees():
    ex = make_exchange(quote=10_000.0, fee_rate=0.001)
    result = PaperTradingEngine(CANDLES, MomentumStrategy(fast=10, slow=30), ex).run()
    assert result.check_accounting()
    assert result.net_pnl == pytest.approx(result.gross_pnl - result.total_fees, abs=1e-6)
    assert result.final_equity == pytest.approx(ex.equity(), abs=1e-9)
    assert result.final_equity == pytest.approx(result.equity.iloc[-1], abs=1e-9)
    assert result.initial_equity == pytest.approx(10_000.0)
    # 毛盈亏 = 成交现金流（不含手续费） + 持仓市值变化
    flow = sum(-t.side.sign * t.price * t.quantity for t in result.trades)
    assert result.gross_pnl == pytest.approx(
        flow + result.final_holdings_value - result.initial_holdings_value, abs=1e-6)


def test_flat_at_end_net_pnl_equals_realized_minus_fees():
    closes = np.linspace(100.0, 150.0, 40)
    candles = make_candles(closes, wick=0.001)
    ex = make_exchange(quote=5_000.0, fee_rate=0.001)

    class RoundTrip(Strategy):
        def on_candle(self, candle, ctx):
            if ctx.index == 1:
                ctx.order_target_percent(1.0)
            elif ctx.index == len(candles) - 1:
                ctx.flatten()

    result = PaperTradingEngine(candles, RoundTrip(), ex).run()
    assert result.final_position == pytest.approx(0.0, abs=1e-12)
    assert result.final_holdings_value == pytest.approx(0.0, abs=1e-9)
    assert result.check_accounting()
    assert result.net_pnl == pytest.approx(result.realized_pnl - result.total_fees, abs=1e-6)
    assert result.unrealized_pnl == pytest.approx(0.0, abs=1e-9)
    assert result.net_pnl > 0                      # 上涨行情低买高卖应盈利


def test_accounting_holds_with_prefunded_base_balance():
    """账户初始就持有币时，期初持仓市值也要计入恒等式。"""
    candles = make_candles(np.linspace(100.0, 120.0, 30), wick=0.001)
    ex = PaperExchange(balances={"USDT": 1_000.0, "BTC": 2.0}, fee_rate=0.001)
    result = PaperTradingEngine(candles, BuyAndHold(percent=0.3), ex).run()
    assert result.initial_holdings_value == pytest.approx(2.0 * candles[0].close)
    assert result.initial_equity == pytest.approx(1_000.0 + 2.0 * candles[0].close)
    assert result.check_accounting()
    assert result.final_position > 2.0             # 又买入了一部分


def test_fees_are_accumulated_in_history():
    ex = make_exchange(quote=10_000.0, fee_rate=0.001)
    result = PaperTradingEngine(CANDLES, DcaStrategy(quote_per_buy=100.0, every_n=50), ex).run()
    fees = result.history["fee_paid"].to_numpy()
    assert np.all(np.diff(fees) >= -1e-12)         # 累计手续费单调不减
    assert fees[-1] == pytest.approx(result.total_fees)
    assert fees[-1] == pytest.approx(ex.fee_paid())
    assert result.total_fees > 0


# -------------------------------------------------------------- 撮合时点语义
def test_market_order_fills_at_current_candle_close():
    candles = make_candles([100.0, 101.0, 102.0, 103.0, 104.0])
    ex = make_exchange(quote=10_000.0, fee_rate=0.0)
    result = PaperTradingEngine(candles, BuyOnce(index=2, quantity=1.0), ex).run()
    assert result.trade_count == 1
    trade = result.trades[0]
    assert trade.price == pytest.approx(candles[2].close)
    assert trade.ts == candles[2].ts


def test_limit_order_placed_on_bar_never_fills_on_same_bar():
    """防未来函数：限价单挂出后，只能由**后续** K 线撮合。"""
    rows = [(100.0, 101.0, 99.0, 100.0),        # i=0，挂单价 98.01 已低于本根 low
            (100.0, 100.5, 97.0, 98.0),         # i=1，low 触及 → 成交
            (98.0, 99.0, 97.5, 98.5)]
    candles = ohlc_candles(rows)
    ex = make_exchange(quote=10_000.0, fee_rate=0.0)
    strat = LimitBelowBar()
    result = PaperTradingEngine(candles, strat, ex).run()
    assert strat.trades_right_after_order == 0        # 挂出当根没有成交
    assert strat.open_orders_after_placement == 1     # 挂单留存
    assert result.trade_count == 1
    assert result.trades[0].ts == candles[1].ts       # 在下一根成交
    assert result.trades[0].price == pytest.approx(candles[0].low * 0.99)


def test_snapshot_taken_after_strategy_trades():
    """快照在策略下单之后记录，因此当根手续费已计入净值。"""
    candles = flat_candles([100.0] * 4)
    ex = make_exchange(quote=1_000.0, fee_rate=0.001)
    strat = Snapshot(buy_index=0, quantity=1.0)
    result = PaperTradingEngine(candles, strat, ex).run()
    # 第 0 根买入 1 个币：花掉 100.1（含手续费），持仓市值 100 → 净值 999.9
    assert result.equity.iloc[0] == pytest.approx(1_000.0 - 0.1)
    assert result.history["fee_paid"].iloc[0] == pytest.approx(0.1)
    assert len(strat.seen) == 4
    # 策略看到的是下单前的净值与持仓，快照记录的是下单后的结果
    assert strat.seen[0][1] == pytest.approx(1_000.0)
    assert strat.seen[0][2] == pytest.approx(0.0)
    assert strat.seen[1][2] == pytest.approx(1.0)


# ------------------------------------------------------------------ 引擎配置
def test_periods_per_year_inferred_and_overridable():
    engine = PaperTradingEngine(CANDLES, BuyAndHold(), make_exchange())
    assert engine.periods_per_year == pytest.approx(8760.0)
    daily = SyntheticCandles(n=100, seed=5, interval="1d").candles
    engine_daily = PaperTradingEngine(daily, BuyAndHold(), make_exchange())
    assert engine_daily.periods_per_year == pytest.approx(365.0)
    custom = PaperTradingEngine(daily, BuyAndHold(), make_exchange(), periods_per_year=52.0)
    assert custom.periods_per_year == 52.0
    assert custom.run().metrics["periods_per_year"] == 52.0


def test_engine_creates_paper_exchange_by_default():
    engine = PaperTradingEngine(CANDLES, BuyAndHold(), quote="USDT", initial_quote=2_000.0,
                                fee_rate=0.002)
    assert isinstance(engine.exchange, PaperExchange)
    assert engine.exchange.fee_rate == 0.002
    result = engine.run()
    assert result.initial_equity == pytest.approx(2_000.0)
    assert result.quote == "USDT"
    assert result.symbol == SYMBOL


def test_engine_requires_strategy_and_candles():
    with pytest.raises(ValueError):
        PaperTradingEngine(CANDLES, None)
    with pytest.raises(ValueError):
        PaperTradingEngine([], BuyAndHold())


def test_accessors_require_run_first():
    engine = PaperTradingEngine(flat_candles([100.0] * 3), BuyAndHold(), make_exchange())
    with pytest.raises(RuntimeError):
        engine.metrics()
    with pytest.raises(RuntimeError):
        engine.equity_curve()
    with pytest.raises(RuntimeError):
        engine.trades_frame()
    with pytest.raises(RuntimeError):
        engine.returns()


def test_run_is_deterministic_and_repeatable():
    def build():
        return PaperTradingEngine(CANDLES, MomentumStrategy(fast=10, slow=30),
                                  make_exchange(quote=10_000.0, fee_rate=0.001)).run()

    first, second = build(), build()
    pd.testing.assert_series_equal(first.equity, second.equity)
    assert first.metrics == second.metrics
    assert [t.to_dict() for t in first.trades] == [t.to_dict() for t in second.trades]


def test_result_frames_and_summary():
    result = PaperTradingEngine(CANDLES, BuyAndHold(), make_exchange()).run()
    trades = result.trades_frame()
    assert {"trade_id", "side", "price", "quantity", "fee"} <= set(trades.columns)
    orders = result.orders_frame()
    assert len(orders) == result.metrics["order_count"]
    summary = result.summary_frame()
    assert list(summary.columns) == ["metric", "value"]
    for key in ("total_return", "sharpe", "max_drawdown", "net_pnl", "total_fees"):
        assert key in set(summary["metric"])


def test_empty_trades_frames_have_columns():
    candles = flat_candles([100.0] * 3)

    class Idle(Strategy):
        def on_candle(self, candle, ctx):
            return None

    result = PaperTradingEngine(candles, Idle(), make_exchange()).run()
    assert result.trade_count == 0
    assert not result.trades_frame().empty or result.trades_frame().columns.size > 0
    assert result.orders_frame().empty
    assert result.net_pnl == pytest.approx(0.0)
    assert result.check_accounting()


def test_engine_accepts_replayer_instance():
    replay = SyntheticCandles(n=50, seed=9).replayer()
    engine = PaperTradingEngine(replay, BuyAndHold(), make_exchange())
    assert engine.replayer is replay
    assert len(engine.run().equity) == 50


def test_metrics_contain_benchmark_and_position():
    result = PaperTradingEngine(CANDLES, BuyAndHold(), make_exchange()).run()
    closes = np.array([c.close for c in CANDLES])
    assert result.metrics["benchmark_return"] == pytest.approx(closes[-1] / closes[0] - 1.0)
    assert result.metrics["final_position"] == pytest.approx(result.final_position)
    assert result.metrics["trade_count"] == float(result.trade_count)


# ------------------------------------------------------------------ 绩效函数
def test_total_return_and_cagr():
    equity = pd.Series([100.0, 110.0, 121.0])
    assert total_return(equity) == pytest.approx(0.21)
    # 3 个点 = 2 期收益，每期 +10%；每年 1 期 → 2 年 → CAGR = 10%
    assert cagr(equity, periods_per_year=1.0) == pytest.approx(0.10, rel=1e-9)
    # 每年 2 期 → 1 年 → CAGR = 21%
    assert cagr(equity, periods_per_year=2.0) == pytest.approx(0.21, rel=1e-9)
    assert total_return([100.0]) == 0.0
    assert cagr([], 365.0) == 0.0


def test_volatility_sharpe_sortino():
    returns = pd.Series([0.01, -0.01] * 50)
    assert annualized_volatility(returns, 365.0) == pytest.approx(
        float(np.std(returns, ddof=1)) * np.sqrt(365.0))
    assert sharpe_ratio(returns) < 0 or sharpe_ratio(returns) == pytest.approx(0.0, abs=1e-9)
    assert sharpe_ratio(pd.Series([0.01] * 10)) == 0.0        # 零波动约定为 0
    assert sortino_ratio(pd.Series([0.01, 0.02, 0.03])) == float("inf")
    assert sortino_ratio([0.01, -0.02, 0.03, -0.01]) < 10.0


def test_drawdown_and_max_drawdown():
    equity = pd.Series([100.0, 120.0, 60.0, 90.0])
    dd = drawdown_series(equity)
    assert dd.min() == pytest.approx(-0.5)
    assert max_drawdown(equity) == pytest.approx(0.5)
    assert max_drawdown(pd.Series([100.0, 110.0])) == pytest.approx(0.0)
    assert drawdown_series([]).empty


def test_equity_returns_and_performance_summary():
    equity = pd.Series([100.0, 105.0, 102.0])
    rets = equity_returns(equity)
    assert rets.iloc[0] == 0.0
    assert rets.iloc[1] == pytest.approx(0.05)
    summary = performance_summary(equity, periods_per_year=365.0)
    for key in ("periods", "initial_equity", "final_equity", "total_return", "cagr",
                "volatility", "sharpe", "sortino", "max_drawdown"):
        assert key in summary
    assert summary["periods"] == 3.0
    assert summary["total_return"] == pytest.approx(0.02)


def test_infer_periods_per_year():
    assert infer_periods_per_year(CANDLES) == pytest.approx(8760.0)
    assert infer_periods_per_year([], fallback=12.0) == 12.0
    assert infer_periods_per_year(CANDLES[:1], fallback=4.0) == 4.0
