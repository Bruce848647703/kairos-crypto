"""策略测试：网格阶梯、定投纪律、动量信号，以及策略上下文接口。"""
from __future__ import annotations

import numpy as np
import pytest

from kairos_crypto import (
    Context,
    DcaStrategy,
    GridStrategy,
    MomentumStrategy,
    OrderStatus,
    OrderType,
    PaperTradingEngine,
    Replayer,
    Side,
    Strategy,
    SyntheticCandles,
)
from helpers import SYMBOL, flat_candles, make_candles, make_exchange, ohlc_candles, run


# ------------------------------------------------------------------ 基类
def test_strategy_base_requires_on_candle():
    class Empty(Strategy):
        pass

    with pytest.raises(NotImplementedError):
        Empty().on_candle(None, None)


def test_strategy_resolves_symbol_from_context():
    strat = MomentumStrategy(fast=2, slow=4)
    replay = Replayer(flat_candles([100.0] * 3))
    ctx = Context(exchange=make_exchange(), replayer=replay)
    ctx.symbol = SYMBOL
    assert strat.symbol is None
    assert strat.resolve_symbol(ctx) == SYMBOL


# ------------------------------------------------------------------ 上下文
def test_context_trading_helpers():
    ex = make_exchange(quote=1_000.0, fee_rate=0.0)
    replay = Replayer(flat_candles([100.0] * 4))
    ctx = Context(exchange=ex, replayer=replay)
    candle = next(iter(replay))
    ctx.index, ctx.symbol, ctx.candle = 0, candle.symbol, candle
    ex.on_candle(candle)

    assert ctx.last_price() == pytest.approx(100.0)
    assert ctx.equity() == pytest.approx(1_000.0)
    assert ctx.base_currency() == "BTC" and ctx.quote_currency() == "USDT"
    assert ctx.free("USDT") == pytest.approx(1_000.0)

    order = ctx.buy(quantity=2.0)
    assert order.status is OrderStatus.FILLED
    assert ctx.holding() == pytest.approx(2.0)
    assert ctx.free("USDT") == pytest.approx(800.0)

    ctx.sell(quote=100.0)                     # 卖出价值 100 USDT ≈ 1 个币
    assert ctx.holding() == pytest.approx(1.0)

    ctx.flatten()
    assert ctx.holding() == pytest.approx(0.0, abs=1e-12)
    assert ctx.equity() == pytest.approx(1_000.0)

    ctx.order_target_percent(0.5)               # 半仓
    assert ctx.holding() * 100.0 == pytest.approx(500.0)

    ctx.limit_buy(90.0, 0.1)
    ctx.limit_buy(80.0, 0.1)
    assert len(ctx.open_orders()) == 2
    assert ctx.locked("USDT") == pytest.approx(17.0)
    assert ctx.cancel_all() == 2
    assert ctx.open_orders() == []
    assert ctx.locked("USDT") == 0.0


def test_context_buy_by_quote_accounts_for_fee():
    ex = make_exchange(quote=1_000.0, fee_rate=0.001)
    replay = Replayer(flat_candles([100.0] * 2))
    ctx = Context(exchange=ex, replayer=replay)
    candle = next(iter(replay))
    ctx.index, ctx.symbol, ctx.candle = 0, candle.symbol, candle
    ex.on_candle(candle)
    ctx.buy(quote=100.0)
    # 花掉的钱恰好是 100（含手续费）
    assert ctx.free("USDT") == pytest.approx(900.0, abs=1e-6)
    assert ctx.holding() == pytest.approx(100.0 / (100.0 * 1.001))


def test_context_requires_symbol():
    ctx = Context(exchange=make_exchange(), replayer=Replayer(flat_candles([100.0])))
    with pytest.raises(ValueError):
        ctx.closes()
    with pytest.raises(RuntimeError):
        _ = ctx.bar


# ------------------------------------------------------------------ 网格
def test_grid_places_arithmetic_ladder():
    """网格应挂出价格等距、数量一致的阶梯状限价单。"""
    candles = flat_candles([100.0] * 5)
    ex = make_exchange(quote=10_000.0, fee_rate=0.0)
    grid = GridStrategy(levels=4, spacing=0.02, qty_per_grid=0.01)
    result = PaperTradingEngine(candles, grid, ex).run()

    step = 100.0 * 0.02
    buys, sells = grid.buy_prices(), grid.sell_prices()
    assert buys == pytest.approx([100.0 - k * step for k in (4, 3, 2, 1)])
    assert sells == pytest.approx([100.0 + k * step for k in (1, 2, 3, 4)])
    # 价格等距
    np.testing.assert_allclose(np.diff(buys), step, rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.diff(sells), step, rtol=0, atol=1e-9)
    # 买卖两侧间距也是同一步长（中心价上下对称）
    assert sells[0] - buys[-1] == pytest.approx(2 * step)

    resting = ex.fetch_orders(status=OrderStatus.NEW)
    assert len(resting) == 8
    assert all(o.order_type is OrderType.LIMIT for o in resting)
    assert all(o.quantity == pytest.approx(0.01) for o in resting)
    assert sum(1 for o in resting if o.side is Side.BUY) == 4
    assert sum(1 for o in resting if o.side is Side.SELL) == 4
    assert grid.grid_frame()["status"].unique().tolist() == ["new"]
    assert result.trade_count >= 1              # 至少建立了卖出档底仓


def test_grid_respects_explicit_center_and_step():
    """显式给定 center/step 时，网格按绝对等差铺开（不随 spacing 变化）。"""
    candles = flat_candles([100.0] * 2)
    ex = make_exchange(quote=10_000.0, fee_rate=0.0)
    grid = GridStrategy(levels=3, qty_per_grid=0.5, center=50.0, step=10.0, side="buy",
                        build_inventory=False)
    PaperTradingEngine(candles, grid, ex).run()
    assert grid.center == pytest.approx(50.0)
    assert grid.step == pytest.approx(10.0)
    assert grid.buy_prices() == pytest.approx([20.0, 30.0, 40.0])
    assert grid.sell_prices() == []             # side="buy" 只挂买档
    assert ex.locked("USDT") == pytest.approx((20.0 + 30.0 + 40.0) * 0.5)


def test_grid_levels_above_market_fill_immediately():
    """买档若高于现价属于「可成交限价单」，按最新价立即成交（不留挂单）。"""
    candles = flat_candles([100.0] * 2)
    ex = make_exchange(quote=10_000.0, fee_rate=0.0)
    grid = GridStrategy(levels=2, qty_per_grid=0.1, center=200.0, step=10.0, side="buy",
                        build_inventory=False, rearm=False)
    result = PaperTradingEngine(candles, grid, ex).run()
    assert grid.buy_prices() == []
    assert result.trade_count == 2
    assert all(t.price == pytest.approx(100.0) for t in result.trades)


def test_grid_rearms_opposite_side_after_fill():
    """买档成交后在其上方一步补挂卖档；卖档成交后在其下方一步补挂买档。"""
    rows = [
        (100.0, 100.5, 99.5, 100.0),    # i=0 建仓：center=100, step=2
        (100.0, 100.2, 97.0, 97.5),     # i=1 low=97 触及买档 98
        (97.5, 101.0, 97.0, 100.5),     # i=2 high=101 触及补挂的卖档 100
        (100.5, 101.0, 97.5, 98.0),     # i=3 low=97.5 再次触及买档 98
    ]
    candles = ohlc_candles(rows)
    ex = make_exchange(quote=1_000.0, fee_rate=0.0)
    grid = GridStrategy(levels=4, spacing=0.02, qty_per_grid=0.01)
    result = PaperTradingEngine(candles, grid, ex).run()

    assert grid.filled_buys == 2
    assert grid.filled_sells == 1
    assert grid.grid_cycles == 1
    prices = [t.price for t in result.trades if t.order_id != grid.inventory_order.order_id]
    assert prices == [98.0, 100.0, 98.0]            # 低买→高卖→再低买
    assert 100.0 in grid.sell_prices()              # 买档成交后补挂了卖档
    assert 98.0 not in grid.buy_prices()            # 该买档已成交并转为卖档
    assert grid.step == pytest.approx(2.0)


def test_grid_inventory_is_built_before_sell_orders():
    candles = flat_candles([100.0] * 2)
    ex = make_exchange(quote=10_000.0, base=0.0, fee_rate=0.0)
    grid = GridStrategy(levels=3, spacing=0.01, qty_per_grid=0.02)
    PaperTradingEngine(candles, grid, ex).run()
    assert grid.inventory_order is not None
    assert grid.inventory_order.status is OrderStatus.FILLED
    assert grid.inventory_order.filled_quantity == pytest.approx(0.06)
    # 卖档全部挂出成功（没有因余额不足被拒）
    assert len(grid.sell_prices()) == 3
    assert not [o for o in ex.fetch_orders(status=OrderStatus.REJECTED)]


def test_grid_sell_orders_rejected_without_inventory_flag():
    candles = flat_candles([100.0] * 2)
    ex = make_exchange(quote=10_000.0, base=0.0, fee_rate=0.0)
    grid = GridStrategy(levels=2, spacing=0.01, qty_per_grid=0.02, build_inventory=False)
    PaperTradingEngine(candles, grid, ex).run()
    rejected = ex.fetch_orders(status=OrderStatus.REJECTED)
    assert len(rejected) == 2
    assert all(o.side is Side.SELL for o in rejected)
    assert len(grid.sell_prices()) == 0
    assert len(grid.buy_prices()) == 2


def test_grid_invalid_parameters():
    with pytest.raises(ValueError):
        GridStrategy(levels=0)
    with pytest.raises(ValueError):
        GridStrategy(qty_per_grid=0.0)
    with pytest.raises(ValueError):
        GridStrategy(spacing=0.0)


# ------------------------------------------------------------------ 定投
def test_dca_buys_fixed_quote_on_schedule():
    candles = flat_candles([100.0] * 31)
    ex = make_exchange(quote=10_000.0, fee_rate=0.001)
    dca = DcaStrategy(quote_per_buy=100.0, every_n=5)
    result = PaperTradingEngine(candles, dca, ex).run()

    assert dca.buys == 7                        # index 0,5,10,15,20,25,30
    assert result.trade_count == 7
    assert all(t.side is Side.BUY for t in result.trades)
    # 每次投入 100 USDT（含手续费），数量一致
    for trade in result.trades:
        assert trade.gross_value + trade.fee == pytest.approx(100.0, abs=1e-9)
        assert trade.quantity == pytest.approx(100.0 / (100.0 * 1.001))
    assert ex.free("USDT") == pytest.approx(10_000.0 - 700.0)
    assert dca.invested == pytest.approx(700.0)
    assert dca.average_cost == pytest.approx(100.0 * 1.001)


def test_dca_position_grows_over_time():
    candles = flat_candles([100.0] * 41)
    ex = make_exchange(quote=10_000.0, fee_rate=0.0)
    dca = DcaStrategy(quote_per_buy=50.0, every_n=4)
    result = PaperTradingEngine(candles, dca, ex).run()
    positions = result.history["position"].to_numpy()
    # 持仓单调不减，且每个定投日都增加
    assert np.all(np.diff(positions) >= -1e-12)
    increases = np.flatnonzero(np.diff(positions) > 1e-12) + 1
    assert increases.tolist() == [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    assert positions[0] == pytest.approx(0.5)     # 第 0 根即首次定投
    assert positions[-1] == pytest.approx(11 * 0.5)
    assert dca.buys == 11
    assert len(dca.schedule_frame()) == 11


def test_dca_respects_max_buys_and_start_index():
    candles = flat_candles([100.0] * 50)
    ex = make_exchange(quote=10_000.0, fee_rate=0.0)
    dca = DcaStrategy(quote_per_buy=100.0, every_n=10, start_index=5, max_buys=2)
    result = PaperTradingEngine(candles, dca, ex).run()
    assert dca.buys == 2
    assert [int(row["index"]) for row in dca.schedule] == [5, 15]
    assert result.trade_count == 2


def test_dca_stops_when_cash_exhausted():
    candles = flat_candles([100.0] * 10)
    ex = make_exchange(quote=250.0, fee_rate=0.001)
    dca = DcaStrategy(quote_per_buy=100.0, every_n=1)
    PaperTradingEngine(candles, dca, ex).run()
    assert dca.buys == 3                        # 100 + 100 + 50
    assert dca.stopped is True
    assert ex.free("USDT") == pytest.approx(0.0, abs=1e-6)


def test_dca_stop_loss_flattens_position():
    closes = [100.0] * 5 + [70.0] * 5           # 定投后暴跌 30%
    candles = make_candles(closes)
    ex = make_exchange(quote=1_000.0, fee_rate=0.0)
    dca = DcaStrategy(quote_per_buy=100.0, every_n=1, stop_pct=0.1)
    result = PaperTradingEngine(candles, dca, ex).run()
    assert dca.stopped is True
    assert result.final_position == pytest.approx(0.0, abs=1e-12)
    assert any(t.side is Side.SELL for t in result.trades)


def test_dca_invalid_parameters():
    with pytest.raises(ValueError):
        DcaStrategy(quote_per_buy=0.0)
    with pytest.raises(ValueError):
        DcaStrategy(every_n=0)


# ------------------------------------------------------------------ 动量
def test_momentum_goes_long_in_uptrend():
    closes = [100.0 * 1.005 ** i for i in range(80)]
    candles = make_candles(closes, wick=0.002)
    ex = make_exchange(quote=10_000.0, fee_rate=0.001)
    strat = MomentumStrategy(fast=5, slow=20)
    result = PaperTradingEngine(candles, strat, ex).run()
    assert strat.state == "long"
    assert result.final_position > 0
    assert result.trade_count >= 1
    assert result.metrics["total_return"] > 0
    assert result.metrics["total_return"] > result.metrics["benchmark_return"] * 0.5


def test_momentum_stays_flat_in_downtrend():
    closes = [100.0 * 0.995 ** i for i in range(80)]
    candles = make_candles(closes, wick=0.002)
    ex = make_exchange(quote=10_000.0, fee_rate=0.001)
    strat = MomentumStrategy(fast=5, slow=20)
    result = PaperTradingEngine(candles, strat, ex).run()
    assert result.trade_count == 0
    assert result.final_position == pytest.approx(0.0)
    assert strat.state == "flat"
    assert result.metrics["net_pnl"] == pytest.approx(0.0)


def test_momentum_needs_full_lookback_before_trading():
    closes = [100.0 * 1.01 ** i for i in range(10)]
    candles = make_candles(closes)
    ex = make_exchange(quote=10_000.0, fee_rate=0.0)
    strat = MomentumStrategy(fast=5, slow=20)
    result = PaperTradingEngine(candles, strat, ex).run()
    assert result.trade_count == 0
    assert strat.signal(_ctx_of(candles), SYMBOL) is None


def _ctx_of(candles):
    """把回放跑到最后一根，返回策略视角的 Context（用于单测信号函数）。"""
    ex = make_exchange()
    replay = Replayer(candles)
    ctx = Context(exchange=ex, replayer=replay)
    for i, candle in enumerate(replay):
        ctx.index, ctx.symbol, ctx.candle = i, candle.symbol, candle
        ex.on_candle(candle)
    return ctx


def test_momentum_signal_is_none_until_enough_history():
    closes = [100.0 * 1.01 ** i for i in range(30)]
    ctx = _ctx_of(make_candles(closes))
    strat = MomentumStrategy(fast=5, slow=20)
    assert strat.signal(ctx, SYMBOL) == "long"
    assert strat.target_exposure(ctx, SYMBOL) == pytest.approx(1.0)
    short_ctx = _ctx_of(make_candles([100.0] * 10))
    assert strat.signal(short_ctx, SYMBOL) is None


def test_momentum_stop_loss_triggers_on_crash():
    # 先单边上涨 40 根建立多头，再单根暴跌 32% 触发止损
    closes = [100.0 * 1.01 ** i for i in range(40)] + [100.0] * 10
    candles = make_candles(closes, wick=0.002)
    ex = make_exchange(quote=10_000.0, fee_rate=0.0)
    strat = MomentumStrategy(fast=5, slow=20, stop_pct=0.05)
    result = PaperTradingEngine(candles, strat, ex).run()
    assert strat.exits, "应至少触发一次止损"
    assert strat.exits[0]["reason"] == "stop_loss"
    assert result.final_position == pytest.approx(0.0, abs=1e-9)


def test_momentum_vol_target_reduces_exposure():
    candles = SyntheticCandles(n=600, seed=11, interval="1h").candles
    plain = MomentumStrategy(fast=12, slow=48)
    capped = MomentumStrategy(fast=12, slow=48, vol_target=0.05)
    res_plain, _ = run(candles, plain, make_exchange(quote=10_000.0))
    res_capped, _ = run(candles, capped, make_exchange(quote=10_000.0))

    def average_exposure(result):
        holdings = result.history["position"] * result.history["price"]
        return float((holdings / result.history["equity"]).mean())

    assert average_exposure(res_capped) < average_exposure(res_plain)
    assert average_exposure(res_capped) < 0.15      # 目标波动 5% ≪ 合成行情波动


def test_momentum_invalid_windows():
    with pytest.raises(ValueError):
        MomentumStrategy(fast=20, slow=20)
    with pytest.raises(ValueError):
        MomentumStrategy(fast=0, slow=10)
