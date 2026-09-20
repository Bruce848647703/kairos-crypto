"""PaperExchange 撮合与会计测试（纯内存，零网络）。"""
from __future__ import annotations

import pytest

from kairos_crypto import (
    Candle,
    Exchange,
    OrderStatus,
    OrderType,
    PaperExchange,
    Side,
    Ticker,
)

START_TS = 1_609_459_200_000
STEP_MS = 3_600_000
SYMBOL = "BTC/USDT"


def candle(i, close, open_=None, high=None, low=None, volume=1.0, symbol=SYMBOL):
    """构造一根 K 线；未显式给出的价位由收盘价推导（默认无实体）。"""
    c = float(close)
    o = c if open_ is None else float(open_)
    h = max(o, c) if high is None else float(high)
    lo = min(o, c) if low is None else float(low)
    return Candle(symbol=symbol, ts=START_TS + i * STEP_MS, open=o, high=h, low=lo,
                  close=c, volume=volume)


def exchange_with_price(price=100.0, quote=10_000.0, base=0.0, **kwargs):
    """创建一个已观测到最新价的纸面交易所（喂入一根收盘价=price 的 K 线）。"""
    ex = PaperExchange(quote="USDT",
                       balances={"USDT": quote, "BTC": base},
                       **kwargs)
    ex.on_candle(candle(0, price))
    return ex


# ----------------------------------------------------------------- 抽象接口
def test_exchange_is_abstract():
    with pytest.raises(TypeError):
        Exchange()                                    # 抽象基类不可实例化
    assert PaperExchange.is_paper is True
    assert issubclass(PaperExchange, Exchange)


def test_paper_exchange_implements_all_abstract_methods():
    required = {"fetch_candles", "fetch_ticker", "create_order", "cancel_order",
                "fetch_balance", "fetch_orders"}
    assert required <= set(dir(PaperExchange))
    for name in required:
        assert callable(getattr(PaperExchange, name))


# ------------------------------------------------------------------ 市价单
def test_market_buy_costs_price_times_one_plus_fee():
    ex = exchange_with_price(100.0, quote=10_000.0, fee_rate=0.001)
    order = ex.market_buy(SYMBOL, 2.0)
    assert order.status is OrderStatus.FILLED
    assert len(ex.trades) == 1
    trade = ex.trades[0]
    assert trade.price == pytest.approx(100.0)
    assert trade.quantity == pytest.approx(2.0)
    assert trade.fee == pytest.approx(0.2)
    # 等效成交价 = 100 × (1 + 0.001)
    assert trade.effective_price == pytest.approx(100.0 * 1.001)
    # 现金按 价格 × (1 + 费率) × 数量 扣减；基础币入账
    assert ex.free("USDT") == pytest.approx(10_000.0 - 2.0 * 100.0 * 1.001)
    assert ex.total("BTC") == pytest.approx(2.0)
    assert ex.locked("USDT") == 0.0


def test_market_sell_credits_price_times_one_minus_fee():
    ex = exchange_with_price(100.0, quote=0.0, base=3.0, fee_rate=0.002)
    order = ex.market_sell(SYMBOL, 1.5)
    assert order.status is OrderStatus.FILLED
    trade = ex.trades[0]
    assert trade.side is Side.SELL
    assert ex.total("BTC") == pytest.approx(1.5)
    assert ex.free("USDT") == pytest.approx(1.5 * 100.0 * (1 - 0.002))
    assert trade.effective_price == pytest.approx(100.0 * (1 - 0.002))


def test_market_order_applies_slippage():
    ex = exchange_with_price(100.0, fee_rate=0.0, slippage_bps=10.0)
    buy = ex.market_buy(SYMBOL, 1.0)
    assert buy.average_fill_price == pytest.approx(100.0 * 1.001)     # +10bp
    ex2 = exchange_with_price(100.0, base=1.0, fee_rate=0.0, slippage_bps=10.0)
    sell = ex2.market_sell(SYMBOL, 1.0)
    assert sell.average_fill_price == pytest.approx(100.0 * 0.999)     # -10bp


def test_market_buy_is_clamped_to_available_cash_no_overdraft():
    ex = exchange_with_price(100.0, quote=1_000.0, fee_rate=0.001)
    order = ex.market_buy(SYMBOL, 100.0)            # 需要 1 万，只有 1 千
    assert order.status is OrderStatus.FILLED
    assert order.quantity < 100.0
    assert ex.free("USDT") >= 0.0                   # 未透支
    assert ex.free("USDT") == pytest.approx(0.0, abs=1e-4)
    assert ex.total("BTC") == pytest.approx(order.quantity)


def test_market_order_rejected_when_no_cash_and_clamp_disabled():
    ex = exchange_with_price(100.0, quote=1_000.0, clamp_to_balance=False)
    order = ex.market_buy(SYMBOL, 100.0)
    assert order.status is OrderStatus.REJECTED
    assert order.filled_quantity == 0.0
    assert ex.trades == []
    assert ex.free("USDT") == 1_000.0


def test_market_order_without_price_is_rejected():
    ex = PaperExchange(quote="USDT", initial_quote=1_000.0)
    order = ex.market_buy(SYMBOL, 0.1)              # 尚未喂入任何行情
    assert order.status is OrderStatus.REJECTED
    assert ex.trades == []


def test_invalid_order_arguments_raise():
    ex = exchange_with_price(100.0)
    with pytest.raises(ValueError):
        ex.create_order(SYMBOL, Side.BUY, 0.0)
    with pytest.raises(ValueError):
        ex.create_order(SYMBOL, Side.BUY, 1.0, OrderType.LIMIT, price=None)
    with pytest.raises(ValueError):
        ex.create_order("BTCUSDT", Side.BUY, 1.0)


# ------------------------------------------------------------------ 限价单
def test_limit_buy_rests_until_price_touches():
    ex = exchange_with_price(105.0, quote=10_000.0, fee_rate=0.001)
    order = ex.limit_buy(SYMBOL, 100.0, 1.0)
    assert order.status is OrderStatus.NEW          # 价格高于限价 → 挂单留存
    assert len(ex.open_orders()) == 1
    assert ex.trades == []
    # 挂单冻结 限价 × 数量 × (1 + 费率)
    assert ex.locked("USDT") == pytest.approx(100.0 * 1.0 * 1.001)
    assert ex.free("USDT") == pytest.approx(10_000.0 - 100.1)

    # 后续 K 线仍未触及 → 继续挂单
    ex.on_candle(candle(1, 103.0, low=101.0))
    assert ex.trades == []
    assert len(ex.open_orders()) == 1

    # 触及限价 → 以限价成交，多冻结部分退回
    fills = ex.on_candle(candle(2, 101.0, low=99.5))
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(100.0)
    assert fills[0].is_maker is True
    assert fills[0].fee == pytest.approx(0.1)
    assert ex.open_orders() == []
    assert ex.locked("USDT") == 0.0
    assert ex.total("BTC") == pytest.approx(1.0)
    assert ex.free("USDT") == pytest.approx(10_000.0 - 100.0 * 1.001)


def test_limit_buy_fills_at_open_when_gapped_through():
    ex = exchange_with_price(105.0, quote=10_000.0, fee_rate=0.0)
    ex.limit_buy(SYMBOL, 100.0, 1.0)
    fills = ex.on_candle(candle(1, 99.0, open_=98.0, high=99.5, low=97.0))
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(98.0)    # 以更优的开盘价成交


def test_limit_buy_not_filled_when_low_stays_above_limit():
    """只有 low 触及限价才成交：本根 K 线 low=100.5 > 限价 100 → 挂单留存。"""
    ex = exchange_with_price(110.0, quote=10_000.0, fee_rate=0.0)
    ex.limit_buy(SYMBOL, 100.0, 1.0)
    ex.on_candle(candle(1, 101.0, open_=105.0, high=106.0, low=100.5))
    assert ex.trades == []
    assert len(ex.open_orders()) == 1


def test_limit_sell_fills_only_when_high_reaches_price():
    ex = exchange_with_price(95.0, quote=0.0, base=2.0, fee_rate=0.001)
    order = ex.limit_sell(SYMBOL, 100.0, 1.0)
    assert order.status is OrderStatus.NEW
    assert ex.locked("BTC") == pytest.approx(1.0)   # 卖出档冻结基础币
    assert ex.free("BTC") == pytest.approx(1.0)

    ex.on_candle(candle(1, 99.0, high=99.5))
    assert ex.trades == []

    fills = ex.on_candle(candle(2, 101.0, open_=99.0, high=102.0, low=98.0))
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(100.0)
    assert fills[0].side is Side.SELL
    assert ex.free("USDT") == pytest.approx(100.0 * (1 - 0.001))
    assert ex.locked("BTC") == 0.0
    assert ex.total("BTC") == pytest.approx(1.0)


def test_crossing_limit_order_fills_immediately_at_last_price():
    ex = exchange_with_price(98.0, quote=10_000.0, fee_rate=0.001)
    order = ex.limit_buy(SYMBOL, 100.0, 1.0)        # 限价高于现价 → 立即成交
    assert order.status is OrderStatus.FILLED
    assert order.average_fill_price == pytest.approx(98.0)
    assert ex.open_orders() == []
    assert ex.locked("USDT") == 0.0                 # 立即成交不留冻结


def test_limit_order_rejected_when_insufficient_balance():
    """限价单不做数量裁剪：无法全额预占资金时直接拒单。"""
    ex = exchange_with_price(105.0, quote=50.0, fee_rate=0.001)
    order = ex.limit_buy(SYMBOL, 100.0, 1.0)        # 需要约 100.1
    assert order.status is OrderStatus.REJECTED
    assert "余额不足" in order.note
    assert ex.locked("USDT") == 0.0
    assert ex.free("USDT") == 50.0
    assert ex.trades == []


def test_crossing_limit_order_rejected_when_insufficient_balance():
    ex = exchange_with_price(100.0, quote=50.0, fee_rate=0.001)
    order = ex.limit_buy(SYMBOL, 100.0, 1.0)        # 可立即成交，但资金不够
    assert order.status is OrderStatus.REJECTED
    assert ex.trades == []
    assert ex.free("USDT") == 50.0


def test_limit_sell_rejected_without_base_balance():
    ex = exchange_with_price(100.0, quote=1_000.0, base=0.0, clamp_to_balance=False)
    order = ex.limit_sell(SYMBOL, 110.0, 1.0)
    assert order.status is OrderStatus.REJECTED


def test_cancel_order_releases_locked_funds():
    ex = exchange_with_price(105.0, quote=10_000.0, fee_rate=0.001)
    order = ex.limit_buy(SYMBOL, 100.0, 2.0)
    assert ex.locked("USDT") == pytest.approx(200.2)
    assert ex.cancel_order(order.order_id) is True
    assert ex.locked("USDT") == 0.0
    assert ex.free("USDT") == pytest.approx(10_000.0)
    assert ex.fetch_orders(status=OrderStatus.CANCELED)[0].order_id == order.order_id
    # 重复撤单 / 撤不存在的单
    assert ex.cancel_order(order.order_id) is False
    assert ex.cancel_order("nope") is False


def test_cancel_all_counts_only_active_orders():
    ex = exchange_with_price(105.0, quote=100_000.0)
    for price in (100.0, 99.0, 98.0):
        ex.limit_buy(SYMBOL, price, 0.1)
    assert len(ex.open_orders()) == 3
    assert ex.cancel_all(SYMBOL) == 3
    assert ex.open_orders() == []
    assert ex.cancel_all(SYMBOL) == 0


def test_maker_and_taker_fee_rates_differ():
    ex = exchange_with_price(105.0, quote=10_000.0, fee_rate=0.001, maker_fee_rate=0.0002)
    ex.limit_buy(SYMBOL, 100.0, 1.0)
    fills = ex.on_candle(candle(1, 100.0, low=99.0))
    assert fills[0].is_maker is True
    assert fills[0].fee == pytest.approx(100.0 * 0.0002)
    ex2 = exchange_with_price(105.0, quote=10_000.0, fee_rate=0.001, maker_fee_rate=0.0002)
    ex2.market_buy(SYMBOL, 1.0)
    trade = ex2.trades[0]
    assert trade.is_maker is False
    assert trade.fee == pytest.approx(105.0 * 0.001)


# ------------------------------------------------------------------ 持仓会计
def test_position_average_price_and_realized_pnl_through_exchange():
    ex = exchange_with_price(100.0, quote=10_000.0, fee_rate=0.0)
    ex.market_buy(SYMBOL, 1.0)
    ex.mark(SYMBOL, 200.0)
    ex.market_sell(SYMBOL, 1.0)
    pos = ex.position(SYMBOL)
    assert pos.quantity == pytest.approx(0.0, abs=1e-12)
    assert pos.realized_pnl == pytest.approx(100.0)
    assert ex.realized_pnl() == pytest.approx(100.0)
    assert ex.equity() == pytest.approx(10_000.0 + 100.0)


def test_equity_marks_holdings_at_last_price():
    ex = exchange_with_price(100.0, quote=1_000.0, fee_rate=0.0)
    ex.market_buy(SYMBOL, 5.0)                      # 花掉 500
    assert ex.equity() == pytest.approx(1_000.0)
    ex.mark(SYMBOL, 120.0)
    assert ex.equity() == pytest.approx(500.0 + 5.0 * 120.0)
    assert ex.unrealized_pnl() == pytest.approx(100.0)
    snap = ex.snapshot()
    assert snap["equity"] == pytest.approx(1_100.0)
    assert snap["holdings_value"] == pytest.approx(600.0)


def test_fee_paid_accumulates():
    ex = exchange_with_price(100.0, quote=10_000.0, fee_rate=0.001)
    ex.market_buy(SYMBOL, 1.0)
    ex.market_sell(SYMBOL, 1.0)
    assert ex.fee_paid() == pytest.approx(0.1 + 0.1)
    assert ex.position(SYMBOL).fee_paid == pytest.approx(0.2)


# ------------------------------------------------------------------ 查询接口
def test_fetch_balance_returns_copies_and_filters():
    ex = exchange_with_price(100.0, quote=1_000.0)
    balances = ex.fetch_balance()
    assert set(balances) == {"USDT", "BTC"}
    balances["USDT"].free = -1.0                    # 改副本不影响内部记账
    assert ex.free("USDT") == 1_000.0
    only = ex.fetch_balance("BTC")
    assert list(only) == ["BTC"]
    assert ex.fetch_balance("DOGE")["DOGE"].total == 0.0


def test_fetch_orders_filters_by_symbol_and_status():
    ex = exchange_with_price(105.0, quote=100_000.0)
    ex.limit_buy(SYMBOL, 100.0, 0.1)
    ex.limit_buy("ETH/USDT", 90.0, 0.1)
    ex.market_buy(SYMBOL, 0.1)
    assert len(ex.fetch_orders()) == 3
    assert len(ex.fetch_orders(symbol=SYMBOL)) == 2
    assert len(ex.fetch_orders(status=OrderStatus.NEW)) == 2
    assert len(ex.fetch_orders(status="filled")) == 1
    assert len(ex.fetch_open_orders("ETH/USDT")) == 1


def test_fetch_candles_returns_observed_history_only():
    empty = PaperExchange(quote="USDT", initial_quote=1_000.0)
    assert empty.fetch_candles(SYMBOL) == []        # 未喂 K 线
    ex = exchange_with_price(100.0)
    ex.on_candles([candle(i, 100.0 + i) for i in range(1, 6)])
    history = ex.fetch_candles(SYMBOL, limit=3)
    assert len(history) == 3
    assert history[-1].close == pytest.approx(105.0)
    assert ex.fetch_candles("ETH/USDT") == []


def test_fetch_ticker_uses_last_price():
    ex = exchange_with_price(100.0)
    ticker = ex.fetch_ticker(SYMBOL)
    assert isinstance(ticker, Ticker)
    assert ticker.last == pytest.approx(100.0)
    ex.mark(SYMBOL, 111.0)
    assert ex.fetch_ticker(SYMBOL).last == pytest.approx(111.0)
    with pytest.raises(ValueError):
        ex.fetch_ticker("ETH/USDT")


def test_on_ticker_matches_resting_orders():
    ex = exchange_with_price(105.0, quote=10_000.0, fee_rate=0.0)
    ex.limit_buy(SYMBOL, 100.0, 1.0)
    assert ex.on_ticker(Ticker(SYMBOL, START_TS, 101.0)) == []
    fills = ex.on_ticker(Ticker(SYMBOL, START_TS + 1, 99.0))
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(99.0)    # 按最新价成交


def test_reset_restores_initial_state():
    ex = exchange_with_price(100.0, quote=1_000.0)
    ex.market_buy(SYMBOL, 1.0)
    assert ex.trades
    ex.reset()
    assert ex.trades == [] and ex.open_orders() == []
    assert ex.free("USDT") == 1_000.0
    assert ex.clock == 0


def test_frames_are_dataframes():
    ex = exchange_with_price(100.0, quote=1_000.0)
    ex.market_buy(SYMBOL, 1.0)
    assert list(ex.trades_frame().columns)[:3] == ["trade_id", "order_id", "symbol"]
    assert len(ex.orders_frame()) == 1
