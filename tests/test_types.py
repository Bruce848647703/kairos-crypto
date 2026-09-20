"""基础数据类型与会计口径测试（纯离线、无交易所依赖）。"""
from __future__ import annotations

import pytest

from kairos_crypto import (
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


def make_candle(i=0, open_=100.0, high=101.0, low=99.0, close=100.5,
                symbol="BTC/USDT", volume=1.0, step_ms=3_600_000):
    return Candle(symbol=symbol, ts=1_609_459_200_000 + i * step_ms, open=open_,
                  high=high, low=low, close=close, volume=volume)


# ------------------------------------------------------------------ symbol
def test_parse_and_join_symbol():
    assert parse_symbol("BTC/USDT") == ("BTC", "USDT")
    assert parse_symbol(" eth/usdt ") == ("ETH", "USDT")
    assert join_symbol("btc", "usdt") == "BTC/USDT"
    with pytest.raises(ValueError):
        parse_symbol("BTCUSDT")


# -------------------------------------------------------------------- Side
def test_side_sign_and_parse():
    assert Side.BUY.sign == 1 and Side.SELL.sign == -1
    assert Side.parse("buy") is Side.BUY
    assert Side.parse("SELL") is Side.SELL
    assert Side.parse(-1) is Side.SELL
    with pytest.raises(ValueError):
        Side.parse("hold")


def test_order_type_parse():
    assert OrderType.parse("limit") is OrderType.LIMIT
    assert OrderType.parse("MARKET") is OrderType.MARKET
    with pytest.raises(ValueError):
        OrderType.parse("stop")


# ------------------------------------------------------------------ Candle
def test_candle_consistency_and_touch():
    candle = make_candle(high=110.0, low=90.0, open_=100.0, close=105.0)
    assert candle.is_consistent()
    assert candle.touched(95.0) and candle.touched(105.0)
    assert not candle.touched(120.0)
    assert candle.crossed_below(90.0) and not candle.crossed_below(89.0)
    assert candle.crossed_above(110.0) and not candle.crossed_above(111.0)
    assert candle.is_bullish and candle.body == pytest.approx(5.0)
    assert candle.mid == pytest.approx(100.0)


def test_candle_detects_bad_ohlc():
    bad = Candle(symbol="BTC/USDT", ts=0, open=100.0, high=100.0, low=101.0, close=100.0)
    assert not bad.is_consistent()          # low > high
    leaky = Candle(symbol="BTC/USDT", ts=0, open=100.0, high=100.5, low=99.5, close=102.0)
    assert not leaky.is_consistent()        # close 超出 high


def test_ticker_from_candle_has_no_spread():
    ticker = Ticker.from_candle(make_candle(close=100.0))
    assert ticker.last == 100.0
    assert ticker.spread == 0.0
    assert ticker.mid == 100.0


# ------------------------------------------------------------------- Order
def test_order_status_helpers():
    order = Order(symbol="BTC/USDT", side=Side.BUY, quantity=1.0,
                  order_type=OrderType.LIMIT, price=100.0, order_id="X1")
    assert order.is_active and order.is_limit
    assert order.remaining_quantity == 1.0
    order.filled_quantity = 0.4
    assert order.remaining_quantity == pytest.approx(0.6)
    order.status = OrderStatus.FILLED
    assert not order.is_active
    assert order.to_dict()["side"] == "buy"


# ------------------------------------------------------------------- Trade
def test_trade_effective_price_is_price_times_one_plus_fee():
    """手续费口径：等效成交价 = 价格 × (1 ± 费率)。"""
    fee_rate = 0.001
    buy = Trade(trade_id="T1", order_id="O1", symbol="BTC/USDT", side=Side.BUY,
                price=100.0, quantity=2.0, fee=100.0 * 2.0 * fee_rate, ts=0)
    sell = Trade(trade_id="T2", order_id="O2", symbol="BTC/USDT", side=Side.SELL,
                 price=100.0, quantity=2.0, fee=100.0 * 2.0 * fee_rate, ts=0)
    assert buy.effective_price == pytest.approx(100.0 * (1 + fee_rate))
    assert sell.effective_price == pytest.approx(100.0 * (1 - fee_rate))
    # 现金流：买入为负、卖出为正，且都已扣手续费
    assert buy.cash_flow == pytest.approx(-(200.0 + 0.2))
    assert sell.cash_flow == pytest.approx(200.0 - 0.2)
    assert buy.gross_value == pytest.approx(200.0)


# ----------------------------------------------------------------- Balance
def test_balance_reserve_release_settle():
    bal = Balance("USDT", free=1000.0)
    assert bal.total == 1000.0
    bal.reserve(300.0)
    assert (bal.free, bal.locked, bal.total) == (700.0, 300.0, 1000.0)
    bal.settle_locked(280.0)
    bal.credit(20.0)                        # 退回多冻结部分
    assert bal.locked == pytest.approx(20.0)
    assert bal.free == pytest.approx(720.0)
    bal.release(20.0)
    assert bal.locked == 0.0 and bal.free == pytest.approx(740.0)
    with pytest.raises(ValueError):
        bal.reserve(10_000.0)               # 余额不足


# ------------------------------------------------------------ SpotPosition
def test_position_average_price_and_realized_pnl():
    pos = SpotPosition(symbol="BTC/USDT")
    pos.apply_fill(Side.BUY, 1.0, 100.0, fee=0.1)
    pos.apply_fill(Side.BUY, 1.0, 200.0, fee=0.2)
    assert pos.quantity == 2.0
    assert pos.average_price == pytest.approx(150.0)
    assert pos.unrealized_pnl(180.0) == pytest.approx(60.0)
    pos.apply_fill(Side.SELL, 1.0, 180.0, fee=0.18)
    assert pos.quantity == 1.0
    assert pos.realized_pnl == pytest.approx(30.0)     # (180 - 150) × 1
    assert pos.fee_paid == pytest.approx(0.48)
    assert pos.highest_price == 200.0 and pos.lowest_price == 100.0
    pos.apply_fill(Side.SELL, 1.0, 120.0)
    assert pos.quantity == 0.0 and pos.average_price == 0.0
    assert pos.realized_pnl == pytest.approx(0.0)      # 30 + (120-150)


def test_position_return_pct():
    pos = SpotPosition(symbol="ETH/USDT")
    pos.apply_fill(Side.BUY, 2.0, 50.0)
    assert pos.return_pct(55.0) == pytest.approx(0.1)
    assert pos.market_value(55.0) == pytest.approx(110.0)
