"""合成行情与回放器测试：确定性、OHLC 约束、无未来数据（全程离线）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos_crypto import (
    DEFAULT_START_TS,
    Candle,
    PaperExchange,
    Replayer,
    SyntheticCandles,
    candles_from_frame,
    candles_to_frame,
    interval_seconds,
    make_candles,
    periods_per_year,
    symbols_of,
)

SEED = 20260101


# ------------------------------------------------------------- 周期与工具
def test_interval_helpers():
    assert interval_seconds("1h") == 3600
    assert interval_seconds("1m") == 60
    assert interval_seconds(900) == 900
    assert periods_per_year("1h") == pytest.approx(8760.0)
    assert periods_per_year("1d") == pytest.approx(365.0)
    with pytest.raises(ValueError):
        interval_seconds("7m")


def test_symbols_of_keeps_first_seen_order():
    candles = [Candle("BTC/USDT", 0, 1, 1, 1, 1), Candle("ETH/USDT", 0, 1, 1, 1, 1),
               Candle("BTC/USDT", 1, 1, 1, 1, 1)]
    assert symbols_of(candles) == ("BTC/USDT", "ETH/USDT")


# ------------------------------------------------------------ 合成 K 线
def test_same_seed_produces_identical_candles():
    a = SyntheticCandles(n=300, seed=SEED, interval="1h")
    b = SyntheticCandles(n=300, seed=SEED, interval="1h")
    assert len(a) == len(b) == 300
    pd.testing.assert_frame_equal(a.frame(), b.frame())
    assert [c.to_dict() for c in a.candles] == [c.to_dict() for c in b.candles]


def test_different_seed_produces_different_candles():
    a = SyntheticCandles(n=200, seed=SEED)
    b = SyntheticCandles(n=200, seed=SEED + 1)
    assert not np.allclose(a.closes, b.closes)


def test_ohlc_constraints_hold_everywhere():
    candles = SyntheticCandles(n=500, seed=SEED, base_vol=0.02, jump_prob=0.05)
    for candle in candles:
        assert candle.is_consistent(), candle
        assert candle.low <= min(candle.open, candle.close) + 1e-12
        assert candle.high >= max(candle.open, candle.close) - 1e-12
        assert candle.low <= candle.high
        assert candle.volume > 0
        assert min(candle.open, candle.high, candle.low, candle.close) > 0


def test_timestamps_are_continuous_247():
    """加密市场 24/7：时间戳等距递增，没有交易日/周末缺口。"""
    candles = SyntheticCandles(n=200, seed=SEED, interval="1h", start_ts=DEFAULT_START_TS)
    stamps = [c.ts for c in candles]
    diffs = np.diff(np.asarray(stamps))
    assert set(diffs.tolist()) == {3_600_000}
    assert stamps[0] == DEFAULT_START_TS
    assert candles[0].datetime == pd.Timestamp("2021-01-01 00:00:00+0000", tz="UTC")


def test_start_price_and_volatility_are_crypto_like():
    gen = SyntheticCandles(n=1000, seed=SEED, start_price=30_000.0, interval="1h")
    assert gen[0].open == pytest.approx(30_000.0)
    vol = gen.realized_vol()
    assert vol > 0.25, f"加密风格应为高波动，实际年化波动 {vol:.3f}"
    # 既有上涨段也有下跌段（趋势 + 震荡），并非单边
    closes = gen.closes
    roll = closes[24:] / closes[:-24] - 1.0
    assert roll.max() > 0.02 and roll.min() < -0.02


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        SyntheticCandles(n=0)
    with pytest.raises(ValueError):
        SyntheticCandles(start_price=0.0)


def test_make_candles_shortcut():
    candles = make_candles(n=50, seed=SEED, symbol="ETH/USDT", start_price=2_000.0)
    assert len(candles) == 50
    assert all(c.symbol == "ETH/USDT" for c in candles)
    assert candles[0].open == pytest.approx(2_000.0)


def test_frame_round_trip():
    candles = SyntheticCandles(n=40, seed=SEED, symbol="BTC/USDT").candles
    frame = candles_to_frame(candles)
    assert list(frame.columns) == ["symbol", "ts", "open", "high", "low", "close", "volume"]
    assert len(frame) == 40
    back = candles_from_frame(frame, "BTC/USDT")
    assert len(back) == 40
    for a, b in zip(candles, back):
        assert a.ts == b.ts
        assert a.close == pytest.approx(b.close)
        assert a.volume == pytest.approx(b.volume)


# ------------------------------------------------------------------ 回放器
def test_replayer_orders_by_time_and_exposes_symbols():
    late = Candle("BTC/USDT", 2_000, 1, 1, 1, 1)
    early = Candle("BTC/USDT", 1_000, 1, 1, 1, 1)
    replay = Replayer([late, early])
    assert [c.ts for c in replay.candles] == [1_000, 2_000]
    assert replay.symbols == ("BTC/USDT",)
    assert replay.start_ts == 1_000 and replay.end_ts == 2_000
    assert len(replay) == 2


def test_replayer_history_never_leaks_future():
    candles = SyntheticCandles(n=30, seed=SEED).candles
    replay = Replayer(candles)
    for i, candle in enumerate(replay):
        hist = replay.history()
        assert len(hist) == i + 1
        assert hist[-1] is candle
        assert replay.closes().size == i + 1
        assert all(h.ts <= candle.ts for h in hist)
    assert replay.index == 29
    assert replay.closes().size == 30


def test_replayer_reset_allows_second_pass():
    candles = SyntheticCandles(n=10, seed=SEED).candles
    replay = Replayer(candles)
    first = list(replay)
    second = list(replay)
    assert first == second
    replay.reset()
    assert replay.history() == []
    assert replay.current is None and replay.index == -1


def test_replayer_steps_and_returns():
    candles = SyntheticCandles(n=5, seed=SEED, start_price=100.0).candles
    replay = Replayer(candles)
    steps = list(replay.steps())
    assert [i for i, _ in steps] == [0, 1, 2, 3, 4]
    rets = replay.returns()
    closes = replay.closes()
    assert rets[0] == 0.0
    np.testing.assert_allclose(rets[1:], closes[1:] / closes[:-1] - 1.0)


def test_replayer_multi_symbol_history_is_per_symbol():
    btc = SyntheticCandles(n=6, seed=SEED, symbol="BTC/USDT").candles
    eth = SyntheticCandles(n=6, seed=SEED + 5, symbol="ETH/USDT", start_price=2_000.0).candles
    replay = Replayer(btc + eth)
    assert replay.symbols == ("BTC/USDT", "ETH/USDT")
    seen = 0
    for candle in replay:
        seen += 1
        assert len(replay.history(candle.symbol)) <= seen
    assert len(replay.history("BTC/USDT")) == 6
    assert len(replay.history("ETH/USDT")) == 6


def test_replayer_infers_interval_and_periods():
    replay = Replayer(SyntheticCandles(n=10, seed=SEED, interval="15m").candles)
    assert replay.interval_seconds() == 900
    assert replay.periods_per_year() == pytest.approx(365 * 96)


def test_replayer_rejects_bad_candles():
    broken = Candle("BTC/USDT", 0, open=100.0, high=100.0, low=101.0, close=100.0)
    with pytest.raises(ValueError):
        Replayer([broken])
    duplicated = [Candle("BTC/USDT", 0, 1, 2, 0.5, 1.5), Candle("BTC/USDT", 0, 1, 2, 0.5, 1.5)]
    with pytest.raises(ValueError):
        Replayer(duplicated)
    with pytest.raises(TypeError):
        Replayer([{"ts": 0}])


def test_replayer_from_frame_and_from_exchange():
    frame = SyntheticCandles(n=20, seed=SEED, symbol="BTC/USDT").frame()
    replay = Replayer.from_frame(frame, "BTC/USDT")
    assert len(replay) == 20

    ex = PaperExchange(quote="USDT", initial_quote=1_000.0)
    ex.on_candles(SyntheticCandles(n=8, seed=SEED).candles)
    replayed = Replayer.from_exchange(ex, "BTC/USDT", interval="1h", limit=5)
    assert len(replayed) == 5                        # 只取交易所已观测到的最后 5 根


def test_replayer_frame_view():
    replay = Replayer(SyntheticCandles(n=12, seed=SEED).candles)
    frame = replay.frame()
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 12
    assert frame["close"].iloc[-1] == pytest.approx(replay[-1].close)
