"""测试公用工具：构造确定性 K 线与跑通引擎的快捷函数（不涉及任何网络）。"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from kairos_crypto import Candle, PaperExchange, PaperTradingEngine, Strategy

START_TS = 1_609_459_200_000          # 2021-01-01T00:00:00Z
STEP_MS = 3_600_000                   # 1 小时
SYMBOL = "BTC/USDT"


def make_candles(closes: Sequence[float], symbol: str = SYMBOL, wick: float = 0.0,
                 start_ts: int = START_TS, step_ms: int = STEP_MS,
                 volume: float = 1.0) -> List[Candle]:
    """由收盘价构造 K 线：开盘 = 前一根收盘，上下影线按 ``wick`` 比例外扩。

    ``wick=0`` 时得到 high=low=open=close 的「无波动」K 线，便于精确验证撮合价。
    """
    out: List[Candle] = []
    prev = float(closes[0])
    for i, close in enumerate(closes):
        c = float(close)
        o = prev
        body_high, body_low = max(o, c), min(o, c)
        out.append(Candle(symbol=symbol, ts=start_ts + i * step_ms, open=o,
                          high=body_high * (1.0 + wick), low=body_low * (1.0 - wick),
                          close=c, volume=volume))
        prev = c
    return out


def ohlc_candles(rows: Iterable[Tuple[float, float, float, float]], symbol: str = SYMBOL,
                 start_ts: int = START_TS, step_ms: int = STEP_MS,
                 volume: float = 1.0) -> List[Candle]:
    """由 (open, high, low, close) 序列精确构造 K 线。"""
    out: List[Candle] = []
    for i, (o, h, low, c) in enumerate(rows):
        out.append(Candle(symbol=symbol, ts=start_ts + i * step_ms, open=float(o),
                          high=float(h), low=float(low), close=float(c), volume=volume))
    return out


def flat_candles(prices: Sequence[float], symbol: str = SYMBOL) -> List[Candle]:
    """价格恒定、无实体无影线的 K 线（每根 open=high=low=close）。"""
    out: List[Candle] = []
    for i, price in enumerate(prices):
        p = float(price)
        out.append(Candle(symbol=symbol, ts=START_TS + i * STEP_MS, open=p, high=p,
                          low=p, close=p, volume=1.0))
    return out


def make_exchange(quote: float = 10_000.0, base: float = 0.0, fee_rate: float = 0.001,
                  **kwargs) -> PaperExchange:
    """创建带初始余额的纸面交易所。"""
    return PaperExchange(quote="USDT", balances={"USDT": quote, "BTC": base},
                         fee_rate=fee_rate, **kwargs)


def run(candles: Sequence[Candle], strategy: Strategy, exchange: Optional[PaperExchange] = None,
        **kwargs):
    """跑一遍纸面交易，返回 (result, exchange)。"""
    ex = exchange if exchange is not None else make_exchange()
    engine = PaperTradingEngine(candles, strategy, ex, **kwargs)
    return engine.run(), ex
