"""合成行情生成与 K 线回放（完全离线、确定性可复现）。

加密货币市场 24/7 连续交易、波动显著高于股票，因此本模块提供：

- :class:`SyntheticCandles`：用「两状态马尔可夫切换 + 肥尾跳跃」生成 BTC 风格的
  合成 K 线（趋势段与震荡段交替、成交量随波动放大），同一 ``seed`` 结果完全一致。
- :class:`Replayer`：把历史 K 线按时间顺序逐根喂给策略与纸面交易所，
  并保证策略在第 ``i`` 根只能看到前 ``i`` 根（含当根）的数据，杜绝未来函数。

所有随机性都来自 ``numpy.random.default_rng(seed)``，不涉及任何网络请求。
"""
from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .types import Candle

#: 常见 K 线周期 -> 秒数（加密行业惯例字符串）
INTERVAL_SECONDS: Dict[str, int] = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "12h": 43200,
    "1d": 86400, "1w": 604800,
}

#: 默认起始时间戳：2021-01-01T00:00:00Z（毫秒），保证无参数调用也可复现
DEFAULT_START_TS = 1_609_459_200_000

#: 一年的秒数；加密市场全年无休，用 365 天折算年化
SECONDS_PER_YEAR = 365 * 24 * 3600


def interval_seconds(interval: Union[str, int]) -> int:
    """把周期字符串（``"1h"``）或秒数转为秒数。"""
    if isinstance(interval, (int, float)) and not isinstance(interval, bool):
        seconds = int(interval)
    else:
        key = str(interval).strip().lower()
        if key not in INTERVAL_SECONDS:
            raise ValueError(f"未知 K 线周期: {interval!r}，可选 {sorted(INTERVAL_SECONDS)}")
        seconds = INTERVAL_SECONDS[key]
    if seconds <= 0:
        raise ValueError("K 线周期必须为正数")
    return seconds


def periods_per_year(interval: Union[str, int]) -> float:
    """按 24/7 连续交易折算某周期的年化期数（如 ``"1h" -> 8760``）。"""
    return SECONDS_PER_YEAR / float(interval_seconds(interval))


def symbols_of(candles: Iterable[Candle]) -> Tuple[str, ...]:
    """提取 K 线中出现过的交易对（保持首次出现顺序）。"""
    seen: List[str] = []
    for candle in candles:
        if candle.symbol not in seen:
            seen.append(candle.symbol)
    return tuple(seen)


def candles_to_frame(candles: Sequence[Candle]) -> pd.DataFrame:
    """K 线列表 -> DataFrame（index 为 UTC 时间）。"""
    rows = [c.to_dict() for c in candles]
    if not rows:
        return pd.DataFrame(columns=["symbol", "ts", "time", "open", "high", "low",
                                     "close", "volume"])
    frame = pd.DataFrame(rows)
    frame["time"] = pd.to_datetime(frame["ts"], unit="ms", utc=True)
    return frame.set_index("time")[["symbol", "ts", "open", "high", "low", "close", "volume"]]


def candles_from_frame(frame: pd.DataFrame, symbol: str,
                       ts_column: Optional[str] = "ts") -> List[Candle]:
    """DataFrame -> K 线列表。

    时间戳来源优先级：``ts_column`` 列（毫秒）> ``DatetimeIndex`` > 按小时递推。
    要求包含 ``open/high/low/close`` 列，``volume`` 可选。
    """
    if frame.empty:
        return []
    df = frame
    n = len(df)
    if ts_column is not None and ts_column in df.columns:
        stamps = [int(v) for v in df[ts_column].tolist()]
    elif isinstance(df.index, pd.DatetimeIndex):
        stamps = [int(v) // 1_000_000 for v in df.index.asi8.tolist()]
    else:
        stamps = [DEFAULT_START_TS + i * 3_600_000 for i in range(n)]
    volumes = df["volume"].tolist() if "volume" in df.columns else [0.0] * n
    out: List[Candle] = []
    for i in range(n):
        out.append(Candle(
            symbol=symbol, ts=stamps[i],
            open=float(df["open"].iloc[i]), high=float(df["high"].iloc[i]),
            low=float(df["low"].iloc[i]), close=float(df["close"].iloc[i]),
            volume=float(volumes[i]),
        ))
    return out


class SyntheticCandles:
    """确定性合成 K 线生成器（加密风格：高波动、趋势与震荡交替、肥尾跳跃）。

    生成模型（全部自研，只用 numpy）::

        regime_t : 两状态马尔可夫链，0=震荡、1=趋势（切换概率 regime_switch_prob）
        dir_t    : 趋势方向 ±1，仅在趋势段生效，偶尔反转（trend_flip_prob）
        r_t      = dir_t·base_vol·trend_strength          # 漂移
                 + base_vol·(1.5 if 趋势 else 1)·z_t      # 常规波动
                 + 跳跃项（以 jump_prob 触发，幅度 jump_scale 倍）
        close_t  = start_price · exp(cumsum(r_t))

    OHLC 由收盘价反推：开盘 = 前收盘 × 微小跳空，上下影线由独立噪声决定，
    因此天然满足 ``low <= min(open, close) <= max(open, close) <= high``。

    参数
    ----
    symbol:      交易对名称（仅作为标签，不影响数值）。
    n:           生成根数。
    interval:    K 线周期，决定时间戳步长与年化期数。
    start_price: 首个收盘价基准。
    seed:        随机种子；**同 seed 同参数两次生成结果完全一致**。
    start_ts:    首根 K 线的毫秒时间戳。
    base_vol:    震荡段的单根 K 线波动率（对数收益标准差）。
    trend_strength: 趋势段单根漂移相对波动的强度。
    regime_switch_prob: 每根 K 线切换市场状态的概率。
    trend_flip_prob:    趋势方向反转的概率。
    jump_prob:   发生跳跃（肥尾）的概率。
    jump_scale:  跳跃幅度相对 base_vol 的倍数。
    wick_ratio:  上下影线长度相对 base_vol 的比例。
    volume_base: 成交量基准（基础币）。
    volume_sensitivity: 成交量对 |收益| 的敏感系数。
    """

    def __init__(self,
                 symbol: str = "BTC/USDT",
                 n: int = 500,
                 interval: Union[str, int] = "1h",
                 start_price: float = 30_000.0,
                 seed: int = 42,
                 start_ts: int = DEFAULT_START_TS,
                 base_vol: float = 0.010,
                 trend_strength: float = 0.35,
                 regime_switch_prob: float = 0.03,
                 trend_flip_prob: float = 0.06,
                 jump_prob: float = 0.02,
                 jump_scale: float = 4.0,
                 wick_ratio: float = 0.6,
                 volume_base: float = 120.0,
                 volume_sensitivity: float = 6.0):
        if n <= 0:
            raise ValueError("n 必须为正整数")
        if start_price <= 0:
            raise ValueError("start_price 必须为正数")
        self.symbol = symbol
        self.n = int(n)
        self.interval = interval
        self.step_ms = interval_seconds(interval) * 1000
        self.start_price = float(start_price)
        self.seed = int(seed)
        self.start_ts = int(start_ts)
        self.base_vol = float(base_vol)
        self.trend_strength = float(trend_strength)
        self.regime_switch_prob = float(regime_switch_prob)
        self.trend_flip_prob = float(trend_flip_prob)
        self.jump_prob = float(jump_prob)
        self.jump_scale = float(jump_scale)
        self.wick_ratio = float(wick_ratio)
        self.volume_base = float(volume_base)
        self.volume_sensitivity = float(volume_sensitivity)
        self._candles: List[Candle] = self._build()

    # ------------------------------------------------------------------ 生成
    def _build(self) -> List[Candle]:
        rng = np.random.default_rng(self.seed)
        n = self.n

        # 1) 市场状态：0=震荡，1=趋势
        regime = np.zeros(n, dtype=np.int8)
        state = 0
        for i in range(n):
            if i > 0 and rng.random() < self.regime_switch_prob:
                state = 1 - state
            regime[i] = state

        # 2) 趋势方向（仅趋势段生效），偶尔反转以形成多轮行情
        direction = np.zeros(n, dtype=float)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        for i in range(n):
            if rng.random() < self.trend_flip_prob:
                sign = -sign
            direction[i] = sign if regime[i] == 1 else 0.0

        # 3) 对数收益：漂移 + 波动 + 跳跃
        vol = self.base_vol * np.where(regime == 1, 1.5, 1.0)
        drift = direction * self.base_vol * self.trend_strength
        shock = vol * rng.standard_normal(n)
        jump_draw = rng.standard_normal(n)
        jump_hit = rng.random(n) < self.jump_prob
        jumps = np.where(
            jump_hit,
            np.sign(jump_draw) * self.jump_scale * self.base_vol * (1.0 + np.abs(jump_draw)),
            0.0,
        )
        log_returns = drift + shock + jumps
        closes = self.start_price * np.exp(np.cumsum(log_returns))

        # 4) 由收盘价构造 OHLC
        opens = np.empty(n, dtype=float)
        opens[0] = self.start_price
        gap = 0.15 * self.base_vol * rng.standard_normal(n)
        opens[1:] = closes[:-1] * np.exp(gap[1:])
        wick_up = np.minimum(np.abs(rng.standard_normal(n)) * self.wick_ratio * self.base_vol, 0.5)
        wick_dn = np.minimum(np.abs(rng.standard_normal(n)) * self.wick_ratio * self.base_vol, 0.5)
        body_high = np.maximum(opens, closes)
        body_low = np.minimum(opens, closes)
        highs = body_high * (1.0 + wick_up)
        lows = body_low * (1.0 - wick_dn)

        # 5) 成交量：对数正态基准 × 波动放大
        intensity = 1.0 + self.volume_sensitivity * np.abs(log_returns) / self.base_vol
        volumes = self.volume_base * np.exp(0.4 * rng.standard_normal(n)) * intensity

        candles = [
            Candle(symbol=self.symbol, ts=self.start_ts + i * self.step_ms,
                   open=float(opens[i]), high=float(highs[i]), low=float(lows[i]),
                   close=float(closes[i]), volume=float(volumes[i]))
            for i in range(n)
        ]
        bad = [c for c in candles if not c.is_consistent()]
        if bad:                                   # pragma: no cover - 防御性检查
            raise RuntimeError(f"合成 K 线不满足 OHLC 约束，例如 ts={bad[0].ts}")
        return candles

    # ------------------------------------------------------------------ 访问
    @property
    def candles(self) -> List[Candle]:
        """生成的 K 线列表（副本，按时间升序）。"""
        return list(self._candles)

    @property
    def closes(self) -> np.ndarray:
        """收盘价数组。"""
        return np.array([c.close for c in self._candles], dtype="float64")

    @property
    def log_returns(self) -> np.ndarray:
        """对数收益率数组（首根为 0）。"""
        log_close = np.log(self.closes)
        out = np.zeros_like(log_close)
        out[1:] = np.diff(log_close)
        return out

    def frame(self) -> pd.DataFrame:
        """转 DataFrame，便于查看/落盘（默认 CSV，无需 pyarrow）。"""
        return candles_to_frame(self._candles)

    def replayer(self, validate: bool = False) -> "Replayer":
        """包装成 :class:`Replayer`，可直接喂给纸面交易引擎。"""
        return Replayer(self._candles, validate=validate)

    def realized_vol(self, annualized: bool = True) -> float:
        """合成行情的已实现波动率（年化按 24/7 折算）。"""
        rets = self.log_returns[1:]
        if rets.size == 0:
            return 0.0
        vol = float(np.std(rets, ddof=1))
        return vol * float(np.sqrt(periods_per_year(self.interval))) if annualized else vol

    def __len__(self) -> int:
        return len(self._candles)

    def __iter__(self) -> Iterator[Candle]:
        return iter(self._candles)

    def __getitem__(self, item):
        return self._candles[item]

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return (f"SyntheticCandles(symbol={self.symbol!r}, n={self.n}, "
                f"interval={self.interval!r}, seed={self.seed})")


def make_candles(n: int = 500, symbol: str = "BTC/USDT", interval: Union[str, int] = "1h",
                 seed: int = 42, start_price: float = 30_000.0, **kwargs) -> List[Candle]:
    """快捷函数：生成合成 K 线列表（参数见 :class:`SyntheticCandles`）。"""
    return SyntheticCandles(symbol=symbol, n=n, interval=interval, seed=seed,
                            start_price=start_price, **kwargs).candles


class Replayer:
    """历史 K 线回放器：按时间顺序把行情喂给策略/交易所，并维护「截至当前」的历史。

    防未来函数约定：:meth:`history` / :meth:`closes` 只返回**已经回放**的 K 线，
    即策略在第 ``i`` 步最多看到前 ``i+1`` 根，绝不可能读到后续数据。

    参数
    ----
    candles:  K 线可迭代对象（``SyntheticCandles``、列表、或 :meth:`from_frame` 构造）。
    validate: 是否校验 OHLC 一致性与时间戳单调性（默认开启）。
    """

    def __init__(self, candles: Iterable[Candle], validate: bool = True):
        items = list(candles)
        for candle in items:
            if not isinstance(candle, Candle):
                raise TypeError(f"Replayer 只接受 Candle 对象，收到 {type(candle).__name__}")
        self._candles: List[Candle] = sorted(items, key=lambda c: (int(c.ts), c.symbol))
        if validate:
            self._validate(self._candles)
        self._history: Dict[str, List[Candle]] = {}
        self._cursor: int = -1
        self.current: Optional[Candle] = None

    @staticmethod
    def _validate(candles: Sequence[Candle]) -> None:
        seen: set = set()
        for candle in candles:
            if not candle.is_consistent():
                raise ValueError(f"K 线不满足 low<=open/close<=high 或存在负值/NaN: {candle}")
            key = (candle.symbol, int(candle.ts))
            if key in seen:
                raise ValueError(f"存在重复的 (symbol, ts): {key}")
            seen.add(key)
        for symbol in symbols_of(candles):
            stamps = [c.ts for c in candles if c.symbol == symbol]
            if any(b <= a for a, b in zip(stamps, stamps[1:])):
                raise ValueError(f"{symbol} 的时间戳必须严格递增")

    # ------------------------------------------------------------- 构造入口
    @classmethod
    def from_frame(cls, frame: pd.DataFrame, symbol: str,
                   ts_column: Optional[str] = "ts", validate: bool = True) -> "Replayer":
        """由 DataFrame 构造回放器。"""
        return cls(candles_from_frame(frame, symbol, ts_column), validate=validate)

    @classmethod
    def from_exchange(cls, exchange, symbol: str, interval: str = "1h",
                      limit: int = 500, validate: bool = True) -> "Replayer":
        """由任意 :class:`~kairos_crypto.exchange.Exchange` 的历史 K 线构造回放器。

        对内置的 ``PaperExchange`` 而言，返回的是它已经观测到的 K 线（不联网）。
        """
        return cls(exchange.fetch_candles(symbol, interval=interval, limit=limit),
                   validate=validate)

    # ------------------------------------------------------------------ 属性
    def __len__(self) -> int:
        return len(self._candles)

    def __getitem__(self, item):
        return self._candles[item]

    @property
    def candles(self) -> List[Candle]:
        """全部 K 线（副本）。注意：直接遍历它会绕过回放语义。"""
        return list(self._candles)

    @property
    def symbols(self) -> Tuple[str, ...]:
        return symbols_of(self._candles)

    @property
    def index(self) -> int:
        """当前回放位置（0 起），未开始为 -1。"""
        return self._cursor

    @property
    def start_ts(self) -> int:
        return int(self._candles[0].ts) if self._candles else 0

    @property
    def end_ts(self) -> int:
        return int(self._candles[-1].ts) if self._candles else 0

    def interval_seconds(self, symbol: Optional[str] = None) -> int:
        """由相邻时间戳中位数推断周期（秒）。"""
        sym = symbol or (self._candles[0].symbol if self._candles else None)
        stamps = [c.ts for c in self._candles if c.symbol == sym]
        if len(stamps) < 2:
            return 3600
        diffs = np.diff(np.asarray(stamps, dtype="int64")) // 1000
        return int(np.median(diffs)) if diffs.size else 3600

    def periods_per_year(self, symbol: Optional[str] = None) -> float:
        """按 24/7 折算的年化期数。"""
        return periods_per_year(self.interval_seconds(symbol))

    def frame(self) -> pd.DataFrame:
        """全部 K 线的 DataFrame 视图。"""
        return candles_to_frame(self._candles)

    # ------------------------------------------------------------------ 回放
    def reset(self) -> "Replayer":
        """回到起点，清空已回放历史（可重复运行同一份行情）。"""
        self._history = {}
        self._cursor = -1
        self.current = None
        return self

    def __iter__(self) -> Iterator[Candle]:
        """按时间顺序回放；每次迭代都会自动从头开始。"""
        self.reset()
        for i, candle in enumerate(self._candles):
            self._cursor = i
            self.current = candle
            self._history.setdefault(candle.symbol, []).append(candle)
            yield candle

    def steps(self) -> Iterator[Tuple[int, Candle]]:
        """回放并附带步序号，等价于 ``enumerate(iter(replayer))``。"""
        for candle in self:
            yield self._cursor, candle

    def history(self, symbol: Optional[str] = None, n: Optional[int] = None) -> List[Candle]:
        """已回放（含当前）的最近 ``n`` 根 K 线；``n=None`` 返回全部历史。"""
        sym = symbol or (self.current.symbol if self.current is not None else None)
        if sym is None:
            return []
        items = self._history.get(sym, [])
        return items if n is None else items[-int(n):]

    def closes(self, symbol: Optional[str] = None, n: Optional[int] = None) -> np.ndarray:
        """已回放 K 线的收盘价数组（无未来数据）。"""
        hist = self.history(symbol, n)
        return np.array([c.close for c in hist], dtype="float64")

    def volumes(self, symbol: Optional[str] = None, n: Optional[int] = None) -> np.ndarray:
        """已回放 K 线的成交量数组。"""
        hist = self.history(symbol, n)
        return np.array([c.volume for c in hist], dtype="float64")

    def returns(self, symbol: Optional[str] = None, n: Optional[int] = None) -> np.ndarray:
        """已回放 K 线的简单收益率数组（首个元素为 0）。"""
        closes = self.closes(symbol, n)
        if closes.size < 2:
            return np.zeros(closes.size, dtype="float64")
        out = np.zeros_like(closes)
        out[1:] = closes[1:] / closes[:-1] - 1.0
        return out

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return (f"Replayer(candles={len(self._candles)}, symbols={list(self.symbols)}, "
                f"cursor={self._cursor})")
