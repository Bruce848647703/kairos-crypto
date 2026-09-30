"""真实历史 OHLCV 数据的离线加载与回放对接（纯 pandas/numpy，零网络）。

背景与定位
----------
当前运行环境**无法访问任何真实加密货币交易所接口**（binance 等公开行情端点均被
网络阻断），且本仓库铁律禁止任何真实联网与下单。本模块提供离线替代方案：
把磁盘上**任意历史 OHLCV CSV**（约定列 ``date,open,high,low,close,volume``，
兼容常见别名与 epoch 时间戳列）转换成本包的 :class:`~kairos_crypto.types.Candle`
序列，直接对接 :class:`~kairos_crypto.data.Replayer` 与
:class:`~kairos_crypto.engine.PaperTradingEngine` 做纸面交易回放。

演示数据用真实 A 股/ETF 日线（后复权），只把它们当作「通用价格序列」来验证
引擎链路；用户把 ``csv_path`` 换成自行下载的加密历史 K 线 CSV 即可复用同一套
策略与引擎代码（本模块不下载、不请求任何外部资源）。

脏数据约定
----------
真实 CSV 难免有缺失/异常行，:func:`load_ohlcv_frame` 的处理规则（确定性、可复现）：

- 列名统一转小写并去空格后匹配；缺 ``open/high/low/close`` 任一列则报错；
  ``volume`` 可缺省（按 0 处理）。
- 日期列解析失败（NaT）或任一 OHLC 为 NaN 的行直接丢弃；重复时间戳保留最后一行。
- 对每行做最小修复以满足 :meth:`Candle.is_consistent`：
  ``high = max(high, open, close)``、``low = min(low, open, close)``；
  非有限或为负的 volume 归零；四价任一非正的行丢弃。
- 输出按时间**升序**。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .data import Replayer
from .types import Candle, parse_symbol

#: OHLCV CSV 必须具备的四价列（小写匹配）
REQUIRED_COLUMNS = ("open", "high", "low", "close")

#: 日期列的常见别名（当 ``date_column`` 不存在时按序尝试）
DATE_ALIASES = ("date", "datetime", "timestamp", "time", "trade_date", "day")

#: 一年的毫秒数（365 天口径，与 data.INTERVAL_SECONDS/SECONDS_PER_YEAR 一致）
MS_PER_YEAR = 365 * 24 * 3600 * 1000


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """把列名统一成「小写 + 去首尾空格」，便于宽松匹配真实 CSV 的各种表头。"""
    rename = {col: str(col).strip().lower() for col in frame.columns
              if str(col) != str(col).strip().lower()}
    return frame.rename(columns=rename) if rename else frame


def _resolve_date_column(frame: pd.DataFrame, date_column: str) -> str:
    """在归一化后的表头里定位日期列；找不到则报 ``ValueError``。"""
    key = str(date_column).strip().lower()
    if key in frame.columns:
        return key
    for alias in DATE_ALIASES:
        if alias in frame.columns:
            return alias
    raise ValueError(
        f"未找到日期列 {date_column!r}（也未匹配到别名 {DATE_ALIASES}），"
        f"实际列: {list(frame.columns)}")


def _parse_index(raw: pd.Series) -> pd.DatetimeIndex:
    """把日期列解析成 UTC ``DatetimeIndex``；数值列按 epoch 时间戳（秒/毫秒）处理。"""
    if pd.api.types.is_numeric_dtype(raw):
        values = raw.astype("float64")
        finite = values[np.isfinite(values)]
        unit = "ms" if finite.size and float(finite.max()) >= 1e12 else "s"
        return pd.to_datetime(values, unit=unit, utc=True)
    return pd.to_datetime(raw, errors="coerce", utc=True)


def load_ohlcv_frame(csv_path: str, date_column: str = "date") -> pd.DataFrame:
    """读取真实 OHLCV CSV，返回清洗/修复后的 DataFrame（UTC DatetimeIndex，升序）。

    参数
    ----
    csv_path:    本地 CSV 路径（只读，绝不联网）。
    date_column: 日期列名，默认 ``"date"``；不存在时尝试 :data:`DATE_ALIASES`。

    返回
    ----
    列为 ``open/high/low/close/volume`` 的 DataFrame，index 名为 ``"date"``。
    """
    path = os.fspath(csv_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"OHLCV CSV 不存在: {path}")
    frame = _normalize_columns(pd.read_csv(path))
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"CSV 缺少必需列 {missing}，实际列: {list(frame.columns)}")

    date_key = _resolve_date_column(frame, date_column)
    work = frame.copy()
    work["_index_"] = _parse_index(work[date_key])
    work = work.dropna(subset=["_index_"] + list(REQUIRED_COLUMNS))
    if work.empty:
        raise ValueError(f"CSV 中没有可解析的有效 OHLCV 行: {path}")
    # 稳定排序：重复时间戳的行保持文件内出现顺序，drop_duplicates(keep="last") 才确定
    work = work.sort_values("_index_", kind="stable").drop_duplicates("_index_", keep="last")

    ohlc = work[list(REQUIRED_COLUMNS)].astype("float64")
    body_high = ohlc[["open", "close"]].max(axis=1).to_numpy()
    body_low = ohlc[["open", "close"]].min(axis=1).to_numpy()
    high = np.maximum(ohlc["high"].to_numpy(), body_high)
    low = np.minimum(ohlc["low"].to_numpy(), body_low)
    if "volume" in work.columns:
        volume = pd.to_numeric(work["volume"], errors="coerce").to_numpy(dtype="float64")
    else:
        volume = np.zeros(len(work), dtype="float64")
    volume = np.where(np.isfinite(volume), np.maximum(volume, 0.0), 0.0)

    out = pd.DataFrame({"open": ohlc["open"].to_numpy(), "high": high, "low": low,
                        "close": ohlc["close"].to_numpy(), "volume": volume},
                       index=pd.DatetimeIndex(work["_index_"], name="date"))
    out = out[(out[list(REQUIRED_COLUMNS)] > 0).all(axis=1)]
    if out.empty:
        raise ValueError(f"CSV 清洗后没有剩余的有效行情行（四价必须为正）: {path}")
    return out


def symbol_from_path(csv_path: str, quote: str = "CNY") -> str:
    """由 CSV 文件名推导交易对标签，如 ``".../sh518880.csv" -> "SH518880/CNY"``。

    真实 A 股/ETF 数据以人民币计价，故默认计价币为 ``CNY``；接入加密历史数据时
    可传 ``quote="USDT"``（或直接给 :func:`load_candles` 传完整 ``symbol``）。
    """
    stem = os.path.splitext(os.path.basename(os.fspath(csv_path)))[0]
    stem = "".join(ch for ch in stem.strip().upper() if ch.isalnum()) or "DATA"
    return f"{stem}/{str(quote).strip().upper()}"


def load_candles(csv_path: str, symbol: Optional[str] = None, quote: str = "CNY",
                 date_column: str = "date") -> List[Candle]:
    """真实 OHLCV CSV → 本包 :class:`Candle` 序列（按时间升序，可直接回放）。

    参数
    ----
    csv_path:    本地 CSV 路径（约定列 ``date,open,high,low,close,volume``，离线只读）。
    symbol:      交易对标签；``None`` 时由文件名推导（见 :func:`symbol_from_path`）。
                 必须能被 :func:`~kairos_crypto.types.parse_symbol` 解析（含 ``/``）。
    quote:       ``symbol=None`` 时使用的计价币标签（A 股/ETF 数据默认 ``"CNY"``）。
    date_column: 日期列名（兼容别名与 epoch 数值列，见 :func:`load_ohlcv_frame`）。

    返回
    ----
    ``List[Candle]``，``ts`` 为该行日期的 UTC 零点毫秒时间戳；每根均满足
    :meth:`Candle.is_consistent`，可被 :class:`Replayer`（``validate=True``）接受。
    """
    frame = load_ohlcv_frame(csv_path, date_column)
    if symbol is None:
        sym = symbol_from_path(csv_path, quote)
    else:
        sym = str(symbol).strip().upper()
    parse_symbol(sym)                              # 提前校验 BASE/QUOTE 格式

    stamps = (frame.index.asi8 // 1_000_000).tolist()
    opens = frame["open"].to_numpy(dtype="float64").tolist()
    highs = frame["high"].to_numpy(dtype="float64").tolist()
    lows = frame["low"].to_numpy(dtype="float64").tolist()
    closes = frame["close"].to_numpy(dtype="float64").tolist()
    volumes = frame["volume"].to_numpy(dtype="float64").tolist()
    return [Candle(symbol=sym, ts=int(stamps[i]), open=opens[i], high=highs[i],
                   low=lows[i], close=closes[i], volume=volumes[i])
            for i in range(len(frame))]


def replayer_from_csv(csv_path: str, symbol: Optional[str] = None, validate: bool = True,
                      **kwargs: Any) -> Replayer:
    """一步到位：真实 OHLCV CSV → :class:`Replayer`（可直接喂给纸面交易引擎）。"""
    return Replayer(load_candles(csv_path, symbol=symbol, **kwargs), validate=validate)


def estimate_periods_per_year(candles: Sequence[Candle], fallback: float = 252.0) -> float:
    """按**真实时间跨度**估计年化期数（股票/ETF 交易日口径，而非加密 24/7 口径）。

    ``ppy = (根数 - 1) / 跨度年数``。日线工作日数据约得 243~252；加密 24/7 小时线
    约得 8760，与本包 :func:`~kairos_crypto.data.periods_per_year` 的口径自然衔接。
    样本不足或跨度非法时返回 ``fallback``。
    """
    n = len(candles)
    if n < 2:
        return float(fallback)
    stamps = sorted(int(c.ts) for c in candles)
    span_years = (stamps[-1] - stamps[0]) / float(MS_PER_YEAR)
    if span_years <= 0:
        return float(fallback)
    ppy = (n - 1) / span_years
    if not np.isfinite(ppy) or ppy <= 0:
        return float(fallback)
    return float(ppy)


def describe_candles(candles: Sequence[Candle]) -> Dict[str, Any]:
    """真实序列的摘要信息（标的/区间/根数/首末收盘价），供报告与日志复用。"""
    if not candles:
        raise ValueError("candles 为空，无法生成摘要")
    first, last = candles[0], candles[-1]
    return {
        "symbol": first.symbol,
        "bars": int(len(candles)),
        "start": f"{first.datetime:%Y-%m-%d}",
        "end": f"{last.datetime:%Y-%m-%d}",
        "first_close": float(first.close),
        "last_close": float(last.close),
        "buy_hold_return": float(last.close / first.close - 1.0),
        "periods_per_year": estimate_periods_per_year(candles),
    }
