"""realdata 模块离线测试：tmp_path 小 CSV + 内嵌极小真实样本，全程不联网。

覆盖：
- :func:`load_candles` 的字段映射、升序、数量、symbol 推导与报错分支；
- 脏数据清洗/修复（NaN 行、重复日期、OHLC 不一致、缺失 volume）；
- :func:`estimate_periods_per_year` / :func:`describe_candles` / :func:`replayer_from_csv`；
- 用一段**真实** ETF 日线（sh518880，2024-11 ~ 2024-12，43 根）跑通
  ``PaperTradingEngine`` 回放并做会计自洽校验。
"""
from __future__ import annotations

import os

import pytest

from kairos_crypto import (
    DcaStrategy,
    GridStrategy,
    MomentumStrategy,
    PaperExchange,
    PaperTradingEngine,
    describe_candles,
    estimate_periods_per_year,
    load_candles,
    replayer_from_csv,
    symbol_from_path,
)

#: 极小真实样本：黄金 ETF sh518880 后复权日线（2024-11-01 ~ 2024-12-31，43 根），
#: 只读自 kairos-data 项目并内嵌于此，保证测试离线、不依赖外部路径。
REAL_SAMPLE_CSV = """date,open,high,low,close,volume
2024-11-01,2.272,2.291,2.272,2.285,3157902.0
2024-11-04,2.27,2.274,2.261,2.268,3206472.0
2024-11-05,2.267,2.269,2.253,2.268,3465225.0
2024-11-06,2.276,2.289,2.256,2.263,5310282.0
2024-11-07,2.215,2.221,2.212,2.213,4971911.0
2024-11-08,2.237,2.242,2.223,2.227,5441984.0
2024-11-11,2.228,2.232,2.224,2.23,2938955.0
2024-11-12,2.202,2.206,2.19,2.193,3675693.0
2024-11-13,2.192,2.195,2.188,2.189,2945700.0
2024-11-14,2.162,2.164,2.141,2.145,4360110.0
2024-11-15,2.157,2.157,2.14,2.146,4101363.0
2024-11-18,2.171,2.18,2.167,2.174,3807962.0
2024-11-19,2.194,2.213,2.194,2.212,4128956.0
2024-11-20,2.228,2.228,2.211,2.212,3933263.0
2024-11-21,2.233,2.241,2.231,2.24,3471025.0
2024-11-22,2.254,2.27,2.253,2.27,4794501.0
2024-11-25,2.269,2.271,2.225,2.234,7735648.0
2024-11-26,2.203,2.211,2.2,2.207,3675693.0
2024-11-27,2.213,2.229,2.211,2.228,3544560.0
2024-11-28,2.202,2.219,2.199,2.216,3513385.0
2024-11-29,2.22,2.238,2.22,2.237,3726694.0
2024-12-02,2.209,2.222,2.208,2.217,3002256.0
2024-12-03,2.234,2.244,2.232,2.24,2904022.0
2024-12-04,2.234,2.242,2.233,2.239,2169824.0
2024-12-05,2.242,2.244,2.237,2.238,2238436.0
2024-12-06,2.215,2.232,2.213,2.229,2917620.0
2024-12-09,2.243,2.245,2.237,2.242,2406329.0
2024-12-10,2.254,2.255,2.246,2.251,3022870.0
2024-12-11,2.276,2.277,2.257,2.27,3684896.0
2024-12-12,2.28,2.286,2.275,2.285,2850292.0
2024-12-13,2.272,2.273,2.262,2.266,2686424.0
2024-12-16,2.245,2.247,2.24,2.246,2513897.0
2024-12-17,2.249,2.254,2.242,2.243,2538572.0
2024-12-18,2.244,2.246,2.239,2.24,1835436.0
2024-12-19,2.216,2.23,2.216,2.222,3622339.0
2024-12-20,2.219,2.226,2.215,2.225,2433829.0
2024-12-23,2.235,2.248,2.235,2.247,2797678.0
2024-12-24,2.239,2.24,2.234,2.236,2279641.0
2024-12-25,2.24,2.243,2.239,2.242,1445139.0
2024-12-26,2.249,2.249,2.242,2.243,1837517.0
2024-12-27,2.247,2.248,2.243,2.245,1765068.0
2024-12-30,2.241,2.243,2.236,2.237,1660249.0
2024-12-31,2.233,2.24,2.228,2.239,1648335.0
"""

REAL_SAMPLE_BARS = REAL_SAMPLE_CSV.strip().count("\n")     # 43 根（去掉表头行）


def write_csv(tmp_path, text: str, name: str = "sample.csv") -> str:
    """把 CSV 文本写到 tmp_path，返回文件路径。"""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


# ------------------------------------------------------------------ 基础加载
def test_load_candles_fields_order_and_count(tmp_path):
    """乱序小 CSV → 升序 Candle；字段、数量、时间戳逐一核对。"""
    text = ("date,open,high,low,close,volume\n"
            "2024-01-03,3.0,3.5,2.5,3.2,300\n"
            "2024-01-01,1.0,1.5,0.5,1.2,100\n"
            "2024-01-02,2.0,2.5,1.5,2.2,200\n")
    candles = load_candles(write_csv(tmp_path, text, "tiny.csv"))
    assert len(candles) == 3
    assert [c.ts for c in candles] == sorted(c.ts for c in candles)
    assert [c.close for c in candles] == [1.2, 2.2, 3.2]
    first = candles[0]
    assert first.symbol == "TINY/CNY"                    # 文件名推导 + 默认计价币
    assert (first.open, first.high, first.low, first.volume) == (1.0, 1.5, 0.5, 100.0)
    assert first.ts == 1_704_067_200_000                 # 2024-01-01T00:00:00Z 毫秒
    assert all(c.is_consistent() for c in candles)


def test_load_candles_symbol_and_quote_options(tmp_path):
    """symbol 显式指定 / quote 推导 / 非法 symbol 报错。"""
    path = write_csv(tmp_path, REAL_SAMPLE_CSV, "sh518880.csv")
    assert symbol_from_path(path) == "SH518880/CNY"
    assert symbol_from_path(path, quote="usdt") == "SH518880/USDT"
    candles = load_candles(path, symbol="btc/usdt")
    assert all(c.symbol == "BTC/USDT" for c in candles)
    with pytest.raises(ValueError):
        load_candles(path, symbol="NO-SEPARATOR")


def test_load_candles_cleans_and_repairs_dirty_rows(tmp_path):
    """NaN 行丢弃、重复日期保留最后一行、OHLC 不一致自动修复、volume 缺失补零。"""
    text = ("Date,Open,High,Low,Close\n"                 # 大小写混排 + 无 volume 列
            "2024-05-02,10,9,11,10\n"                    # high<open 且 low>open → 修复
            "2024-05-01,1,2,0.5,1.5\n"
            "2024-05-01,1,2,0.5,9.9\n"                   # 重复日期 → 保留最后一行
            "bad-date,1,2,0.5,1.5\n"                     # 日期不可解析 → 丢弃
            "2024-05-03,1,2,,1.5\n")                     # low 为 NaN → 丢弃
    candles = load_candles(write_csv(tmp_path, text))
    assert [f"{c.datetime:%Y-%m-%d}" for c in candles] == ["2024-05-01", "2024-05-02"]
    assert candles[0].close == 9.9                       # 重复日期取最后一行
    repaired = candles[1]
    assert repaired.high >= max(repaired.open, repaired.close)
    assert repaired.low <= min(repaired.open, repaired.close)
    assert all(c.volume == 0.0 for c in candles)         # volume 缺省补零
    assert all(c.is_consistent() for c in candles)


def test_load_candles_error_branches(tmp_path):
    """文件不存在 / 缺必需列 → 明确报错。"""
    with pytest.raises(FileNotFoundError):
        load_candles(str(tmp_path / "missing.csv"))
    bad = write_csv(tmp_path, "date,open,high,low\n2024-01-01,1,2,0.5\n", "bad.csv")
    with pytest.raises(ValueError):
        load_candles(bad)


# ------------------------------------------------------------------ 派生工具
def test_replayer_and_period_estimation_on_real_sample(tmp_path):
    """真实样本：回放器可直接构造；日线工作日数据的年化期数应落在 200~300。"""
    path = write_csv(tmp_path, REAL_SAMPLE_CSV, "sh518880.csv")
    candles = load_candles(path)
    assert len(candles) == REAL_SAMPLE_BARS == 43
    replayer = replayer_from_csv(path)
    assert len(replayer) == 43
    assert replayer.symbols == ("SH518880/CNY",)
    replayed = list(replayer)
    assert [c.close for c in replayed] == [c.close for c in candles]
    ppy = estimate_periods_per_year(candles)
    assert 200.0 < ppy < 300.0
    info = describe_candles(candles)
    assert info["bars"] == 43
    assert info["start"] == "2024-11-01" and info["end"] == "2024-12-31"
    assert info["first_close"] == pytest.approx(2.285)
    assert info["last_close"] == pytest.approx(2.239)


# ------------------------------------------------------------------ 引擎回放
def _assert_accounting(result, exchange: PaperExchange, quote: str):
    """三条会计恒等式：净=毛-费、费用与逐笔一致、期末净值=现金+持仓市值。"""
    assert result.check_accounting()
    fees = sum(t.fee for t in result.trades)
    assert fees == pytest.approx(result.total_fees, abs=1e-9)
    identity = exchange.total(quote) + exchange.holdings_value()
    assert result.final_equity == pytest.approx(identity, rel=1e-9, abs=1e-6)
    assert result.equity.notna().all() and (result.equity > 0).all()


def test_engine_replay_real_sample_dca(tmp_path):
    """极小真实样本 + 定投策略：每 5 根买 200，回放跑通且会计自洽。"""
    path = write_csv(tmp_path, REAL_SAMPLE_CSV, "sh518880.csv")
    candles = load_candles(path, symbol="SH518880/CNY")
    strategy = DcaStrategy(symbol="SH518880/CNY", quote_per_buy=200.0, every_n=5)
    exchange = PaperExchange(quote="CNY", initial_quote=10_000.0, fee_rate=0.0005)
    result = PaperTradingEngine(candles, strategy, exchange,
                                periods_per_year=estimate_periods_per_year(candles)).run()
    assert len(result.equity) == len(candles)
    assert strategy.buys == 9                            # index 0,5,...,40
    assert len(result.trades) == 9
    assert result.total_fees > 0
    _assert_accounting(result, exchange, "CNY")


def test_engine_replay_real_sample_momentum_and_grid(tmp_path):
    """极小真实样本 + 动量/网格：均能跑通、净值完整、会计自洽。"""
    path = write_csv(tmp_path, REAL_SAMPLE_CSV, "sh518880.csv")
    candles = load_candles(path, symbol="SH518880/CNY")
    ppy = estimate_periods_per_year(candles)
    strategies = {
        "momentum": MomentumStrategy(symbol="SH518880/CNY", fast=5, slow=20,
                                     stop_pct=0.05, rebalance_band=0.01),
        "grid": GridStrategy(symbol="SH518880/CNY", levels=3, spacing=0.02,
                             qty_per_grid=200.0),
    }
    for strategy in strategies.values():
        exchange = PaperExchange(quote="CNY", initial_quote=10_000.0, fee_rate=0.0005)
        result = PaperTradingEngine(candles, strategy, exchange,
                                    periods_per_year=ppy).run()
        assert len(result.equity) == len(candles)
        _assert_accounting(result, exchange, "CNY")
    assert os.path.basename(path) == "sh518880.csv"      # 样本文件名约定（防误改）
