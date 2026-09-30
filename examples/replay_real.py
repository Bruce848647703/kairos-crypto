"""真实历史 OHLCV 回放纸面交易演示（完全离线，**纯模拟、无真实下单**）。

为什么用「真实历史 bar 回放」而不是加密实时行情？
--------------------------------------------------
当前运行环境**无法访问任何真实加密货币交易所接口**（binance 等公开 REST/WebSocket
端点均被网络阻断），且本仓库铁律禁止任何真实联网抓取与下单。因此本示例用
**真实 A 股/ETF 日线 OHLCV**（本地 kairos-data 只读 CSV，后复权 hfq）作为通用
价格序列，通过 ``load_candles → Replayer/PaperTradingEngine`` 驱动内置策略
（动量 / 网格 / 定投）完成纸面交易，验证「真实数据 → 引擎 → 策略 → 会计」全链路。
若要接入真实加密数据：自行下载历史 K 线 CSV 交给 ``load_candles``，或继承
``Exchange`` 实现实时适配器 —— 策略与引擎代码一行都不用改。

用法::

    python examples/replay_real.py --csv <历史OHLCV.csv>            # 默认三个策略全跑
    python examples/replay_real.py --strategy momentum              # 只跑动量
    python examples/replay_real.py --strategy all --outdir research/real_replay

输出：终端打印各策略关键数字（期末净值/总收益/夏普/最大回撤/成交笔数/手续费），
并在 ``--outdir``（默认 ``research/real_replay/``）写入 REPORT.md、metrics.json、
equity.csv（下采样后的小净值表）。每份结果都做会计自洽校验：
净盈亏 = 毛盈亏 - 手续费、期末净值 = 现金 + 持仓市值、手续费与逐笔成交一致。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kairos_crypto import (                                # noqa: E402
    DcaStrategy,
    GridStrategy,
    MomentumStrategy,
    PaperExchange,
    PaperTradingEngine,
    describe_candles,
    estimate_periods_per_year,
    load_candles,
    parse_symbol,
)

#: 默认真实数据：黄金 ETF（sh518880，后复权日线，只读；来自 kairos-data 项目）
DEFAULT_CSV = "/home/zhuoming.wang/quant-hub/kairos/kairos-data/data/etf/sh518880.csv"
DEFAULT_OUTDIR = os.path.join(ROOT, "research", "real_replay")
ALL_STRATEGIES = ("momentum", "grid", "dca")


# ------------------------------------------------------------------ 参数与策略
def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数（全部离线，无任何网络选项）。"""
    parser = argparse.ArgumentParser(
        description="真实历史 OHLCV 回放纸面交易（纯模拟，无真实下单）")
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help="历史 OHLCV CSV 路径（列: date,open,high,low,close,volume）")
    parser.add_argument("--strategy", default="all",
                        choices=("momentum", "grid", "dca", "all"),
                        help="跑哪个内置策略（默认 all 三个全跑）")
    parser.add_argument("--symbol", default=None,
                        help="交易对标签（默认由文件名推导，如 SH518880/CNY）")
    parser.add_argument("--initial-quote", type=float, default=10_000.0,
                        help="初始计价币资金（默认 10000）")
    parser.add_argument("--fee-rate", type=float, default=0.0005,
                        help="手续费率（默认 0.0005 ≈ 万五，模拟 ETF 佣金量级）")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help="研究产物输出目录（REPORT.md / metrics.json / equity.csv）")
    parser.add_argument("--equity-rows", type=int, default=240,
                        help="equity.csv 下采样后的最大行数（保持文件小）")
    return parser.parse_args(argv)


def build_strategy(name: str, symbol: str, candles, initial_quote: float):
    """按日线真实数据的特点构造内置策略（参数随价格量级/根数自适应，确定性）。"""
    first_close = float(candles[0].close)
    n = len(candles)
    if name == "momentum":
        # 日线双均线 20/60 + 8% 止损 + 2% 调仓带（现货只做多）
        return MomentumStrategy(symbol=symbol, fast=20, slow=60, stop_pct=0.08,
                                rebalance_band=0.02)
    if name == "grid":
        # 中心价 ±2% 等差 4 档，四档底仓合计约占初始资金 40%
        levels = 4
        qty = max(round(0.4 * initial_quote / (levels * first_close), 2), 0.01)
        return GridStrategy(symbol=symbol, levels=levels, spacing=0.02, qty_per_grid=qty)
    if name == "dca":
        # 每 21 个交易日（约每月）定投一次，总额铺开至全历史的 95%
        every_n = 21
        buys = max(1, math.ceil(n / every_n))
        per_buy = round(0.95 * initial_quote / buys, 2)
        return DcaStrategy(symbol=symbol, quote_per_buy=per_buy, every_n=every_n)
    raise ValueError(f"未知策略: {name}")


# ------------------------------------------------------------------ 运行与校验
def run_one(name: str, candles, symbol: str, quote_currency: str,
            initial_quote: float, fee_rate: float, ppy: float):
    """在真实 Candle 序列上跑一个策略的纸面交易，返回 (strategy, exchange, result)。"""
    strategy = build_strategy(name, symbol, candles, initial_quote)
    exchange = PaperExchange(quote=quote_currency, initial_quote=initial_quote,
                             fee_rate=fee_rate)
    engine = PaperTradingEngine(candles, strategy, exchange, periods_per_year=ppy)
    return strategy, exchange, engine.run()


def accounting_checks(result, exchange: PaperExchange, quote_currency: str):
    """会计自洽校验（三条独立恒等式 + 引擎自带 check_accounting）。"""
    fees_from_trades = float(sum(t.fee for t in result.trades))
    holdings = float(exchange.holdings_value())
    identity = float(exchange.total(quote_currency)) + holdings
    scale = max(1.0, abs(identity), abs(result.total_fees))
    checks = {
        "net_pnl_eq_gross_minus_fees": bool(result.check_accounting()),
        "fees_eq_sum_of_trades": abs(fees_from_trades - result.total_fees) <= 1e-9 * scale,
        "final_equity_eq_cash_plus_holdings":
            abs(result.final_equity - identity) <= 1e-6 * scale,
        "equity_curve_positive_and_complete":
            bool(result.equity.notna().all() and (result.equity > 0).all()),
    }
    return checks, all(checks.values())


def strategy_extras(name: str, strategy, exchange: PaperExchange, quote_currency: str):
    """每个策略的补充观察指标（写入 metrics.json 与报告）。"""
    base = parse_symbol(strategy.symbol)[0]
    extras = {"final_base_balance": float(exchange.total(base)),
              "final_cash": float(exchange.total(quote_currency)),
              "open_orders": float(len(exchange.open_orders()))}
    if name == "momentum":
        extras.update({"risk_exits": float(len(strategy.exits)),
                       "end_state": strategy.state})
    elif name == "grid":
        extras.update({"grid_buy_fills": float(strategy.filled_buys),
                       "grid_sell_fills": float(strategy.filled_sells),
                       "grid_cycles": float(strategy.grid_cycles)})
    elif name == "dca":
        extras.update({"dca_buys": float(strategy.buys),
                       "dca_invested": float(strategy.invested),
                       "dca_average_cost": float(strategy.average_cost)})
    return extras


def print_block(name: str, result, checks, ok: bool, quote_currency: str) -> None:
    """打印一个策略的关键数字。"""
    m = result.metrics
    print(f"--- {name} ---")
    print(f"  期初净值   : {result.initial_equity:>14,.2f} {quote_currency}")
    print(f"  期末净值   : {result.final_equity:>14,.2f} {quote_currency}")
    print(f"  总收益率   : {m['total_return']:>14.2%}")
    print(f"  年化收益   : {m['cagr']:>14.2%}")
    print(f"  夏普比率   : {m['sharpe']:>14.2f}")
    print(f"  最大回撤   : {m['max_drawdown']:>14.2%}")
    print(f"  成交笔数   : {int(m['trade_count']):>14d}")
    print(f"  手续费合计 : {result.total_fees:>14,.2f} {quote_currency}")
    print(f"  买入持有   : {m['benchmark_return']:>14.2%}   (基准)")
    print(f"  会计自洽   : {'通过' if ok else '不通过'}   "
          f"(净盈亏 {result.net_pnl:,.2f} = 毛盈亏 {result.gross_pnl:,.2f} "
          f"- 手续费 {result.total_fees:,.2f})")
    for key, passed in checks.items():
        if not passed:
            print(f"    [失败] {key}")


# ------------------------------------------------------------------ 产物落盘
def downsample_history(result, max_rows: int) -> pd.DataFrame:
    """把逐根快照均匀下采样到不超过 ``max_rows`` 行（保留首末行）。"""
    hist = result.history
    n = len(hist)
    step = max(1, -(-n // max(int(max_rows), 2)))
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return hist.iloc[idx]


def build_equity_frame(runs: dict, max_rows: int) -> pd.DataFrame:
    """合并各策略的下采样净值曲线（同一序列回放，日期天然对齐）。"""
    merged = None
    for name, (_strategy, _exchange, result) in runs.items():
        sub = downsample_history(result, max_rows)
        frame = pd.DataFrame({
            "date": sub["time"].dt.strftime("%Y-%m-%d").to_numpy(),
            "close": sub["price"].to_numpy(dtype="float64"),
            f"equity_{name}": sub["equity"].to_numpy(dtype="float64"),
        })
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame[["date", f"equity_{name}"]], on="date", how="outer")
    return merged.sort_values("date").reset_index(drop=True)


def build_report(meta: dict, rows: list, benchmark: float) -> str:
    """生成 REPORT.md 全文（中文，由脚本自动填充真实回放数字）。"""
    lines = [
        "# 真实历史 OHLCV 回放 · 纸面交易报告",
        "",
        "> 本文件由 `python examples/replay_real.py` 自动生成（重跑即可复现），",
        f"> 生成时间（UTC）：{meta['generated_utc']}。",
        "",
        "## 0. 重要声明（先读）",
        "",
        "- 本演示为**纯纸面交易 / 模拟撮合，无任何真实下单**，结果不构成投资建议。",
        "- **加密实时源在当前环境不可达**：binance 等真实加密货币交易所接口在本运行",
        "  环境被网络阻断，且本仓库铁律禁止任何真实联网抓取。因此改用**真实历史",
        "  OHLCV bar 回放**来驱动并验证纸面交易引擎。",
        "- 价格序列来自**真实 A 股/ETF 日线（后复权 hfq）**，仅作为「通用价格序列」",
        "  使用；引擎与策略代码与标的无关。接入真实加密数据的方式：自行下载历史",
        "  K 线 CSV 交给 `load_candles()`，或继承 `Exchange` 实现实时适配器。",
        "",
        "## 1. 数据（只读）",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| CSV 文件 | `{meta['csv']}` |",
        f"| 映射交易对 | `{meta['symbol']}`（文件名作 BASE，{meta['quote']} 作计价币标签） |",
        f"| 区间 | {meta['start']} → {meta['end']} |",
        f"| Bar 数 | {meta['bars']}（日线） |",
        f"| 首末收盘 | {meta['first_close']:.4f} → {meta['last_close']:.4f}"
        f"（买入持有 {benchmark:+.2%}） |",
        "",
        "数据声明：引自 kairos-data 项目 `data/etf/DATA_NOTICE.md` —— 真实跨资产 ETF",
        "日线（后复权），来源为腾讯公开行情接口（web.ifzq.gtimg.cn，由该项目原创代码",
        "抓取，本脚本只读本地文件、不联网）；仅供研究/学习/演示，数据不保证准确完整，",
        "不用于商业用途。",
        "",
        "## 2. 引擎与参数口径",
        "",
        f"- 初始资金 {meta['initial_quote']:,.2f} {meta['quote']}，"
        f"手续费率 {meta['fee_rate']:.4%}（模拟 ETF 佣金量级，未建模印花税/滑点，"
        "`slippage_bps=0`）。",
        f"- 年化期数按真实时间跨度估计 ≈ {meta['periods_per_year']:.1f}"
        "（A 股交易日口径；加密 24/7 小时线数据会自动得到 ≈8760 的口径）。",
        "- 成交时序由 `PaperTradingEngine` 保证防未来函数：市价单按当根收盘价成交，",
        "  限价单挂出后只能由**后续** K 线触及撮合。",
        "",
        "## 3. 各策略纸面交易结果（真实 bar 回放）",
        "",
        "| 策略 | 期末净值 | 总收益 | 年化 | 夏普 | 最大回撤 | 成交笔数 | 手续费 | 会计自洽 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| {row['name']} | {row['final_equity']:,.2f} | {m['total_return']:+.2%} "
            f"| {m['cagr']:+.2%} | {m['sharpe']:.2f} | {m['max_drawdown']:.2%} "
            f"| {int(m['trade_count'])} | {row['total_fees']:,.2f} "
            f"| {'通过' if row['accounting_ok'] else '不通过'} |")
    lines += [
        "",
        f"买入持有基准（同区间收盘价涨幅）：**{benchmark:+.2%}**。",
        "",
    ]
    for row in rows:
        lines += [f"### {row['name']} 补充观察", ""]
        for key, value in row["extras"].items():
            if isinstance(value, float):
                value = 0.0 if abs(value) < 1e-9 else value    # 抹掉 -0.0000 浮点残差
                shown = f"{value:,.4f}"
            else:
                shown = str(value)
            lines.append(f"- {key}: {shown}")
        lines.append("")
    lines += [
        "## 4. 会计自洽校验",
        "",
        "每个策略独立验证四条恒等式（全部通过才记「通过」）：",
        "",
        "1. `净盈亏 = 毛盈亏 - 手续费合计`（`PaperTradingResult.check_accounting`）；",
        "2. `手续费合计 = Σ 逐笔成交 fee`；",
        "3. `期末净值 = 计价币余额（含冻结） + 持仓市值`；",
        "4. 净值曲线无 NaN 且恒为正。",
        "",
        "| 策略 | " + " | ".join(meta["check_names"]) + " |",
        "|---|" + "---|" * len(meta["check_names"]),
    ]
    for row in rows:
        cells = " | ".join("通过" if row["checks"][key] else "失败"
                          for key in meta["check_names"])
        lines.append(f"| {row['name']} | {cells} |")
    lines += [
        "",
        "## 5. 复现方式",
        "",
        "```bash",
        "cd kairos-crypto",
        "python3 -m pytest -q                      # 离线测试（含 tests/test_realdata.py）",
        f"python3 examples/replay_real.py --csv {meta['csv']}",
        "```",
        "",
        "产物：本报告、`metrics.json`（全部数字）、`equity.csv`（下采样净值曲线，",
        f"≤{meta['equity_rows']} 行）。",
        "",
        "## 6. 局限性",
        "",
        "- 日线收盘价近似成交，未建模盘中流动性、ETF 折溢价、分红税与最小申报单位；",
        "- 网格策略的中心价锚定在**首根 bar**，在长期单边趋势中底仓会被早期卖光、",
        "  网格随之失效（这是策略特性使然，报告中如实呈现，不做美化）；",
        "- 纸面撮合假设限价单被触及即全额成交，无排队与部分成交；",
        "- 再次强调：**加密实时源在本环境不可达，本报告用真实历史 bar 回放演示引擎**，",
        "  全部结果仅为模拟，不构成投资建议。",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ 主流程
def main(argv=None) -> int:
    """加载真实 CSV → 逐策略回放 → 打印关键数字 → 落盘研究产物。"""
    args = parse_args(argv)
    if not os.path.isfile(args.csv):
        print(f"[错误] 找不到 CSV: {args.csv}\n"
              "       请用 --csv 指定任意历史 OHLCV 文件"
              "（列: date,open,high,low,close,volume）。")
        return 2

    candles = load_candles(args.csv, symbol=args.symbol)
    symbol = candles[0].symbol
    quote_currency = parse_symbol(symbol)[1]
    info = describe_candles(candles)
    ppy = estimate_periods_per_year(candles)
    names = ALL_STRATEGIES if args.strategy == "all" else (args.strategy,)

    print("=" * 72)
    print("真实历史 OHLCV 回放 · 纸面交易（纯模拟，无真实下单，全程离线）")
    print("=" * 72)
    print(f"数据文件   : {args.csv}")
    print(f"映射交易对 : {symbol}    Bar 数: {info['bars']}（日线）")
    print(f"区间       : {info['start']} → {info['end']}")
    print(f"价格       : {info['first_close']:.4f} → {info['last_close']:.4f} "
          f"({info['buy_hold_return']:+.2%}，买入持有基准)")
    print(f"年化期数   : {ppy:.1f}（按真实时间跨度估计）")
    print("说明       : 加密实时源在本环境不可达（网络阻断），故用真实历史 bar 回放")
    print("             演示引擎；可自行接入真实加密历史/实时数据，代码无需改动。")

    runs = {}
    rows = []
    all_ok = True
    for name in names:
        print("=" * 72)
        strategy, exchange, result = run_one(
            name, candles, symbol, quote_currency, args.initial_quote,
            args.fee_rate, ppy)
        checks, ok = accounting_checks(result, exchange, quote_currency)
        all_ok = all_ok and ok
        print_block(name, result, checks, ok, quote_currency)
        runs[name] = (strategy, exchange, result)
        extras = strategy_extras(name, strategy, exchange, quote_currency)
        rows.append({
            "name": name,
            "final_equity": float(result.final_equity),
            "total_fees": float(result.total_fees),
            "net_pnl": float(result.net_pnl),
            "gross_pnl": float(result.gross_pnl),
            "metrics": {k: float(v) for k, v in result.metrics.items()},
            "checks": checks,
            "accounting_ok": bool(ok),
            "extras": extras,
        })

    os.makedirs(args.outdir, exist_ok=True)
    meta = {
        "csv": os.path.abspath(args.csv),
        "symbol": symbol,
        "quote": quote_currency,
        "bars": int(info["bars"]),
        "start": info["start"],
        "end": info["end"],
        "first_close": float(info["first_close"]),
        "last_close": float(info["last_close"]),
        "buy_hold_return": float(info["buy_hold_return"]),
        "initial_quote": float(args.initial_quote),
        "fee_rate": float(args.fee_rate),
        "periods_per_year": float(ppy),
        "strategies": list(names),
        "equity_rows": int(args.equity_rows),
        "generated_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S"),
        "generator": "examples/replay_real.py",
        "disclaimer": ("纯纸面交易/模拟撮合，无真实下单；加密实时源在当前环境不可达，"
                       "故以真实 A 股/ETF 历史 OHLCV（后复权）回放演示引擎。"
                       "结果不构成投资建议。"),
        "check_names": ["net_pnl_eq_gross_minus_fees", "fees_eq_sum_of_trades",
                        "final_equity_eq_cash_plus_holdings",
                        "equity_curve_positive_and_complete"],
    }

    metrics_path = os.path.join(args.outdir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump({"meta": meta, "strategies": {row["name"]: row for row in rows}},
                  handle, ensure_ascii=False, indent=2)

    equity_frame = build_equity_frame(runs, args.equity_rows)
    equity_path = os.path.join(args.outdir, "equity.csv")
    equity_frame.to_csv(equity_path, index=False, float_format="%.6f")

    report_path = os.path.join(args.outdir, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(build_report(meta, rows, float(info["buy_hold_return"])))

    print("=" * 72)
    print(f"产物已写入: {args.outdir}")
    print(f"  - REPORT.md    ({os.path.getsize(report_path):,} 字节)")
    print(f"  - metrics.json ({os.path.getsize(metrics_path):,} 字节)")
    print(f"  - equity.csv   ({len(equity_frame)} 行，下采样净值曲线)")
    print(f"会计自洽: {'全部通过' if all_ok else '存在失败项，请检查上方明细'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
