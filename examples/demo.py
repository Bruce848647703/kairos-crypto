"""Kairos Crypto 演示：合成 BTC 行情 + 纸面交易（完全离线、固定 seed 可复现）。

运行： python examples/demo.py

内容：
① 动量策略（均线 + 动量确认）跑纸面交易引擎，打印期末净值/总收益/夏普/成交笔数/手续费
② 网格策略与定投策略的同台对比，并做一次会计自洽校验

声明：全过程只在内存中撮合，**不连接任何真实交易所**。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kairos_crypto import (
    DcaStrategy,
    GridStrategy,
    MomentumStrategy,
    PaperExchange,
    PaperTradingEngine,
    SyntheticCandles,
)

SYMBOL = "BTC/USDT"
SEED = 22


def indent(text, prefix="    "):
    """给多行文本统一加缩进，便于对齐打印。"""
    return "\n".join(prefix + line for line in text.splitlines())


def run_once(candles, strategy, quote=10_000.0, fee_rate=0.001):
    """用给定的策略跑一遍纸面交易，返回结果对象。"""
    exchange = PaperExchange(quote="USDT", initial_quote=quote, fee_rate=fee_rate)
    engine = PaperTradingEngine(candles, strategy, exchange)
    return engine.run(), exchange


def report(result, title):
    """打印一份紧凑的绩效报告。"""
    metrics = result.metrics
    print(f"--- {title} ---")
    print(f"  期初净值   : {result.initial_equity:>14,.2f} USDT")
    print(f"  期末净值   : {result.final_equity:>14,.2f} USDT")
    print(f"  总收益率   : {metrics['total_return']:>14.2%}")
    print(f"  年化收益   : {metrics['cagr']:>14.2%}")
    print(f"  年化波动   : {metrics['volatility']:>14.2%}")
    print(f"  夏普比率   : {metrics['sharpe']:>14.2f}")
    print(f"  最大回撤   : {metrics['max_drawdown']:>14.2%}")
    print(f"  成交笔数   : {int(metrics['trade_count']):>14d}")
    print(f"  手续费合计 : {result.total_fees:>14,.2f} USDT")
    print(f"  已实现盈亏 : {result.realized_pnl:>14,.2f} USDT")
    print(f"  浮动盈亏   : {result.unrealized_pnl:>14,.2f} USDT")
    print(f"  期末持仓   : {result.final_position:>14.6f} BTC")
    print(f"  买入持有   : {metrics['benchmark_return']:>14.2%}   (基准)")
    print(f"  会计自洽   : {'通过' if result.check_accounting() else '不通过'}"
          f"   (净盈亏 {result.net_pnl:,.2f} = 毛盈亏 {result.gross_pnl:,.2f} "
          f"- 手续费 {result.total_fees:,.2f})")


def main():
    generator = SyntheticCandles(symbol=SYMBOL, n=1500, interval="1h",
                                 start_price=30_000.0, seed=SEED)
    candles = generator.candles
    first, last = candles[0], candles[-1]
    print("=" * 72)
    print("Kairos Crypto 纸面交易演示（合成行情，无网络、无真实下单）")
    print("=" * 72)
    print(f"标的: {SYMBOL}    K 线: {len(candles)} 根 1h（24/7 连续）")
    print(f"区间: {first.datetime:%Y-%m-%d %H:%M} → {last.datetime:%Y-%m-%d %H:%M} UTC")
    print(f"价格: {first.close:,.2f} → {last.close:,.2f} USDT "
          f"({last.close / first.close - 1:+.2%})")
    print(f"合成行情年化波动: {generator.realized_vol():.2%}"
          f"    随机种子: {SEED}（固定，可复现）")

    print("=" * 72)
    print("① 动量策略（fast=12 / slow=48 均线 + 区间动量确认，5% 止损）")
    momentum = MomentumStrategy(symbol=SYMBOL, fast=12, slow=48, stop_pct=0.05,
                                rebalance_band=0.02)
    result, exchange = run_once(candles, momentum)
    report(result, "MomentumStrategy")
    print(f"  风控触发次数: {len(momentum.exits)}   当前状态: {momentum.state}")
    trades = result.trades_frame()
    if not trades.empty:
        cols = ["side", "price", "quantity", "fee", "gross_value"]
        print("  最近 5 笔成交:")
        print(indent(trades[cols].tail(5).to_string(index=False)))

    print("=" * 72)
    print("② 网格策略（中心价 ±2% 等差 4 档，每档 0.005 BTC，成交后自动补挂）")
    grid = GridStrategy(symbol=SYMBOL, levels=4, spacing=0.02, qty_per_grid=0.005)
    grid_result, grid_exchange = run_once(candles, grid)
    report(grid_result, "GridStrategy")
    print(f"  网格成交: 买档 {grid.filled_buys} 次 / 卖档 {grid.filled_sells} 次，"
          f"完成 {grid.grid_cycles} 轮低买高卖")
    print(f"  期末挂单: {len(grid_exchange.open_orders())} 笔；仍在挂的档位:")
    resting = grid.grid_frame().query("status == 'new'")
    if not resting.empty:
        print(indent(resting[["level", "price", "side", "quantity"]].to_string(index=False)))

    print("=" * 72)
    print("③ 定投策略（每 24 根 1h K 线投入 100 USDT）")
    dca = DcaStrategy(symbol=SYMBOL, quote_per_buy=100.0, every_n=24)
    dca_result, dca_exchange = run_once(candles, dca)
    report(dca_result, "DcaStrategy")
    print(f"  定投次数: {dca.buys}   累计投入: {dca.invested:,.2f} USDT   "
          f"定投均价: {dca.average_cost:,.2f} USDT/BTC")
    print(f"  期末币量: {dca_exchange.total('BTC'):.6f} BTC   "
          f"剩余现金: {dca_exchange.free('USDT'):,.2f} USDT")

    print("=" * 72)
    print("提示：以上全部为**纸面交易**结果。若要接入真实交易所，")
    print("      请自行继承 kairos_crypto.Exchange 实现适配器（本仓库不含任何网络代码）。")


if __name__ == "__main__":
    main()
