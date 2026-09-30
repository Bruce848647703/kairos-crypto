# 真实历史 OHLCV 回放 · 纸面交易报告

> 本文件由 `python examples/replay_real.py` 自动生成（重跑即可复现），
> 生成时间（UTC）：2026-09-30 01:17:05。

## 0. 重要声明（先读）

- 本演示为**纯纸面交易 / 模拟撮合，无任何真实下单**，结果不构成投资建议。
- **加密实时源在当前环境不可达**：binance 等真实加密货币交易所接口在本运行
  环境被网络阻断，且本仓库铁律禁止任何真实联网抓取。因此改用**真实历史
  OHLCV bar 回放**来驱动并验证纸面交易引擎。
- 价格序列来自**真实 A 股/ETF 日线（后复权 hfq）**，仅作为「通用价格序列」
  使用；引擎与策略代码与标的无关。接入真实加密数据的方式：自行下载历史
  K 线 CSV 交给 `load_candles()`，或继承 `Exchange` 实现实时适配器。

## 1. 数据（只读）

| 项 | 值 |
|---|---|
| CSV 文件 | `/home/zhuoming.wang/quant-hub/kairos/kairos-data/data/etf/sh518880.csv` |
| 映射交易对 | `SH518880/CNY`（文件名作 BASE，CNY 作计价币标签） |
| 区间 | 2013-07-29 → 2026-09-24 |
| Bar 数 | 3202（日线） |
| 首末收盘 | 0.9920 → 3.3190（买入持有 +234.58%） |

数据声明：引自 kairos-data 项目 `data/etf/DATA_NOTICE.md` —— 真实跨资产 ETF
日线（后复权），来源为腾讯公开行情接口（web.ifzq.gtimg.cn，由该项目原创代码
抓取，本脚本只读本地文件、不联网）；仅供研究/学习/演示，数据不保证准确完整，
不用于商业用途。

## 2. 引擎与参数口径

- 初始资金 10,000.00 CNY，手续费率 0.0500%（模拟 ETF 佣金量级，未建模印花税/滑点，`slippage_bps=0`）。
- 年化期数按真实时间跨度估计 ≈ 243.2（A 股交易日口径；加密 24/7 小时线数据会自动得到 ≈8760 的口径）。
- 成交时序由 `PaperTradingEngine` 保证防未来函数：市价单按当根收盘价成交，
  限价单挂出后只能由**后续** K 线触及撮合。

## 3. 各策略纸面交易结果（真实 bar 回放）

| 策略 | 期末净值 | 总收益 | 年化 | 夏普 | 最大回撤 | 成交笔数 | 手续费 | 会计自洽 |
|---|---|---|---|---|---|---|---|---|
| momentum | 22,531.96 | +125.32% | +6.36% | 0.57 | 32.48% | 103 | 672.04 | 通过 |
| grid | 11,660.98 | +16.61% | +1.17% | 0.31 | 12.07% | 139 | 70.99 | 通过 |
| dca | 24,997.59 | +149.98% | +7.21% | 0.70 | 29.77% | 153 | 4.75 | 通过 |

买入持有基准（同区间收盘价涨幅）：**+234.58%**。

### momentum 补充观察

- final_base_balance: 6,788.7791
- final_cash: 0.0000
- open_orders: 0.0000
- risk_exits: 0.0000
- end_state: long

### grid 补充观察

- final_base_balance: 0.0000
- final_cash: 11,660.9788
- open_orders: 8.0000
- grid_buy_fills: 67.0000
- grid_sell_fills: 71.0000
- grid_cycles: 71.0000

### dca 补充观察

- final_base_balance: 7,380.9446
- final_cash: 500.2300
- open_orders: 0.0000
- dca_buys: 153.0000
- dca_invested: 9,499.7700
- dca_average_cost: 1.2871

## 4. 会计自洽校验

每个策略独立验证四条恒等式（全部通过才记「通过」）：

1. `净盈亏 = 毛盈亏 - 手续费合计`（`PaperTradingResult.check_accounting`）；
2. `手续费合计 = Σ 逐笔成交 fee`；
3. `期末净值 = 计价币余额（含冻结） + 持仓市值`；
4. 净值曲线无 NaN 且恒为正。

| 策略 | net_pnl_eq_gross_minus_fees | fees_eq_sum_of_trades | final_equity_eq_cash_plus_holdings | equity_curve_positive_and_complete |
|---|---|---|---|---|
| momentum | 通过 | 通过 | 通过 | 通过 |
| grid | 通过 | 通过 | 通过 | 通过 |
| dca | 通过 | 通过 | 通过 | 通过 |

## 5. 复现方式

```bash
cd kairos-crypto
python3 -m pytest -q                      # 离线测试（含 tests/test_realdata.py）
python3 examples/replay_real.py --csv /home/zhuoming.wang/quant-hub/kairos/kairos-data/data/etf/sh518880.csv
```

产物：本报告、`metrics.json`（全部数字）、`equity.csv`（下采样净值曲线，
≤240 行）。

## 6. 局限性

- 日线收盘价近似成交，未建模盘中流动性、ETF 折溢价、分红税与最小申报单位；
- 网格策略的中心价锚定在**首根 bar**，在长期单边趋势中底仓会被早期卖光、
  网格随之失效（这是策略特性使然，报告中如实呈现，不做美化）；
- 纸面撮合假设限价单被触及即全额成交，无排队与部分成交；
- 再次强调：**加密实时源在本环境不可达，本报告用真实历史 bar 回放演示引擎**，
  全部结果仅为模拟，不构成投资建议。
