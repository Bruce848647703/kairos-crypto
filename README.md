# Kairos Crypto

[![CI](https://github.com/Bruce848647703/kairos-crypto/actions/workflows/ci.yml/badge.svg)](https://github.com/Bruce848647703/kairos-crypto/actions/workflows/ci.yml)

> Kairos 量化系列的加密货币模块 —— 一个**自研、轻量、纯离线**的加密量化库：
> **交易所无关抽象（Exchange）+ 纸面交易撮合（PaperExchange）**。

`kairos_crypto` 提供一整套「行情 → 策略 → 撮合 → 记账 → 绩效」的纸面交易流水线，
用来研究、教学与验证加密货币策略（动量 / 网格 / 定投 / 波动率目标仓位）。
核心代码全部原创，只依赖 `numpy` 与 `pandas`，测试与示例全程离线、固定随机种子可复现。

---

## ⚠️ 重要声明：纸面交易 / 模拟撮合

**本项目是纯模拟（paper trading）框架，不包含任何真实交易所连接、鉴权、行情抓取或下单代码。**

- 仓库中**没有**任何网络请求、WebSocket、REST 客户端、API Key / Secret 相关代码；
  `tests/test_offline.py` 会用静态扫描 + 运行时 `socket` 拦截来强制这一约束。
- 所有撮合都发生在内存中的 `PaperExchange`，行情来自 `SyntheticCandles` 合成数据
  （或用户自己提供的历史 K 线）。
- **若要接入真实交易所，请自行继承 `Exchange` 实现适配器**（网络、密钥管理、限频重试、
  实盘风控与合规责任全部由使用者承担）。本仓库只保证接口清晰、语义明确、易于替换。
- 本项目的输出**不构成任何投资建议**，模拟结果与实盘必然存在差异（深度、滑点、
  资金费率、撮合优先级、延迟等均未建模）。

---

## 特性

- **交易所无关抽象 `Exchange`**：`fetch_candles / fetch_ticker / create_order / cancel_order /
  fetch_balance / fetch_orders` 六个方法即可描述一个现货交易所；基类另附
  `market_buy / limit_sell / fetch_open_orders` 等便捷方法。
- **自研内存撮合 `PaperExchange`**：余额与冻结（预占/退款）、市价单与限价单撮合、
  maker/taker 双费率、市价单滑点、成交回报 `Trade`、持仓均价与已实现盈亏、
  余额不足时市价单自动缩量、限价单拒单，**零网络**。
- **确定性合成行情 `SyntheticCandles`**：24/7 连续时间戳、两状态马尔可夫切换的
  「趋势 / 震荡」段、肥尾跳跃、成交量随波动放大；同 `seed` 两次生成完全一致。
- **回放器 `Replayer`**：按时间顺序逐根喂给策略与交易所，`history()/closes()` 只返回
  **已回放**的数据，从接口层面杜绝未来函数。
- **三个自研示例策略**：`MomentumStrategy`（双均线 + 区间动量确认 + 止损/止盈/移动止损）、
  `GridStrategy`（等差网格限价单 + 成交后自动反向补挂 + 底仓建立）、
  `DcaStrategy`（定期定额买入，含资金耗尽与止损停机）。
- **纸面交易引擎 `PaperTradingEngine`**：逐 K 线推进「撮合 → 策略 → 记账」，
  输出净值曲线、逐根快照（现金/持仓/累计手续费/已实现与浮动盈亏）、
  成交与委托表、绩效指标（总收益、年化、波动、夏普、索提诺、最大回撤），
  并提供**会计自洽校验**。
- **仓位与风控 `risk`**：`fixed_fractional_size`（固定比例）、
  `volatility_target_size`（波动率目标，波动越大仓位越小）、`kelly_fraction`、
  止损/止盈/移动止损触发判定与 `check_exits` 优先级、数量步长对齐等纯函数。
- **克制依赖**：必需依赖只有 `numpy`、`pandas`；Python 3.8/3.9 兼容；全中文 docstring。

## 安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # 或 pip install numpy pandas
pip install -e ".[dev]"     # 需要跑测试时
```

## 快速开始

### ① 三步跑通纸面交易

```python
from kairos_crypto import SyntheticCandles, MomentumStrategy, PaperTradingEngine, PaperExchange

candles  = SyntheticCandles(symbol="BTC/USDT", n=1500, interval="1h", seed=22).candles
exchange = PaperExchange(quote="USDT", initial_quote=10_000, fee_rate=0.001)
result   = PaperTradingEngine(candles, MomentumStrategy(fast=12, slow=48, stop_pct=0.05),
                              exchange).run()

print(result.metrics["total_return"], result.metrics["sharpe"], result.total_fees)
print(result.check_accounting())        # 净盈亏 == 毛盈亏 - 手续费
print(result.summary_frame().to_string(index=False))
```

### ② 网格与定投

```python
from kairos_crypto import GridStrategy, DcaStrategy

grid = GridStrategy(symbol="BTC/USDT", levels=4, spacing=0.02, qty_per_grid=0.005)
dca  = DcaStrategy(symbol="BTC/USDT", quote_per_buy=100.0, every_n=24)   # 每天定投 100 USDT
```

### ③ 自定义策略

```python
from kairos_crypto import Strategy

class MyStrategy(Strategy):
    """只做多：20 根 K 线收益率 > 2% 时半仓，< -2% 时清仓。"""

    def on_candle(self, candle, ctx):
        closes = ctx.closes(21)                 # 只含「截至当根」的数据
        if closes.size < 21:
            return
        roc = closes[-1] / closes[0] - 1.0
        ctx.order_target_percent(0.5 if roc > 0.02 else 0.0, candle.symbol)
```

### ④ 接入自己的交易所适配器（网络部分请自行实现）

```python
from kairos_crypto import Exchange

class MyExchangeAdapter(Exchange):
    """把统一接口翻译成你自己的交易所客户端调用。"""

    name = "my-exchange"

    def fetch_candles(self, symbol, interval="1h", limit=500, since=None):
        raise NotImplementedError("在此调用你的行情接口，并转成 kairos_crypto.Candle")

    def fetch_ticker(self, symbol):
        raise NotImplementedError

    def create_order(self, symbol, side, quantity, order_type, price=None, client_id=None):
        raise NotImplementedError

    def cancel_order(self, order_id, symbol=None):
        raise NotImplementedError

    def fetch_balance(self, currency=None):
        raise NotImplementedError

    def fetch_orders(self, symbol=None, status=None):
        raise NotImplementedError
```

先用自己的历史 K 线 + `Replayer` 在 `PaperExchange` 上验证策略，再替换成真实适配器，
策略代码一行都不用改。

完整可运行示例见 [`examples/demo.py`](examples/demo.py)。

## 历史回放（真实数据演示）

> ⚠️ **纯模拟 / 无真实下单；加密实时源在当前环境不可达。**
> binance 等真实加密货币交易所接口在本运行环境被网络阻断，且本仓库铁律禁止任何
> 真实联网抓取。因此 `kairos_crypto/realdata.py` + `examples/replay_real.py` 改用
> **真实 A 股/ETF 历史日线 OHLCV**（本地只读 CSV，后复权 hfq）作为「通用价格序列」，
> 回放驱动纸面交易引擎完成演示。用户可自行准备**真实加密历史 K 线 CSV**（或继承
> `Exchange` 实现实时适配器）喂给 `load_candles()`，引擎与策略代码一行都不用改。

```bash
# 默认真实数据：黄金 ETF sh518880（来自 kairos-data 项目，只读）
python examples/replay_real.py --csv /path/to/kairos-data/data/etf/sh518880.csv
python examples/replay_real.py --csv <任意OHLCV.csv> --strategy momentum   # 只跑一个策略
```

- `load_candles(csv_path, symbol=None)`：真实 OHLCV CSV（`date,open,high,low,close,volume`，
  兼容常见列名别名与 epoch 时间戳列）→ **升序** `Candle` 列表；自动丢弃 NaN/无效行、
  去重日期，并对不满足 `low ≤ open/close ≤ high` 的行做最小修复。
  `symbol` 缺省由文件名推导（如 `SH518880/CNY`，加密数据可传 `quote="USDT"`）。
- `replayer_from_csv()` 直接产出 `Replayer`；`estimate_periods_per_year()` 按真实时间
  跨度折算年化期数（A 股日线 ≈ 243，加密 24/7 小时线自动 ≈ 8760）。
- 脚本对每个策略（momentum/grid/dca）输出期末净值、总收益、夏普、最大回撤、成交笔数、
  手续费，并做**会计自洽校验**（净盈亏 = 毛盈亏 − 手续费、期末净值 = 现金 + 持仓市值等），
  结果写入 `research/real_replay/`（REPORT.md + metrics.json + equity.csv）。
- **数据声明**：演示数据只读引用 `kairos-data` 项目 `data/etf/`（真实 ETF 日线后复权，
  来源为腾讯公开行情接口，由其原创代码抓取；仅供研究/学习/演示，数据不保证准确完整，
  结果不构成投资建议）。本仓库对数据只读、不抓取、不联网。

## API 概览

| 模块 | 关键对象 | 说明 |
|---|---|---|
| `types` | `Candle` `Ticker` `Side` `OrderType` `OrderStatus` `Order` `Trade` `Balance` `SpotPosition` `parse_symbol` | 交易所无关的数据类型与会计口径 |
| `exchange` | `Exchange`（抽象）`PaperExchange`（内存模拟） | 行情/下单/账户接口 + 纸面撮合 |
| `data` | `SyntheticCandles` `Replayer` `make_candles` `candles_to_frame` `candles_from_frame` `interval_seconds` `periods_per_year` | 确定性合成行情与回放 |
| `realdata` | `load_candles` `load_ohlcv_frame` `replayer_from_csv` `symbol_from_path` `estimate_periods_per_year` `describe_candles` | 本地真实历史 OHLCV CSV 的离线加载 → `Candle`/`Replayer` |
| `strategy` | `Strategy` `Context` `MomentumStrategy` `GridStrategy` `DcaStrategy` | 策略基类、运行时上下文与示例策略 |
| `engine` | `PaperTradingEngine` `PaperTradingResult` `performance_summary` `sharpe_ratio` `max_drawdown` `total_return` `cagr` | 纸面交易主循环与绩效 |
| `risk` | `fixed_fractional_size` `volatility_target_size` `realized_volatility` `kelly_fraction` `check_exits` `stop_loss_triggered` `take_profit_triggered` `trailing_stop_triggered` `quote_to_quantity` `align_step` | 仓位管理与风控判定 |

## 设计要点

### 时序与防未来函数

主循环严格三步（见 `engine.py`）：

```
for i, candle in enumerate(replayer):
    1) exchange.on_candle(candle)      # 更新最新价，并撮合「此前」挂出的限价单
    2) strategy.on_candle(candle, ctx) # 策略决策
    3) 记录净值快照（现金 + 持仓市值，已扣当根手续费）
```

- 策略在第 `i` 根只能通过 `ctx.closes()/ctx.history()` 看到第 `0..i` 根数据；
- **市价单**以第 `i` 根收盘价（可叠加滑点）立即成交；
- **限价单**在第 `i` 根挂出后，只能由第 `i+1` 根及以后的 K 线撮合
  —— 即便第 `i` 根的最低价已经「触及」委托价，也不会当根成交。
  这条规则由 `tests/test_engine.py::test_limit_order_placed_on_bar_never_fills_on_same_bar` 守护。

### 纸面撮合规则

| 情形 | 规则 |
|---|---|
| 市价单 | 以最近观测价成交，可叠加 `slippage_bps`（买入抬价、卖出压价）；手续费 = 成交额 × 费率，等效成交价 = `价格 × (1 ± 费率)` |
| 限价买单 | 后续 K 线 `low <= 限价` 时以**限价**成交；若 `open <= 限价`（跳空穿越）则以更优的**开盘价**成交 |
| 限价卖单 | 后续 K 线 `high >= 限价` 时以**限价**成交；若 `open >= 限价` 则以开盘价成交 |
| 挂出即穿越 | 限价单挂出时若最新价已优于委托价，按吃单价立即成交 |
| 资金预占 | 限价买单冻结 `限价 × 数量 × (1 + 费率)`，限价卖单冻结基础币；成交后差额退回，撤单全额解冻 |
| 余额不足 | 市价单缩量到「可成交量」（`clamp_to_balance=True`），限价单直接 `REJECTED`（与现货交易所一致） |
| 现货语义 | 默认不允许做空（`allow_short=False`），买入不透支 |

撮合假设流动性充足：限价单被触及即全额成交，不建模盘口深度、排队优先级与部分成交。

### 会计恒等式（可用 `result.check_accounting()` 校验）

```
净盈亏 = 期末净值 - 期初净值
毛盈亏 = Σ 成交现金流(不含手续费, 买负卖正) + (期末持仓市值 - 期初持仓市值)
净盈亏 = 毛盈亏 - 手续费合计
```

若期末已清仓，则进一步有 `净盈亏 = 已实现盈亏 - 手续费合计`。
`PaperTradingResult` 同时暴露 `gross_pnl / total_fees / realized_pnl / unrealized_pnl /
net_pnl`，便于逐笔核对，测试对这三条恒等式都有断言。

### 年化口径（24/7）

加密市场全年无休，因此年化期数按 `365 × 86400 / K线秒数` 计算：
`1h → 8760`、`15m → 35040`、`1d → 365`。引擎会用 `infer_periods_per_year` 自动推断，
也可以显式传入 `periods_per_year` 覆盖。

### 仓位与风控

- `fixed_fractional_size(equity, fraction, price)`：固定比例仓位（净值的 x%）。
- `volatility_target_size(equity, returns, target_vol, price, periods_per_year, max_leverage)`：
  杠杆 = `min(目标波动 / 已实现波动, 上限)`，**波动越大仓位越小**；
  `MomentumStrategy(vol_target=...)` 直接复用它来缩放敞口。
- `check_exits(entry, price, stop_pct, take_pct, highest_price, trail_pct)`：
  固定止损 > 移动止损 > 止盈的优先级判定，返回触发原因字符串。

## 测试

```bash
make test          # 或 python -m pytest -q
```

覆盖：市价单成交价与余额扣减、限价单触及/未触及、挂单预占与撤单解冻、拒单与缩量、
网格阶梯（价格等距、数量一致）与自动补挂、定投周期与累计持仓、
引擎净值长度与会计恒等式、防未来函数、合成行情确定性与 OHLC 约束、
仓位函数单调性、风控触发边界，以及**离线约束**（源码静态扫描 + 运行时 socket 拦截）。

## 项目结构

```
kairos_crypto/           核心包（types / exchange / data / realdata / strategy / engine / risk）
examples/demo.py         可直接运行的纸面交易演示（合成行情）
examples/replay_real.py  真实历史 OHLCV 回放纸面交易演示（离线，写 research/real_replay/）
tests/                   pytest 测试（含离线约束、真实样本回放与测试工具 helpers.py）
research/real_replay/    回放研究产物（REPORT.md / metrics.json / equity.csv，入库）
```

## 许可

MIT © 2026 Bruce848647703，见 [LICENSE](LICENSE)。

## 参考与致谢

本项目为**独立原创实现**，未复制任何第三方代码。设计思路受业界通用的加密量化范式
（交易所无关的统一接口抽象、纸面/模拟撮合与余额冻结会计、网格与定投等被动策略、
均线与动量趋势跟随、波动率目标仓位与止损止盈风控、防未来函数的事件驱动回放）启发，
在此向开源量化社区致谢。算法与接口均为本仓库自研。
