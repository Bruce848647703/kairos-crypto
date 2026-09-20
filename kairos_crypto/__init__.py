"""Kairos Crypto —— 自研轻量加密货币量化库（**纸面交易 / 模拟撮合**）。

本包提供一套「交易所无关」的抽象与一个完全离线的内存模拟交易所，
用于研究、教学与策略验证：

- :mod:`kairos_crypto.types`    基础数据类型（Candle / Ticker / Order / Trade / Balance …）
- :mod:`kairos_crypto.exchange` 交易所抽象基类 ``Exchange`` + 纸面实现 ``PaperExchange``
- :mod:`kairos_crypto.data`     确定性合成行情 ``SyntheticCandles`` 与回放器 ``Replayer``
- :mod:`kairos_crypto.strategy` 策略基类 ``Strategy`` 与示例（动量 / 网格 / 定投）
- :mod:`kairos_crypto.engine`   ``PaperTradingEngine`` 与净值/绩效/会计核对
- :mod:`kairos_crypto.risk`     仓位管理（固定比例、波动率目标）与止损止盈判定

重要声明
--------
本项目**只做纸面交易与模拟撮合，不含任何真实交易所连接、鉴权或下单代码**，
测试与示例全程离线、固定随机种子可复现。若要接入真实交易所，
请自行继承 :class:`Exchange` 实现适配器（网络与密钥管理由使用者负责）。
"""
from .data import (
    DEFAULT_START_TS,
    INTERVAL_SECONDS,
    Replayer,
    SyntheticCandles,
    candles_from_frame,
    candles_to_frame,
    interval_seconds,
    make_candles,
    periods_per_year,
    symbols_of,
)
from .engine import (
    PaperTradingEngine,
    PaperTradingResult,
    annualized_volatility,
    cagr,
    drawdown_series,
    equity_returns,
    infer_periods_per_year,
    max_drawdown,
    performance_summary,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)
from .exchange import Exchange, PaperExchange
from .risk import (
    EXIT_STOP_LOSS,
    EXIT_TAKE_PROFIT,
    EXIT_TRAILING_STOP,
    align_step,
    check_exits,
    fixed_fractional_notional,
    fixed_fractional_size,
    kelly_fraction,
    quantity_to_quote,
    quote_to_quantity,
    realized_volatility,
    stop_loss_price,
    stop_loss_triggered,
    take_profit_price,
    take_profit_triggered,
    trailing_stop_price,
    trailing_stop_triggered,
    volatility_target_notional,
    volatility_target_size,
)
from .strategy import (
    Context,
    DcaStrategy,
    GridStrategy,
    MomentumStrategy,
    Strategy,
)
from .types import (
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

__version__ = "0.1.0"

__all__ = [
    # types
    "Candle", "Ticker", "Side", "OrderType", "OrderStatus", "Order", "Trade",
    "Balance", "SpotPosition", "parse_symbol", "join_symbol",
    # exchange
    "Exchange", "PaperExchange",
    # data
    "SyntheticCandles", "Replayer", "make_candles", "candles_to_frame",
    "candles_from_frame", "symbols_of", "interval_seconds", "periods_per_year",
    "INTERVAL_SECONDS", "DEFAULT_START_TS",
    # strategy
    "Strategy", "Context", "MomentumStrategy", "GridStrategy", "DcaStrategy",
    # engine
    "PaperTradingEngine", "PaperTradingResult", "performance_summary", "equity_returns",
    "total_return", "cagr", "annualized_volatility", "sharpe_ratio", "sortino_ratio",
    "max_drawdown", "drawdown_series", "infer_periods_per_year",
    # risk
    "fixed_fractional_size", "fixed_fractional_notional", "volatility_target_size",
    "volatility_target_notional", "realized_volatility", "kelly_fraction",
    "stop_loss_price", "take_profit_price", "trailing_stop_price",
    "stop_loss_triggered", "take_profit_triggered", "trailing_stop_triggered",
    "check_exits", "quote_to_quantity", "quantity_to_quote", "align_step",
    "EXIT_STOP_LOSS", "EXIT_TAKE_PROFIT", "EXIT_TRAILING_STOP",
    "__version__",
]
