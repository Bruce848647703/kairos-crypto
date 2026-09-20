"""仓位管理与止损/止盈判定（纯函数、无状态、可独立测试）。

包含两组工具：

1. **仓位规模**
   - :func:`fixed_fractional_size`：固定比例仓位（净值的 x% 买入某币）。
   - :func:`volatility_target_size`：波动率目标仓位（目标年化波动 / 已实现波动 = 杠杆），
     波动越大仓位越小，是加密永续/现货常用的风险平价式做法。
   - :func:`kelly_fraction`：凯利比例（可选上限与折扣），用于评估下注比例上限。
2. **风控触发判定**
   - :func:`stop_loss_triggered` / :func:`take_profit_triggered` / :func:`trailing_stop_triggered`
   - :func:`check_exits`：一次性判定该走哪条出场规则（止损优先于止盈）。

所有函数都做输入校验并返回确定值，不依赖全局状态，便于在策略与测试中复用。
年化默认按 24/7 的 ``periods_per_year=365``（日频）或 ``8760``（小时频）折算。
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

ReturnsLike = Union[Sequence[float], np.ndarray, pd.Series]

#: 触发判定的相对容差：避免 100×1.1 = 110.00000000000001 这类浮点边界误判
TOUCH_TOL = 1e-12

EXIT_STOP_LOSS = "stop_loss"
EXIT_TAKE_PROFIT = "take_profit"
EXIT_TRAILING_STOP = "trailing_stop"


def _as_array(returns: ReturnsLike) -> np.ndarray:
    """把收益率输入统一成一维 float 数组并剔除 NaN。"""
    if isinstance(returns, pd.Series):
        arr = returns.to_numpy(dtype="float64")
    else:
        arr = np.asarray(returns, dtype="float64")
    arr = np.atleast_1d(arr).ravel()
    return arr[~np.isnan(arr)]


def realized_volatility(returns: ReturnsLike, periods_per_year: float = 365.0,
                        min_obs: int = 2) -> float:
    """已实现波动率（年化）。样本不足或零波动返回 0.0。"""
    arr = _as_array(returns)
    if arr.size < max(2, int(min_obs)):
        return 0.0
    vol = float(np.std(arr, ddof=1))
    if not np.isfinite(vol):
        return 0.0
    return vol * float(np.sqrt(periods_per_year))


def fixed_fractional_notional(equity: float, fraction: float) -> float:
    """固定比例仓位对应的名义金额（计价币）。非法输入返回 0.0。"""
    equity = float(equity)
    fraction = float(fraction)
    if equity <= 0 or fraction <= 0 or not np.isfinite(equity) or not np.isfinite(fraction):
        return 0.0
    return equity * fraction


def fixed_fractional_size(equity: float, fraction: float, price: float) -> float:
    """固定比例仓位：用净值的 ``fraction`` 比例买入，返回可买数量（基础币）。

    参数
    ----
    equity:   账户净值（计价币）。
    fraction: 目标仓位占净值的比例，如 ``0.2`` 表示两成仓。
    price:    当前价格（计价币/基础币）。

    返回
    ----
    买入数量；``equity/fraction/price`` 非正或非法时返回 ``0.0``。
    """
    notional = fixed_fractional_notional(equity, fraction)
    price = float(price)
    if notional <= 0 or price <= 0 or not np.isfinite(price):
        return 0.0
    return notional / price


def volatility_target_notional(equity: float, returns: ReturnsLike, target_vol: float,
                               periods_per_year: float = 365.0,
                               max_leverage: float = 1.0,
                               min_vol: float = 1e-8) -> float:
    """波动率目标对应的名义敞口（计价币）。

    杠杆 = ``min(target_vol / 已实现波动率, max_leverage)``；
    已实现波动率过低（信息不足）时退化为 ``max_leverage``，避免仓位爆炸。
    """
    equity = float(equity)
    target_vol = float(target_vol)
    if equity <= 0 or target_vol <= 0:
        return 0.0
    max_leverage = max(float(max_leverage), 0.0)
    vol = realized_volatility(returns, periods_per_year)
    if vol <= min_vol:
        leverage = max_leverage
    else:
        leverage = min(target_vol / vol, max_leverage)
    return equity * max(leverage, 0.0)


def volatility_target_size(equity: float, returns: ReturnsLike, target_vol: float,
                           price: float, periods_per_year: float = 365.0,
                           max_leverage: float = 1.0, min_vol: float = 1e-8) -> float:
    """波动率目标仓位：返回应持有的数量（基础币）。

    直觉：**波动越大，仓位越小**，使组合承担的风险（年化波动）趋于 ``target_vol``。

    参数
    ----
    equity:            账户净值（计价币）。
    returns:           近期收益率样本（频率需与 ``periods_per_year`` 匹配）。
    target_vol:        目标年化波动率，如 ``0.6`` 表示 60%。
    price:             当前价格。
    periods_per_year:  年化期数（1h K 线为 8760，1d 为 365）。
    max_leverage:      杠杆上限（现货默认 1.0，即不加杠杆）。
    """
    notional = volatility_target_notional(equity, returns, target_vol,
                                          periods_per_year, max_leverage, min_vol)
    price = float(price)
    if notional <= 0 or price <= 0 or not np.isfinite(price):
        return 0.0
    return notional / price


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float,
                   discount: float = 1.0, cap: float = 1.0) -> float:
    """凯利比例 ``f* = p - (1-p)/b``，其中 ``b = 平均盈利 / 平均亏损``。

    参数
    ----
    win_rate: 胜率 p（0~1）。
    avg_win:  平均盈利（正数）。
    avg_loss: 平均亏损（负数，内部取绝对值）。
    discount: 折扣系数，如 0.5 表示半凯利（更稳健）。
    cap:      上限，避免极端值。

    返回
    ----
    建议下注比例（已裁剪到 ``[0, cap]``）。
    """
    p = float(win_rate)
    win = abs(float(avg_win))
    loss = abs(float(avg_loss))
    if not 0.0 <= p <= 1.0 or win <= 0 or loss <= 0:
        return 0.0
    b = win / loss
    f = p - (1.0 - p) / b
    f *= max(float(discount), 0.0)
    return float(min(max(f, 0.0), max(float(cap), 0.0)))


# --------------------------------------------------------------------- 风控
def _check_pct(pct: float, name: str) -> float:
    pct = float(pct)
    if not np.isfinite(pct) or pct <= 0:
        raise ValueError(f"{name} 必须为正数，收到: {pct}")
    return pct


def stop_loss_price(entry_price: float, stop_pct: float) -> float:
    """止损价 = 入场价 × (1 - stop_pct)。"""
    return float(entry_price) * (1.0 - _check_pct(stop_pct, "stop_pct"))


def take_profit_price(entry_price: float, take_pct: float) -> float:
    """止盈价 = 入场价 × (1 + take_pct)。"""
    return float(entry_price) * (1.0 + _check_pct(take_pct, "take_pct"))


def trailing_stop_price(highest_price: float, trail_pct: float) -> float:
    """移动止损价 = 持仓期间最高价 × (1 - trail_pct)。"""
    return float(highest_price) * (1.0 - _check_pct(trail_pct, "trail_pct"))


def stop_loss_triggered(entry_price: float, price: float, stop_pct: float) -> bool:
    """多头止损是否触发：现价 <= 止损价（含 :data:`TOUCH_TOL` 相对容差）。"""
    return float(price) <= stop_loss_price(entry_price, stop_pct) * (1.0 + TOUCH_TOL)


def take_profit_triggered(entry_price: float, price: float, take_pct: float) -> bool:
    """多头止盈是否触发：现价 >= 止盈价（含 :data:`TOUCH_TOL` 相对容差）。"""
    return float(price) >= take_profit_price(entry_price, take_pct) * (1.0 - TOUCH_TOL)


def trailing_stop_triggered(highest_price: float, price: float, trail_pct: float) -> bool:
    """移动止损是否触发：现价从最高点回撤达到 ``trail_pct``（含相对容差）。"""
    return float(price) <= trailing_stop_price(highest_price, trail_pct) * (1.0 + TOUCH_TOL)


def check_exits(entry_price: float, price: float, stop_pct: Optional[float] = None,
                take_pct: Optional[float] = None, highest_price: Optional[float] = None,
                trail_pct: Optional[float] = None) -> Optional[str]:
    """一次性判定多头出场信号，返回触发原因（无触发返回 ``None``）。

    优先级：固定止损 > 移动止损 > 止盈。这样在极端行情下先保住本金，
    与实盘挂单的风控习惯一致。
    """
    if entry_price is None or price is None:
        return None
    if stop_pct is not None and stop_loss_triggered(entry_price, price, stop_pct):
        return EXIT_STOP_LOSS
    if trail_pct is not None and highest_price is not None and \
            trailing_stop_triggered(highest_price, price, trail_pct):
        return EXIT_TRAILING_STOP
    if take_pct is not None and take_profit_triggered(entry_price, price, take_pct):
        return EXIT_TAKE_PROFIT
    return None


# --------------------------------------------------------------------- 工具
def quote_to_quantity(quote_amount: float, price: float, fee_rate: float = 0.0) -> float:
    """把「花多少钱」换算成「买多少币」，手续费计入成本。

    例：``quote_to_quantity(100, 50000, 0.001) ≈ 0.0019980``。
    """
    quote_amount = float(quote_amount)
    price = float(price)
    fee_rate = float(fee_rate)
    if quote_amount <= 0 or price <= 0:
        return 0.0
    return quote_amount / (price * (1.0 + fee_rate))


def quantity_to_quote(quantity: float, price: float, fee_rate: float = 0.0) -> float:
    """:func:`quote_to_quantity` 的逆运算：卖出 ``quantity`` 能拿回多少钱（已扣手续费）。"""
    quantity = float(quantity)
    price = float(price)
    if quantity <= 0 or price <= 0:
        return 0.0
    return quantity * price * (1.0 - float(fee_rate))


def align_step(quantity: float, step_size: float) -> float:
    """把数量向下对齐到交易所的最小下单步长（如 0.00001 BTC）。"""
    quantity = float(quantity)
    step_size = float(step_size)
    if quantity <= 0 or step_size <= 0:
        return max(quantity, 0.0)
    steps = np.floor(quantity / step_size + 1e-12)
    return float(steps * step_size)
