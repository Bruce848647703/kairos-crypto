"""仓位管理与止损止盈判定测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos_crypto import (
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


# ------------------------------------------------------------ 固定比例仓位
def test_fixed_fractional_size_math():
    # 1 万净值、两成仓、币价 5 万 → 0.04 个币
    assert fixed_fractional_size(10_000.0, 0.2, 50_000.0) == pytest.approx(0.04)
    assert fixed_fractional_notional(10_000.0, 0.2) == pytest.approx(2_000.0)
    # 价格越高，数量越少
    assert fixed_fractional_size(10_000.0, 0.2, 100_000.0) < \
        fixed_fractional_size(10_000.0, 0.2, 50_000.0)


@pytest.mark.parametrize("equity,fraction,price", [
    (0.0, 0.5, 100.0), (-1.0, 0.5, 100.0), (10_000.0, 0.0, 100.0),
    (10_000.0, -0.5, 100.0), (10_000.0, 0.5, 0.0), (10_000.0, 0.5, -1.0),
    (float("nan"), 0.5, 100.0),
])
def test_fixed_fractional_size_guards(equity, fraction, price):
    assert fixed_fractional_size(equity, fraction, price) == 0.0
    assert fixed_fractional_notional(equity, fraction) >= 0.0


# ------------------------------------------------------------ 波动率目标仓位
def test_higher_volatility_gives_smaller_position():
    """核心性质：波动越大，仓位越小。"""
    equity, price, target = 10_000.0, 50_000.0, 0.6
    calm_noisy = np.linspace(-0.02, 0.02, 100)          # 年化波动 ~22%
    wild = calm_noisy * 4.0                              # 年化波动 ~89%
    sizes = [volatility_target_size(equity, r, target, price, periods_per_year=365.0,
                                    max_leverage=10.0)
             for r in (np.zeros(100) + 1e-9, calm_noisy, wild)]
    assert sizes[0] > sizes[1] > sizes[2]
    # 波动率翻倍 → 仓位减半（未触及杠杆上限时）
    mid = volatility_target_size(equity, calm_noisy, target, price, 365.0, max_leverage=10.0)
    double = volatility_target_size(equity, calm_noisy * 2.0, target, price, 365.0,
                                    max_leverage=10.0)
    assert double == pytest.approx(mid / 2.0, rel=1e-9)


def test_volatility_target_respects_leverage_cap():
    equity, price = 10_000.0, 100.0
    quiet = np.full(200, 1e-6)                        # 几乎无波动
    notional = volatility_target_notional(equity, quiet, target_vol=0.5,
                                          periods_per_year=365.0, max_leverage=1.0)
    assert notional == pytest.approx(equity)          # 不超过 1 倍杠杆
    size = volatility_target_size(equity, quiet, 0.5, price, 365.0, max_leverage=1.0)
    assert size == pytest.approx(equity / price)
    assert volatility_target_size(equity, quiet, 0.5, price, 365.0, max_leverage=3.0) == \
        pytest.approx(3 * equity / price)


def test_volatility_target_guards():
    assert volatility_target_size(0.0, [0.01, 0.02], 0.5, 100.0) == 0.0
    assert volatility_target_size(1000.0, [0.01, 0.02], 0.0, 100.0) == 0.0
    assert volatility_target_size(1000.0, [0.01, 0.02], 0.5, 0.0) == 0.0
    # 样本不足 → 波动率视为 0 → 退化到杠杆上限
    assert volatility_target_size(1000.0, [0.01], 0.5, 100.0, max_leverage=1.0) == \
        pytest.approx(10.0)


def test_realized_volatility_annualization():
    rets = np.array([0.01, -0.01] * 50)
    expected = float(np.std(rets, ddof=1)) * np.sqrt(365.0)
    assert realized_volatility(rets, 365.0) == pytest.approx(expected)
    assert realized_volatility(pd.Series(rets), 365.0) == pytest.approx(expected)
    assert realized_volatility(rets, 365.0) * np.sqrt(2) == pytest.approx(
        realized_volatility(rets, 730.0))
    assert realized_volatility([0.01], 365.0) == 0.0        # 样本不足
    assert realized_volatility([], 365.0) == 0.0
    with_nan = np.array([0.01, np.nan, -0.01, 0.02])
    assert realized_volatility(with_nan, 1.0) == pytest.approx(
        float(np.std([0.01, -0.01, 0.02], ddof=1)))


def test_kelly_fraction():
    # p=0.6，盈亏比 1:1 → f* = 0.6 - 0.4 = 0.2
    assert kelly_fraction(0.6, 1.0, -1.0) == pytest.approx(0.2)
    assert kelly_fraction(0.6, 1.0, -1.0, discount=0.5) == pytest.approx(0.1)
    assert kelly_fraction(0.4, 1.0, -1.0) == 0.0           # 负期望 → 不下注
    assert kelly_fraction(0.9, 5.0, -1.0, cap=0.25) == pytest.approx(0.25)
    assert kelly_fraction(1.5, 1.0, -1.0) == 0.0           # 非法胜率
    assert kelly_fraction(0.5, 0.0, -1.0) == 0.0


# ------------------------------------------------------------------ 风控判定
def test_stop_and_take_profit_prices():
    assert stop_loss_price(100.0, 0.05) == pytest.approx(95.0)
    assert take_profit_price(100.0, 0.10) == pytest.approx(110.0)
    assert trailing_stop_price(120.0, 0.10) == pytest.approx(108.0)
    with pytest.raises(ValueError):
        stop_loss_price(100.0, 0.0)
    with pytest.raises(ValueError):
        take_profit_price(100.0, -0.1)


def test_trigger_predicates():
    assert stop_loss_triggered(100.0, 95.0, 0.05)          # 恰好触及
    assert stop_loss_triggered(100.0, 94.0, 0.05)
    assert not stop_loss_triggered(100.0, 96.0, 0.05)
    assert take_profit_triggered(100.0, 110.0, 0.10)
    assert not take_profit_triggered(100.0, 109.0, 0.10)
    assert trailing_stop_triggered(120.0, 108.0, 0.10)
    assert not trailing_stop_triggered(120.0, 112.0, 0.10)
    # 浮点边界：100 × 1.1 在二进制下略大于 110，容差保证「恰好触及」成立
    assert take_profit_price(100.0, 0.10) != 110.0
    assert take_profit_triggered(100.0, take_profit_price(100.0, 0.10), 0.10)


def test_check_exits_priority_and_none():
    # 同时满足止损与止盈 → 止损优先
    assert check_exits(100.0, 90.0, stop_pct=0.05, take_pct=0.10) == EXIT_STOP_LOSS
    # 移动止损优先于止盈
    assert check_exits(100.0, 112.0, take_pct=0.10, highest_price=130.0,
                       trail_pct=0.10) == EXIT_TRAILING_STOP
    assert check_exits(100.0, 111.0, stop_pct=0.05, take_pct=0.10) == EXIT_TAKE_PROFIT
    assert check_exits(100.0, 105.0, stop_pct=0.05, take_pct=0.10) is None
    assert check_exits(100.0, 50.0) is None                 # 未配置任何规则
    assert check_exits(100.0, 50.0, stop_pct=None, take_pct=None, trail_pct=None) is None


def test_check_exits_trailing_uses_highest_price():
    # 最高点 110、回撤 15% → 移动止损位 93.5
    assert check_exits(100.0, 100.0, highest_price=110.0, trail_pct=0.15) is None
    assert check_exits(100.0, 93.5, highest_price=110.0, trail_pct=0.15) == EXIT_TRAILING_STOP
    assert check_exits(100.0, 90.0, highest_price=110.0, trail_pct=0.15) == EXIT_TRAILING_STOP


# -------------------------------------------------------------------- 工具
def test_quote_and_quantity_conversion_round_trip():
    qty = quote_to_quantity(100.0, 50_000.0, fee_rate=0.001)
    assert qty == pytest.approx(100.0 / (50_000.0 * 1.001))
    # 买回来花掉的钱 = 卖出拿回的钱 + 两次手续费的差额，口径自洽
    spent = qty * 50_000.0 * 1.001
    assert spent == pytest.approx(100.0)
    assert quantity_to_quote(qty, 50_000.0, fee_rate=0.001) == pytest.approx(
        qty * 50_000.0 * 0.999)
    assert quote_to_quantity(0.0, 100.0) == 0.0
    assert quote_to_quantity(100.0, 0.0) == 0.0
    assert quantity_to_quote(-1.0, 100.0) == 0.0


def test_align_step():
    assert align_step(0.123456, 0.001) == pytest.approx(0.123)
    assert align_step(0.9999999, 1.0) == pytest.approx(0.0)
    assert align_step(2.5, 0.5) == pytest.approx(2.5)
    assert align_step(-1.0, 0.1) == 0.0
    assert align_step(1.0, 0.0) == pytest.approx(1.0)
