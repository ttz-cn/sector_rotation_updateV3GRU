import pandas as pd
import numpy as np
from src.src_get_indicators import (
    calculate_beta,
    calculate_close_volume_divergence_corr,
    calculate_ir,
    calculate_long_short_position,
    calculate_second_order_mom,
    calculate_ts_vol,
    calculate_turnover_rate,
    calculate_volume_HDL_diff_divergence,
    calculate_ema_diff,
    calculate_volume_price_strength,
)
from utils.utils_tools import get_factor_value_extracted, get_unconstant_variables, rolling_standard_scaler
import cvxpy as cp


# 计算单一指标
def apply_indicator(data, func, resample_freq="M", rolling_window=360, apply_func=True, **kwargs):
    if not apply_func:
        indicator = func(data, **kwargs)
    else:
        indicator = data.apply(func, **kwargs)
    indicator = rolling_standard_scaler(indicator, rolling_window)
    indicator = indicator.resample(resample_freq).last()
    return indicator


# 计算分组指标
def apply_grouped_indicator(grouped_data, func, rolling_window=360, resample_freq="M", **Kwargs):
    result = pd.DataFrame()
    for fund_code, group in grouped_data:
        group = group.set_index("Date").resample("B").asfreq()
        temp = func(group, column_name=fund_code, **Kwargs)
        result = pd.concat([result, temp], axis=1)
    result = rolling_standard_scaler(result, rolling_window)
    result = result.resample(resample_freq).last()
    return result


# 组合优化函数，根据图片内容实现权重优化
def optimize_portfolio_weights(factor_values, benchmark_weights, params):
    """
    根据因子值和约束条件优化投资组合权重

    参数:
    factor_values: 综合因子值Series，索引为行业代码
    benchmark_weights: 基准权重Series，索引为行业代码
    params: 约束参数字典，包含tracking_error, industry_bias, weight_limit

    返回:
    优化后的权重Series
    """
    n = len(factor_values)
    assets = factor_values.index.tolist()

    # 创建优化变量
    w = cp.Variable(n)

    # 目标函数：最大化因子值乘以权重
    objective = cp.Maximize(factor_values.values @ w)

    # 约束条件
    constraints = [
        # 1. 跟踪误差约束: w^T∑w < tracking_error
        cp.quad_form(w - benchmark_weights.values, np.eye(n)) <= params["tracking_error"],
        # 2. 权重范围约束: 0 <= w <= x
        w >= 0,
        w <= params["weight_limit"],
        # 3. 行业偏离约束: max|wH - wL| < n
        cp.max(cp.abs(w - benchmark_weights.values)) <= params["industry_bias"],
        # 4. 权重之和为1
        cp.sum(w) == 1,
    ]

    # 求解优化问题
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.ECOS)

        # 检查求解状态
        if problem.status == "optimal":
            optimal_weights = pd.Series(w.value, index=assets)
            return optimal_weights
        else:
            print(f"优化求解失败，状态: {problem.status}")
            return benchmark_weights  # 失败时返回基准权重
    except Exception as e:
        print(f"优化过程出错: {e}")
        return benchmark_weights  # 出错时返回基准权重


# 根据拥挤度和最小行业数调整权重
def adjust_portfolio_by_crowding(factor_values, weights, min_industries=3, crowding_threshold=0.25):
    """
    根据拥挤度阈值和最小行业数调整组合

    参数:
    factor_values: 综合因子值Series
    weights: 初步优化的权重Series
    min_industries: 最小行业数
    crowding_threshold: 拥挤度阈值，前1/4为高拥挤度

    返回:
    调整后的权重Series
    """
    # 按因子值排序
    sorted_factors = factor_values.sort_values(ascending=False)
    n = len(sorted_factors)

    # 确定高拥挤度的行业数量
    crowding_count = int(n * crowding_threshold)
    crowded_industries = sorted_factors.iloc[:crowding_count].index

    # 检查当前行业数量
    selected_industries = weights[weights > 0].index
    industry_count = len(selected_industries)

    # 如果行业数量小于最小要求，需要加入更多行业
    if industry_count < min_industries:
        # 需要添加的行业数
        to_add = min_industries - industry_count

        # 从非高拥挤度行业中选择因子值最高的行业添加
        non_crowded = [ind for ind in sorted_factors.index if ind not in crowded_industries]
        additional_industries = sorted_factors[non_crowded].iloc[:to_add].index

        # 为这些行业分配权重
        if len(additional_industries) > 0:
            # 从现有的权重中均匀减少一部分
            reduction_per_industry = 0.05 * len(additional_industries) / len(weights[weights > 0])
            weights[weights > 0] *= 1 - reduction_per_industry * len(weights[weights > 0])

            # 平均分配给新行业
            for ind in additional_industries:
                weights[ind] = 0.05

            # 确保权重之和为1
            weights = weights / weights.sum()

    return weights
