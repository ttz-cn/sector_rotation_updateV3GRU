import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.src_indicator_processing import apply_grouped_indicator, apply_indicator
from src.src_get_indicators import (
    calculate_beta,
    calculate_close_volume_divergence_corr,
    calculate_ir,
    calculate_long_short_position,
    calculate_second_order_mom,
    calculate_ts_vol,
    calculate_turnover_rate,
    calculate_volume_HDL_diff_divergence,
    straight_return,
    calculate_ema_diff,
    calculate_volume_price_strength,
)
from utils.utils_tools import (
    get_factor_value_extracted,
    get_unconstant_variables,
    load_panel_dropd,
    rolling_standard_scaler,
)
import config


# Load data,都drop_duplicate，因为我也不确定到底哪里有duplicate
etf_panel_data = load_panel_dropd("./db/panel_data.h5", key="sector")  # sector panel data
wind_data = pd.read_hdf("./db/time_series_data.h5", key="wind_a")
wind_data = wind_data.groupby(wind_data.index).tail(1).ffill()  # 保留每个index的最后一条
# 按照基金代码分组
grouped = etf_panel_data.groupby("industry_code")


# 获取收盘价时序数据
fund_close = etf_panel_data.pivot_table(index="Date", columns="industry_code", values="CLOSE")
fund_close = fund_close.resample("B").asfreq()  # 按工作日重采样
# 获取换手率时序数据
fund_turn = etf_panel_data.pivot_table(index="Date", columns="industry_code", values="TURN")
fund_turn = fund_turn.resample("B").asfreq()  # 按工作日重采样
# 获取roe一致预期时序数据
roe_fy1 = etf_panel_data.pivot_table(index="Date", columns="industry_code", values="WEST_AVGROE_FY1")
roe_fy1 = roe_fy1.resample("B").asfreq()  # 按工作日重采样
# 获取eps一致预期时序数据
eps_fy1 = etf_panel_data.pivot_table(index="Date", columns="industry_code", values="WEST_EPS_FY1")
eps_fy1 = eps_fy1.resample("B").asfreq()  # 按工作日重采样
# 获取Wind_A股收盘价时序数据
wind_close = pd.DataFrame(wind_data["CLOSE"])
wind_close = wind_close.resample("B").asfreq()  # 按工作日重采样

# 获取收益率时序数据
ret_fund = fund_close.pct_change()
ret_wind = wind_close.pct_change()

ret_fund_ = ret_fund.copy()

indicator_funcs = {
    "second_order_mom": (
        fund_close,
        calculate_second_order_mom,
        config.rebalance_period,
        360,
        {"window1": 10, "window2": 20, "window3": 20},
        True,
    ),
    "long_short_position": (
        grouped,
        calculate_long_short_position,
        config.rebalance_period,
        360,
        {"window1": 15},
        False,
    ),
    "close_volume_divergence_corr": (
        grouped,
        calculate_close_volume_divergence_corr,
        config.rebalance_period,
        60,
        {"window": 10},
        False,
    ),
    "volume_HDL_diff_divergence": (
        grouped,
        calculate_volume_HDL_diff_divergence,
        config.rebalance_period,
        360,
        {"window": 10},
        False,
    ),
    "ir": (ret_fund_, calculate_ir, config.rebalance_period, 360, {"window": 10}, False),
    "turnover_rate": (
        fund_turn,
        calculate_turnover_rate,
        config.rebalance_period,
        180,
        {"window1": 80, "window2": 5},
        True,
    ),
    "ts_vol": (fund_close, calculate_ts_vol, config.rebalance_period, 360, {"window": 40}, True),
    "beta": (
        ret_fund,
        calculate_beta,
        config.rebalance_period,
        90,
        {"benchmark": ret_wind["CLOSE"], "window_size": 40, "min_periods": 5},
        True,
    ),
    "roe_fy1": (roe_fy1, straight_return, config.rebalance_period, 40, {}, True),
    "eps_fy1": (eps_fy1, straight_return, config.rebalance_period, 40, {}, True),
    # 新增因子
    "ema_diff": (
        fund_close,
        calculate_ema_diff,
        config.rebalance_period,
        180,
        {"fast_window": 5, "slow_window": 20},
        True,
    ),
    "volume_price_strength": (
        grouped,
        calculate_volume_price_strength,
        config.rebalance_period,
        90,
        {"window": 15},
        False,
    ),
}


if __name__ == "__main__":

    # 计算所有指标
    indicator_results = {}
    for name, (data, func, freq, rolling_window, params, apply_func) in indicator_funcs.items():
        print(f"计算因子: {name}")
        if name in [
            "long_short_position",
            "close_volume_divergence_corr",
            "volume_HDL_diff_divergence",
            "volume_price_strength",
        ]:
            indicator_results[name] = apply_grouped_indicator(
                data, func, resample_freq=freq, rolling_window=rolling_window, **params
            )
        else:
            indicator_results[name] = apply_indicator(
                data, func, resample_freq=freq, rolling_window=rolling_window, apply_func=apply_func, **params
            )

    # 将数据存储到HDF5文件中
    for key, result in indicator_results.items():
        result.to_hdf(
            "./db/indicator_timeseries_data.h5", key="{}-{}".format(key, config.rebalance_period), format="table"
        )

    """
    # 计算二阶动量因子-second_order_mom
    second_order_mom = -fund_close.apply(calculate_second_order_mom, window1=10, window2=20, window3=20)
    second_order_mom = rolling_standard_scaler(second_order_mom, 360)
    second_order_mom_ = second_order_mom.resample("M").last()  # 按月取最后一个交易日数据
    second_order_mom_ = get_factor_value_extracted(
        isExtract=False, df=second_order_mom_, date="2022-01-01"
    )  # 截取2022年1月1日以后的数据

    # 计算多空头寸因子-long_short_position
    long_short_position = pd.DataFrame()  # 初始化空的多空头寸数据框
    for fund_code, temp in grouped:
        temp = temp.set_index("Date").resample("B").asfreq()  # 按工作日重采样
        temp_ = -calculate_long_short_position(df=temp, column_name=fund_code, window1=15)
        long_short_position = pd.concat([long_short_position, temp_], axis=1)  # 合并多空头寸数据
    long_short_position = rolling_standard_scaler(long_short_position, 360)
    long_short_position_ = long_short_position.resample("M").last()  # 按月取最后一个交易日数据

    # 计算量价背离因子-close_volume_divergence_corr
    close_volume_divergence_corr = pd.DataFrame()
    for fund_code, temp in grouped:
        temp = temp.set_index("Date").resample("B").asfreq()
        temp_ = -calculate_close_volume_divergence_corr(df=temp, window=60, column_name=fund_code)
        close_volume_divergence_corr = pd.concat([close_volume_divergence_corr, temp_], axis=1)  # 合并多空头寸数据
    close_volume_divergence_corr = rolling_standard_scaler(close_volume_divergence_corr, 180)
    close_volume_divergence_corr_ = close_volume_divergence_corr.resample("M").last()

    # 计算量价背离因子-volume_HDL_diff_divergence
    volume_HDL_diff_divergence = pd.DataFrame()
    for fund_code, temp in grouped:
        temp = temp.set_index("Date").resample("B").asfreq()
        temp = get_unconstant_variables(temp)  # 处理数据集中的常量变量
        temp_ = calculate_volume_HDL_diff_divergence(df=temp, window=10, column_name=fund_code)
        volume_HDL_diff_divergence = pd.concat([volume_HDL_diff_divergence, temp_], axis=1)
    volume_HDL_diff_divergence = rolling_standard_scaler(volume_HDL_diff_divergence, 360)
    volume_HDL_diff_divergence_ = volume_HDL_diff_divergence.resample("M").last()

    # 计算IR指标
    ret_fund_ = ret_fund.copy()
    ir = calculate_ir(ret_fund_, window=360)
    ir = rolling_standard_scaler(ir, 360)
    ir_ = ir.resample("M").last()

    # 计算换手率指标
    turnover_rate = fund_turn.apply(calculate_turnover_rate, window1=80, window2=5)  # 计算turnover_rate
    turnover_rate = rolling_standard_scaler(turnover_rate, 180)  # 滚动标准化
    turnover_rate_ = turnover_rate.resample("M").last()  # 按月取最后一个交易日数据

    # 计算收益率的波动率
    ts_vol = -calculate_ts_vol(fund_close, window=90)
    ts_vol = rolling_standard_scaler(ts_vol, 180)  # 滚动标准化
    ts_vol_ = ts_vol.resample("M").last()

    # 计算beta指标
    beta = pd.DataFrame()  # 创建空df储存beta数据
    # 计算斜率（注：暂时没有找到在rolling下利用skitlearn计算斜率的方法）
    for col in ret_fund.columns:
        beta[col] = calculate_beta(ret_fund[col], ret_wind["CLOSE"], window_size=90, min_periods=60)
    beta = rolling_standard_scaler(beta, 90)
    beta_ = beta.resample("M").last()

    # 计算一致预期ROE指标
    roe_fy1_st = rolling_standard_scaler(roe_fy1, window=360)
    roe_fy1_st_ = -roe_fy1_st.resample("M").last()

    # 计算一致预期EPS指标
    eps_fy1_st = rolling_standard_scaler(eps_fy1, window=360)
    eps_fy1_st_ = -eps_fy1_st.resample("M").last()
    """
