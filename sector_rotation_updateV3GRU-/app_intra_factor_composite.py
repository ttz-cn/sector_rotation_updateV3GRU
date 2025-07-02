import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.src_get_composit_indicators import calculate_icir_for_factors, calculate_icir_weighted_factors
import config

rebalance_period = config.rebalance_period  # 初始化调仓周期

# Load data
etf_panel_data = pd.read_hdf("./db/panel_data.h5", key="sector")  # ETF panel data
fund_open = (
    etf_panel_data.pivot_table(index="Date", columns="industry_code", values="OPEN").resample("B").asfreq()
)  # 按工作日重采样
fund_open_nf = fund_open.resample(rebalance_period).first()  # 按调仓周期重采样
forward_ret = fund_open_nf.pct_change(1).shift(-2)  # 对齐收益率，当期因子值对下期收益率

# 筛选需要的技术指标
dic_tech_factor = {
    key: pd.read_hdf("./db/indicator_timeseries_data.h5", key="{}-{}".format(key, rebalance_period))
    for key in config.list_tech_factor
}

dic_crd_factor = {
    key: pd.read_hdf("./db/indicator_timeseries_data.h5", key="{}-{}".format(key, rebalance_period))
    for key in config.list_crd_factor
}

# 筛选需要的拥挤度指标
dic_funda_factor = {
    key: pd.read_hdf("./db/indicator_timeseries_data.h5", key="{}-{}".format(key, rebalance_period))
    for key in config.list_funda_factor
}

# dic_tech_factor = {
#     "second_order_mom": pd.read_hdf(
#         "./db/indicator_timeseries_data.h5", key="second_order_mom-{}".format(rebalance_period)
#     ),
#     "long_short_position": pd.read_hdf(
#         "./db/indicator_timeseries_data.h5", key="long_short_position-{}".format(rebalance_period)
#     ),
#     "close_volume_divergence_corr": pd.read_hdf(
#         "./db/indicator_timeseries_data.h5", key="close_volume_divergence_corr-{}".format(rebalance_period)
#     ),
#     "volume_HDL_diff_divergence": pd.read_hdf(
#         "./db/indicator_timeseries_data.h5", key="volume_HDL_diff_divergence-{}".format(rebalance_period)
#     ),
# }

# 筛选需要的拥挤度指标
# dic_crd_factor = {
#     "turnover_rate": pd.read_hdf("./db/indicator_timeseries_data.h5", key="turnover_rate-{}".format(rebalance_period)),
#     "ts_vol": pd.read_hdf("./db/indicator_timeseries_data.h5", key="ts_vol-{}".format(rebalance_period)),
#     "beta": pd.read_hdf("./db/indicator_timeseries_data.h5", key="beta-{}".format(rebalance_period)),
# }

# # 筛选需要的基本面指标
# dic_funda_factor = {
#     "roe_fy1": pd.read_hdf("./db/indicator_timeseries_data.h5", key="roe_fy1-{}".format(rebalance_period)),
#     "eps_fy1": pd.read_hdf("./db/indicator_timeseries_data.h5", key="eps_fy1-{}".format(rebalance_period)),
# }


if __name__ == "__main__":

    # 合并tech—factor数据
    dic_tech_icir = calculate_icir_for_factors(
        dic_tech_factor,
        forward_ret,
        rollingwindow1=config.get_icir_rolling_window[rebalance_period][0],
        rollingwindow2=config.get_icir_rolling_window[rebalance_period][1],
    )
    composit_tech_factor = calculate_icir_weighted_factors(dic_tech_factor, dic_tech_icir)

    # 合并crd-factor数据
    dic_crd_icir = calculate_icir_for_factors(
        dic_crd_factor,
        forward_ret,
        rollingwindow1=config.get_icir_rolling_window[rebalance_period][0],
        rollingwindow2=config.get_icir_rolling_window[rebalance_period][1],
    )
    composit_crd_factor = calculate_icir_weighted_factors(dic_crd_factor, dic_crd_icir)

    # 合并funda-factor数据
    dic_funda_icir = calculate_icir_for_factors(
        dic_funda_factor,
        forward_ret,
        rollingwindow1=config.get_icir_rolling_window[rebalance_period][0],
        rollingwindow2=config.get_icir_rolling_window[rebalance_period][1],
    )
    composit_funda_factor = calculate_icir_weighted_factors(dic_funda_factor, dic_funda_icir)

    # 存储合并后的因子数据
    composit_tech_factor.to_hdf("./db/composite_factor.h5", key="composite_tech_factor-{}".format(rebalance_period))
    composit_crd_factor.to_hdf("./db/composite_factor.h5", key="composite_crd_factor-{}".format(rebalance_period))
    composit_funda_factor.to_hdf("./db/composite_factor.h5", key="composite_funda_factor-{}".format(rebalance_period))
