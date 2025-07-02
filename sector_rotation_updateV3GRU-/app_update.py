import datetime
import pandas as pd
import numpy as np
import torch
import config
from WindPy import w

from utils import utils_tools
from config import rebalance_period
from utils.utils_tools import get_factor_value_extracted
from src.src_get_composit_indicators import (
    calculate_icir_for_factors,
    calculate_icir_weighted_factors,
    calculate_quantile_groups,
    gru_factor_composite,
)
from src.src_indicator_processing import optimize_portfolio_weights, adjust_portfolio_by_crowding

_ = w.start()  # 启动WindPy


etf_panel_data = pd.read_hdf("./db/panel_data.h5", key="sector")  # sector panel data
wind_data = pd.read_hdf("./db/time_series_data.h5", key="wind_a")  # Wind A-share time series data


start_date = (etf_panel_data["Date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # 获取更新起始日期
today_date = datetime.date.today()
if rebalance_period == "W":
    end_date = utils_tools.get_previous_sunday(today_date).strftime("%Y-%m-%d")  # 获取截至日期
elif rebalance_period == "B":
    end_date = (today_date + pd.Timedelta(days=-1)).strftime("%Y-%m-%d")

# 检查开始日期是否大于结束日期，如果是，则直接退出程序
if pd.to_datetime(start_date) > pd.to_datetime(end_date):
    print(f"开始日期 {start_date} 大于结束日期 {end_date}，没有新数据需要更新，程序退出。")
    import sys

    sys.exit(0)

if __name__ == "__main__":
    # get_panel_data for etf
    data_list = []
    for industry, fund in config.dic_Industry2Etf.items():
        ind_code = industry  # 行业编码
        of_code = "{}.OF".format(fund[1])  # etf基金编码

        # 提取行业一致预期，提取etf行情数据
        error, ind = w.wsd(
            ind_code,
            "west_avgroe_FY1,west_eps_FY1",
            start_date,
            end_date,
            "unit=1;Days=Alldays;Fill=Previous",
            usedf=True,
        )

        error, of = w.wsd(
            ind_code,
            "open,high,low,close,volume,amt,pct_chg,turn",
            start_date,
            end_date,
            "unit=1;Days=Alldays;Fill=Previous",
            usedf=True,
        )
        # 合并一致预期数据和行情数据
        _ = pd.merge(left=ind, right=of, left_index=True, right_index=True)

        # 重置索引并设置列名
        _.reset_index(names="Date", inplace=True)
        # 添加factor_name
        _["fund_code"] = of_code
        _["industry_code"] = ind_code
        # 存入list
        data_list.append(_)

    # 合并为Panel数据并处理数据格式
    panel_data = pd.concat(data_list, ignore_index=True)
    panel_data["Date"] = pd.to_datetime(panel_data["Date"])  # 将Date转换为datetime格式
    panel_data["fund_code"] = panel_data["fund_code"].astype("category")
    panel_data["industry_code"] = panel_data["industry_code"].astype("category")  # 将分类code转化为category类

    # get wind_a data
    errorcode, wind_data_ = w.wsd("881001.WI", "close", start_date, end_date, "Days=Alldays;Fill=Previous", usedf=True)
    wind_data_.index = pd.to_datetime(wind_data_.index)

    # 将数据存入hdf5文件
    panel_data.to_hdf("./db/panel_data.h5", key="sector", append=True, format="table")
    wind_data_.to_hdf("./db/time_series_data.h5", key="wind_a", append=True, format="table")
