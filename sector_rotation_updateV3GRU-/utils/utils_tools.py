# -*- coding: utf-8 -*-
# @Time    : 2023/10/1 17:00
# @Author  : zhangyong
# @File    : utils_tools.py
# @Software: PyCharm
# @Description: 工具函数

import datetime
import numpy as np
import pandas as pd

from scipy.stats import spearmanr


def load_panel_dropd(path, key):
    return (
        pd.read_hdf(path, key=key)
        .drop_duplicates(keep="last")  # 全表去重，保留最后一条
        .ffill()  # 向前填充
        .sort_values(["industry_code", "Date"])  # 按行业+日期排序，保证后面 drop_duplicates 保留“最后一条”是最新的
        .drop_duplicates(subset=["industry_code", "Date"], keep="last")
    )


def get_previous_sunday(date):
    day_of_week = date.weekday()  # 获取当前日位置
    day_to_subtrack = day_of_week + 1
    previous_sunday = date - datetime.timedelta(days=day_to_subtrack)
    return previous_sunday


# rolling_standard_scaler: 滚动标准化
def rolling_standard_scaler(df, window):

    scaled_df = pd.DataFrame(index=pd.to_datetime(df.index), columns=df.columns)

    for col in df.columns:  # rolling 很慢的
        scaled_col = (
            df[col]
            .rolling(window=window)
            .apply(
                lambda col: ((col[-1] - np.mean(col)) / np.std(col) if np.std(col) != 0 else np.nan),
                raw=True,
            )
        )
        scaled_df[col] = scaled_col

    return scaled_df


# 截取固定日期起始的数据
def get_factor_value_extracted(isExtract=False, **params):

    df = params["df"]
    date = params["date"]

    if not isExtract:
        return df
    else:
        df = df[df.index > date]
        return df


# 处理数据集中的常量变量
def get_unconstant_variables(df):
    # 创建一个布尔掩码，标记与上一行相同的行
    mask = (df == df.shift(1)).all(axis=1)
    # 过滤掉非数值列（用来处理category类型）
    numeric_cols = df.select_dtypes(include=["number"])
    # 对与上一行相同的行进行加1操作
    random_noise = np.random.rand(*numeric_cols.shape) * 2 - 1  # randm.rand 生成*df.shape的随机数
    numeric_cols[mask] = numeric_cols[mask] + random_noise[mask]
    df[numeric_cols.columns] = numeric_cols
    return df


# 获取数据集中时序数据
def get_fund_timeseries_data(df, values, date_col="Date", columns="fund_code"):
    fund_timeserise_data = df.pivot_table(index=date_col, columns=columns, values=values)
    return fund_timeserise_data


def safe_spearmanr(a, b):
    # 检查输入数组是否为常数
    if np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan  # 返回 NaN 表示无法定义
    return spearmanr(a, b)[0]  # 返回相关系数


def update_etfs(dic, row):
    # 行业映射到标的etf
    if isinstance(row[0], list):  # 确保 row[0] 是一个列表
        return ["{}.OF".format(dic[key][1]) for key in row[0] if key in dic.keys()]
    else:
        print("Unexpected data type:", row[0])
        return np.nan
