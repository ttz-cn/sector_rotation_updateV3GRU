import pandas as pd
import numpy as np
import talib as ta


# 计算二阶动量指标


def calculate_second_order_mom(close_prices, window1, window2, window3):
    def ewma(series, span):
        return series.ewm(span=span, adjust=False).mean()

    close_series = pd.Series(close_prices)  # convert to series
    mean_window1 = close_series.rolling(window=window1).mean()  # calculate mean value
    fraction = (close_series - mean_window1) / mean_window1  # Calculate the fraction

    delayed_fraction = fraction.shift(window2)  # delay (shift) operation

    result = ewma(fraction - delayed_fraction, span=window3)  # Calculate the final result using EWM

    return result


# 计算多空头寸指标
def calculate_long_short_position(df, column_name, window1):

    epsilon = 1e-9  # alphalens会自动把0和nan都drop掉，认为0和nan都没用，所以需要先判断高、低、收的价格确保不相等
    df["LOW"] = np.where(
        np.abs(df["CLOSE"] - df["LOW"]) < epsilon,
        df["LOW"] - np.random.rand(),
        df["LOW"],
    )
    df["HIGH"] = df.apply(
        lambda row: (row["HIGH"] + np.random.rand() if np.abs(row["HIGH"] - row["CLOSE"]) < epsilon else row["HIGH"]),
        axis=1,
    )

    long = df["CLOSE"] - df["LOW"]
    short = df["HIGH"] - df["CLOSE"]

    df[column_name] = long.rolling(window1).sum() / short.rolling(window1).sum()  # 计算long/short在窗口期的和

    return df[column_name]


# 计算相关系数量价背离指标
def calculate_close_volume_divergence_corr(df, column_name, window):

    def rolling_correlation(x, y, window):
        return x.rolling(window).corr(y)  # 窗口window下x和y的协方差

    corr = rolling_correlation(df["CLOSE"], df["VOLUME"], window)

    df[column_name] = corr

    return df[column_name]


# 计算量价背离指标
def calculate_volume_HDL_diff_divergence(df, column_name, window):

    def rank_series(series):
        return series.rank()  # 对series进行排名

    def rolling_correlation(x, y, window):
        return x.rolling(window).corr(y)  # 窗口window下x和y的协方差

    # 计算 hdl 和 Volume 的排名
    rank_HDL = rank_series(df["HIGH"] / df["LOW"] - 1)
    rank_volume = rank_series(df["VOLUME"] / df["VOLUME"].shift(1) - 1)
    corr = rolling_correlation(rank_volume, rank_HDL, window)
    df[column_name] = corr

    return df[column_name]


# 计算ir指标
def calculate_ir(df, window):
    # 确保索引datatime格式
    df.index = pd.to_datetime(df.index)
    # 计算基准列
    df["benchmark"] = df.apply(lambda row: row.mean(), axis=1)

    # 计算与benchmark的差异
    _ = pd.DataFrame({ind: (df[ind] - df["benchmark"]) * 100 for ind in df.columns if ind != "benchmark"})
    ir = _.rolling(window).sum() / _.rolling(window).std() * np.sqrt(252)

    return ir


# 计算换手率指标
def calculate_turnover_rate(turnover, window1, window2):  # window1>window2

    turnover_series = pd.Series(turnover)  # convert to series

    mean_window1 = turnover_series.rolling(window1).mean()  # calculate mean turnover for window1
    mean_window2 = turnover_series.rolling(window2).mean()  # calculate mean turnover for window2

    turnover_rate = mean_window1 / mean_window2

    return -turnover_rate


# 计算波动率指标
def calculate_ts_vol(df, window):
    ret = df.pct_change() * 100  # 计算收益率的波动率
    ts_vol = ret.rolling(window).std()
    return ts_vol


# 计算beta指标
def calculate_beta(ret_ind_, benchmark=None, window_size=90, min_periods=60):

    # 合并个体收益率和基准收益率
    combined = pd.DataFrame({"ret_ind": ret_ind_, "ret_bm": benchmark})

    # 计算滚动协方差和滚动方差
    rolling_cov = combined["ret_ind"].rolling(window=window_size, min_periods=min_periods).cov(combined["ret_bm"])
    rolling_var = combined["ret_bm"].rolling(window=window_size, min_periods=min_periods).var()

    # 计算 Beta
    rolling_beta = rolling_cov / rolling_var

    return -rolling_beta


# 计算EMA差值因子 (新增)
def calculate_ema_diff(close_prices, fast_window, slow_window):
    """
    计算快速EMA和慢速EMA的差值因子

    参数:
    close_prices: 收盘价序列
    fast_window: 快速EMA窗口
    slow_window: 慢速EMA窗口

    返回:
    EMA差值因子
    """
    close_series = pd.Series(close_prices)
    fast_ema = close_series.ewm(span=fast_window, adjust=False).mean()
    slow_ema = close_series.ewm(span=slow_window, adjust=False).mean()

    # 快EMA减慢EMA的差值，并标准化
    ema_diff = (fast_ema - slow_ema) / close_series

    return ema_diff


# 计算量价强度因子 (新增)
def calculate_volume_price_strength(df, column_name, window):
    """
    计算量价强度因子：价格上涨时放量程度与价格下跌时缩量程度的综合指标

    参数:
    df: 包含价格和成交量的DataFrame
    column_name: 输出列名
    window: 滚动窗口大小

    返回:
    量价强度因子
    """
    # 计算日度收益率
    returns = df["CLOSE"].pct_change()

    # 计算成交量变化率
    volume_change = df["VOLUME"].pct_change()

    # 创建上涨放量指标和下跌缩量指标
    up_volume_strength = (returns > 0) * volume_change  # 上涨时的成交量变化
    down_volume_strength = (returns < 0) * (-volume_change)  # 下跌时的成交量变化（负值取反）

    # 综合指标：上涨放量 + 下跌缩量
    strength = up_volume_strength + down_volume_strength

    # 使用窗口平滑
    df[column_name] = strength.rolling(window=window).mean()

    return df[column_name]


# 直接返回
def straight_return(df):
    return df


def calculate_icir(factor, forward_ret, rollingwindow1, rollingwindow2):
    """
    计算因子ICIR值。
    """

    def safe_corr(row):
        ret = forward_ret.loc[row.name]  # 取出对应的收益率
        if np.all(row == row.iloc[0]) or np.all(ret == ret.iloc[0]):
            return 0
        return row.corr(ret, method="spearman")

    # 计算IC（Spearman相关性）
    ic = factor.apply(safe_corr, axis=1)

    # 计算滚动均值和滚动标准差
    ic_rolling_mean = ic.rolling(rollingwindow1).mean()
    ic_rolling_std = ic.rolling(rollingwindow2).std()

    # 计算ICIR
    ic_ir = ic_rolling_mean / ic_rolling_std

    return ic_ir
