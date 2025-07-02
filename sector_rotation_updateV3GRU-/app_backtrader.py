import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import backtrader as bt
import pyfolio as pf

from src.src_EtfRotationStrategy import ETFRotationStrategy, calculate_etf_selection
from utils.utils_boyi import RiskAnalyzer
from utils.utils_tools import update_etfs
import config


# 自定义 PandasData Feed
class CustomPandasData(bt.feeds.PandasData):
    lines = ("WEST_AVGROE_FY1", "WEST_EPS_FY1", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "AMT", "PCT_CHG", "TURN")

    params = (
        ("dataname", None),
        ("fromdate", None),
        ("todate", None),
        ("WEST_AVGROE_FY1", -1),
        ("WEST_EPS_FY1", -1),
        ("OPEN", -1),
        ("HIGH", -1),
        ("LOW", -1),
        ("CLOSE", -1),
        ("VOLUME", -1),
        ("AMT", -1),
        ("PCT_CHG", -1),
        ("TURN", -1),
    )


if __name__ == "__main__":

    rebalance_period = "W"  # 初始化调仓周期
    fromdate = pd.Timestamp("2022-12-30")
    todate = pd.Timestamp("2025-03-31")

    # 创建 Cerebro 引擎
    cerebro = bt.Cerebro()

    # AddEtfData
    etf_panel_data = pd.read_hdf("./db/panel_data.h5", key="etf").dropna()  # ETF数据提取
    wind_data = pd.read_hdf("./db/time_series_data.h5", key="wind_a")
    wind_data = wind_data.groupby(wind_data.index).tail(1)  # 保留每个index的最后一条
    etf_data_dic = {}
    for fund_code, group in etf_panel_data.groupby("fund_code"):
        group = group.copy()
        group["Date"] = pd.to_datetime(group["Date"])
        group.set_index("Date", inplace=True)
        etf_data_dic[fund_code] = group[
            ["WEST_AVGROE_FY1", "WEST_EPS_FY1", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "AMT", "PCT_CHG", "TURN"]
        ]

    for fund_code, df in etf_data_dic.items():
        data_feed = CustomPandasData(
            dataname=df,
            fromdate=fromdate,  # 回测开始日期
            todate=todate,  # 回测结束日期
        )
        cerebro.adddata(data_feed, name=fund_code)  # 添加数据到 Cerebro 引擎

    # 加载threshold和filter数据
    composite_funda_factor = pd.read_hdf(
        "./db/composite_factor.h5", key="composite_funda_factor-{}".format(rebalance_period)
    ).dropna()  # 起到阈值作用的指标
    list_factor = [
        pd.read_hdf("./db/composite_factor.h5", key="composite_tech_factor-{}".format(rebalance_period)).dropna(),
        pd.read_hdf("./db/composite_factor.h5", key="composite_crd_factor-{}".format(rebalance_period)).dropna(),
    ]  # composite from icir
    dic_indicator = {"factor_{}".format(i): indc for i, indc in enumerate(list_factor, 0)}  # 起到etf选择作用的指标

    selected_industrys = calculate_etf_selection(
        composite_factor=composite_funda_factor, dic_indicator=dic_indicator, group1=5, group2=3
    )  # 计算选中的行业
    selected_etfs = selected_industrys.apply(
        lambda row: update_etfs(config.dic_Industry2Etf, row), axis=1
    )  # 行业映射到标的ETF
    cerebro.addstrategy(ETFRotationStrategy, selected_etfs=selected_etfs, frequency="weekly")  # 选中的ETF列表 )

    # 设置初始资金
    cerebro.broker.setcash(100000.0)
    # 设置交易手续费
    cerebro.broker.setcommission(commission=0.001)  # 0.1%手续费
    # 添加分析器，用于记录账户净值和交易数据
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")

    # 运行回测
    print("初始资金: %.2f" % cerebro.broker.getvalue())
    results = cerebro.run()
    print("最终资金: %.2f" % cerebro.broker.getvalue())

    # 提取 Pyfolio 分析器结果
    pyfolio_analyzer = results[0].analyzers.pyfolio
    # 获取回测期间的账户净值和交易数据
    returns, positions, transactions, gross_lev = pyfolio_analyzer.get_pf_items()

    # 获取benchmark_rets
    # 使用wind_data计算基准收益率
    wind_returns = (wind_data.resample("D").ffill()).pct_change(1).loc[fromdate:todate]
    # 如果wind_data是DataFrame，取第一列；如果是Series，直接使用
    if isinstance(wind_returns, pd.DataFrame):
        benchmark_rets = wind_returns.iloc[:, 0]  # 取第一列作为基准
    else:
        benchmark_rets = wind_returns

    # create full tear sheet
    returns.index = returns.index.tz_localize(None)  # 去除tzinfo
    pf.create_full_tear_sheet(returns, positions=positions, transactions=transactions, benchmark_rets=benchmark_rets)
    plt.tight_layout()
    plt.show()  # 显示图表

    # 计算风险分析
    riskanalyzer_ = RiskAnalyzer(returns=returns, benchmarkreturns=benchmark_rets, period="daily")
    riskanalyzer_.run()
