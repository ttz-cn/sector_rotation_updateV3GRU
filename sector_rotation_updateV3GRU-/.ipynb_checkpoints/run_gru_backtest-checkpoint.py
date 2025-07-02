import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import backtrader as bt
import pyfolio as pf
from datetime import datetime, timedelta
import config
from utils.utils_tools import load_panel_dropd
from utils.utils_boyi import RiskAnalyzer
import glob



# 自定义 PandasData Feed
class CustomPandasData(bt.feeds.PandasData):
    lines = ("WEST_AVGROE_FY1", "WEST_EPS_FY1", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "AMT", "PCT_CHG", "TURN")

    params = (
        ("datetime", None),
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


# GRU预测结果回测策略
class GRUBacktestStrategy(bt.Strategy):
    params = (
        ("weights_file", None),  # 历史权重文件路径
        ("verbose", False),  # 是否打印详细信息
        ("rebalance_freq", 20),  # 调仓频率（天数），默认20天
        ("log_positions", True),  # 是否记录持仓
    )
    
    def __init__(self):
        # 初始化变量
        self.order_dict = {}  # 订单字典
        self.current_weights = {}  # 当前权重
        self.last_rebalance_date = None  # 上次调仓日期
        self.data_dict = {}  # 数据字典
        self.day_count = 0  # 天数计数器
        self.position_history = []  # 持仓历史记录
        
        # 初始化数据字典
        for i, d in enumerate(self.datas):
            self.data_dict[d._name] = d
            
        # 加载历史权重数据
        if self.p.weights_file is None:
            raise ValueError("必须提供历史权重文件路径")
            
        self.historical_weights = pd.read_csv(self.p.weights_file, index_col=0, parse_dates=True)
        
        if self.p.verbose:
            print(f"加载历史权重数据，形状: {self.historical_weights.shape}")
            print(f"可用调仓日期: {len(self.historical_weights)} 个")
            print(f"调仓频率: {self.p.rebalance_freq} 天")
        
    def next(self):
        # 获取当前日期
        current_date = self.data.datetime.datetime()
        self.day_count += 1
        
        # 检查是否需要调仓 (基于天数)
        if self.day_count >= self.p.rebalance_freq or self.last_rebalance_date is None:
            # 找到最接近当前日期但不超过当前日期的调仓日期
            mark = self.historical_weights.index <= current_date
            valid_dates = self.historical_weights.index[mark]
            
            if not valid_dates.empty:
                latest_rebalance_date = valid_dates[-1]
                
                # 进行调仓
                if self.p.verbose:
                    print(f"\n执行调仓 - 当前日期: {current_date}, 调仓日期: {latest_rebalance_date}")
                
                # 获取该日期的权重
                target_weights = self.historical_weights.loc[latest_rebalance_date]
                
                # 更新投资组合
                self.update_portfolio(target_weights)
                
                # 更新上次调仓日期和重置天数计数器
                self.last_rebalance_date = latest_rebalance_date
                self.day_count = 0
                
                # 记录当前持仓状态
                if self.p.log_positions:
                    self.record_positions(current_date)
    
    def record_positions(self, date):
        """记录当前持仓状态"""
        if not self.p.log_positions:
            return
            
        portfolio_value = self.broker.getvalue()
        
        # 构建持仓记录 - 只记录日期和行业权重
        position_record = {
            "Date": date,
        }
        
        # 只记录每个行业的权重
        for name, data in self.data_dict.items():
            position = self.getposition(data)
            size = position.size
            value = size * data.close[0] if size != 0 else 0
            weight = value / portfolio_value if portfolio_value > 0 else 0
            
            # 只保存权重
            position_record[name] = weight
        
        # 添加到持仓历史记录
        self.position_history.append(position_record)
        
    def update_portfolio(self, target_weights):
        """根据目标权重更新投资组合"""
        # 取消所有未完成订单
        for data, order in self.order_dict.items():
            if order and order.status != order.Completed:
                self.cancel(order)
        
        # 更新订单字典
        self.order_dict = {}
        
        # 对每个资产进行调仓
        for name, data in self.data_dict.items():
            # 获取目标权重，如果不存在或为NaN，则设为0
            target_weight = target_weights.get(name, 0)
            if pd.isna(target_weight):
                target_weight = 0
            
            # 获取当前持仓
            position = self.getposition(data)
            current_size = position.size
            
            # 计算当前权重
            portfolio_value = self.broker.getvalue()
            current_weight = position.size * data.close[0] / portfolio_value if position.size != 0 else 0
            
            # 如果权重差异超过阈值，则调整
            weight_diff = target_weight - current_weight
            if abs(weight_diff) > 0.01:  # 1%的权重差异阈值
                # 使用order_target_percent直接按权重下单
                self.order_dict[data] = self.order_target_percent(data, target_weight)
                if self.p.verbose:
                    print(f"调整 {name}: {current_weight:.2%} -> {target_weight:.2%} (差异: {weight_diff:.2%})")
            elif current_weight > 0 and target_weight == 0:
                # 确保清空权重为0的持仓
                self.order_dict[data] = self.order_target_percent(data, 0)
                if self.p.verbose:
                    print(f"清空 {name} 的持仓: {current_weight:.2%} -> 0%")
    
    def stop(self):
        """策略结束时的处理"""
        if self.p.log_positions and self.position_history:
            # 将持仓历史记录转换为DataFrame并保存
            positions_df = pd.DataFrame(self.position_history)
            positions_df.to_csv(f"./output/positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
            
            if self.p.verbose:
                print(f"持仓历史记录已保存，共 {len(self.position_history)} 条记录")


def get_latest_weights_file():
    """获取最新的权重文件"""
    weight_files = glob.glob("./output/historical_weights_*.csv")
    if not weight_files:
        raise FileNotFoundError("未找到历史权重文件，请先运行run_optimizer.py生成权重")
    
    # 按文件名排序，选择最新的文件
    return sorted(weight_files)[-1]


def prepare_data(fromdate, todate):
    """准备回测数据"""
    # 加载ETF数据
    etf_panel_data = load_panel_dropd("./db/panel_data.h5", key="sector").dropna()
    
    # 加载基准数据 - 使用wind全A指数
    wind_data = pd.read_hdf("./db/time_series_data.h5", key="wind_a")
    if isinstance(wind_data, pd.DataFrame):
        wind_data = wind_data.groupby(wind_data.index.date).tail(1)  # 保留每个日期的最后一条
    
    # 准备ETF数据
    etf_data_dic = {}
    for fund_code, group in etf_panel_data.groupby("industry_code"):
        group = group.copy()
        group["Date"] = pd.to_datetime(group["Date"])
        group.set_index("Date", inplace=True)
        etf_data_dic[fund_code] = group[
            ["WEST_AVGROE_FY1", "WEST_EPS_FY1", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "AMT", "PCT_CHG", "TURN"]
        ]
    
    return etf_data_dic, wind_data


def run_gru_backtest(weights_file=None, fromdate=None, todate=None, benchmark="wind_a", rebalance_freq=20, log_positions=True):
    """
    运行基于GRU预测结果的回测
    
    参数:
    - weights_file: 历史权重文件路径，如果为None则使用最新的权重文件
    - fromdate: 回测开始日期，格式为"YYYY-MM-DD"
    - todate: 回测结束日期，格式为"YYYY-MM-DD"
    - benchmark: 基准指数，默认为'wind_a'
    - rebalance_freq: 调仓频率（天数），默认20天
    - log_positions: 是否记录持仓，默认为True
    
    返回:
    - 回测结果
    """
    print("开始GRU预测回测...")
    
    # 如果未指定权重文件，则使用最新的权重文件
    if weights_file is None:
        weights_file = get_latest_weights_file()
    
    print(f"使用权重文件: {weights_file}")
    print(f"调仓频率: {rebalance_freq} 天")
    
    # 加载权重数据
    historical_weights = pd.read_csv(weights_file, index_col=0, parse_dates=True)
    
    # 打印权重文件信息
    print(f"权重文件包含 {len(historical_weights)} 个调仓日期")
    print(f"权重文件包含 {len(historical_weights.columns)} 个行业")
    print(f"行业代码: {list(historical_weights.columns)}")
    
    # 设置回测日期范围
    if fromdate is None:
        # 使用权重数据的第一个日期
        fromdate = historical_weights.index[0]
    else:
        fromdate = pd.to_datetime(fromdate)
        
    if todate is None:
        # 使用权重数据的最后一个日期
        todate = historical_weights.index[-1] + pd.Timedelta(days=30)  # 额外增加30天，确保最后的交易能够执行
    else:
        todate = pd.to_datetime(todate)
    
    print(f"回测日期范围: {fromdate.strftime('%Y-%m-%d')} 至 {todate.strftime('%Y-%m-%d')}")
    
    # 创建 Cerebro 引擎
    cerebro = bt.Cerebro()
    
    # 准备数据
    etf_data_dic, wind_data = prepare_data(fromdate, todate)
    
    # 验证权重文件中的行业代码与ETF数据中的行业代码匹配
    industry_codes_in_data = set(etf_data_dic.keys())
    industry_codes_in_weights = set(historical_weights.columns)

    print(f"ETF数据中的行业数量: {len(industry_codes_in_data)}")
    print(f"权重文件中的行业数量: {len(industry_codes_in_weights)}")

    common_codes = industry_codes_in_data & industry_codes_in_weights
    print(f"两者共有的行业数量: {len(common_codes)}")

    missing_in_data = industry_codes_in_weights - industry_codes_in_data
    if missing_in_data:
        print(f"警告: 权重文件中有 {len(missing_in_data)} 个行业在ETF数据中不存在: {missing_in_data}")

    # 添加数据到Cerebro引擎
    for fund_code, df in etf_data_dic.items():
        # 只添加权重文件中存在的行业
        if fund_code in industry_codes_in_weights:
            # 使用自定义数据馈送
            data_feed = CustomPandasData(
                dataname=df,  # 使用dataname参数传递数据
                fromdate=fromdate,  # 回测开始日期
                todate=todate,  # 回测结束日期
                name=fund_code  # 添加名称参数
            )
            cerebro.adddata(data_feed, name=fund_code)  # 添加数据到 Cerebro 引擎
    
    # 添加策略
    cerebro.addstrategy(
        GRUBacktestStrategy,
        weights_file=weights_file,
        verbose=True,
        rebalance_freq=rebalance_freq,
        log_positions=log_positions
    )
    
    # 设置初始资金
    cerebro.broker.setcash(100000.0)
    # 设置交易手续费
    cerebro.broker.setcommission(commission=0.001)  # 0.1%手续费
    # 添加分析器，用于记录账户净值和交易数据
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")
    
    # 运行回测
    print(f"初始资金: {cerebro.broker.getvalue():.2f}")
    results = cerebro.run()
    print(f"最终资金: {cerebro.broker.getvalue():.2f}")
    
    # 提取 Pyfolio 分析器结果
    pyfolio_analyzer = results[0].analyzers.pyfolio
    # 获取回测期间的账户净值和交易数据
    returns, positions, transactions, gross_lev = pyfolio_analyzer.get_pf_items()
    
    # 获取benchmark_rets
    # 使用wind_data计算基准收益率
    wind_returns = wind_data.resample("D").ffill().pct_change(1).loc[fromdate:todate]
    # 如果wind_data是DataFrame，取第一列；如果是Series，直接使用
    if isinstance(wind_returns, pd.DataFrame):
        benchmark_rets = wind_returns.iloc[:, 0]  # 取第一列作为基准
    else:
        benchmark_rets = wind_returns
    
    # 确保时区一致性
    returns.index = returns.index.tz_localize(None)  # 去除tzinfo
    benchmark_rets.index = benchmark_rets.index.tz_localize(None)  # 确保基准指数也没有时区信息
    
    # 保存结果文件名中包含调仓频率信息
    output_file_prefix = f"gru_backtest_{rebalance_freq}d_{datetime.now().strftime('%Y%m%d')}"
    
    # 使用pyfolio生成完整的回测报告
    try:
        print("\n生成pyfolio回测报告...")
        pf.create_full_tear_sheet(
            returns, 
            positions=positions, 
            transactions=transactions, 
            benchmark_rets=benchmark_rets
        )
        plt.tight_layout()
        
        # 保存pyfolio报告
        plt.savefig(f"./output/{output_file_prefix}_pyfolio_report.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"生成pyfolio报告时出错: {e}")
        print("无法生成pyfolio报告，请检查数据格式或pyfolio安装情况。")
    
    # 计算风险分析
    try:
        riskanalyzer_ = RiskAnalyzer(returns=returns, benchmarkreturns=benchmark_rets, period="daily")
        risk_metrics = riskanalyzer_.run()
        
        # 保存风险指标
        if risk_metrics:
            risk_metrics_df = pd.DataFrame(risk_metrics).T
            risk_metrics_df.to_csv(f"./output/{output_file_prefix}_metrics.csv")
            
            # 显示主要风险指标
            print("\n主要风险指标:")
            if '年化收益率' in risk_metrics:
                print(f"年化收益率: {risk_metrics['年化收益率']:.2%}")
            if '最大回撤' in risk_metrics:
                print(f"最大回撤: {risk_metrics['最大回撤']:.2%}")
            if '夏普比率' in risk_metrics:
                print(f"夏普比率: {risk_metrics['夏普比率']:.2f}")
            if '信息比率' in risk_metrics:
                print(f"信息比率: {risk_metrics['信息比率']:.2f}")
    except Exception as e:
        print(f"计算风险指标时出错: {e}")
        risk_metrics = None
    
    # 保存收益率数据
    returns.to_csv(f"./output/{output_file_prefix}_returns.csv")
    benchmark_rets.to_csv(f"./output/{output_file_prefix}_benchmark_rets.csv")
    
    return results, returns, benchmark_rets, risk_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="基于GRU预测结果的回测")
    parser.add_argument("--weights", type=str, help="历史权重文件路径")
    parser.add_argument("--fromdate", type=str, help="回测开始日期 (YYYY-MM-DD 格式)")
    parser.add_argument("--todate", type=str, help="回测结束日期 (YYYY-MM-DD 格式)")
    parser.add_argument("--freq", type=int, default=20, help="调仓频率（天数），默认20天")
    parser.add_argument("--no-log-positions", action="store_true", help="不记录持仓信息")
    args = parser.parse_args()
    
    # 运行回测
    results, returns, benchmark_rets, risk_metrics = run_gru_backtest(
        weights_file=args.weights,
        fromdate=args.fromdate,
        todate=args.todate,
        rebalance_freq=args.freq,
        log_positions=not args.no_log_positions
    )
    
    print("\n回测完成，结果已保存。") 