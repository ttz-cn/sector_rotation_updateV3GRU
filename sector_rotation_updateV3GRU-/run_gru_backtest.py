import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import backtrader as bt
import pyfolio
from datetime import datetime, timedelta
import config
from utils.utils_tools import load_panel_dropd
from utils.utils_boyi import RiskAnalyzer
import glob



# 自定义 PandasData Feed
class CustomPandasData(bt.feeds.PandasData):
    # 添加额外的自定义列
    lines = ("WEST_AVGROE_FY1", "WEST_EPS_FY1", "AMT", "PCT_CHG", "TURN")
    
    # 定义这些列在DataFrame中的位置
    params = (
        # 保留默认参数
        ('open', 'OPEN'),
        ('high', 'HIGH'),
        ('low', 'LOW'),
        ('close', 'CLOSE'),
        ('volume', 'VOLUME'),
        # 添加自定义参数
        ('WEST_AVGROE_FY1', 'WEST_AVGROE_FY1'),
        ('WEST_EPS_FY1', 'WEST_EPS_FY1'),
        ('AMT', 'AMT'),
        ('PCT_CHG', 'PCT_CHG'),
        ('TURN', 'TURN'),
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
            
        self.historical_weights = pd.read_csv(self.p.weights_file, index_col=0, parse_dates=True).shift(-1)
        
        if self.p.verbose:
            print(f"加载历史权重数据，形状: {self.historical_weights.shape}")
            print(f"可用调仓日期: {len(self.historical_weights)} 个")
            print(f"调仓频率: {self.p.rebalance_freq} 天")
    
    # 添加订单通知函数，用于调试订单执行情况
    def notify_order(self, order):
        if self.p.verbose:
            if order.status == order.Accepted:
                print(f"order accepted: {order.data._name}, size: {order.created.size}")
                return
                
            if order.status == order.Completed:
                if order.isbuy():
                    print(f"buy order executed: {order.data._name}")
                    print(f"  执行价格: {order.executed.price:.2f}")
                    print(f"  执行大小: {order.executed.size:.6f}")
                    print(f"  执行价值: {order.executed.value:.2f}")
                    print(f"  执行佣金: {order.executed.comm:.2f}")
                else:
                    print(f"sell order executed: {order.data._name}")
                    print(f"  执行价格: {order.executed.price:.2f}")
                    print(f"  执行大小: {order.executed.size:.6f}")
                    print(f"  执行价值: {order.executed.value:.2f}")
                    print(f"  执行佣金: {order.executed.comm:.2f}")
            elif order.status == order.Canceled:
                print(f"order canceled: {order.data._name}")
            elif order.status == order.Margin:
                print(f"margin error: {order.data._name}, 需要资金: {order.created.size * order.data.close[0]:.2f}")
            elif order.status == order.Rejected:
                print(f"order rejected: {order.data._name}")
            elif order.status == order.Expired:
                print(f"order expired: {order.data._name}")
        
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
                    print(f"当前投资组合价值: {self.broker.getvalue():.2f}, 现金: {self.broker.getcash():.2f}")
                
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
                    
                # 打印当前持仓情况
                if self.p.verbose:
                    print("当前持仓情况:")
                    total_value = self.broker.getvalue()
                    print(f"  现金: {self.broker.getcash():.2f} ({self.broker.getcash()/total_value:.2%})")
                    for name, data in self.data_dict.items():
                        position = self.getposition(data)
                        if position.size != 0:
                            value = position.size * data.close[0]
                            print(f"  {name}: 大小={position.size:.6f}, 价值={value:.2f} ({value/total_value:.2%})")
    
    def record_positions(self, date):
        """记录当前持仓状态"""
        if not self.p.log_positions:
            return
            
        # 构建持仓记录 - 简化格式
        position_record = {
            "Date": date,  
            "cash": self.broker.getcash(),
        }
        
        # 记录每个行业的持仓价值
        for name, data in self.data_dict.items():
            position = self.getposition(data)
            size = position.size
            price = data.close[0]
            value = size * price if size != 0 else 0
            position_record[name] = value
        
        # 添加到持仓历史记录
        self.position_history.append(position_record)
    
    def stop(self):
        # 记录最后一天的持仓状态
        if self.p.log_positions:
            # 获取当前日期
            current_date = self.data0.datetime.date(0)
            
            # 最后一天的持仓记录
            position_record = {
                "Date": current_date,  
                "cash": self.broker.getcash(),
            }
            # 记录每个行业的持仓价值
            for name, data in self.data_dict.items():
                position = self.getposition(data)
                size = position.size
                price = data.close[0]
                value = size * price if size != 0 else 0
                position_record[name] = value
            
            # 添加到持仓历史记录
            self.position_history.append(position_record)
            
        if self.p.verbose:
            print("\n回测结束")
            print(f"最终投资组合价值: {self.broker.getvalue():.2f}")
            print(f"最终现金: {self.broker.getcash():.2f} ({self.broker.getcash()/self.broker.getvalue():.2%})")

    def update_portfolio(self, target_weights):
        """根据目标权重更新投资组合"""
        # 取消所有未完成订单
        for data, order in self.order_dict.items():
            if order and order.status != order.Completed:
                self.cancel(order)
        
        # 更新订单字典
        self.order_dict = {}
        
        # 第一步：计算每个资产的目标权重和当前权重
        asset_weights = {}
        for name, data in self.data_dict.items():
            # 获取目标权重，如果不存在或为NaN，则设为0
            target_weight = target_weights.get(name, 0)
            if pd.isna(target_weight):
                target_weight = 0
            
            # 获取当前持仓
            position = self.getposition(data)
            
            # 计算当前权重
            portfolio_value = self.broker.getvalue()
            current_weight = position.size * data.close[0] / portfolio_value if position.size != 0 else 0
            
            # 存储权重信息
            asset_weights[name] = {
                'data': data,
                'target_weight': target_weight,
                'current_weight': current_weight,
                'weight_diff': target_weight - current_weight
            }
        
        # 第二步：先执行所有卖出操作（权重需要减少的资产）
        for name, info in asset_weights.items():
            data = info['data']
            target_weight = info['target_weight']
            current_weight = info['current_weight']
            weight_diff = info['weight_diff']
            
            # 如果需要减少权重（卖出）
            if weight_diff < -0.05:  # 权重需要减少超过5%
                self.order_dict[data] = self.order_target_percent(data, target_weight)
                if self.p.verbose:
                    print(f"卖出 {name}: {current_weight:.2%} -> {target_weight:.2%} (差异: {weight_diff:.2%})")
            
            # 如果需要完全清空持仓
            elif current_weight > 0 and target_weight == 0:
                self.order_dict[data] = self.order_target_percent(data, 0)
                if self.p.verbose:
                    print(f"清空 {name} 的持仓: {current_weight:.2%} -> 0%")
        
        # 第三步：再执行所有买入操作（权重需要增加的资产）
        for name, info in asset_weights.items():
            data = info['data']
            target_weight = info['target_weight']
            current_weight = info['current_weight']
            weight_diff = info['weight_diff']
            
            # 如果需要增加权重（买入）
            if weight_diff > 0.05:  # 权重需要增加超过5%
                # 检查当日交易量是否足够
                if data.volume[0] > 0:  # 确保有交易量
                    self.order_dict[data] = self.order_target_percent(data, target_weight)
                    if self.p.verbose:
                        print(f"买入 {name}: {current_weight:.2%} -> {target_weight:.2%} (差异: {weight_diff:.2%})")
                else:
                    if self.p.verbose:
                        print(f"跳过 {name}: 当日无交易量")


def get_latest_weights_file():
    """获取最新的权重文件"""
    weight_files = glob.glob("./output/historical_weights_*.csv")
    if not weight_files:
        raise FileNotFoundError("未找到历史权重文件，请先运行run_optimizer.py生成权重")
    
    # 按文件名排序，选择最新的文件
    return sorted(weight_files)[-1]


def prepare_data(fromdate, todate):
    """准备回测数据"""

     # 加载基准数据 - 使用wind全A指数
    wind_data = pd.read_hdf("./db/time_series_data.h5", key="wind_a")
    wind_data = wind_data.groupby(wind_data.index).tail(1).ffill() 

    # 加载ETF数据
    etf_panel_data = load_panel_dropd("./db/panel_data.h5", key="sector").dropna()

    # 准备ETF数据
    etf_data_dic = {}
    for fund_code, group in etf_panel_data.groupby("industry_code"):
        group = group.copy()
        group["Date"] = pd.to_datetime(group["Date"])
        group.set_index("Date", inplace=True)
        group = group.resample("B").asfreq()
        etf_data_dic[fund_code] = group[
            ["WEST_AVGROE_FY1", "WEST_EPS_FY1", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "AMT", "PCT_CHG", "TURN"]
        ]
    return etf_data_dic, wind_data


def run_gru_backtest(weights_file=None, fromdate=None, todate=None, benchmark="wind_a", rebalance_freq=5, log_positions=True):
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
    historical_weights_org = pd.read_csv(weights_file, index_col=0, parse_dates=True)
    historical_weights = historical_weights_org.shift(-1)

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
    
    # 打印日期范围
    print(f"回测日期范围: {fromdate} 至 {todate}")
    
    # 创建 Cerebro 引擎
    cerebro = bt.Cerebro()

    
    # 准备数据
    etf_data_dic, wind_data = prepare_data(fromdate, todate)
    
    # 验证权重文件中的行业代码与ETF数据中的行业代码匹配
    industry_codes_in_data = set(etf_data_dic.keys())
    industry_codes_in_weights = set(historical_weights.columns)

    missing_in_data = industry_codes_in_weights - industry_codes_in_data
    if missing_in_data:
        print(f"警告: 权重文件中有 {len(missing_in_data)} 个行业在ETF数据中不存在: {missing_in_data}")

    # 添加数据到Cerebro引擎
    for fund_code, df in etf_data_dic.items():
        # 只添加权重文件中存在的行业
        if fund_code in industry_codes_in_weights:
            # 创建dataframe副本以避免警告
            df_copy = df.copy()
            # 确保日期索引格式正确
            if not isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy.index = pd.to_datetime(df_copy.index)
                
            data = CustomPandasData(
                dataname=df_copy,
                fromdate=fromdate,
                todate=todate
            )             
            data._name = fund_code  # 设置名称
            cerebro.adddata(data, name=fund_code)  # 添加数据到 Cerebro 引擎
    
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
    # 计算基准收益率
    benchmark_rets = wind_data.resample("D").ffill().pct_change(1)
    # 截取日期范围
    benchmark_rets = benchmark_rets.loc[fromdate:todate]
    
    # 确保时区一致性
    if hasattr(returns, 'index'):
        returns.index = returns.index.tz_localize(None)  # 去除tzinfo
    if benchmark_rets is not None and hasattr(benchmark_rets, 'index'):
        benchmark_rets.index = benchmark_rets.index.tz_localize(None)
    
    # 保存结果文件名中包含调仓频率信息
    output_file_prefix = f"gru_backtest_{rebalance_freq}d_{datetime.now().strftime('%Y%m%d')}"
    
    # 使用pyfolio生成回测报告
    print("\n生成pyfolio回测报告...")
    try:        
        # 确保returns和benchmark_rets是Series而不是DataFrame
        if isinstance(returns, pd.DataFrame) and returns.shape[1] == 1:
            returns = returns.iloc[:, 0]
            returns.name = 'strategy'  # 添加name属性
        
        if isinstance(benchmark_rets, pd.DataFrame) and benchmark_rets.shape[1] == 1:
            benchmark_rets = benchmark_rets.iloc[:, 0]
            benchmark_rets.name = 'benchmark'  # 添加name属性
            # 使用报表，禁用intraday推断

        pyfolio.create_full_tear_sheet(
            returns=returns,
            benchmark_rets=benchmark_rets,
            positions=positions,
            transactions=transactions,
            estimate_intraday="False"  # 禁用日内交易推断
        )
        plt.tight_layout()
        plt.savefig(f"./output/{output_file_prefix}_pyfolio_report.png", dpi=300, bbox_inches='tight')
        plt.close() # 关闭图像窗口
    except Exception as e:
        print(f"生成pyfolio报告时出错: {e}")
    
    # 计算风险分析
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
    
    # 保存收益率数据
    if returns is not None:
        returns.to_csv(f"./output/{output_file_prefix}_returns.csv")
    if benchmark_rets is not None:
        benchmark_rets.to_csv(f"./output/{output_file_prefix}_benchmark_rets.csv")
    # 保存transactions数据
    if transactions is not None:
        transactions.to_csv(f"./output/{output_file_prefix}_transactions.csv")
    # 保存策略的position_history,pyfolio导出的不中使
    if log_positions and hasattr(results[0], 'position_history') and results[0].position_history:
        position_df = pd.DataFrame(results[0].position_history)
        position_df.to_csv(f"./output/{output_file_prefix}_all_positions.csv", index=False)
        print(f"所有行业持仓记录已保存到: ./output/{output_file_prefix}_all_positions.csv")

    return results, returns, benchmark_rets, risk_metrics


if __name__ == "__main__":
 
    weights_path = './output/historical_weights_high_top5_0.01.csv'
    fromdate = '2025-01-10'
    todate = '2025-06-25'
    freq = 5
    log_positions = True
        
    # 运行回测
    results, returns, benchmark_rets, risk_metrics = run_gru_backtest(
        weights_file=weights_path,
        fromdate=fromdate,
        todate=todate,
        rebalance_freq=freq,
        log_positions=log_positions # 是否记录持仓
    )

    print("\n回测完成，结果已保存。") 