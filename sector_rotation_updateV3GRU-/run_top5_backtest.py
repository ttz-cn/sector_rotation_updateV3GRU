import pandas as pd
import numpy as np
import backtrader as bt
import pyfolio
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from utils.utils_tools import load_panel_dropd
from utils.utils_boyi import RiskAnalyzer

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

# Top5等权配置回测策略
class Top5EqualWeightStrategy(bt.Strategy):
    params = (
        ("predictions_file", None),  # 预测文件路径
        ("verbose", True),           # 是否打印详细信息
        ("log_positions", True),     # 是否记录持仓
    )
    
    def __init__(self):
        # 初始化变量
        self.order_dict = {}         # 订单字典
        self.current_weights = {}    # 当前权重
        self.last_rebalance_date = None  # 上次调仓日期
        self.data_dict = {}          # 数据字典
        self.position_history = []   # 持仓历史记录
        
        # 初始化数据字典
        for i, d in enumerate(self.datas):
            self.data_dict[d._name] = d
            
        # 加载预测数据
        if self.p.predictions_file is None:
            raise ValueError("必须提供预测文件路径")
            
        self.predictions = pd.read_csv(self.p.predictions_file, index_col=0, parse_dates=True)
        
        if self.p.verbose:
            print(f"加载预测数据，形状: {self.predictions.shape}")
            print(f"预测日期范围: {self.predictions.index[0]} 至 {self.predictions.index[-1]}")
    
    def next(self):
        # 获取当前日期
        current_date = self.datas[0].datetime.datetime(0)
        
        # 检查是否是周一
        is_monday = current_date.weekday() == 0
        
        # 如果是周一或者尚未进行过调仓，则进行调仓
        if is_monday or self.last_rebalance_date is None:
            # 找到最接近当前日期但不超过当前日期的预测日期
            valid_dates = self.predictions.index[self.predictions.index <= current_date]
            
            if not valid_dates.empty:
                latest_pred_date = valid_dates[-1]
                
                # 检查是否已经使用过这个调仓日期
                if self.last_rebalance_date != latest_pred_date:
                    # 进行调仓
                    if self.p.verbose:
                        print(f"\n执行调仓 - 当前日期: {current_date}, 预测日期: {latest_pred_date}")
                    
                    # 获取该日期的预测值
                    predictions_row = self.predictions.loc[latest_pred_date]
                    
                    # 选择前5名行业
                    top5_industries = predictions_row.nlargest(5).index.tolist()
                    
                    if self.p.verbose:
                        print(f"选择的前5名行业: {top5_industries}")
                    
                    # 创建等权重字典 (每个行业20%)
                    target_weights = {ind: 0.2 for ind in top5_industries}
                    
                    # 更新投资组合
                    self.update_portfolio(target_weights)
                    
                    # 更新上次调仓日期
                    self.last_rebalance_date = latest_pred_date
                    
                    # 记录当前持仓状态
                    if self.p.log_positions:
                        self.record_positions(current_date)
            else:
                if self.p.verbose:
                    print(f"警告: 当前日期 {current_date} 之前没有有效的预测日期")
    
    def record_positions(self, date):
        """记录当前持仓状态"""
        if not self.p.log_positions:
            return
            
        # 构建持仓记录
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
            # 获取目标权重，如果不在top5中则为0
            target_weight = target_weights.get(name, 0)
            
            # 获取当前持仓
            position = self.getposition(data)
            current_size = position.size
            
            # 计算当前权重
            portfolio_value = self.broker.getvalue()
            current_weight = position.size * data.close[0] / portfolio_value if position.size != 0 else 0
            
            # 如果权重差异超过阈值，则调整
            weight_diff = target_weight - current_weight
            if abs(weight_diff) > 0.05:  # 5%的权重差异阈值
                # 使用order_target_percent直接按权重下单
                self.order_dict[data] = self.order_target_percent(data, target_weight)
                if self.p.verbose:
                    print(f"调整 {name}: {current_weight:.2%} -> {target_weight:.2%} (差异: {weight_diff:.2%})")
            elif current_weight > 0 and target_weight == 0:
                # 确保清空权重为0的持仓
                self.order_dict[data] = self.order_target_percent(data, 0)
                if self.p.verbose:
                    print(f"清空 {name} 的持仓: {current_weight:.2%} -> 0%")

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
        etf_data_dic[fund_code] = group[
            ["WEST_AVGROE_FY1", "WEST_EPS_FY1", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "AMT", "PCT_CHG", "TURN"]
        ]
    return etf_data_dic, wind_data

if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="基于Top5等权配置的回测")
    parser.add_argument("--predictions", type=str, default="./output/historical_predictions_high_top5.csv", 
                        help="预测文件路径")
    parser.add_argument("--fromdate", type=str, help="回测开始日期 (YYYY-MM-DD 格式)")
    parser.add_argument("--todate", type=str, help="回测结束日期 (YYYY-MM-DD 格式)")
    parser.add_argument("--no-log-positions", action="store_true", help="不记录持仓信息")
    args = parser.parse_args()

    prin
    # 加载预测数据
    predictions_file = args.predictions
    predictions = pd.read_csv(predictions_file, index_col=0, parse_dates=True)
    
    # 打印预测文件信息
    print(f"预测文件包含 {len(predictions)} 个日期")
    print(f"预测文件包含 {len(predictions.columns)} 个行业")
    
    # 设置回测日期范围
    fromdate = args.fromdate
    todate = args.todate
    
    if fromdate is None:
        # 使用预测数据的第一个日期
        fromdate = predictions.index[0]
    else:
        fromdate = pd.to_datetime(fromdate)
        
    if todate is None:
        # 使用预测数据的最后一个日期
        todate = predictions.index[-1] + pd.Timedelta(days=30)  # 额外增加30天，确保最后的交易能够执行
    else:
        todate = pd.to_datetime(todate)
    
    # 打印日期范围
    print(f"回测日期范围: {fromdate} 至 {todate}")
    
    # 创建 Cerebro 引擎
    cerebro = bt.Cerebro()
    
    # 准备数据
    etf_data_dic, wind_data = prepare_data(fromdate, todate)
    
    # 验证预测文件中的行业代码与ETF数据中的行业代码匹配
    industry_codes_in_data = set(etf_data_dic.keys())
    industry_codes_in_predictions = set(predictions.columns)

    missing_in_data = industry_codes_in_predictions - industry_codes_in_data
    if missing_in_data:
        print(f"警告: 预测文件中有 {len(missing_in_data)} 个行业在ETF数据中不存在: {missing_in_data}")

    # 添加数据到Cerebro引擎
    for fund_code, df in etf_data_dic.items():
        # 只添加预测文件中存在的行业
        if fund_code in industry_codes_in_predictions:
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
        Top5EqualWeightStrategy,
        predictions_file=predictions_file,
        verbose=True,
        log_positions=not args.no_log_positions
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
    
    # 计算基准收益率
    benchmark_rets = wind_data.resample("D").ffill().pct_change(1)
    # 截取日期范围
    benchmark_rets = benchmark_rets.loc[fromdate:todate]
    
    # 确保时区一致性
    if hasattr(returns, 'index'):
        returns.index = returns.index.tz_localize(None)  # 去除tzinfo
    if benchmark_rets is not None and hasattr(benchmark_rets, 'index'):
        benchmark_rets.index = benchmark_rets.index.tz_localize(None)
    
    # 保存结果文件前缀
    output_file_prefix = f"top5_equal_weight_{datetime.now().strftime('%Y%m%d')}"
    
    # 使用pyfolio生成回测报告
    print("\n生成pyfolio回测报告...")

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
    
    print("\n回测完成，结果已保存。") 