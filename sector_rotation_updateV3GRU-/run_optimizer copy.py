import pandas as pd
import numpy as np
import config
import cvxpy as cp
from src.src_gru_factor_fusion import optimize_portfolio_weights, adjust_portfolio_by_crowding
from utils.utils_tools import load_panel_dropd
import os
import sys
import torch
from datetime import datetime


def run_optimization(pred_values, fund_close, risk_preference="high", top_n_low=3, threshold=0.015):
    """ 
    基于预测结果运行投资组合优化

    参数:
    - pred_values: 预测收益率Series，索引为行业代码
    - fund_close: 基金收盘价数据框
    - risk_preference: 风险偏好，'high'或'low'，会根据标的数量自动调整
    - top_n: 选择预测收益率最高的前N个行业进行投资，最大值为5
    - threshold: 预测收益率阈值

    返回:
    - final_weights: 优化后的投资组合权重
    """
    print("\n开始组合优化...")
    
    # 限制top_n最大为5
    top_n_high = max(top_n_low, 5)
    
    # 选择预测收益率大于threshold的行业
    positive_industries = pred_values[pred_values > threshold]
    print(f"\n选择预测收益率大于{threshold}的行业，共{len(positive_industries)}个行业")
    
    # 如果没有预测收益率大于threshold的行业，则空仓
    if len(positive_industries) == 0:
        print(f"没有预测收益率大于{threshold}的行业，将空仓")
        return pd.Series(0.0, index=fund_close.columns)
    
    # 根据正收益行业数量自动确定风险偏好
    actual_risk_preference = "low" if len(positive_industries) < top_n_low else "high"
    print(f"根据标的数量自动选择风险偏好: {actual_risk_preference}")
    
    # 如果大于threshold的行业过多（超过top_n个），则只选择前top_n个
    if len(positive_industries) > top_n_low & len(positive_industries)<top_n_high:
        print(f"预测收益率大于{threshold}的行业超过{top_n_low}个，将只选择预测收益率最高的前{top_n_high}个行业")
        positive_industries = positive_industries.sort_values(ascending=False)
    elif len(positive_industries) > top_n_high:
        print(f"预测收益率大于{threshold}的行业超过{top_n_high}个，将只选择预测收益率最高的前{top_n_high}个行业")
        positive_industries = positive_industries.sort_values(ascending=False).head(top_n_high)
    # 如果大于threshold的行业不足3个，则选择3个 
    elif len(positive_industries) < top_n_low:
        print(f"预测收益率大于{threshold}的行业不足{top_n_low}个，将选择预测收益率最高的前{top_n_low}个行业")
        positive_industries = pred_values.sort_values(ascending=False).head(top_n_low)

    print(f"\n选择的行业及其预测收益率:")
    for ind, value in positive_industries.items():
        ind_name = config.dic_Industry2Etf[ind][0] if ind in config.dic_Industry2Etf else ind
        print(f"{ind_name}: {value:.4f}")

    # 只保留这些行业的预测值作为因子值
    latest_factor = positive_industries.copy()

    # 配置优化参数，使用实际风险偏好
    opt_params = {
        "max_weight": config.optimization_params["weight_limit"][actual_risk_preference],
        "min_weight": config.optimization_params["min_weight"][actual_risk_preference],  
    }

    # 计算选中行业的历史收益率，用于估计协方差矩阵和平均收益率
    # 使用过去120个交易日的数据
    returns = fund_close.loc[:, positive_industries.index].pct_change().dropna().iloc[-120:]
    cov_matrix = returns.cov()

    print(f"使用过去{len(returns)}个交易日的数据计算协方差矩阵")
    print(f"风险偏好参数: 最大权重={opt_params['max_weight']:.2f}, 最小权重={opt_params['min_weight']:.2f}")

    # 实现最大化夏普比率的凸优化问题
    final_weights = cvxpy_sharpe_optimization(
        expected_returns=latest_factor,  # 使用预测收益率作为期望收益
        cov_matrix=cov_matrix,
        max_weight=opt_params["max_weight"],
        min_weight=opt_params["min_weight"],
    )

    # 创建完整的权重Series（未选中的行业权重为0）
    complete_weights = pd.Series(0.0, index=fund_close.columns)
    complete_weights[final_weights.index] = final_weights

    print(f"行业权重:")
    for i, weight in enumerate(final_weights):
        ind = positive_industries.index[i]
        ind_name = config.dic_Industry2Etf[ind][0] if ind in config.dic_Industry2Etf else ind
        print(f"{ind_name}: {weight:.2%}")

    return complete_weights


def cvxpy_sharpe_optimization(expected_returns, cov_matrix=None, max_weight=0.3, min_weight=0.1, risk_free_rate=0.0):
    """
    使用CVXPY实现最大化夏普比率的投资组合优化问题：
    max (w^T μ - r_f) / sqrt(w^T Σ w)
    s.t. min_weight ≤ w ≤ max_weight
         Σ w = 1

    通过引入辅助变量将非凸问题转换为凸优化问题：
    max z^T μ - r_f
    s.t. z^T Σ z ≤ 1
         min_weight * y ≤ z ≤ max_weight * y
         Σ z = y
         y > 0

    其中 w = z/y

    参数:
    - expected_returns: 期望收益率Series
    - cov_matrix: 协方差矩阵DataFrame，使用过去120个工作日的数据计算
    - max_weight: 单个行业最大权重限制
    - min_weight: 单个行业最小权重限制
    - risk_free_rate: 无风险利率

    返回:
    - optimal_weights: 优化后的权重Series
    """
    n = len(expected_returns)
    if n == 0:
        return pd.Series()

    # 创建辅助优化变量
    z = cp.Variable(n)  # z = y*w, 其中y是一个标量变量
    y = cp.Variable(1, pos=True)  # 必须为正

    # 处理协方差矩阵
    if cov_matrix is None or cov_matrix.shape != (n, n):
        print("警告: 协方差矩阵无效，使用单位矩阵代替")
        sigma = np.eye(n)
    else:
        # 确保协方差矩阵是半正定的
        sigma = cov_matrix.values
        min_eig = np.min(np.linalg.eigvals(sigma))
        if min_eig < 0:
            # 如果不是半正定矩阵，进行调整
            sigma = sigma - min_eig * np.eye(n) * 1.1

    # 目标函数：最大化调整后的夏普比率
    objective = cp.Maximize(expected_returns.values @ z - risk_free_rate * y)

    # 约束条件
    constraints = [
        # 风险约束: z^T Σ z ≤ 1
        # 这是关键的约束！确保最终的夏普比率正确
        cp.quad_form(z, sigma) <= 1,
        # 权重范围约束: min_weight * y ≤ z ≤ max_weight * y
        z >= min_weight * y,
        z <= max_weight * y,
        # 权重之和约束: Σ z = y
        cp.sum(z) == y,
        # y必须为正
        y >= 1e-8 ,
    ]

    # 构建并求解问题
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve()

        if problem.status == cp.OPTIMAL:
            # 恢复原始权重 w = z/y
            y_val = y.value[0]
            w_values = z.value / y_val

            # 返回最优解
            opt_weights = pd.Series(w_values, index=expected_returns.index)

            # 计算并打印最优投资组合的夏普比率
            portfolio_return = np.sum(opt_weights.values * expected_returns.values)
            portfolio_risk = np.sqrt(opt_weights.values @ sigma @ opt_weights.values)
            sharpe = (portfolio_return - risk_free_rate) / portfolio_risk

            # 计算并显示优化后投资组合的年化波动率
            ann_vol_portfolio = np.sqrt(252) * portfolio_risk

            print(f"优化结果:")
            print(f"- 预期夏普比率: {sharpe:.4f}")
            print(f"- 预期收益率: {portfolio_return:.4f}")
            print(f"- 投资组合年化波动率: {ann_vol_portfolio:.2%}")

            return opt_weights
        else:
            print(f"优化问题求解失败: {problem.status}")
            # 返回等权重作为备选
            equal_weight = 1 / n
            return pd.Series(equal_weight, index=expected_returns.index)
    except Exception as e:
        print(f"优化求解出错: {e}")
        # 返回等权重作为备选
        equal_weight = 1 / n
        return pd.Series(equal_weight, index=expected_returns.index)


def get_latest_predictions(predictions_dict, target_date=None):
    """
    从预测字典中获取最新的预测结果
    
    参数:
    - predictions_dict: 预测结果字典，键为行业代码，值为时序预测值
    - target_date: 目标日期，如果为None则使用最新日期
    
    返回:
    - latest_predictions: 最新预测结果Series
    """
    latest_predictions = {}
    
    for industry, pred_series in predictions_dict.items():
        # 如果指定了目标日期，则获取该日期或之前最近的预测值
        if target_date is not None:
            valid_dates = pred_series.index[pred_series.index <= target_date]
            if not valid_dates.empty:
                latest_date = valid_dates[-1]
                latest_predictions[industry] = pred_series[latest_date]
        else:
            # 否则获取最新的预测值
            latest_predictions[industry] = pred_series.iloc[-1]
    
    # 转换为Series
    return pd.Series(latest_predictions)


def generate_historical_weights(predictions_dict, fund_close, start_date=None, end_date=None, 
                               top_n=5, threshold=0.005, freq='B'):
    """
    生成历史上每个时间点的最优权重
    
    参数:
    - predictions_dict: 预测结果字典，键为行业代码，值为时序预测值
    - fund_close: 基金收盘价数据框
    - start_date: 开始日期，如果为None则使用预测数据的最早日期
    - end_date: 结束日期，如果为None则使用预测数据的最晚日期
    - top_n: 选择预测收益率最高的前N个行业，最大为5
    - threshold: 预测收益率阈值
    - freq: 生成权重的频率，默认为每个工作日
    
    返回:
    - historical_weights: 历史权重DataFrame，索引为日期，列为行业代码
    """
    print(f"生成历史权重序列...")
    
    # 获取所有预测日期
    all_dates = set()
    for industry, pred_series in predictions_dict.items():
        all_dates.update(pred_series.index)
    
    all_dates = sorted(all_dates)

    # 设置日期范围
    if start_date is None:
        start_date = all_dates[0]
    else:
        start_date = pd.to_datetime(start_date)
        
    if end_date is None:
        end_date = all_dates[-1]
    else:
        end_date = pd.to_datetime(end_date)
    
    # 生成调仓日期序列
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # 存储每个调仓日期的权重
    historical_weights = {}
    # 存储每个调仓日期的预测结果
    historical_predictions = {}
    
    for date in rebalance_dates:
        print(f"\n处理日期: {date.strftime('%Y-%m-%d')}")
        
        # 获取该日期的预测结果
        pred_values = get_latest_predictions(predictions_dict, date)
        
        if len(pred_values) == 0:
            print(f"  警告: {date.strftime('%Y-%m-%d')} 没有可用的预测值，跳过")
            continue
        
        # 存储该日期的预测结果
        historical_predictions[date] = pred_values
        
        # 获取该日期之前的收盘价数据
        available_close = fund_close[fund_close.index <= date]
        if len(available_close) < 120:  # 至少需要120个交易日的数据计算协方差矩阵
            print(f"  警告: {date.strftime('%Y-%m-%d')} 之前的收盘价数据不足，跳过")
            continue
        
        # 运行优化获取权重，现在自动确定风险偏好
        try:
            weights = run_optimization(
                pred_values, 
                available_close,
                top_n=top_n,
                threshold=threshold
            )
            historical_weights[date] = weights
            
            # 检查是否空仓
            if weights.sum() == 0:
                print(f"  {date.strftime('%Y-%m-%d')} 空仓")
            else:
                print(f"  {date.strftime('%Y-%m-%d')} 选择了 {(weights > 0).sum()} 个行业")
        except Exception as e:
            print(f"  错误: 优化失败 - {e}")
            continue
    
    # 转换为DataFrame
    if historical_weights:
        historical_weights_df = pd.DataFrame(historical_weights).T
        
        # 保存权重结果，使用"auto"代替具体的风险偏好
        output_weights_file = f"./output/historical_weights_auto_top{top_n}_{threshold}.csv"
        historical_weights_df.to_csv(output_weights_file)
        print(f"\n历史权重序列已保存到 {output_weights_file}")
        
        # 保存预测结果
        if historical_predictions:
            historical_predictions_df = pd.DataFrame(historical_predictions).T
            output_predictions_file = f"./output/historical_predictions_auto_top{top_n}_{threshold}.csv"
            historical_predictions_df.to_csv(output_predictions_file)
            print(f"历史预测序列已保存到 {output_predictions_file}")
        
        return historical_weights_df
    else:
        print("警告: 没有生成任何历史权重")
        return None


if __name__ == "__main__":
    freq_pred = '5d'
    freq_weights = 'B'
    start = '2025-01-06'
    end = '2025-06-25'
    top_n = 5
    risk = 'high'
    threshold = 0.01
    
    print("正在加载数据...")
    
    # 加载ETF面板数据
    etf_panel_data = load_panel_dropd("./db/panel_data.h5", key="sector")  # sector panel data
    
    # 获取收盘价时序数据
    fund_close = etf_panel_data.pivot_table(index="Date", columns="industry_code", values="CLOSE")
    fund_close = fund_close.resample("B").asfreq()  # 按工作日重采样
    
    # 加载GRU模型生成的预测结果
    try:
        # 尝试从models目录加载预测结果
        predictions = torch.load(f"./models/all_industry_predictions_{freq_pred}.pt")
        print("成功从models目录加载预测结果")
        
        # 生成历史权重序列
        historical_weights = generate_historical_weights(
            predictions_dict=predictions,
            fund_close=fund_close,
            start_date=start,
            end_date=end,
            top_n=top_n,
            threshold=threshold,
            freq=freq_weights
        )
        
    except Exception as e:
        print(f"从models目录加载预测结果失败: {e}")
        print("尝试从output目录加载预测结果...")
        
        # 尝试从output目录加载预测结果
        try:
            # 获取最新的预测文件
            import glob
            pred_files = glob.glob("./output/gru_predictions_*.csv")
            if not pred_files:
                raise FileNotFoundError("未找到预测结果文件")
                
            # 按文件名排序，选择最新的文件
            latest_file = sorted(pred_files)[-1]
            print(f"使用预测文件: {latest_file}")
            
            # 加载预测结果
            pred_df = pd.read_csv(latest_file, index_col=0)
            latest_predictions = pred_df["predicted_return"]
            
            # 运行优化，只选择前N个行业
            run_optimization(latest_predictions, fund_close, risk_preference=risk, top_n=top_n)
            
        except Exception as e2:
            print(f"从output目录加载预测结果失败: {e2}")
            print("无法加载预测结果，退出程序")
            sys.exit(1)
