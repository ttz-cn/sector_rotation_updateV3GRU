import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
except:
    pass

def select_top5_industries(predictions_file, output_dir='./output'):
    """
    根据预测值选择每周一的Top5行业，并生成等权配置的回测模拟结果
    
    参数:
    - predictions_file: 预测文件路径
    - output_dir: 输出目录
    
    返回:
    - 回测结果DataFrame
    """
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 加载预测数据
    predictions = pd.read_csv(predictions_file, index_col=0, parse_dates=True)
    
    print(f"预测数据范围: {predictions.index[0]} 至 {predictions.index[-1]}")
    print(f"预测数据包含 {len(predictions.columns)} 个行业")
    
    # 初始化结果存储
    results = []
    weekly_top5 = {}
    
    # 确定所有的周一日期
    all_dates = pd.date_range(start=predictions.index[0], end=predictions.index[-1])
    mondays = [date for date in all_dates if date.weekday() == 0]
    
    # 对每个周一进行处理
    for monday in mondays:
        # 找到最接近但不超过当前周一的预测日期
        valid_dates = predictions.index[predictions.index <= monday]
        if valid_dates.empty:
            continue
            
        latest_pred_date = valid_dates[-1]
        predictions_row = predictions.loc[latest_pred_date]
        
        # 选择前5名行业
        top5_industries = predictions_row.nlargest(5).index.tolist()
        top5_values = predictions_row.nlargest(5).values.tolist()
        
        # 记录结果
        weekly_top5[monday] = {
            'date': monday,
            'pred_date': latest_pred_date,
            'industries': top5_industries,
            'values': top5_values
        }
        
        # 添加到结果列表
        results.append({
            'date': monday,
            'pred_date': latest_pred_date,
            'top1': top5_industries[0],
            'top2': top5_industries[1],
            'top3': top5_industries[2], 
            'top4': top5_industries[3],
            'top5': top5_industries[4],
            'top1_value': top5_values[0],
            'top2_value': top5_values[1],
            'top3_value': top5_values[2],
            'top4_value': top5_values[3],
            'top5_value': top5_values[4],
        })
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存结果
    results_df.to_csv(f"{output_dir}/top5_weekly_allocation.csv", index=False)
    
    # 生成可视化
    plot_top5_changes(results_df, output_dir)
    plot_prediction_heatmap(predictions, output_dir)
    
    return results_df, weekly_top5

def plot_top5_changes(results_df, output_dir):
    """绘制Top5行业变化图"""
    plt.figure(figsize=(15, 10))
    
    # 获取所有不同的行业代码
    all_industries = set()
    for i in range(1, 6):
        all_industries.update(results_df[f'top{i}'].unique())
    
    # 为每个行业分配一个唯一的数字
    industry_to_num = {ind: i for i, ind in enumerate(sorted(all_industries))}
    
    # 为每个TOP位置绘制行业变化曲线
    for i in range(1, 6):
        plt.plot(
            results_df['date'], 
            results_df[f'top{i}'].map(industry_to_num), 
            'o-', 
            label=f'TOP {i}',
            markersize=4
        )
    
    # 添加行业标签
    plt.yticks(
        list(industry_to_num.values()),
        list(industry_to_num.keys())
    )
    
    # 格式化x轴日期
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    plt.title('每周一调仓Top5行业变化')
    plt.xlabel('日期')
    plt.ylabel('行业代码')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f"{output_dir}/top5_changes.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制Top5行业的预测值随时间变化图
    plt.figure(figsize=(15, 8))
    
    for i in range(1, 6):
        plt.plot(
            results_df['date'], 
            results_df[f'top{i}_value'], 
            'o-', 
            label=f'TOP {i}',
            markersize=4
        )
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    plt.title('Top5行业预测值随时间变化')
    plt.xlabel('日期')
    plt.ylabel('预测值')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f"{output_dir}/top5_values.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_heatmap(predictions, output_dir):
    """绘制预测值热力图"""
    # 提取每周一的数据
    weekly_data = []
    all_dates = pd.date_range(start=predictions.index[0], end=predictions.index[-1])
    mondays = [date for date in all_dates if date.weekday() == 0]
    
    for monday in mondays:
        # 找到最接近但不超过当前周一的预测日期
        valid_dates = predictions.index[predictions.index <= monday]
        if valid_dates.empty:
            continue
            
        latest_pred_date = valid_dates[-1]
        weekly_data.append(predictions.loc[latest_pred_date])
    
    # 创建每周一的预测数据DataFrame
    weekly_predictions = pd.DataFrame(weekly_data, index=mondays)
    
    # 绘制热力图
    plt.figure(figsize=(16, 10))
    plt.imshow(weekly_predictions.T, aspect='auto', cmap='RdYlGn')
    
    # 设置坐标轴
    plt.colorbar(label='预测值')
    plt.xticks(range(len(weekly_predictions.index)), 
               [d.strftime('%Y-%m-%d') for d in weekly_predictions.index], 
               rotation=90)
    plt.yticks(range(len(weekly_predictions.columns)), weekly_predictions.columns)
    
    plt.title('每周一行业预测值热力图')
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f"{output_dir}/prediction_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制行业排名热力图
    weekly_ranks = weekly_predictions.rank(axis=1, ascending=False).T
    
    plt.figure(figsize=(16, 10))
    plt.imshow(weekly_ranks, aspect='auto', cmap='YlGnBu_r')
    
    # 设置坐标轴
    plt.colorbar(label='排名 (1为最高)')
    plt.xticks(range(len(weekly_predictions.index)), 
               [d.strftime('%Y-%m-%d') for d in weekly_predictions.index], 
               rotation=90)
    plt.yticks(range(len(weekly_predictions.columns)), weekly_predictions.columns)
    
    plt.title('每周一行业预测排名热力图')
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f"{output_dir}/rank_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def load_wind_a_data():
    """加载wind全A指数数据"""
    try:
        # 尝试从数据库加载wind全A数据
        wind_data = pd.read_hdf("./db/time_series_data.h5", key="wind_a")
        # 确保日期格式正确
        if not isinstance(wind_data.index, pd.DatetimeIndex):
            wind_data.index = pd.to_datetime(wind_data.index)
        # 计算日收益率
        wind_returns = wind_data.pct_change(1).dropna()
        return wind_returns
    except Exception as e:
        print(f"加载wind全A数据失败: {e}")
        # 如果加载失败，返回空DataFrame
        return pd.DataFrame()

def load_industry_returns():
    """加载行业指数收益率数据"""
    try:
        # 尝试从数据库加载行业指数数据
        etf_panel_data = pd.read_hdf("./db/panel_data.h5", key="sector")
        
        # 提取行业代码和日期
        etf_panel_data["Date"] = pd.to_datetime(etf_panel_data["Date"])
        
        # 按行业代码分组，提取收盘价
        industry_close_prices = {}
        for industry_code, group in etf_panel_data.groupby("industry_code"):
            group = group.sort_values("Date")
            group = group.set_index("Date")
            industry_close_prices[industry_code] = group["CLOSE"]
        
        # 转换为DataFrame并计算日收益率
        industry_prices = pd.DataFrame(industry_close_prices)
        industry_returns = industry_prices.pct_change(1).dropna()
        
        return industry_returns
    except Exception as e:
        print(f"加载行业指数数据失败: {e}")
        # 如果加载失败，返回空DataFrame
        return pd.DataFrame()

def calculate_portfolio_returns(results_df, industry_returns):
    """
    计算投资组合收益率
    
    参数:
    - results_df: Top5行业选择结果
    - industry_returns: 行业指数收益率
    
    返回:
    - 投资组合日收益率序列
    """
    # 初始化投资组合收益率序列
    portfolio_returns = pd.Series(index=industry_returns.index)
    
    # 为每个交易日分配权重
    portfolio_weights = pd.DataFrame(0, index=industry_returns.index, columns=industry_returns.columns)
    
    # 遍历每个调仓日期
    for i in range(len(results_df)):
        rebalance_date = results_df.iloc[i]['date']
        
        # 确定下一个调仓日期
        next_rebalance_date = results_df.iloc[i+1]['date'] if i < len(results_df)-1 else industry_returns.index[-1]
        
        # 获取该期间的交易日
        trading_days = industry_returns.loc[rebalance_date:next_rebalance_date].index
        
        # 提取Top5行业
        top5_industries = [results_df.iloc[i][f'top{j}'] for j in range(1, 6)]
        
        # 设置等权重 (20%)
        for industry in top5_industries:
            if industry in portfolio_weights.columns:
                portfolio_weights.loc[trading_days, industry] = 0.2
    
    # 计算投资组合每日收益率
    for date in industry_returns.index:
        if date in portfolio_weights.index:
            # 当日各行业权重
            weights = portfolio_weights.loc[date]
            # 当日各行业收益率
            returns = industry_returns.loc[date]
            # 计算投资组合收益率
            portfolio_returns[date] = (weights * returns).sum()
    
    return portfolio_returns.dropna()

def plot_nav_curves(portfolio_returns, benchmark_returns, output_dir):
    """
    绘制净值曲线
    
    参数:
    - portfolio_returns: 投资组合收益率
    - benchmark_returns: 基准收益率
    - output_dir: 输出目录
    """
    # 计算累积收益率
    portfolio_nav = (1 + portfolio_returns).cumprod()
    benchmark_nav = (1 + benchmark_returns).cumprod()
    
    # 对齐数据
    aligned_data = pd.concat([portfolio_nav, benchmark_nav], axis=1).dropna()
    aligned_data.columns = ['Top5等权策略', 'Wind全A']
    
    # 计算最大回撤
    portfolio_max_drawdown = calculate_max_drawdown(portfolio_nav)
    benchmark_max_drawdown = calculate_max_drawdown(benchmark_nav)
    
    # 计算年化收益率
    days_count = (aligned_data.index[-1] - aligned_data.index[0]).days
    years = days_count / 365.0
    
    portfolio_annual_return = (aligned_data['Top5等权策略'].iloc[-1] / aligned_data['Top5等权策略'].iloc[0]) ** (1/years) - 1
    benchmark_annual_return = (aligned_data['Wind全A'].iloc[-1] / aligned_data['Wind全A'].iloc[0]) ** (1/years) - 1
    
    # 绘制净值曲线
    plt.figure(figsize=(15, 8))
    
    plt.plot(aligned_data.index, aligned_data['Top5等权策略'], 
             label=f'Top5等权策略 (年化: {portfolio_annual_return:.2%}, 最大回撤: {portfolio_max_drawdown:.2%})')
    plt.plot(aligned_data.index, aligned_data['Wind全A'], 
             label=f'Wind全A (年化: {benchmark_annual_return:.2%}, 最大回撤: {benchmark_max_drawdown:.2%})')
    
    # 格式化x轴日期
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    plt.title('Top5等权策略 vs Wind全A 净值曲线')
    plt.xlabel('日期')
    plt.ylabel('净值')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f"{output_dir}/nav_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存净值数据
    aligned_data.to_csv(f"{output_dir}/nav_data.csv")
    
    # 绘制超额收益曲线
    excess_returns = portfolio_returns - benchmark_returns
    excess_nav = (1 + excess_returns).cumprod()
    
    plt.figure(figsize=(15, 8))
    plt.plot(excess_nav.index, excess_nav, label='超额收益')
    
    # 格式化x轴日期
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    plt.title('Top5等权策略相对Wind全A的超额收益')
    plt.xlabel('日期')
    plt.ylabel('净值')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f"{output_dir}/excess_returns.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return aligned_data

def calculate_max_drawdown(nav_series):
    """计算最大回撤"""
    nav_values = nav_series.values
    max_so_far = nav_values[0]
    max_drawdown = 0
    
    for value in nav_values:
        if value > max_so_far:
            max_so_far = value
        drawdown = (max_so_far - value) / max_so_far
        max_drawdown = max(max_drawdown, drawdown)
    
    return max_drawdown

def simulate_top5_equal_weight_strategy(predictions_file, output_dir='./output'):
    """
    模拟Top5等权配置策略的表现
    
    参数:
    - predictions_file: 预测文件路径
    - output_dir: 输出目录
    """
    # 获取Top5行业分配
    results_df, weekly_top5 = select_top5_industries(predictions_file, output_dir)
    
    print(f"生成了 {len(results_df)} 个周一的调仓信号")
    print("\n前10个调仓日的TOP5行业:")
    for _, row in results_df.head(10).iterrows():
        print(f"{row['date'].strftime('%Y-%m-%d')}: {row['top1']}, {row['top2']}, {row['top3']}, {row['top4']}, {row['top5']}")
    
    # 统计每个行业被选入TOP5的次数
    industry_count = {}
    for i in range(1, 6):
        column = f'top{i}'
        for industry in results_df[column]:
            if industry not in industry_count:
                industry_count[industry] = 0
            industry_count[industry] += 1
    
    # 绘制行业选择频率图
    plt.figure(figsize=(14, 8))
    industries = list(industry_count.keys())
    counts = list(industry_count.values())
    
    # 对频率进行排序
    sorted_indices = np.argsort(counts)[::-1]
    sorted_industries = [industries[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]
    
    plt.bar(sorted_industries, sorted_counts)
    plt.title('各行业被选入TOP5的频率')
    plt.xlabel('行业代码')
    plt.ylabel('次数')
    plt.xticks(rotation=90)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f"{output_dir}/industry_frequency.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 加载行业收益率数据
    print("\n加载行业收益率数据...")
    industry_returns = load_industry_returns()
    
    if not industry_returns.empty:
        # 加载Wind全A基准数据
        print("加载Wind全A基准数据...")
        benchmark_returns = load_wind_a_data()
        
        if not benchmark_returns.empty:
            # 计算投资组合收益率
            print("计算投资组合收益率...")
            portfolio_returns = calculate_portfolio_returns(results_df, industry_returns)
            
            # 对齐基准收益率
            aligned_benchmark = benchmark_returns.reindex(portfolio_returns.index)
            
            # 绘制净值曲线
            print("绘制净值曲线...")
            nav_data = plot_nav_curves(portfolio_returns, aligned_benchmark, output_dir)
            
            print(f"\n净值数据已保存至 {output_dir}/nav_data.csv")
            print(f"最终净值 - Top5等权策略: {nav_data['Top5等权策略'].iloc[-1]:.4f}, Wind全A: {nav_data['Wind全A'].iloc[-1]:.4f}")
        else:
            print("警告: 无法加载Wind全A基准数据，跳过净值曲线绘制")
    else:
        print("警告: 无法加载行业收益率数据，跳过净值曲线绘制")
    
    print("\n行业选择频率统计完成")
    print(f"所有图表已保存至 {output_dir} 目录")

if __name__ == "__main__":
    # 运行策略模拟
    simulate_top5_equal_weight_strategy("./output/historical_predictions_high_top5.csv") 