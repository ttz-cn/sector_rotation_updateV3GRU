 import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
from pathlib import Path

def plot_returns_comparison(strategy_file, benchmark_file, output_dir='./output'):
    """
    绘制策略收益率与基准收益率的对比图
    
    参数:
    - strategy_file: 策略收益率CSV文件路径
    - benchmark_file: 基准收益率CSV文件路径
    - output_dir: 输出目录
    """
    print(f"读取策略收益率文件: {strategy_file}")
    print(f"读取基准收益率文件: {benchmark_file}")
    
    # 读取数据
    strategy_returns = pd.read_csv(strategy_file, index_col=0)
    benchmark_returns = pd.read_csv(benchmark_file, index_col=0)
    
    # 转换索引为日期类型
    strategy_returns.index = pd.to_datetime(strategy_returns.index)
    benchmark_returns.index = pd.to_datetime(benchmark_returns.index)
    
    # 计算累计收益率
    strategy_cum_returns = (1 + strategy_returns).cumprod() - 1
    benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
    
    # 提取文件名作为标题
    strategy_name = Path(strategy_file).stem
    benchmark_name = Path(benchmark_file).stem
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制累计收益率
    plt.subplot(2, 1, 1)
    plt.plot(strategy_cum_returns.index, strategy_cum_returns, label='策略', linewidth=2)
    plt.plot(benchmark_cum_returns.index, benchmark_cum_returns, label='基准', linewidth=2, alpha=0.7)
    plt.title('累计收益率对比', fontsize=14)
    plt.ylabel('累计收益率', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # 设置x轴日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()  # 自动格式化x轴日期标签
    
    # 计算超额收益
    excess_returns = strategy_returns.values - benchmark_returns.values
    excess_returns_series = pd.Series(excess_returns.flatten(), index=strategy_returns.index)
    cumulative_excess = (1 + excess_returns_series).cumprod() - 1
    
    # 绘制超额收益
    plt.subplot(2, 1, 2)
    plt.plot(cumulative_excess.index, cumulative_excess, label='超额收益', color='green', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('累计超额收益', fontsize=14)
    plt.ylabel('超额收益率', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # 设置x轴日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()  # 自动格式化x轴日期标签
    
    # 计算一些统计指标
    total_days = len(strategy_returns)
    strategy_annual_return = ((1 + strategy_returns).prod()) ** (252 / total_days) - 1
    benchmark_annual_return = ((1 + benchmark_returns).prod()) ** (252 / total_days) - 1
    
    strategy_volatility = strategy_returns.std() * np.sqrt(252)
    benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
    
    strategy_sharpe = strategy_annual_return / strategy_volatility if strategy_volatility != 0 else 0
    benchmark_sharpe = benchmark_annual_return / benchmark_volatility if benchmark_volatility != 0 else 0
    
    # 计算最大回撤
    strategy_cum = (1 + strategy_returns).cumprod()
    benchmark_cum = (1 + benchmark_returns).cumprod()
    
    strategy_max_drawdown = 0
    benchmark_max_drawdown = 0
    
    for i in range(1, len(strategy_cum)):
        strategy_drawdown = 1 - strategy_cum.iloc[i] / strategy_cum.iloc[:i].max()
        benchmark_drawdown = 1 - benchmark_cum.iloc[i] / benchmark_cum.iloc[:i].max()
        
        strategy_max_drawdown = max(strategy_max_drawdown, strategy_drawdown.values[0])
        benchmark_max_drawdown = max(benchmark_max_drawdown, benchmark_drawdown.values[0])
    
    # 添加统计信息文本框
    plt.figtext(0.15, 0.01, f"策略年化收益率: {strategy_annual_return[0]:.2%}\n"
                          f"基准年化收益率: {benchmark_annual_return[0]:.2%}\n"
                          f"超额年化收益率: {strategy_annual_return[0] - benchmark_annual_return[0]:.2%}",
                fontsize=12, ha='left')
    
    plt.figtext(0.45, 0.01, f"策略波动率: {strategy_volatility[0]:.2%}\n"
                          f"基准波动率: {benchmark_volatility[0]:.2%}\n"
                          f"策略夏普比率: {strategy_sharpe[0]:.2f}",
                fontsize=12, ha='left')
    
    plt.figtext(0.75, 0.01, f"策略最大回撤: {strategy_max_drawdown:.2%}\n"
                          f"基准最大回撤: {benchmark_max_drawdown:.2%}\n"
                          f"信息比率: {(strategy_annual_return[0] - benchmark_annual_return[0]) / (excess_returns_series.std() * np.sqrt(252)):.2f}",
                fontsize=12, ha='left')
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # 为底部文本框留出空间
    
    # 保存图表
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"returns_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_file}")
    
    # 显示图表
    plt.show()
    
    # 返回统计结果
    stats = {
        "策略年化收益率": strategy_annual_return[0],
        "基准年化收益率": benchmark_annual_return[0],
        "超额年化收益率": strategy_annual_return[0] - benchmark_annual_return[0],
        "策略波动率": strategy_volatility[0],
        "基准波动率": benchmark_volatility[0],
        "策略夏普比率": strategy_sharpe[0],
        "基准夏普比率": benchmark_sharpe[0],
        "策略最大回撤": strategy_max_drawdown,
        "基准最大回撤": benchmark_max_drawdown,
        "信息比率": (strategy_annual_return[0] - benchmark_annual_return[0]) / (excess_returns_series.std() * np.sqrt(252))
    }
    
    # 打印统计结果
    print("\n==== 策略统计指标 ====")
    for key, value in stats.items():
        if "率" in key or "回撤" in key:
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value:.2f}")
    
    return stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="绘制策略收益率与基准收益率的对比图")
    parser.add_argument("strategy_file", type=str, help="策略收益率CSV文件路径")
    parser.add_argument("benchmark_file", type=str, help="基准收益率CSV文件路径")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    
    args = parser.parse_args()
    
    plot_returns_comparison(args.strategy_file, args.benchmark_file, args.output_dir)