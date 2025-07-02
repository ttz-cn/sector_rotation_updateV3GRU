import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from datetime import datetime, timedelta

# 设置字体，使用更通用的设置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_prediction_data(file_path):
    """加载预测数据"""
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df

def load_wind_data():
    """加载wind数据"""
    wind_data = pd.read_hdf("./db/time_series_data.h5", key="wind_a")
    wind_data = wind_data.groupby(wind_data.index).tail(1).ffill()  # 保留每个index的最后一条
    return wind_data

def select_industries(pred_df, threshold=0, top_n=5):
    """根据预测值选择行业
    
    Args:
        pred_df: 预测数据DataFrame
        threshold: 预测收益率阈值
        top_n: 最多选择的行业数量
        
    Returns:
        selected_industries: 每个日期选择的行业字典
    """
    selected_industries = {}
    
    for date, row in pred_df.iterrows():
        # 获取当天所有行业的预测值
        pred_values = row.sort_values(ascending=False)
        
        # 选择预测收益率大于threshold的行业
        positive_industries = pred_values[pred_values > threshold]
        
        # 如果没有预测收益率大于threshold的行业，则空仓
        if len(positive_industries) == 0:
            selected_industries[date] = []
            continue
            
        # 如果行业过多，只选择前top_n个
        if len(positive_industries) > top_n:
            positive_industries = positive_industries.head(top_n)
            
        selected_industries[date] = positive_industries.index.tolist()
    
    return selected_industries

def plot_predictions_with_wind(pred_df, wind_df, threshold=0, top_n=5):
    """绘制预测数据和wind数据在同一张图上，并标记选择的行业"""
    # 确保索引对齐
    common_indices = pred_df.index.intersection(wind_df.index)
    if len(common_indices) == 0:
        print("预测数据和wind数据没有共同的日期索引！")
        return
    
    pred_df = pred_df.loc[common_indices]
    wind_df = wind_df.loc[common_indices]
    
    # 选择行业
    selected_industries = select_industries(pred_df, threshold, top_n)
    
    # 创建一个大图
    plt.figure(figsize=(16, 10))
    
    # 创建两个Y轴
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # 绘制所有行业的原始预测数据（使用较低的透明度）
    colors = plt.cm.tab10(np.linspace(0, 1, len(pred_df.columns)))
    for i, col in enumerate(pred_df.columns):
        ax1.plot(pred_df.index, pred_df[col], linewidth=1.0, color=colors[i % len(colors)], alpha=0.3, label=col)
    
    # 绘制阈值线
    ax1.axhline(y=threshold, color='black', linestyle='--', alpha=0.7, label=f'Threshold: {threshold}')
    
    # 绘制wind数据
    ax2.plot(wind_df.index, wind_df['CLOSE'], linewidth=3, color='black', label='Wind Close')
    
    # 高亮显示选择的行业
    # 创建一个新的DataFrame来存储每天选择的行业的预测值
    highlight_df = pd.DataFrame(index=pred_df.index, columns=pred_df.columns)
    highlight_df = highlight_df.fillna(np.nan)  # 填充NaN
    
    for date, industries in selected_industries.items():
        for ind in industries:
            highlight_df.loc[date, ind] = pred_df.loc[date, ind]
    
    # 绘制高亮的行业
    for i, col in enumerate(pred_df.columns):
        if not highlight_df[col].isna().all():  # 如果该行业至少被选中过一次
            ax1.scatter(highlight_df.index, highlight_df[col].dropna(), 
                      color=colors[i % len(colors)], s=50, alpha=1.0, zorder=5)
    
    # 设置标题和标签
    plt.title(f'Predictions and Wind Close Data (Threshold: {threshold}, Max Industries: {top_n})', fontsize=16)
    ax1.set_xlabel('Date', fontsize=14)
    ax1.set_ylabel('Prediction Return', fontsize=14)
    ax2.set_ylabel('Wind Close Price', fontsize=14)
    
    # 添加图例
    # 获取两个轴的图例句柄和标签
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    
    # 将两个图例合并
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.05, 1), 
               fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # 设置网格
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 设置x轴日期格式
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_formatter(date_format)
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('predictions_with_selected_industries.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 返回选择的行业
    return selected_industries

def main():
    # 加载预测数据
    pred_file_path = 'output/historical_predictions_high_top5.csv'
    pred_df = load_prediction_data(pred_file_path)
    
    # 加载wind数据
    try:
        wind_df = load_wind_data()
        print(f"Wind data loaded successfully. Shape: {wind_df.shape}")
        print(f"Wind data columns: {wind_df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading wind data: {e}")
        return
    
    # 设置阈值和最大行业数
    threshold = 0.0  # 预测收益率阈值
    top_n = 5  # 最多选择的行业数量
    
    # 绘制预测数据和wind数据，并选择行业
    selected_industries = plot_predictions_with_wind(pred_df, wind_df, threshold, top_n)
    
    # 输出一些基本信息
    print("Data Overview:")
    print(f"Prediction data time range: {pred_df.index.min()} to {pred_df.index.max()}")
    print(f"Wind data time range: {wind_df.index.min()} to {wind_df.index.max()}")
    print(f"Number of industries in prediction data: {pred_df.shape[1]}")
    
    # 统计选择的行业信息
    total_days = len(pred_df.index)
    empty_days = sum(1 for industries in selected_industries.values() if len(industries) == 0)
    avg_industries = sum(len(industries) for industries in selected_industries.values()) / total_days
    
    print("\n行业选择统计:")
    print(f"总交易日数: {total_days}")
    print(f"空仓天数: {empty_days} ({empty_days/total_days*100:.2f}%)")
    print(f"平均每天选择行业数: {avg_industries:.2f}")
    
    # 输出每个行业被选中的次数
    industry_counts = {}
    for industries in selected_industries.values():
        for ind in industries:
            if ind in industry_counts:
                industry_counts[ind] += 1
            else:
                industry_counts[ind] = 1
    
    print("\n各行业被选中的次数:")
    for ind, count in sorted(industry_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{ind}: {count} 次 ({count/total_days*100:.2f}%)")

if __name__ == "__main__":
    main() 