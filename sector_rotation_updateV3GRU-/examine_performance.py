import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy.stats import spearmanr, pearsonr
import traceback

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

def load_performance_data(performance_file):
    """加载行业表现数据"""
    try:
        print(f"尝试加载性能数据: {performance_file}")
        performance_data = torch.load(performance_file)
        print(f"成功加载性能数据: {performance_file}")
        return performance_data
    except Exception as e:
        print(f"加载性能数据失败: {e}")
        traceback.print_exc()
        return None

def load_prediction_data(prediction_file):
    """加载预测数据"""
    try:
        print(f"尝试加载预测数据: {prediction_file}")
        predictions = pd.read_csv(prediction_file, index_col=0, parse_dates=True)
        print(f"成功加载预测数据: {prediction_file}")
        print(f"预测数据范围: {predictions.index[0]} 至 {predictions.index[-1]}")
        print(f"预测数据包含 {len(predictions.columns)} 个行业")
        return predictions
    except Exception as e:
        print(f"加载预测数据失败: {e}")
        traceback.print_exc()
        return None

def examine_performance_data(performance_data):
    """详细检查性能数据结构"""
    if not isinstance(performance_data, dict):
        print(f"性能数据不是字典类型，而是 {type(performance_data)}")
        return
    
    print(f"\n性能数据包含 {len(performance_data)} 个行业")
    
    # 遍历几个样本行业
    sample_industries = list(performance_data.keys())[:5]
    for industry in sample_industries:
        print(f"\n行业 {industry} 的数据结构:")
        if isinstance(performance_data[industry], dict):
            for key, value in performance_data[industry].items():
                print(f"  - {key}: {type(value)}")
                
                # 如果是性能数据，显示一些统计信息
                if key not in ['train_mse', 'test_mse'] and hasattr(value, 'shape'):
                    print(f"    - 形状: {value.shape}")
                    if hasattr(value, 'mean'):
                        print(f"    - 均值: {value.mean()}")
                        print(f"    - 最大值: {value.max()}")
                        print(f"    - 最小值: {value.min()}")
        else:
            print(f"  数据类型: {type(performance_data[industry])}")
            
    # 检查是否有共同的键
    common_keys = set()
    first = True
    
    for industry, data in performance_data.items():
        if isinstance(data, dict):
            if first:
                common_keys = set(data.keys())
                first = False
            else:
                common_keys = common_keys.intersection(set(data.keys()))
    
    print(f"\n所有行业共有的键: {common_keys}")
    
    # 如果有性能数据，分析其分布
    performance_values = []
    industries = []
    
    for industry, data in performance_data.items():
        if isinstance(data, dict) and '5d_return' in data:
            if hasattr(data['5d_return'], 'item'):
                perf = data['5d_return'].item()
                performance_values.append(perf)
                industries.append(industry)
            elif isinstance(data['5d_return'], (int, float)):
                performance_values.append(data['5d_return'])
                industries.append(industry)
    
    if performance_values:
        perf_df = pd.DataFrame({
            'industry': industries,
            'performance': performance_values
        })
        
        print(f"\n5天收益率统计:")
        print(f"行业数量: {len(perf_df)}")
        print(f"平均值: {perf_df['performance'].mean():.4f}")
        print(f"中位数: {perf_df['performance'].median():.4f}")
        print(f"最大值: {perf_df['performance'].max():.4f} ({perf_df.loc[perf_df['performance'].idxmax(), 'industry']})")
        print(f"最小值: {perf_df['performance'].min():.4f} ({perf_df.loc[perf_df['performance'].idxmin(), 'industry']})")
        
        # 排序并显示
        sorted_perf = perf_df.sort_values('performance', ascending=False)
        print("\n行业性能排名 (前10名):")
        for i, (_, row) in enumerate(sorted_perf.head(10).iterrows()):
            print(f"{i+1}. {row['industry']}: {row['performance']:.4f}")
        
        return perf_df
    
    return None

def compare_prediction_with_performance(predictions, perf_df, output_dir='./output'):
    """比较预测与实际性能"""
    if perf_df is None or len(perf_df) == 0:
        print("没有有效的性能数据，无法进行比较")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取最新的预测日期
    latest_pred_date = predictions.index[-1]
    print(f"\n使用最新预测日期: {latest_pred_date}")
    
    # 获取该日期的预测值
    latest_pred = predictions.loc[latest_pred_date]
    
    # 创建包含预测和实际性能的DataFrame
    comparison_df = pd.DataFrame()
    comparison_df['industry'] = perf_df['industry']
    comparison_df['performance'] = perf_df['performance']
    
    # 添加预测值
    pred_values = []
    for industry in comparison_df['industry']:
        if industry in latest_pred.index:
            pred_values.append(latest_pred[industry])
        else:
            pred_values.append(np.nan)
    
    comparison_df['prediction'] = pred_values
    
    # 过滤掉没有预测值的行
    valid_comparison = comparison_df.dropna()
    print(f"有效比较数据行数: {len(valid_comparison)}")
    
    if len(valid_comparison) < 5:
        print("有效数据太少，无法进行有意义的比较")
        return
    
    # 计算相关系数
    spearman_corr, spearman_p = spearmanr(valid_comparison['prediction'], valid_comparison['performance'])
    pearson_corr, pearson_p = pearsonr(valid_comparison['prediction'], valid_comparison['performance'])
    
    print(f"\n相关性分析:")
    print(f"Spearman相关系数: {spearman_corr:.4f} (p值: {spearman_p:.4f})")
    print(f"Pearson相关系数: {pearson_corr:.4f} (p值: {pearson_p:.4f})")
    
    # 比较Top5预测和实际Top5
    top5_pred = valid_comparison.nlargest(5, 'prediction')['industry'].tolist()
    top5_perf = valid_comparison.nlargest(5, 'performance')['industry'].tolist()
    
    hit_count = len(set(top5_pred) & set(top5_perf))
    hit_rate = hit_count / 5.0
    
    print(f"\nTop5命中分析:")
    print(f"Top5预测: {', '.join(top5_pred)}")
    print(f"Top5实际: {', '.join(top5_perf)}")
    print(f"命中数: {hit_count}/5")
    print(f"命中率: {hit_rate:.2%}")
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(valid_comparison['prediction'], valid_comparison['performance'], alpha=0.7)
    
    # 添加行业标签
    for _, row in valid_comparison.iterrows():
        plt.annotate(row['industry'], 
                    (row['prediction'], row['performance']),
                    textcoords="offset points",
                    xytext=(0,5), 
                    ha='center')
    
    # 添加回归线
    z = np.polyfit(valid_comparison['prediction'], valid_comparison['performance'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(valid_comparison['prediction'].min(), valid_comparison['prediction'].max(), 100)
    plt.plot(x_range, p(x_range), "r--", alpha=0.8)
    
    plt.title(f'预测值与实际表现散点图 (Spearman={spearman_corr:.4f}, Pearson={pearson_corr:.4f})')
    plt.xlabel('预测值')
    plt.ylabel('实际表现 (5天收益率)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f"{output_dir}/prediction_performance_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制预测值和实际性能的柱状图
    plt.figure(figsize=(14, 8))
    
    # 按预测值排序
    sorted_by_pred = valid_comparison.sort_values('prediction', ascending=False)
    
    # 绘制双条形图
    x = np.arange(len(sorted_by_pred))
    width = 0.35
    
    plt.bar(x - width/2, sorted_by_pred['prediction'], width, label='预测值')
    plt.bar(x + width/2, sorted_by_pred['performance'], width, label='实际表现')
    
    plt.xticks(x, sorted_by_pred['industry'], rotation=90)
    plt.title('预测值与实际表现对比 (按预测值排序)')
    plt.xlabel('行业')
    plt.ylabel('值')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f"{output_dir}/prediction_performance_bar_by_prediction.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 按实际性能排序的柱状图
    plt.figure(figsize=(14, 8))
    sorted_by_perf = valid_comparison.sort_values('performance', ascending=False)
    
    x = np.arange(len(sorted_by_perf))
    
    plt.bar(x - width/2, sorted_by_perf['prediction'], width, label='预测值')
    plt.bar(x + width/2, sorted_by_perf['performance'], width, label='实际表现')
    
    plt.xticks(x, sorted_by_perf['industry'], rotation=90)
    plt.title('预测值与实际表现对比 (按实际表现排序)')
    plt.xlabel('行业')
    plt.ylabel('值')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f"{output_dir}/prediction_performance_bar_by_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存比较数据
    comparison_df.to_csv(f"{output_dir}/prediction_performance_comparison.csv", index=False)
    
    return comparison_df

def main():
    try:
        # 加载数据
        performance_data = load_performance_data('/root/autodl-tmp/sector_rotation_updateV2/models/all_industry_performance_5d.pt')
        prediction_data = load_prediction_data('./output/historical_predictions_high_top5.csv')
        
        if performance_data is None or prediction_data is None:
            print("无法加载数据，退出分析")
            return
        
        # 检查性能数据
        perf_df = examine_performance_data(performance_data)
        
        # 比较预测与实际性能
        if perf_df is not None:
            comparison_df = compare_prediction_with_performance(prediction_data, perf_df)
            
            if comparison_df is not None:
                print("\n分析完成，结果已保存到output目录")
    
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
