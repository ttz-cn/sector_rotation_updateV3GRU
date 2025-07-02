import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
from datetime import datetime, timedelta
import os
from scipy.stats import spearmanr, pearsonr
import traceback

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
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

def evaluate_prediction_accuracy(predictions, performances, output_dir='./output'):
    """评估预测准确性"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n开始评估预测准确性...")
    
    # 转换性能数据为DataFrame
    print("转换性能数据为DataFrame...")
    perf_dates = []
    perf_industries = []
    perf_values = []
    
    try:
        # 假设performances是一个字典，键是日期，值是各行业的表现
        for date_str, industry_perfs in performances.items():
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
                for industry, perf in industry_perfs.items():
                    perf_dates.append(date)
                    perf_industries.append(industry)
                    perf_values.append(perf)
            except Exception as e:
                print(f"处理日期 {date_str} 时出错: {e}")
                continue
        
        # 创建性能DataFrame
        performance_df = pd.DataFrame({
            'date': perf_dates,
            'industry': perf_industries,
            'performance': perf_values
        })
        
        print(f"性能数据DataFrame创建完成，共 {len(performance_df)} 行")
        print(f"性能数据日期范围: {min(perf_dates)} 至 {max(perf_dates)}")
        print(f"性能数据行业数量: {len(set(perf_industries))}")
        
        # 按日期和行业排序
        performance_df = performance_df.sort_values(['date', 'industry'])
        
        # 将性能数据转换为与预测数据相同的格式（行业为列，日期为索引）
        print("转换性能数据格式...")
        perf_pivot = performance_df.pivot(index='date', columns='industry', values='performance')
        perf_pivot.index = pd.DatetimeIndex(perf_pivot.index)
        
        # 获取两个数据集的共同日期和行业
        common_dates = sorted(list(set(predictions.index) & set(perf_pivot.index)))
        common_industries = sorted(list(set(predictions.columns) & set(perf_pivot.columns)))
        
        if not common_dates or not common_industries:
            print("警告: 预测数据和性能数据没有共同的日期或行业")
            return None, None
        
        print(f"共同日期范围: {common_dates[0]} 至 {common_dates[-1]}, 共 {len(common_dates)} 天")
        print(f"共同行业数量: {len(common_industries)} 个")
        
        # 对齐数据
        aligned_pred = predictions.loc[common_dates, common_industries]
        aligned_perf = perf_pivot.loc[common_dates, common_industries]
        
        # 保存对齐后的数据
        aligned_pred.to_csv(f"{output_dir}/aligned_predictions.csv")
        aligned_perf.to_csv(f"{output_dir}/aligned_performances.csv")
        
        # 计算每个日期的相关性
        print("计算每个日期的相关性...")
        corr_dates = []
        spearman_corrs = []
        pearson_corrs = []
        pred_top5_hit_rates = []  # Top5命中率
        
        for date in common_dates:
            pred_row = aligned_pred.loc[date]
            perf_row = aligned_perf.loc[date]
            
            # 过滤掉NaN值
            valid_mask = ~(pred_row.isna() | perf_row.isna())
            if valid_mask.sum() < 5:  # 至少需要5个有效值
                print(f"日期 {date} 的有效值少于5个，跳过")
                continue
            
            pred_valid = pred_row[valid_mask]
            perf_valid = perf_row[valid_mask]
            
            # 计算Spearman和Pearson相关系数
            try:
                spearman, _ = spearmanr(pred_valid, perf_valid)
                pearson, _ = pearsonr(pred_valid, perf_valid)
                
                corr_dates.append(date)
                spearman_corrs.append(spearman)
                pearson_corrs.append(pearson)
                
                # 计算Top5命中率
                pred_top5 = pred_valid.nlargest(5).index
                perf_top5 = perf_valid.nlargest(5).index
                hit_count = len(set(pred_top5) & set(perf_top5))
                hit_rate = hit_count / 5.0
                pred_top5_hit_rates.append(hit_rate)
            except Exception as e:
                print(f"计算日期 {date} 的相关性时出错: {e}")
                continue
        
        # 创建相关性DataFrame
        corr_df = pd.DataFrame({
            'date': corr_dates,
            'spearman': spearman_corrs,
            'pearson': pearson_corrs,
            'top5_hit_rate': pred_top5_hit_rates
        })
        
        print(f"完成相关性计算，共 {len(corr_df)} 个有效日期")
        
        # 保存相关性数据
        corr_df.to_csv(f"{output_dir}/prediction_performance_correlation.csv", index=False)
        
        # 绘制相关性随时间的变化
        print("绘制相关性图表...")
        plt.figure(figsize=(15, 8))
        plt.plot(corr_df['date'], corr_df['spearman'], 'o-', label='Spearman相关系数')
        plt.plot(corr_df['date'], corr_df['pearson'], 'o-', label='Pearson相关系数')
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('预测值与实际表现相关性随时间变化')
        plt.xlabel('日期')
        plt.ylabel('相关系数')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(f"{output_dir}/prediction_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制Top5命中率随时间的变化
        plt.figure(figsize=(15, 8))
        plt.plot(corr_df['date'], corr_df['top5_hit_rate'], 'o-', color='green')
        
        plt.axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='随机选择期望值 (1/5)')
        plt.title('Top5预测命中率随时间变化')
        plt.xlabel('日期')
        plt.ylabel('命中率')
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(f"{output_dir}/top5_hit_rate.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 计算总体相关性
        print("计算总体相关性...")
        pred_flat = aligned_pred.values.flatten()
        perf_flat = aligned_perf.values.flatten()
        
        # 过滤掉NaN值
        valid_mask = ~(np.isnan(pred_flat) | np.isnan(perf_flat))
        pred_valid = pred_flat[valid_mask]
        perf_valid = perf_flat[valid_mask]
        
        overall_spearman, _ = spearmanr(pred_valid, perf_valid)
        overall_pearson, _ = pearsonr(pred_valid, perf_valid)
        
        print(f"总体Spearman相关系数: {overall_spearman:.4f}")
        print(f"总体Pearson相关系数: {overall_pearson:.4f}")
        print(f"平均Top5命中率: {np.mean(corr_df['top5_hit_rate']):.4f}")
        
        # 绘制散点图
        print("绘制散点图...")
        plt.figure(figsize=(10, 8))
        plt.scatter(pred_valid, perf_valid, alpha=0.5)
        
        # 添加回归线
        z = np.polyfit(pred_valid, perf_valid, 1)
        p = np.poly1d(z)
        plt.plot(sorted(pred_valid), p(sorted(pred_valid)), "r--", alpha=0.8)
        
        plt.title(f'预测值与实际表现散点图 (Spearman={overall_spearman:.4f}, Pearson={overall_pearson:.4f})')
        plt.xlabel('预测值')
        plt.ylabel('实际表现')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(f"{output_dir}/prediction_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 计算平均IC值 (信息系数)
        ic_mean = np.mean(corr_df['spearman'])
        ic_std = np.std(corr_df['spearman'])
        ic_ratio = ic_mean / ic_std if ic_std != 0 else 0
        
        print(f"平均IC值: {ic_mean:.4f}")
        print(f"IC标准差: {ic_std:.4f}")
        print(f"IC比率(IR): {ic_ratio:.4f}")
        
        # 统计正负IC的比例
        positive_ic = (corr_df['spearman'] > 0).sum() / len(corr_df['spearman'])
        print(f"正IC比例: {positive_ic:.4f}")
        
        # 计算每个行业的预测值与实际表现的相关性
        print("计算各行业相关性...")
        industry_corrs = {}
        for industry in common_industries:
            try:
                pred_ind = aligned_pred[industry].dropna()
                perf_ind = aligned_perf[industry].reindex(pred_ind.index).dropna()
                
                # 确保两个序列长度相同且有足够的数据点
                common_idx = pred_ind.index.intersection(perf_ind.index)
                if len(common_idx) < 5:
                    print(f"行业 {industry} 的有效数据点少于5个，跳过")
                    continue
                    
                pred_ind = pred_ind.loc[common_idx]
                perf_ind = perf_ind.loc[common_idx]
                
                spearman, _ = spearmanr(pred_ind, perf_ind)
                industry_corrs[industry] = spearman
            except Exception as e:
                print(f"计算行业 {industry} 的相关性时出错: {e}")
                continue
        
        # 转换为DataFrame并排序
        industry_corr_df = pd.DataFrame.from_dict(industry_corrs, orient='index', columns=['correlation'])
        industry_corr_df = industry_corr_df.sort_values('correlation', ascending=False)
        
        # 绘制柱状图
        print("绘制行业相关性柱状图...")
        plt.figure(figsize=(14, 8))
        bars = plt.bar(industry_corr_df.index, industry_corr_df['correlation'])
        
        # 为正负相关设置不同颜色
        for i, bar in enumerate(bars):
            if industry_corr_df['correlation'].iloc[i] >= 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('各行业预测值与实际表现的相关性')
        plt.xlabel('行业')
        plt.ylabel('Spearman相关系数')
        plt.xticks(rotation=90)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(f"{output_dir}/industry_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存行业相关性数据
        industry_corr_df.to_csv(f"{output_dir}/industry_correlation.csv")
        
        print("评估完成!")
        return corr_df, industry_corr_df
    
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        traceback.print_exc()
        return None, None

def main():
    try:
        # 加载数据
        performance_data = load_performance_data('/root/autodl-tmp/sector_rotation_updateV2/models/all_industry_performance_5d.pt')
        prediction_data = load_prediction_data('./output/historical_predictions_high_top5.csv')
        
        if performance_data is None or prediction_data is None:
            print("无法加载数据，退出分析")
            return
        
        # 打印性能数据结构以便理解
        print("\n性能数据结构:")
        if isinstance(performance_data, dict):
            print(f"键数量: {len(performance_data)}")
            sample_key = list(performance_data.keys())[0]
            print(f"样本键: {sample_key}")
            print(f"样本值类型: {type(performance_data[sample_key])}")
            if isinstance(performance_data[sample_key], dict):
                print(f"样本值行业数量: {len(performance_data[sample_key])}")
                sample_industry = list(performance_data[sample_key].keys())[0]
                print(f"样本行业: {sample_industry}")
                print(f"样本行业值: {performance_data[sample_key][sample_industry]}")
        else:
            print(f"性能数据类型: {type(performance_data)}")
        
        # 评估预测准确性
        corr_df, industry_corr_df = evaluate_prediction_accuracy(prediction_data, performance_data)
        
        if corr_df is not None:
            # 输出总结
            print("\n分析总结:")
            print(f"总共分析了 {len(corr_df)} 个交易日的预测与实际表现")
            print(f"平均Spearman相关系数: {corr_df['spearman'].mean():.4f}")
            print(f"平均Pearson相关系数: {corr_df['pearson'].mean():.4f}")
            print(f"平均Top5命中率: {corr_df['top5_hit_rate'].mean():.4f}")
            
            # 统计正相关的天数比例
            positive_days = (corr_df['spearman'] > 0).sum()
            total_days = len(corr_df)
            print(f"预测与实际表现正相关的天数: {positive_days}/{total_days} ({positive_days/total_days:.2%})")
            
            # 统计预测与实际表现显著正相关的天数(>0.3)
            significant_days = (corr_df['spearman'] > 0.3).sum()
            print(f"预测与实际表现显著正相关(>0.3)的天数: {significant_days}/{total_days} ({significant_days/total_days:.2%})")
            
            # 统计Top5命中率>0.4的天数
            good_hit_days = (corr_df['top5_hit_rate'] > 0.4).sum()
            print(f"Top5命中率>40%的天数: {good_hit_days}/{total_days} ({good_hit_days/total_days:.2%})")
    
    except Exception as e:
        print(f"主函数执行过程中发生错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 