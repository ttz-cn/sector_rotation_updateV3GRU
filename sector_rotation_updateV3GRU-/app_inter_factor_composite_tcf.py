import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pickle
import os

from src.src_get_composit_indicators import (
    calculate_quantile_groups,
    gru_factor_composite,
    calculate_icir_for_factors,
    calculate_icir_weighted_factors,
)
from src.src_indicator_processing import optimize_portfolio_weights, adjust_portfolio_by_crowding
from utils.utils_tools import get_factor_value_extracted
import config

# 导入基本面因子作为阈值
composite_funda_factor = pd.read_hdf(
    "./db/composite_factor.h5", key="composite_funda_factor-{}".format(config.rebalance_period)
)

# 加载收盘价数据
fund_close = pd.read_hdf("./db/indicator_timeseries_data.h5", key=f"fund_close-{config.rebalance_period}")
# 使用5天收益率
y_ret = fund_close.pct_change(5).mean(axis=1)


def run_tcf_factor_composite():
    """
    运行技术因子、人气因子和基本面因子的合成
    """
    # 读取因子数据
    dic_techfactor = {}
    for k, v in config.dic_tech_factors.items():
        dic_techfactor[k] = pd.read_pickle("./db/{}_SI_M.pkl".format(k))

    dic_crdfactor = {}
    for k, v in config.dic_crd_factors.items():
        dic_crdfactor[k] = pd.read_pickle("./db/{}_SI_M.pkl".format(k))

    dic_fundafactor = {}
    for k, v in config.dic_funda_factors.items():
        dic_fundafactor[k] = pd.read_pickle("./db/{}_SI_M.pkl".format(k))

    # 读取收益率数据
    close_SI = pd.read_pickle("./db/close_SI_d.pkl")
    forward_ret = close_SI.pct_change(5).shift(-5)  # 计算未来5天收益率

    # 使用GRU模型合成因子
    # 1. 技术因子
    tech_factor_df = pd.DataFrame(index=forward_ret.index)
    for k, v in dic_techfactor.items():
        tech_factor_df[k] = v

    # 2. 人气因子
    crd_factor_df = pd.DataFrame(index=forward_ret.index)
    for k, v in dic_crdfactor.items():
        crd_factor_df[k] = v

    # 3. 基本面因子
    funda_factor_df = pd.DataFrame(index=forward_ret.index)
    for k, v in dic_fundafactor.items():
        funda_factor_df[k] = v

    # 使用GPU如果可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 合成技术因子
    print("开始合成技术因子...")
    tech_model, tech_pred_dict = gru_factor_composite(
        tech_factor_df.dropna(),
        forward_ret,
        window_size=20,
        hidden_size=20,
        device=device,
    )

    # 合成人气因子
    print("开始合成人气因子...")
    crd_model, crd_pred_dict = gru_factor_composite(
        crd_factor_df.dropna(),
        forward_ret,
        window_size=20,
        hidden_size=20,
        device=device,
    )

    # 合成基本面因子
    print("开始合成基本面因子...")
    funda_model, funda_pred_dict = gru_factor_composite(
        funda_factor_df.dropna(),
        forward_ret,
        window_size=20,
        hidden_size=20,
        device=device,
    )

    # 将预测结果转换为DataFrame
    tech_pred_df = pd.Series(tech_pred_dict)
    crd_pred_df = pd.Series(crd_pred_dict)
    funda_pred_df = pd.Series(funda_pred_dict)

    # 保存合成因子
    tech_pred_df.to_pickle("./db/composite_techfactor_SI_M.pkl")
    crd_pred_df.to_pickle("./db/composite_crdfactor_SI_M.pkl")
    funda_pred_df.to_pickle("./db/composite_fundafactor_SI_M.pkl")

    # 保存模型
    torch.save(tech_model.state_dict(), "./db/tech_model.pth")
    torch.save(crd_model.state_dict(), "./db/crd_model.pth")
    torch.save(funda_model.state_dict(), "./db/funda_model.pth")

    print("因子合成完成！")


if __name__ == "__main__":
    # 读取GRU模型训练好的因子 (如未训练，需要先运行app_inter_factor_composite_tfc.py)
    try:
        gru_factor_df = pd.read_hdf("./db/composite_factor.h5", key=f"gru_composite_factor-{config.rebalance_period}")
        print(f"成功加载GRU合成因子，日期范围: {gru_factor_df.index.min()} - {gru_factor_df.index.max()}")
    except:
        print("未找到GRU合成因子，请先运行app_inter_factor_composite_tfc.py生成")
        exit(1)

    # 获取最新交易日的因子值
    latest_date = gru_factor_df.index.max()
    latest_factor = gru_factor_df.loc[latest_date]

    print(f"\n使用日期 {latest_date} 的因子值进行组合优化")

    # 计算基准权重（等权）
    industries = fund_close.columns.tolist()
    benchmark_weights = pd.Series(1 / len(industries), index=industries)

    # 判断当前景气度情况
    positive_count = sum(latest_factor > 0.7)  # 高于0.7概率的行业数

    # 根据景气度选择风险偏好
    risk_preference = "high" if positive_count >= config.optimization_params["minimum_industry_count"] else "low"
    print(f"当前有{positive_count}个行业处于高景气状态，风险偏好设置为: {risk_preference}")

    # 配置优化参数
    opt_params = {
        "tracking_error": config.optimization_params["tracking_error"][risk_preference],
        "industry_bias": config.optimization_params["industry_bias"][risk_preference],
        "weight_limit": config.optimization_params["weight_limit"][risk_preference],
    }
    print(
        f"优化参数:\n - 跟踪误差: {opt_params['tracking_error']}\n - 行业偏离: {opt_params['industry_bias']}\n - 权重上限: {opt_params['weight_limit']}"
    )

    # 进行优化
    print("\n执行组合优化...")
    optimal_weights = optimize_portfolio_weights(
        factor_values=latest_factor, benchmark_weights=benchmark_weights, params=opt_params
    )

    # 根据拥挤度调整组合
    print("\n根据拥挤度和最小行业数调整组合...")
    final_weights = adjust_portfolio_by_crowding(
        factor_values=latest_factor,
        weights=optimal_weights,
        min_industries=config.optimization_params["minimum_industry_count"],
        crowding_threshold=config.optimization_params["holding_threshold"],
    )

    # 保存最终权重
    output_path = f"./output/tcf_optimal_weights_{latest_date.strftime('%Y%m%d')}.csv"
    final_weights.to_csv(output_path)
    print(f"最终权重已保存至: {output_path}")

    # 打印结果
    print("\n最终行业配置权重:")
    for ind, weight in final_weights[final_weights > 0].sort_values(ascending=False).items():
        ind_name = config.dic_Industry2Etf[ind][0] if ind in config.dic_Industry2Etf else ind
        print(f"{ind_name}: {weight:.2%}")

    # 生成可视化
    plt.figure(figsize=(12, 8))

    # 只显示持仓的行业
    held_industries = final_weights[final_weights > 0].sort_values(ascending=False)
    labels = [config.dic_Industry2Etf[ind][0] for ind in held_industries.index]

    plt.bar(labels, held_industries.values)
    plt.title(f"行业配置权重 ({latest_date.strftime('%Y-%m-%d')})")
    plt.ylabel("权重")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图表
    plt.savefig(f"./output/tcf_weights_{latest_date.strftime('%Y%m%d')}.png")
    print(f"权重图表已保存至: ./output/tcf_weights_{latest_date.strftime('%Y%m%d')}.png")

    # 显示图表
    plt.show()

    run_tcf_factor_composite()
