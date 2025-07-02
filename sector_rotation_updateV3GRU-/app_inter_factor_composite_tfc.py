import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from src.src_get_composit_indicators import (
    calculate_quantile_groups,
    gru_factor_composite,
)
from utils.utils_tools import get_factor_value_extracted
import config
import pickle
import os

# 加载基本面因子作为阈值
composite_funda_factor = pd.read_hdf(
    "./db/composite_factor.h5", key="composite_funda_factor-{}".format(config.rebalance_period)
)  # 基本面指标作为阈值

# 创建持有行业名称和权重的空字典
selected_industries_dict = {}
industry_weights_dict = {}

# 加载收盘价数据计算收益率
fund_close = pd.read_hdf("./db/indicator_timeseries_data.h5", key=f"fund_close-{config.rebalance_period}")
y_ret = fund_close.pct_change(5).mean(axis=1)  # 使用5天收益率


def run_tfc_factor_composite():
    """
    运行技术因子、基本面因子和人气因子的合成
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

    # 2. 基本面因子
    funda_factor_df = pd.DataFrame(index=forward_ret.index)
    for k, v in dic_fundafactor.items():
        funda_factor_df[k] = v

    # 3. 人气因子
    crd_factor_df = pd.DataFrame(index=forward_ret.index)
    for k, v in dic_crdfactor.items():
        crd_factor_df[k] = v

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

    # 合成基本面因子
    print("开始合成基本面因子...")
    funda_model, funda_pred_dict = gru_factor_composite(
        funda_factor_df.dropna(),
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

    # 将预测结果转换为DataFrame
    tech_pred_df = pd.Series(tech_pred_dict)
    funda_pred_df = pd.Series(funda_pred_dict)
    crd_pred_df = pd.Series(crd_pred_dict)

    # 保存合成因子
    tech_pred_df.to_pickle("./db/composite_techfactor_SI_M.pkl")
    funda_pred_df.to_pickle("./db/composite_fundafactor_SI_M.pkl")
    crd_pred_df.to_pickle("./db/composite_crdfactor_SI_M.pkl")

    # 保存模型
    torch.save(tech_model.state_dict(), "./db/tech_model.pth")
    torch.save(funda_model.state_dict(), "./db/funda_model.pth")
    torch.save(crd_model.state_dict(), "./db/crd_model.pth")

    print("因子合成完成！")


if __name__ == "__main__":
    run_tfc_factor_composite()

    # 方法1：原始ICIR方法（保留作为参考）
    # 加载合成的技术和拥挤度因子
    list_factor = [
        pd.read_hdf("./db/composite_factor.h5", key="composite_tech_factor-{}".format(config.rebalance_period)),
        pd.read_hdf("./db/composite_factor.h5", key="composite_crd_factor-{}".format(config.rebalance_period)),
    ]  # composite from icir
    dic_indicator = {"factor_{}".format(i): indc for i, indc in enumerate(list_factor, 0)}

    # 技术指标+拥挤度 等权合并
    first_key = next(iter(dic_indicator))  # 获取字典第一个键名
    df_composite_factor_icir: pd.DataFrame = pd.DataFrame(
        data=0, index=dic_indicator[first_key].index, columns=dic_indicator[first_key].columns
    )  # 空df存储复合因子值
    for key, value in dic_indicator.items():
        df_composite_factor_icir += dic_indicator[key]  # 复合因子值相加

    # 保存ICIR方法的合成因子
    df_composite_factor_icir.to_hdf(
        "./db/composite_factor.h5", key="icir_composite_factor-{}".format(config.rebalance_period)
    )

    # 方法2：使用GRU模型合成因子
    print("\n使用GRU模型合成因子...")

    # 读取因子数据
    dic_factors = {}
    for factor in config.selected_indicators:
        try:
            dic_factors[factor] = pd.read_hdf(
                "./db/indicator_timeseries_data.h5", key=f"{factor}-{config.rebalance_period}"
            )
        except:
            print(f"无法加载因子: {factor}")

    # 准备因子数据
    factor_df = pd.DataFrame(index=y_ret.index)
    for factor, df in dic_factors.items():
        factor_df[factor] = df.mean(axis=1)  # 对行业因子取均值形成市场因子

    # 检查是否有GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 使用GRU模型合成因子
    gru_model, date_preds = gru_factor_composite(
        factor_df=factor_df.dropna(),
        y_ret=y_ret,
        window_size=20,
        hidden_size=20,
        device=device,
    )

    # 将预测结果转换为Series
    gru_pred = pd.Series(date_preds)

    # 创建GRU因子值DataFrame
    dates = gru_pred.index
    industries = fund_close.columns.tolist()
    gru_factor_df = pd.DataFrame(index=dates, columns=industries)

    # 填充行业因子值
    for date in dates:
        pred = gru_pred[date]
        for industry in industries:
            # 此处可以基于行业特性进行调整，这里简单地为所有行业设置相同的预测值
            gru_factor_df.loc[date, industry] = pred

    # 保存GRU合成因子
    gru_factor_df.to_hdf("./db/composite_factor.h5", key="gru_composite_factor-{}".format(config.rebalance_period))

    print("GRU因子合成完成，用于后续的组合优化")

    # 使用量化打分选择行业（传统方法，可以和GRU方法一起使用）
    df_quantil_groups = calculate_quantile_groups(df_composite_factor_icir.dropna(), 5)  # 技术指标+拥挤度
    df_threshquantil_groups = calculate_quantile_groups(
        composite_funda_factor, 5
    )  # 基本面指标threshold获取排名，qcut中间均两边堆

    # threshold quantile groups
    df_selected_industry_no_fill = df_quantil_groups.apply(
        lambda row: select_with_threshold(row, df_threshquantil_groups, 5, 1), axis=1
    )
    df_selected_industry_no_fill = pd.DataFrame(
        df_selected_industry_no_fill
    )  # df can be operated with apply in terms of row.name

    # 补齐操作
    df_selected_industry_with_filling = df_selected_industry_no_fill.apply(
        lambda row: fill_to_minimum_length(row, df_composite_factor_icir), axis=1
    )

    # 输出结果
    last_date = df_selected_industry_with_filling.index[-1]
    selected_industries = df_selected_industry_with_filling.loc[last_date][0]

    print("\n选择的行业:")
    for ind in selected_industries:
        ind_name = config.dic_Industry2Etf[ind][0] if ind in config.dic_Industry2Etf else ind
        print(f"{ind_name} ({ind})")
