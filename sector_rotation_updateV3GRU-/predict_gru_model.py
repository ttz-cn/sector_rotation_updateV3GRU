import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import config
from utils.utils_tools import load_panel_dropd
from train_gru_model import GRUModel

# 全局变量使用G_前缀标识
G_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {G_DEVICE}")

# 模型路径
G_MODELS_PATH = os.path.join("./models", "all_industry_best_models_state.pt")
G_PARAMS_PATH = os.path.join("./models", "all_industry_best_params.pt")

# 加载模型和参数
if os.path.exists(G_MODELS_PATH) and os.path.exists(G_PARAMS_PATH):
    G_INDUSTRY_MODELS_STATE = torch.load(G_MODELS_PATH)
    G_INDUSTRY_PARAMS = torch.load(G_PARAMS_PATH)
    G_INDUSTRY_CODES = list(G_INDUSTRY_MODELS_STATE.keys())
    print(f"找到 {len(G_INDUSTRY_CODES)} 个行业模型")
else:
    print("ERROR: model file not found, please run train_gru_model.py first")
    G_INDUSTRY_MODELS_STATE = None
    G_INDUSTRY_PARAMS = None
    G_INDUSTRY_CODES = []

# 数据加载
G_ETF_PANEL_DATA = load_panel_dropd("./db/panel_data.h5", key="sector")

# 因子数据
G_TECH_FACTORS = {}
G_CRD_FACTORS = {}

# 载入技术因子
for factor in config.list_tech_factor:
    G_TECH_FACTORS[factor] = pd.read_hdf("./db/indicator_timeseries_data.h5", 
                                      key=f"{factor}-{config.rebalance_period}")
# 载入拥挤度因子
for factor in config.list_crd_factor:
    G_CRD_FACTORS[factor] = pd.read_hdf("./db/indicator_timeseries_data.h5", 
                                     key=f"{factor}-{config.rebalance_period}")

# 窗口大小
G_FEATURE_WINDOW_SIZE = 20
G_PREDICTION_WINDOW_SIZE = 5

# 提前准备每个行业的因子数据
G_INDUSTRY_FACTORS = {}
for industry in G_INDUSTRY_CODES:
    # 准备该行业的技术因子
    tech_df = pd.DataFrame({k: v[industry] for k, v in G_TECH_FACTORS.items() if industry in v.columns})
    
    # 准备该行业的拥挤度因子
    crd_df = pd.DataFrame({k: v[industry] for k, v in G_CRD_FACTORS.items() if industry in v.columns})
    
    # 合并两种因子并丢弃缺失值
    factor_df = pd.concat([tech_df, crd_df], axis=1).dropna()
    
    # 存储该行业的所有因子数据
    G_INDUSTRY_FACTORS[industry] = factor_df
    
    print(f"行业 {industry} 因子数据已准备，形状: {factor_df.shape}")


def predict_returns_for_date(target_date=None):
    """使用已训练好的GRU模型预测指定日期的行业收益率"""
    print("开始预测...")
    
    if G_INDUSTRY_MODELS_STATE is None or len(G_INDUSTRY_CODES) == 0:
        print("错误: 模型文件加载失败")
        return None
    
    # 获取需要预测的日期数据
    if target_date is not None and isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    
    # 准备预测
    predictions = {}
    
    for industry in G_INDUSTRY_CODES:
        print(f"预测行业: {industry}")
        
        # 获取该行业已准备好的因子数据
        factor_df = G_INDUSTRY_FACTORS[industry].copy()
        
        # 如果指定了目标日期，筛选数据
        if target_date is not None:
            factor_df = factor_df[factor_df.index <= target_date]
        
        # 检查数据是否足够
        if len(factor_df) < G_FEATURE_WINDOW_SIZE:
            print(f"  行业 {industry} 的因子数据不足，跳过")
            continue
        
        # 准备序列数据
        scaler = StandardScaler()
        factor_values = scaler.fit_transform(factor_df.values.T).T
        last_window = factor_values[-G_FEATURE_WINDOW_SIZE:]
        X_tensor = torch.FloatTensor(np.array([last_window]))
        
        # 创建模型
        hidden_size = G_INDUSTRY_PARAMS[industry]["hidden_size"]
        input_size = factor_df.shape[1]
        
        model = GRUModel(input_size=input_size, hidden_size=hidden_size)
        model.load_state_dict(G_INDUSTRY_MODELS_STATE[industry])
        model = model.to(G_DEVICE)
        
        # 预测
        model.eval()
        with torch.no_grad():
            X_tensor = X_tensor.to(G_DEVICE)
            pred = model(X_tensor).cpu().numpy().item()
            
        predictions[industry] = pred
    
    # 创建结果DataFrame
    if target_date is None:
        # 使用最新日期
        target_date = factor_df.index[-1]
        
    pred_df = pd.DataFrame(predictions, index=[target_date]).T
    pred_df.columns = ["predicted_return"]
    
    # 排序
    sorted_pred = pred_df.sort_values("predicted_return", ascending=False)
    
    # 保存结果
    output_file = f"./output/gru_predictions_{target_date.strftime('%Y%m%d')}.csv"
    sorted_pred.to_csv(output_file)
    print(f"预测结果已保存到 {output_file}")
    
    return sorted_pred


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="预测行业收益率")
    parser.add_argument("--date", type=str, help="预测日期 (YYYY-MM-DD 格式，默认为最新日期)")
    args = parser.parse_args()
    
    target_date = None
    if args.date:
        target_date = args.date
        
    predictions = predict_returns_for_date(target_date)
    print("\n预测收益率排序:")
    print(predictions)
