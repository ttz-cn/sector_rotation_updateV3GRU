import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
import config
from utils.utils_tools import load_panel_dropd
import os
import pickle
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
import copy

# 设置随机种子以确保结果可重现
np.random.seed(42)
torch.manual_seed(42)

# 创建模型保存目录
os.makedirs("./models", exist_ok=True)


# 定义GRU模型
class GRUModel(nn.Module):
    """GRU模型用于因子融合"""

    def __init__(self, input_size, hidden_size, output_size=1, dropout=0.2):
        """
        初始化GRU模型

        参数:
        - input_size: 输入特征的数量
        - hidden_size: GRU隐藏层的大小
        - output_size: 输出维度，默认为1（预测未来5天收益率）
        - dropout: Dropout比率
        """
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        前向传播

        参数:
        - x: 输入张量，形状为 [batch_size, seq_len, input_size]

        返回:
        - 输出张量，形状为 [batch_size, output_size]
        """
        out, _ = self.gru(x)  # out shape: [batch_size, seq_len, hidden_size]
        out = out[:, -1, :]  # 只取序列的最后一个输出
        out = self.fc(out)  # [batch_size, output_size]
        return out


# 准备数据函数
def prepare_sequence_data(factor_df, y_ret, feature_window=20, prediction_window=5):
    """
    准备序列数据

    参数:
    - factor_df: 因子DataFrame
    - y_ret: 目标收益率序列
    - feature_window: 特征窗口大小
    - prediction_window: 预测窗口大小

    返回:
    - X_tensor: 特征张量
    - y_tensor: 目标张量
    - valid_dates: 有效日期列表
    """
    # 确保索引对齐
    common_idx = factor_df.index.intersection(y_ret.index)
    factor_df = factor_df.loc[common_idx]
    y_ret = y_ret.loc[common_idx]

    # 标准化特征
    scaler = StandardScaler()
    factor_values = scaler.fit_transform(factor_df.values.T).T

    # 创建序列数据
    X, y = [], []
    valid_dates = []

    for i in range(len(factor_df) - feature_window - prediction_window + 1):
        # 使用feature_window天的因子数据作为特征
        X.append(factor_values[i : i + feature_window])
        # 预测future_window天后的收益率
        y.append(y_ret.iloc[i + feature_window + prediction_window - 1])
        valid_dates.append(y_ret.index[i + feature_window + prediction_window - 1])

    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(np.array(X))
    y_tensor = torch.FloatTensor(np.array(y)).view(-1, 1)

    return X_tensor, y_tensor, valid_dates


# 训练GRU模型
def train_gru_model(
    X_train,
    y_train,
    X_val,
    y_val,
    input_size,
    hidden_size,
    device,
    epochs=100,
    learning_rate=0.001,
    batch_size=32,
):
    """
    训练GRU模型

    参数:
    - X_train, y_train: 训练数据
    - X_val, y_val: 验证数据
    - input_size: 输入特征数量
    - hidden_size: GRU隐藏层大小
    - device: 训练设备
    - epochs: 训练轮数
    - learning_rate: 学习率
    - batch_size: 批次大小

    返回:
    - model: 训练好的模型
    - metrics: 训练和验证误差
    """
    # 初始化模型
    model = GRUModel(input_size, hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 构造DataLoader
    train_dataset = TensorDataset(X_train.to(device), y_train.to(device))
    val_dataset = TensorDataset(X_val.to(device), y_val.to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model_state = None
    patience = 10
    patience_counter = 0


    for epoch in range(epochs):
        # 训练模式
        model.train()
        total_train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # 验证模式
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_outputs = model(xb)
                val_loss = criterion(val_outputs, yb)
                total_val_loss += val_loss.item() * xb.size(0)
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        # 早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 返回训练好的模型和训练指标
    metrics = {
        "train_loss": train_losses[-1] if train_losses else None,
        "val_loss": val_losses[-1] if val_losses else None,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    return model, metrics

if __name__ == "__main__":

    # 训练GRU因子融合模型并显示结果
    print("开始GRU因子融合模型训练...")

    # 当前时间，用于保存模型文件名
    current_time = datetime.now().strftime("%Y%m%d")

    # 读取因子数据
    dic_techfactor = {}  # 技术因子字典
    dic_crdfactor = {}  # 拥挤度因子字典

    # 设置模型参数
    feature_window = 20
    prediction_window = 5
    n_splits = 5

    # 为每个行业训练模型
    all_industry_best_models_state = {}
    all_industry_predictions = {}
    all_industry_best_params = {}
    industry_performance = {}


    # 检查是否有CUDA可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 载入数据
    print("加载数据...")
    # 加载ETF面板数据
    etf_panel_data = load_panel_dropd("./db/panel_data.h5", key="sector")

    # 获取收盘价时序数据
    fund_close = etf_panel_data.pivot_table(index="Date", columns="industry_code", values="CLOSE")
    fund_close = fund_close.resample("B").asfreq()  # 按工作日重采样

    # 载入技术因子
    for factor in config.list_tech_factor:
        dic_techfactor[factor] = pd.read_hdf("./db/indicator_timeseries_data.h5", key=f"{factor}-{config.rebalance_period}")
    # 载入拥挤度因子
    for factor in config.list_crd_factor:
        dic_crdfactor[factor] = pd.read_hdf("./db/indicator_timeseries_data.h5", key=f"{factor}-{config.rebalance_period}")
    # 定义目标收益率
    y_ret = fund_close.pct_change(prediction_window)  # 使用5天收益率，保留每个行业的收益率

    # 准备每个行业的因子数据
    industry_codes = fund_close.columns

    # 使用时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=n_splits)

    count = 0  # 不想用enumerate了，循环太多了
    for industry in industry_codes:
        count += 1
        print(f"{'*'*10}处理行业: {industry}{'*'*10} {count}/{len(industry_codes)}")

        # 1. 技术因子
        tech_factor_df = pd.DataFrame(index=dic_techfactor[list(dic_techfactor.keys())[0]].index)
        for k, v in dic_techfactor.items():
            tech_factor_df[k] = v[industry] if industry in v.columns else np.nan
        # 2. 人气因子
        crd_factor_df = pd.DataFrame(index=dic_crdfactor[list(dic_crdfactor.keys())[0]].index)
        for k, v in dic_crdfactor.items():
            crd_factor_df[k] = v[industry] if industry in v.columns else np.nan

        # 合并所有因子
        factor_df = pd.concat([tech_factor_df, crd_factor_df], axis=1).dropna()

        # 获取当前行业的目标收益率
        industry_y_ret = y_ret[industry].dropna()

        # 准备序列数据
        X_tensor, y_tensor, valid_dates = prepare_sequence_data(factor_df, industry_y_ret, feature_window, prediction_window)
        input_size = factor_df.shape[1]

        print(f"  数据形状: X={X_tensor.shape}, y={y_tensor.shape}")

        # 保存该行业的预测结果
        industry_predictions = {}

        # 保存该行业的所有模型
        industry_models = []

        # 定义超参数网格
        param_grid = {
            "hidden_size": [40, 50, 60],
            "learning_rate": [0.01,0.001, 0.005],
        }
        epochs = 100
        batch_size = 64

        # 网格搜索
        best_score = float("inf")
        best_params = {"hidden_size": 40, "learning_rate": 0.001}  # 设置默认参数，以防没有找到更好的
        best_predictions = None

        # 遍历所有参数组合
        for i, params in enumerate(ParameterGrid(param_grid), 1):
            print(f"    [{i}/{len(list(ParameterGrid(param_grid)))}] parameterGrid: {params}")

            # 交叉验证评估当前参数
            cv_scores = []
            # 交叉验证
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_tensor), 1):
                print(f"  训练折 {fold}/{n_splits}")

                # 分割数据
                X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
                X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

                # 训练模型
                model, metrics = train_gru_model(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    input_size,
                    hidden_size=params["hidden_size"],
                    device=device,
                    epochs=epochs,
                    learning_rate=params["learning_rate"],
                    batch_size=batch_size,
                )

                # 可以选择best_val_loss或最终的验证损失
                val_loss = metrics["best_val_loss"]  # 或者 metrics['val_losses'][-1]
                cv_scores.append(val_loss)
                

            # 计算平均验证损失
            avg_cv_score = np.mean(cv_scores)
            print(f"    参数 {params} 的平均CV分数: {avg_cv_score:.6f}")

            if avg_cv_score < best_score:
                best_score, best_params = avg_cv_score, params.copy()
                print(f"    ✓ Better Params! Score: {best_score:.6f}")
            

        print(f"{industry} ▶ Best params: {best_params}, CV loss={best_score:.6f}")
        all_industry_best_params[industry] = best_params  # 保存行业中最佳超参数

        # 使用全量数据训练最终模型
        print(f'{industry} ▶ 使用全量数据训练最终模型')
        final_model, final_metrics = train_gru_model(
            X_tensor,
            y_tensor,
            X_tensor,
            y_tensor,
            input_size=input_size,
            hidden_size=best_params["hidden_size"],
            device=device,
            epochs=epochs,
            learning_rate=best_params["learning_rate"],
            batch_size=batch_size,
        )
        print(f'{industry} ▶ 最终模型训练完成，开始保存')
        all_industry_best_models_state[industry] = final_model.state_dict()  # 保存行业中最佳模型权重

        # 预测并保存
        final_model.eval()
        with torch.no_grad():
            final_preds = final_model(X_tensor.to(device)).cpu().numpy().flatten()
        all_industry_predictions[industry] = pd.Series(final_preds, index=valid_dates)

        # 自测指标
        mse = mean_squared_error(y_tensor.numpy(), final_preds)
        r2 = r2_score(y_tensor.numpy(), final_preds)
        industry_performance[industry] = {"train_mse": mse, "train_r2": r2}


    # 汇总保存
    # 1.保存行业中最佳超参数
    params_path = os.path.join("./models", f"all_industry_best_params_{prediction_window}d.pt")
    torch.save(all_industry_best_params, params_path)

    # 2.保存行业中最佳模型
    models_path = os.path.join("./models", f"all_industry_best_models_state_{prediction_window}d.pt")
    torch.save(all_industry_best_models_state, models_path)

    # 3.保存预测结果
    predictions_path = os.path.join("./models", f"all_industry_predictions_{prediction_window}d.pt")
    torch.save(all_industry_predictions, predictions_path)

    # 4.保存自测指标
    performance_path = os.path.join("./models", f"all_industry_performance_{prediction_window}d.pt")
    torch.save(industry_performance, performance_path)
