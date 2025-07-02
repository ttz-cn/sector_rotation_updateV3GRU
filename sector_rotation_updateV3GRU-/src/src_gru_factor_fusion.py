import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from typing import Dict, Tuple, Any, List, Optional
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import TensorDataset, DataLoader


class GRUModel(nn.Module):
    """GRU模型用于因子融合"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):
        """
        初始化GRU模型

        参数:
        - input_size: 输入特征的数量
        - hidden_size: GRU隐藏层的大小
        - output_size: 输出维度，默认为1（预测未来5天收益率）
        """
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


from sklearn.preprocessing import scale


def prepare_sequence_data(
    factor_df: pd.DataFrame, y_ret: pd.Series, window_size: int
) -> Tuple[torch.Tensor, torch.Tensor, List[pd.Timestamp]]:
    # 确保索引对齐
    common_idx = factor_df.index.intersection(y_ret.index)
    factor_df = factor_df.loc[common_idx]
    y_ret = y_ret.loc[common_idx]

    # 截面标准化：axis=1 表示对每一行进行标准化
    factor_values = scale(factor_df.values, axis=1)

    # 创建序列数据
    X, y = [], []
    valid_dates = []

    for i in range(len(factor_df) - window_size):
        X.append(factor_values[i : i + window_size])
        y.append(y_ret.iloc[i + window_size])
        valid_dates.append(y_ret.index[i + window_size])

    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(np.array(X))  # X_tensor.shape =  [sample_size, window_size, input_size]
    y_tensor = torch.FloatTensor(np.array(y)).view(-1, 1)  # y_tensor.shape =  [sample_size, 1]

    return X_tensor, y_tensor, valid_dates


def train_gru_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    input_size: int,
    hidden_size: int,
    device: torch.device,
    epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 64,  # 新增batch_size参数，默认64
) -> Tuple[GRUModel, Dict[str, float]]:
    """
    训练GRU模型（支持mini-batch）

    参数:
    - X_train, y_train: 训练数据
    - X_val, y_val: 验证数据
    - input_size: 输入特征数量
    - hidden_size: GRU隐藏层大小
    - device: 训练设备
    - epochs: 训练轮数
    - learning_rate: 学习率
    - batch_size: mini-batch大小

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
    best_model = None
    patience = 20
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
            best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # 加载最佳模型
    if best_model is not None:
        model.load_state_dict(best_model)

    # 返回训练好的模型和训练指标
    metrics = {
        "train_loss": train_losses[-1] if train_losses else None,
        "val_loss": val_losses[-1] if val_losses else None,
        "best_val_loss": best_val_loss,
    }

    return model, metrics


def gru_factor_composite(
    factor_df: pd.DataFrame,
    y_ret: pd.Series,
    window_size: int = 20,
    hidden_size: int = 30,
    n_splits: int = 5,
    device: torch.device = torch.device("cpu"),
    epochs: int = 100,
    lr: float = 0.001,
) -> Tuple[GRUModel, Dict[pd.Timestamp, float]]:
    """
    使用GRU模型合成因子

    参数:
    - factor_df: 因子数据框
    - y_ret: 目标收益率序列
    - window_size: 时间窗口大小
    - hidden_size: GRU隐藏层大小
    - n_splits: K折交叉验证的折数
    - device: 训练设备
    - epochs: 训练轮数
    - lr: 学习率

    返回:
    - model: 最终训练好的模型
    - prediction_dict: 日期到预测值的映射
    """
    # 准备数据
    X_tensor, y_tensor, valid_dates = prepare_sequence_data(factor_df, y_ret, window_size)
    input_size = factor_df.shape[1]

    # 使用TimeSeriesSplit进行时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_predictions = {}
    final_model = None

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_tensor), 1):
        print(f"Training fold {fold}/{n_splits}")

        # 分割数据
        X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
        X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

        # 训练模型
        model, metrics = train_gru_model(X_train, y_train, X_val, y_val, input_size, hidden_size, device, epochs, lr)

        # 保存最后一折的模型作为最终模型
        if fold == n_splits:
            final_model = model

        # 预测验证集
        model.eval()
        with torch.no_grad():
            X_val = X_val.to(device)
            val_preds = model(X_val).cpu().numpy().flatten()

        # 存储预测结果
        for i, idx in enumerate(val_idx):
            date = valid_dates[idx]
            all_predictions[date] = val_preds[i]

    return final_model, all_predictions


def optimize_portfolio_weights(
    factor_values: pd.Series, benchmark_weights: pd.Series, params: Dict[str, Any]
) -> pd.Series:
    """
    基于因子值优化投资组合权重

    参数:
    - factor_values: 因子值序列（行业级别）
    - benchmark_weights: 基准权重（通常为等权）
    - params: 优化参数
        - tracking_error: 跟踪误差约束
        - industry_bias: 行业偏离度限制
        - weight_limit: 权重限制

    返回:
    - optimal_weights: 优化后的权重
    """
    # 这里是简化实现，实际应使用凸优化
    # 根据因子值对行业进行排序
    sorted_factors = factor_values.sort_values(ascending=False)

    # 分配权重
    n_industries = len(factor_values)
    rank_weight = pd.Series(
        [(n_industries - i) / sum(range(1, n_industries + 1)) for i in range(n_industries)], index=sorted_factors.index
    )

    # 应用权重限制
    weight_limit = params["weight_limit"]
    rank_weight = rank_weight.clip(0, weight_limit)

    # 归一化权重
    if rank_weight.sum() > 0:
        rank_weight = rank_weight / rank_weight.sum()

    return rank_weight


def adjust_portfolio_by_crowding(
    factor_values: pd.Series, weights: pd.Series, min_industries: int = 3, crowding_threshold: float = 0.05
) -> pd.Series:
    """
    根据拥挤度调整投资组合

    参数:
    - factor_values: 因子值
    - weights: 初始权重
    - min_industries: 最少持有行业数
    - crowding_threshold: 持有阈值

    返回:
    - adjusted_weights: 调整后的权重
    """
    # 确保至少持有指定数量的行业
    sorted_weights = weights.sort_values(ascending=False)
    top_industries = sorted_weights.head(min_industries)

    # 根据阈值过滤权重
    adjusted_weights = weights.copy()
    adjusted_weights[adjusted_weights < crowding_threshold] = 0

    # 确保至少持有min_industries个行业
    if (adjusted_weights > 0).sum() < min_industries:
        for ind in top_industries.index:
            adjusted_weights[ind] = max(adjusted_weights[ind], top_industries[ind])

    # 归一化权重
    if adjusted_weights.sum() > 0:
        adjusted_weights = adjusted_weights / adjusted_weights.sum()

    return adjusted_weights
