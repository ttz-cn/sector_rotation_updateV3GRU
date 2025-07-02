import numpy as np
import pandas as pd
from src.src_get_indicators import calculate_icir
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, TimeSeriesSplit


def calculate_icir_for_factors(dic_techfactor, forward_ret, rollingwindow1=3, rollingwindow2=6):
    """
    计算因子的ICIR值并存储到字典中。
    """
    dic_icir = {}  # 存储不同因子ICIR值
    for k, v in dic_techfactor.items():
        # 确保因子数据没有缺失值
        v = v.dropna()
        # 计算因子的ICIR值并存储
        dic_icir[k] = calculate_icir(v, forward_ret, rollingwindow1=rollingwindow1, rollingwindow2=rollingwindow2)

    return dic_icir


def calculate_icir_weighted_factors(dic_factor, dic_icir):
    """
    使用ICIR权重对因子进行加权
    """
    dic_weighted_factor = {}  # empty dic for weighted factor values

    for key, value in dic_factor.items():  # 获取ic*factor_value dic

        # get clean icir
        icir_values = dic_icir[key].dropna()

        # get extracted value
        start_data = icir_values.index.min()
        value = value[start_data:]

        dic_weighted_factor[key] = value.apply(lambda row: row * icir_values.loc[: row.name][-1], axis=1)

    df_composite_factor = sum(dic_weighted_factor.values())  #  因子值相加

    return df_composite_factor


class FactorDataset(Dataset):
    """
    因子数据集类，用于PyTorch训练
    """

    def __init__(self, x, y, device="cpu"):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class GRURegressor(torch.nn.Module):
    """
    GRU回归模型，直接预测未来收益率
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)  # 输出一个值，即未来收益率

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        _, h = self.gru(x)  # h: [1, batch_size, hidden_size]
        h = h.squeeze(0)  # [batch_size, hidden_size]
        out = self.fc(h)  # [batch_size, 1]
        return out.squeeze(-1)  # [batch_size]


def prepare_time_series_data(factor_df, y_ret, window_size=20):
    """
    准备时间序列数据，将因子数据转换为滑动窗口格式

    参数:
    factor_df: 因子DataFrame，行为日期，列为不同因子
    y_ret: 目标收益率Series，索引为日期
    window_size: 滑动窗口大小

    返回:
    X: 特征数据，形状为 [样本数, 窗口大小, 特征数]
    y: 目标数据，形状为 [样本数]
    dates: 对应的日期
    """
    # 确保索引对齐
    common_index = factor_df.index.intersection(y_ret.index)
    factor_df = factor_df.loc[common_index]
    y_ret = y_ret.loc[common_index]

    # 创建滑动窗口数据
    X, y, dates = [], [], []
    for i in range(len(factor_df) - window_size):
        X.append(factor_df.iloc[i : i + window_size].values)
        y.append(y_ret.iloc[i + window_size])
        dates.append(factor_df.index[i + window_size])

    return np.array(X), np.array(y), np.array(dates)


def gru_factor_composite(
    factor_df,
    y_ret,
    window_size=20,
    test_size=0.2,
    hidden_size=20,
    batch_size=64,
    num_epochs=100,
    patience=10,
    n_splits=5,
    device="cpu",
):
    """
    使用GRU模型直接预测未来收益率，采用时间序列交叉验证

    参数:
    factor_df: 因子数据DataFrame，行为日期，列为不同因子
    y_ret: 目标收益率Series，索引为日期
    window_size: 滑动窗口大小
    test_size: 测试集比例
    hidden_size: GRU隐藏层大小
    batch_size: 批次大小
    num_epochs: 训练轮数
    patience: 早停耐心值
    n_splits: 交叉验证折数
    device: 运行设备

    返回:
    模型、日期预测收益率字典
    """
    # 准备时间序列数据
    X, y, dates = prepare_time_series_data(factor_df, y_ret, window_size)
    print(f"样本总数: {len(dates)}")

    # 划分训练测试集
    X_train_val, X_test, y_train_val, y_test, dates_train_val, dates_test = train_test_split(
        X, y, dates, test_size=test_size, shuffle=False  # 时间序列不打乱
    )

    # 使用TimeSeriesSplit进行时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # 存储每个折的最佳模型参数
    best_models = []
    best_val_losses = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val), 1):
        print(f"\n----- Fold {fold}/{n_splits} -----")

        X_train = X_train_val[train_idx]
        y_train = y_train_val[train_idx]
        X_val = X_train_val[val_idx]
        y_val = y_train_val[val_idx]

        print(f"训练集: {len(X_train)} 样本, 验证集: {len(X_val)} 样本")

        # 准备数据集
        train_ds = FactorDataset(X_train, y_train, device=device)
        val_ds = FactorDataset(X_val, y_val, device=device)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # 初始化模型
        model = GRURegressor(input_size=X.shape[2], hidden_size=hidden_size).to(device)  # 特征数量

        # 训练模型
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_loss = float("inf")
        no_imp = 0
        best_model_path = f"best_gru_model_fold{fold}.pth"

        for epoch in range(1, num_epochs + 1):
            # 训练
            model.train()
            total_train_loss = 0.0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            # 验证
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    pred = model(x_batch)
                    loss = criterion(pred, y_batch)
                    total_val_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)

            if epoch % 10 == 0:  # 每10轮打印一次
                print(f"Epoch {epoch}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # 早停
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                no_imp = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                no_imp += 1
                if no_imp >= patience:
                    print(f"早停触发 (patience={patience})")
                    break

        # 加载最佳模型
        model.load_state_dict(torch.load(best_model_path))
        best_models.append(model.state_dict())
        best_val_losses.append(best_loss)

    # 选择验证集表现最好的模型
    best_fold = np.argmin(best_val_losses)
    print(f"\n选择Fold {best_fold+1}的模型作为最终模型，验证损失: {best_val_losses[best_fold]:.4f}")

    # 使用最佳模型
    final_model = GRURegressor(input_size=X.shape[2], hidden_size=hidden_size).to(device)
    final_model.load_state_dict(best_models[best_fold])

    # 在测试集上评估
    test_ds = FactorDataset(X_test, y_test, device=device)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    final_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            pred = final_model(x_batch)
            loss = criterion(pred, y_batch)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"测试集损失: {avg_test_loss:.4f}")

    # 获取所有数据的预测值
    final_model.eval()
    all_ds = FactorDataset(X, y, device=device)
    all_loader = DataLoader(all_ds, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for x_batch, _ in all_loader:
            pred = final_model(x_batch)
            all_preds.append(pred.cpu().numpy())

    all_preds = np.concatenate(all_preds)

    # 创建日期-预测收益率字典
    date_pred_dict = {date: pred for date, pred in zip(dates, all_preds)}

    return final_model, date_pred_dict


# 按日期分组并计算每个日期截面的分位数分组
def calculate_quantile_groups(df, num_quantiles):
    """
    计算每个日期截面的分位数分组
    """
    quantile_groups = df.apply(lambda row: pd.qcut(row.rank(method="first"), num_quantiles, labels=False) + 1, axis=1)
    return quantile_groups
