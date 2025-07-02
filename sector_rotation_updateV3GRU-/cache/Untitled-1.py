# %%
import pandas as pd
import numpy as np
import talib as ta
import pickle
import warnings

warnings.filterwarnings("ignore")
import config
from src.src_get_features import (
    calculate_not_change,
    calculate_pct_change,
    calculate_day2week_ewm,
    get_bbands_idc,
    get_bbi_idc,
    get_cci_idc,
    get_cmo_idc,
    get_kdj_idc,
    get_roc_idc,
    get_sma_idc,
    get_vol_mom_idc,
)
from src.src_get_signals import (
    cross_rule,
    mom_rule,
    positional_rule_bands,
    positional_rule_reference_line,
    positional_rule_value,
    quantile_rule,
    volume_price_momentum_rule,
)

# %%
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, TimeSeriesSplit, ParameterGrid
from sklearn.metrics import classification_report

# %%
# 特征计算表
features_calculation_functions = {
    "not_change": calculate_not_change,
    "pct_change": calculate_pct_change,
    "day2week_ewm": calculate_day2week_ewm,
    "get_sma_idc": get_sma_idc,
    "get_bbi_idc": get_bbi_idc,
    "get_cci_idc": get_cci_idc,
    "get_kdj_idc": get_kdj_idc,
    "get_cmo_idc": get_cmo_idc,
    "get_roc_idc": get_roc_idc,
    "get_bbands_iccfe": get_bbands_idc,
    "get_vol_mom_idc": get_vol_mom_idc,
}
# 特征计算表_再处理
features_calculation_post_functions = {
    "get_bbands_idc": get_bbands_idc,
    "get_sma_idc": get_sma_idc,
}
# 信号计算表
signal_calculation_functions = {
    "mom_rule": mom_rule,
    "positional_rule_reference_line": positional_rule_reference_line,
    "positional_rule_bands": positional_rule_bands,
    "quantile_rule": quantile_rule,
    "cross_rule": cross_rule,
    "positional_rule_value": positional_rule_value,
    "volume_price_momentum_rule": volume_price_momentum_rule,
}

# %% [markdown]
# # meta-data

# %%
df_881001 = pd.read_excel("./db/881001_aug.xlsx").drop("pct_chg", axis=1).set_index("Date")
df_881001.columns = df_881001.columns.str.upper()  # 大写列名
df_881001.to_hdf("./db/benchmark_ts_data.h5", key="wind")

# %%
data_list = []
for key, item in config.dic_labels.items():
    _ = pd.read_excel(f"./db/{key}.xlsx").set_index("Unnamed: 0").dropna()  # 读取excel中的数据
    _.reset_index(names="Date", inplace=True)  # 重置索引，将原来的索引列转换为普通列
    if key in ["ic_cfe"]:  # 数据源的问题，需要专门改一下名字
        _.rename(columns={"ANAL_BASISANNUALYIELD": "CLOSE"}, inplace=True)
    _["label"] = key  # 添加标签列
    data_list.append(_)
panel_data = pd.concat(data_list, ignore_index=True)  # 合并所有数据

# %%
panel_data["Date"] = pd.to_datetime(panel_data["Date"])  # 确保日期列是datetime类型
panel_data["label"] = panel_data["label"].astype("category")  # 将标签列转换为分类类型
panel_data

# %%
panel_data.to_hdf("./db/panel_data.h5", key="meta_data", format="table")

# %% [markdown]
# # features

# %%
panel_data = pd.read_hdf("E:\ZYXT\\files\script\\timingFactors_update\db\panel_data.h5", key="meta_data")
panel_data

# %%
df_881001 = pd.read_hdf("./db/benchmark_ts_data.h5", key="wind")
df_881001

# %%
data_list = []
for label in config.dic_labels2features.keys():
    print(label)
    function = features_calculation_functions[config.dic_labels2features[label][0]]  # 获取函数
    kwargs = config.dic_labels2features[label][1]  # 获取参数

    if label in ["tby_diff", "tby_winda_spread"]:
        label1 = config.dic_labels2features[label][1]["label1"]
        label2 = config.dic_labels2features[label][1]["label2"]
        df_temp_1 = panel_data[panel_data["label"] == label1].set_index("Date")["CLOSE"]
        df_temp_2 = panel_data[panel_data["label"] == label2].set_index("Date")["CLOSE"]
        _ = (df_temp_1 - df_temp_2).reset_index()  # 特殊处理，计算两个标签的差值
        # _['label'] = label  # 添加标签列
        __ = function(_, **kwargs)

    elif label in [
        "cpi",
        "pmi_neo",
        "pmi_sv",
        "m&ltl_yoy",
        "pmi_p",
        "r007",
        "neqf",
        "mrb",
        "usdcnh_spot",
        "s&p500_vix",
        "pb",
        "pe_ttm",
        "ic_cfe",
    ]:
        _ = panel_data[panel_data["label"] == label]
        __ = function(_, **kwargs)

    elif label in ["sma", "bbi", "cci", "kdj", "cmo", "roc", "volume_price_mom"]:
        __ = function(df=df_881001, **kwargs)

    if label in config.dic_label2feature_post.keys():  # 如果标签在字典中有对应的后处理函数
        post_function = features_calculation_post_functions[config.dic_label2feature_post[label][0]]  # 获取后处理函数
        post_kwargs = config.dic_label2feature_post[label][1]  # 获取参数
        __ = post_function(__.set_index("Date"), **post_kwargs)

    __["label"] = label  # 添加标签列

    data_list.append(__)
panel_data_features = pd.concat(data_list, ignore_index=True)  # 合并所有数据

# %%
panel_data_features.to_hdf("./db/panel_data.h5", key="features", format="table")

# %% [markdown]
# # Signals
#

# %%
panel_data_features = pd.read_hdf("./db/panel_data.h5", key="features")
panel_data_features

# %%
df_temp = panel_data_features[panel_data_features["label"] == "m&ltl_yoy"]
df_temp

# %%
data_list = []
for key in config.dic_feature2signals.keys():
    print(key)
    function = signal_calculation_functions[config.dic_feature2signals[key][0]]  # 获取函数
    kwargs = config.dic_feature2signals[key][1]  # 获取参数
    # 从panel_data中读取特定标签的数据
    df_temp = panel_data_features[panel_data_features["label"] == key].set_index("Date")
    # 调用函数计算信号
    __ = function(df=df_temp, **kwargs).to_frame().reset_index()
    __["label"] = key  # 添加标签列
    __["freq"] = config.dic_features[key][1]
    data_list.append(__)  # 将结果添加到列表中

panel_data_signals = pd.concat(data_list, ignore_index=True)  # 合并所有信号数据

# %%
panel_data_signals.to_hdf("./db/panel_data.h5", key="signals", format="table")
panel_data_signals

# %%
panel_data_signals[panel_data_signals["label"] == "r007"]

# %% [markdown]
# # featurevalue

# %%
panel_data_features = pd.read_hdf("./db/panel_data.h5", key="features")
panel_data_features

# %%
# def keep_value(df,feature_name,Inverse=False):
#     if Inverse:  # 反转信号
#         df['featurevalue'] = - df[feature_name]
#         return df[['Date','label','featurevalue']].reset_index(drop=True)
#     df['featurevalue'] = df[feature_name]
#     return df[['Date','label','featurevalue']].reset_index(drop=True)

# def get_j(df):
#     df['featurevalue'] = 3*df['slowk'] - 2*df['slowd']
#     return  df[['Date','label','featurevalue']].reset_index(drop=True)

# def get_momentum_strength(df):
#     df['featurevalue'] = df['CLOSE'].pct_change()*df['VOLUME'].pct_change()*1e4
#     return  df[['Date','label','featurevalue']].reset_index(drop=True)


def get_featurevalue(df, mode, price_col, vol_col, inverse, feature_col):
    df = df.copy()  # 避免直接操作

    # 原值提取
    if mode == "features":
        series = df[feature_col]
    # KDJ
    elif mode == "kdj":
        series = 3 * df["slowk"] - 2 * df["slowd"]
    # volume_price_mom
    elif mode == "volume_price_mom":
        series = df[price_col].pct_change() * df[vol_col].pct_change() * 1e4

    if inverse:  # 反转因子
        series = -series

    # 组装output
    out = pd.DataFrame({"Date": df["Date"], "label": df["label"], "featurevalue": series})
    return out.reset_index(drop=True)


# %%
data_list = []
for key, params in config.dic_feature2value.items():
    print(key)
    # 拆参数
    df = panel_data_features[panel_data_features["label"] == key]
    mode = params["mode"]
    inverse = params.get("inverse", False)
    price_c = params.get("price_col")
    vol_c = params.get("vol_col")
    feature_col = params.get("feature_col", None)
    # 调用统一接口
    df_feat = get_featurevalue(
        df, feature_col=feature_col, mode=mode, inverse=inverse, price_col=price_c, vol_col=vol_c
    )
    data_list.append(df_feat)
panel_data_featuresvalue = pd.concat(data_list, ignore_index=True)

# %%
result[result["label"] == "m&ltl_yoy"]

# %%


# %% [markdown]
# # Composite

# %%
panel_data_signals = pd.read_hdf("./db/panel_data.h5", key="signals")
df_881001 = pd.read_hdf("./db/benchmark_ts_data.h5", key="wind")
panel_data_signals


# %%
class PanelToSeq:
    def __init__(
        self, df: pd.DataFrame, labels: list = None, seq_len: dict = {"D": 40, "W": 20, "M": 12}, fill_method="ffill"
    ):
        self.seq_len = seq_len  # 每种freq对应的窗口长度
        self.labels = labels or df["label"].unique().tolist()  # 如果没有指定标签，则使用DataFrame中的所有标签
        # 准备三张透视表
        self.pivots = {}
        for freq in ["D", "W", "M"]:
            piv = (
                df[df["freq"] == freq]
                .pivot(index="Date", columns="label", values="signal")
                .sort_index()
                .fillna(method=fill_method)
                .fillna(0)  # 前填充,如果开头还有NaN.,用0再填充一次
            )
            self.pivots[freq] = piv  # shape = (n_f,3)

    def make_samples(self, y_series: pd.Series, pre_date: int = 10, ref_dates: pd.DatetimeIndex = None):
        """
        y_series:index=日频交易日，value=未来10天收益率；
        ref_dates: 运行日索引
        """
        daily_idx = self.pivots["D"].index
        ld = self.seq_len["D"]  # 日频窗口长度

        # 1.确定ref_dates
        latest_label_date = y_series.index[-pre_date]  # 最后一个标签数据的日期
        # 候选运行日：ld to latest_label_data：
        candidates = daily_idx[ld - 1 :]
        if ref_dates is None:
            ref_dates = candidates[
                candidates <= latest_label_date
            ]  # 运行日索引，默认是候选运行日中小于等于最后一个标签数据的日期
        # 2.遍历日频ref_dates,其他频率日期对齐
        xd, xw, xm, yp, date_list = [], [], [], [], []  # 用于存储样本数据
        for date in ref_dates:
            window = {}  # 用于存储每个频率的窗口数据
            for f, piv in self.pivots.items():
                l = self.seq_len[f]  # 获取当前频率的窗口长度
                # 找到<=d最近的idx
                idx = piv.index.get_indexer([date], method="pad")[0]
                if idx < l - 1:
                    continue  # 如果索引小于窗口长度-1，说明没有足够的数据来构建样本，跳过这个日期
                arr = piv.values[idx - l + 1 : idx + 1]  # arr.shape = (l,f)
                window[f] = arr  # 存储当前频率的窗口数据

            # 取y：第date+10天的收益率
            yi = y_series.index.get_indexer([date], method="pad")[0]
            if yi + pre_date >= len(y_series):
                continue
            y = y_series.values[yi + pre_date]  # 未来10天收益率

            # 将窗口数据添加到对应的列表中，当都可以找到对应idx时再append
            if all(k in window for k in ["D", "W", "M"]):
                xd.append(window["D"])
                xw.append(window["W"])
                xm.append(window["M"])
                yp.append(y)
                date_list.append(date)  # 记下这个样本对应的日期

        # 将列表转换为numpy数组
        xd = np.array(xd)  # shape = (nd,ld,fd)
        xw = np.array(xw)  # shape = (nd,lw,fw)
        xm = np.array(xm)  # shape = (nd,lm,fm)
        yp = np.array(yp, dtype=np.float32)  # shape = (nd,)
        date_arr = np.array(date_list)  # shap = (nd,)

        return xd, xw, xm, yp, date_arr


"""
def make_samples_onefreq(
        self,
        y_series: pd.Series,
        pre_days: int = 10,
        ref_dates: pd.DatetimeIndex = None,
        freq: str = 'D'
):
    # 1. 拿到对应频率的 pivot 和窗口长度
    piv = self.pivots[freq]           # e.g. self.pivots['D']
    dates = piv.index                 # e.g. 日频交易日索引
    L = self.seq_len[freq]            # 窗口长度

    # 2. 计算最后一个可用 label 日期
    last_offset = pre_days + 1
    latest_label_date = y_series.index[-last_offset]

    # 3. 候选 ref_dates: 从 dates[L-1] 开始，到 latest_label_date
    candidates = dates[L-1:]
    if ref_dates is None:
        ref_dates = candidates[candidates <= latest_label_date]
    else:
        ref_dates = pd.to_datetime(ref_dates)
        ref_dates = ref_dates[
            (ref_dates >= candidates[0]) &
            (ref_dates <= latest_label_date)
        ]

    # 4. 循环构造样本
    x_list, y_list = [], []
    for d in ref_dates:
        # ——构造 X——
        pos = dates.get_indexer([d], method='pad')[0]
        if pos < L - 1:
            continue
        x_win = piv.values[pos - L + 1 : pos + 1]  # shape=(L, F)

        # ——构造 y——
        yi = y_series.index.get_indexer([d], method='pad')[0]
        if yi + pre_days >= len(y_series):
            continue
        y_val = y_series.values[yi + pre_days]

        x_list.append(x_win)
        y_list.append(y_val)

    # 5. 转 np.ndarray 并返回
    x = np.stack(x_list, axis=0)               # (N, L, F)
    y = np.array(y_list, dtype=np.float32)     # (N,)

    return x, y
    """


class MultiFreqDataset(Dataset):
    def __init__(self, xd, xw, xm, yp, device="cpu"):
        assert len(xd) == len(xw) == len(xm) == len(yp), "All input arrays must have the same length."
        self.xd = torch.tensor(xd, dtype=torch.float32).to(device)
        self.xw = torch.tensor(xw, dtype=torch.float32).to(device)
        self.xm = torch.tensor(xm, dtype=torch.float32).to(device)
        self.yp = torch.tensor(yp, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.yp)

    def __getitem__(self, idx):
        return (self.xd[idx], self.xw[idx], self.xm[idx], self.yp[idx])


class MultiFreqGRUClassifier(torch.nn.Module):
    def __init__(self, in_feats_d, in_feats_w, in_feats_m, hidden_size, num_classes):  # 日维度  # 周维度  # 月维度
        super().__init__()
        # 三条独立的 GRU
        self.gru_d = torch.nn.GRU(input_size=in_feats_d, hidden_size=hidden_size, batch_first=True)
        self.gru_w = torch.nn.GRU(input_size=in_feats_w, hidden_size=hidden_size, batch_first=True)
        self.gru_m = torch.nn.GRU(input_size=in_feats_m, hidden_size=hidden_size, batch_first=True)

        # 三个各自独立的全连接
        self.fc_d = torch.nn.Linear(hidden_size, num_classes)  # in_featrue=hidden_size,out_feature=num_classes
        self.fc_w = torch.nn.Linear(hidden_size, num_classes)
        self.fc_m = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, xd, xw, xm):
        # xd, xw, xm: [B, seq_len, in_feats_*]
        _, hd = self.gru_d(xd)  # hd: [1, B, H]
        _, hw = self.gru_w(xw)
        _, hm = self.gru_m(xm)

        # 去掉第 0 维，得到 [B, H]
        hd = hd.squeeze(0)
        hw = hw.squeeze(0)
        hm = hm.squeeze(0)

        # 各自过各自的全连接，得到 [B, num_classes]
        logits_d = self.fc_d(hd)
        logits_w = self.fc_w(hw)
        logits_m = self.fc_m(hm)

        # 等权合成
        logits = (logits_d + logits_w + logits_m) / 3.0

        return logits


# %%
# 基本参数设置
n_splits = 5
num_epochs = 100
patience = 10
weight_decay = 1e-4

# 周期参数
freq_d = 30
seq_len = {"d": {"D": 10, "W": 2, "M": 1}, "w": {"D": 20, "W": 5, "M": 2}, "m": {"D": 40, "W": 10, "M": 3}}
ops_bench = 0.001
neg_bench = 0

# 对超参数grid_search，要搜索的超参数空间
param_grid = {"hidden_size": [10, 20, 30], "batch_size": [256], "lr": [5e-3, 1e-3]}

# %%
# 1.读信号表
panel_signal = panel_data_signals
# 2.构建y—series
y_ret = df_881001["CLOSE"].pct_change(freq_d)
y_ret_ = y_ret.dropna()  # 删除NaN值
# 3.用PanelToSeq构建样本
converter = PanelToSeq(panel_signal, seq_len=seq_len["w"])
xd, xw, xm, yp, dates = converter.make_samples(y_series=y_ret_, pre_date=freq_d)
print(f"总样本数 N = {xd.shape[0]}")
print("xd.shape=", xd.shape, " xw.shape=", xw.shape, " xm.shape=", xm.shape, " yp.shape=", yp.shape)
# 4先把y转化为三分类
y_cls = np.zeros_like(yp)  # 初始化分类标签
y_cls[yp > ops_bench] = 1  # 正收益为1
y_cls[yp < neg_bench] = -1  # 负收益为-1
y_cls = y_cls + 1  # 将标签转换为0,1,2形式
# 4.保留测试集
xd_trval, xd_test, xw_trval, xw_test, xm_trval, xm_test, y_trval, y_test, date_travel, date_test = train_test_split(
    xd, xw, xm, y_cls, dates, test_size=0.2, shuffle=False, random_state=42  # 对时间序列，请设置 shuffle=False 保持顺序
)

# %%
y_ret_.describe(), np.sum(y_cls == 2), np.sum(y_cls == 1), np.sum(y_cls == 0)

# %%
# gridsearch + timeseriessplit
result = []

for params in ParameterGrid(param_grid):
    hidden_size = params["hidden_size"]
    batch_size = params["batch_size"]
    lr = params["lr"]

    # 5.使用TimeSeriesSplit切分测试集/验证集
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_best_losses = []  # 存储每个fold下best loss

    for fold, (train_idx, val_idx) in enumerate(tscv.split(xd_trval), 1):
        print("\n" + "-" * 20 + f"Fold{fold}/{n_splits}" + "-" * 20)
        xd_train, xw_train, xm_train, y_train, dates_train = (
            xd_trval[train_idx],
            xw_trval[train_idx],
            xm_trval[train_idx],
            y_trval[train_idx],
            dates[train_idx],
        )  # 划分训练集
        xd_val, xw_val, xm_val, y_val, dates_val = (
            xd[val_idx],
            xw[val_idx],
            xm[val_idx],
            y_cls[val_idx],
            dates[val_idx],
        )  # 划分验证集
        print(f"训练集：{len(y_train)}样本，验证集：{len(y_val)}样本")
        # 6.构造Dataset和DataLoader
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_ds = MultiFreqDataset(xd_train, xw_train, xm_train, y_train, device=device)  # 构造训练集数据
        val_ds = MultiFreqDataset(xd_val, xw_val, xm_val, y_val, device=device)  # 构造测试集数据
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        # 7.初始化模型
        model = MultiFreqGRUClassifier(
            in_feats_d=xd.shape[-1],
            in_feats_w=xw.shape[-1],
            in_feats_m=xm.shape[-1],
            hidden_size=hidden_size,
            num_classes=3,
        )  # 模型初始化
        criterion = torch.nn.CrossEntropyLoss()  # 使用交叉熵作损失函数
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )  # 使用Adam optimizer做梯度下降
        # 8.训练循环+es
        best_loss = float("inf")
        no_imp = 0
        best_path = "best_model.path"

        print("\n开始多轮训练+测试机early stopping....")
        for epoch in range(1, num_epochs + 1):
            # --训练--
            model.train()
            total_tr_loss = 0.0
            for xd_b, xw_b, xm_b, y_b in train_loader:
                logits = model(xd_b, xw_b, xm_b)
                loss = criterion(logits, y_b.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_tr_loss += loss.item()
            avg_tr = total_tr_loss / len(train_loader)
            # --测试--
            model.eval()
            total_ev_loss = 0.0
            for xd_b, xw_b, xm_b, y_b in val_loader:
                logits = model(xd_b, xw_b, xm_b)
                loss_ev = criterion(logits, y_b.long())
                total_ev_loss += loss_ev.item()
            avg_ev = total_ev_loss / len(val_loader)
            print(f"Epoch{epoch}/{num_epochs}|Train Loss:{avg_tr:.4f}|Test Loss:{avg_ev:.4f}")

            # Early stopping 判断
            if avg_ev < best_loss:
                best_loss = avg_ev
                no_imp = 0
                print("Loss降低，更新best_loss")
            else:
                no_imp += 1
                if no_imp >= patience:
                    print("EarlyStopping触发：{}轮".format(patience))
                    break
        fold_best_losses.append(best_loss)

    # 计算该超参下的平均 fold 验证 Loss
    cv_mean_loss = np.mean(fold_best_losses)
    result.append({**params, "cv_mean_loss": cv_mean_loss})
    print(f">>> Params={params} | mean_val_loss={cv_mean_loss:.4f}")

# 找到最优超参
best = min(result, key=lambda x: x["cv_mean_loss"])
print("\nGridSearch+TimeSeriesSplit 最优参数：", best)

# %%
with open("best_params.pkl", "wb") as f:
    pickle.dump(best, f)

# %%
final_num_epochs = 100
final_no_imp = 0
final_best_loss = float("inf")

# %%
# 9.使用最佳超参数训练模型
best_params = best
hidden_size = best_params["hidden_size"]
batch_size = best_params["batch_size"]
lr = best_params["lr"]
# 重新声明训练集
xd_train_final, xw_train_final, xm_train_final, y_train_final = xd_trval, xw_trval, xm_trval, y_trval
# 构造Dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_ds_final = MultiFreqDataset(xd_train_final, xw_train_final, xm_train_final, y_train_final, device=device)
train_loader_final = DataLoader(train_ds_final, batch_size=batch_size, shuffle=False)
# 建模
model = MultiFreqGRUClassifier(
    in_feats_d=xd.shape[-1], in_feats_w=xw.shape[-1], in_feats_m=xm.shape[-1], hidden_size=hidden_size, num_classes=3
)  # 模型初始化
criterion = torch.nn.CrossEntropyLoss()  # 使用交叉熵作损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # 使用Adam optimizer做梯度下降
for epoch in range(1, final_num_epochs + 1):
    model.train()
    total_loss = 0.0
    for xd_b, xw_b, xm_b, y_b in train_loader_final:
        logits = model(xd_b, xw_b, xm_b)
        optimizer.zero_grad()
        loss = criterion(logits, y_b.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_final = total_loss / len(train_loader_final)  # len(train_loader_final) =样本数/batch_size
    print(f"Final Epoch {epoch}/{final_num_epochs} | Train Loss={avg_train_final:.4f}")

    # Early stopping 判断
    if avg_train_final < final_best_loss:
        final_best_loss = avg_train_final
        final_no_imp = 0
        print("Loss降低，更新best_loss")
    else:
        final_no_imp += 1
        if final_no_imp >= patience:
            print("EarlyStopping触发：{}轮".format(patience))
            break
# 保存最终模型
torch.save(model.state_dict(), "best_model.pth")


# %%
# ——加载保存的最优权重 ——
checkpoint = torch.load("best_model.pth", map_location=device)
model.load_state_dict(checkpoint)
# 10.在测试集上评估
test_ds = MultiFreqDataset(xd_test, xw_test, xm_test, y_test, device=device)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
model.eval()
all_p, all_t, all_probs = [], [], []
with torch.no_grad():
    for xd_b, xw_b, xm_b, y_b in test_loader:
        logits = model(xd_b, xw_b, xm_b)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_p.append(preds)
        all_t.extend(y_b.cpu().numpy())

        # 存储softmax序列
        probs = torch.softmax(logits, dim=1)  # shape=[batch,3]
        probs_np = probs.cpu().numpy()  # tensor --> ndarray 供list append
        all_probs.append(probs_np)  # list of [b,3]


print("\n最终再测试机上的分类：")
all_p_flat = np.concatenate(all_p, axis=0)  # flatten
print(classification_report(all_t, all_p_flat, target_names=["Neg", "Zero", "Pos"]))

# %%
# softmax格式
all_probs_flat = np.concatenate(all_probs, axis=0)
prob = all_probs_flat[:, 2]  # prob=2的概率
all_probs_flat

# %%
# TODO 业绩评估模块

# %%
# 合并仓位、日期、return
df_result = pd.DataFrame(data={"p": prob, "y_ret": y_ret_.reindex(date_test).values}, index=date_test)
df_result_ = pd.merge(left=df_result, right=df_bond, left_index=True, right_index=True)  # 合并债市price


# %%
df_result_

# %%
df_result_.p.describe()

# %%
import pandas as pd
import matplotlib.pyplot as plt

# —— 0. Matplotlib 中文字体设置（可选）
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# —— 1. 拷贝 & 确保索引为 DatetimeIndex
df = df_result.copy()
df.index = pd.to_datetime(df.index)

df = df.loc["2022-01-01":].copy()

# —— 2. 持仓信号 & 策略日收益
#    用 shift(1) 表示“昨日信号，今日执行”
# df['p_y']   = df['p'].shift(1).fillna(0)           # 对 y_future_1d 的仓位

df["p_y"] = (df["p"].shift(1) > 0.7).astype(int)  # 1 表示满仓权益，0 表示满仓债券
df["p_pct"] = 1 - df["p_y"]  # 对 PCT_CHG 的仓位

# df['strat_ret'] = df['p_y'] * df['y_future_1d'] + df['p_pct'] * df['PCT_CHG']/100
df["strat_ret"] = df["p_y"] * df["y_ret"]

# df['strat_ret'] = df['PCT_CHG']


# —— 3. 累计净值
df["cum_strat"] = (1 + df["strat_ret"]).cumprod()
df["cum_bench"] = (1 + df["y_ret"]).cumprod()  # 基准：满仓 y_future_1d

# —— 4. 年化收益（可选）
n = len(df)
ann_strat = df["cum_strat"].iloc[-1] ** (252 / n) - 1
ann_bench = df["cum_bench"].iloc[-1] ** (252 / n) - 1
print(f"策略年化收益：{ann_strat:.2%}")
print(f"基准年化收益：{ann_bench:.2%}")

# —— 5. 可视化
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["cum_strat"], label="策略累计净值", linewidth=2)
plt.plot(df.index, df["cum_bench"], label="基准累计净值（满仓 y_future_1d）", linestyle="--", alpha=0.8)
plt.title("组合策略 vs 基准 累计净值对比")
plt.xlabel("日期")
plt.ylabel("累计净值")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
df.to_excel("benchmark-0.7.xlsx")

# %% [markdown]
# # get_price_series

# %%
df_881001 = pd.read_hdf("./db/benchmark_ts_data.h5", key="wind")
df_881001

# %%
df_bond = pd.read_excel("./db/bond_ts.xlsx").set_index("Unnamed: 0")
df_bond

# %%
y_10d = df_881001["CLOSE"].pct_change(10)
y_1d = df_881001["CLOSE"].pct_change(1)

# %% [markdown]
# # 因子融合-线性

# %%


# %%


# %%
