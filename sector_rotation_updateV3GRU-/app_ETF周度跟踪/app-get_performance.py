import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 读取数据
df = pd.read_excel("./db/timeseries.xlsx", sheet_name="Sheet1", parse_dates=["日期"]).set_index("日期").sort_index()

ret = df[["ETF行业轮动策略收益", "万得全A收益"]].rename(
    columns={"ETF行业轮动策略收益": "strat", "万得全A收益": "bench"}
)  # 声明收益
ret["excess"] = ret["strat"] - ret["bench"]


# 计算累计收益、年化收益、年化波动、夏普、最大回撤
def calc_stats(ser, start, end):
    seg = ser.loc[start:end].dropna()
    cum = (1 + seg).prod() - 1  # 累计收益
    ann_ret = (1 + cum) ** (252 / len(seg)) - 1  # 年化收益
    ann_vol = seg.std() * np.sqrt(252)  # 年化波动
    sharpe = ann_ret / ann_vol  # 夏普
    # 最大回撤
    nav = (1 + seg).cumprod()  # 净值序列
    peak = nav.cummax()
    max_dd = ((nav - peak) / peak).min()
    return cum, ann_ret, ann_vol, sharpe, max_dd


# 获取不同区间日期,和通联保持一致
today = df.index.max()
start_of_year = datetime(today.year, 1, 1)

periods = {
    "近1月": today - pd.DateOffset(months=1) + pd.DateOffset(days=1),
    "近3月": today - pd.DateOffset(months=3) + pd.DateOffset(days=1),
    "近6月": today - pd.DateOffset(months=6) + pd.DateOffset(days=1),
    "今年以来": start_of_year,
    "近1年": today - pd.DateOffset(years=1) + pd.DateOffset(days=1),
}
# 计算periods中的累计收益
rows = []
for name, start in periods.items():
    strat_cum, *_ = calc_stats(ret["strat"], start, today)
    bench_cum, *_ = calc_stats(ret["bench"], start, today)
    rows.append({"periods": name, "stra_cum": strat_cum, "bench_cum": bench_cum, "excess": strat_cum - bench_cum})
df_top = pd.DataFrame(rows).set_index("periods")

# 计算今年以来累计&四项年化指标
rows_2 = []
for key in ["strat", "bench"]:
    cum, ann_ret, ann_vol, sharpe, max_dd = calc_stats(ret[key], start_of_year, today)
    rows_2.append(
        {"name": key, "2025": cum, "ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd}
    )
df_bottom = pd.DataFrame(rows_2).set_index("name")
df_bottom.loc["excess"] = df_bottom.loc["strat"] - df_bottom.loc["bench"]

print("-各区间累计&超额")
print(df_top.map(lambda x: f"{x:.2%}"))
print("-今年以来&年化")
print(df_bottom.map(lambda x: f"{x:.2%}"))
