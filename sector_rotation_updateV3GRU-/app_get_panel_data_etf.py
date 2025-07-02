import pandas as pd
from WindPy import w
import config

data_list = []
for industry, fund in config.dic_Industry2Etf.items():
    ind_code = industry  # 行业编码
    of_code = "{}.OF".format(fund[1])  # etf基金编码

    # 提取行业一致预期，提取etf行情数据
    error, ind = w.wsd(
        ind_code,
        "west_avgroe_FY1,west_eps_FY1",
        "2020-01-01",
        "2025-03-31",
        "unit=1;Days=Alldays;Fill=Previous",
        usedf=True,
    )

    error, of = w.wsd(
        of_code,
        "open,high,low,close,volume,amt,pct_chg,turn",
        "2020-01-01",
        "2025-03-31",
        "unit=1;Days=Alldays;Fill=Previous",
        usedf=True,
    )
    # 合并一致预期数据和行情数据
    _ = pd.merge(left=ind, right=of, left_index=True, right_index=True)

    # 重置索引并设置列名
    _.reset_index(names="Date", inplace=True)
    # 添加factor_name
    _["fund_code"] = of_code
    _["industry_code"] = ind_code
    # 存入list
    data_list.append(_)

# 合并为Panel数据并处理数据格式
panel_data = pd.concat(data_list, ignore_index=True)
panel_data["Date"] = pd.to_datetime(panel_data["Date"])  # 将Date转换为datetime格式
panel_data["fund_code"] = panel_data["fund_code"].astype("category")
panel_data["industry_code"] = panel_data["industry_code"].astype("category")  # 将分类code转化为category类

# 将数据存储为HDF5格式,这里的key是存储在HDF5文件中的数据集名称
panel_data.to_hdf("./db/panel_data.h5", key="etf", format="table")
