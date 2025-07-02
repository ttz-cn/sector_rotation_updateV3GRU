rebalance_period = "B"  # 初始化调仓周期


dic_Industry2Etf = {
    "801080.SI": ["电子", "159995"],
    "801110.SI": ["家用电器", "159996"],
    "801880.SI": ["汽车", "516110"],
    "801050.SI": ["有色金属", "512400"],
    "801770.SI": ["通信", "515880"],
    "801790.SI": ["非银金融", "512880"],
    "801150.SI": ["医药生物", "512010"],
    "801120.SI": ["食品饮料", "512690"],
    "801780.SI": ["银行", "512800"],
    "801160.SI": ["公共事业", "159611"],
    "801890.SI": ["机械设备", "562500"],
    "801030.SI": ["基础化工", "159870"],
    "801210.SI": ["社会服务", "159766"],
    "801740.SI": ["国防军工", "512660"],
    "801960.SI": ["石油石化", "159930"],
    "801750.SI": ["计算机", "159998"],
    "801170.SI": ["交通运输", "516910"],
    "801720.SI": ["建筑装饰", "516970"],
    "801730.SI": ["电力设备", "515790"],
    "801760.SI": ["传媒", "159869"],
    "801010.SI": ["农林牧渔", "159865"],
    "801180.SI": ["房地产", "512200"],
    "801040.SI": ["钢铁", "515210"],
    "801710.SI": ["建筑材料", "159745"],
    "801950.SI": ["煤炭", "515220"],
}


selected_indicators = [
    "second_order_mom",
    "long_short_position",
    "close_volume_divergence_corr",
    "volume_HDL_diff_divergence",
    "ir",
    "turnover_rate",
    "ts_vol",
    "beta",
    "roe_fy1_st",
    "eps_fy1_st",
    "ema_diff",
    "volume_price_strength",
]

get_icir_rolling_window = {"M": [6, 6], "W": [24, 24], "B": [20, 20]}

list_tech_factor = [
    "second_order_mom",
    "long_short_position",
    "close_volume_divergence_corr",
    "volume_HDL_diff_divergence",
    "ir",
    "ema_diff",
    "volume_price_strength",
]
list_crd_factor = ["turnover_rate", "ts_vol", "beta"]
list_funda_factor = ["roe_fy1", "eps_fy1"]

# 组合优化参数 (根据图片内容新增)
optimization_params = {
    "weight_limit": {"high": 0.80, "low": 0.60},  # 风险偏好高时的权重上限  # 风险偏好低时的权重上限
    "min_weight": {"high": 0.10, "low": 0.30},  # 最小仓位为10%
}
