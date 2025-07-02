import math
import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.src_get_composit_indicators import calculate_quantile_groups, fill_to_minimum_length, select_with_threshold


def calculate_etf_selection(composite_factor, dic_indicator, group1=6, group2=4):

    # 复合因子值相加
    first_key = next(iter(dic_indicator))  # 获取字典第一个键名
    df_composite_factor = pd.DataFrame(
        data=0, index=dic_indicator[first_key].index, columns=dic_indicator[first_key].columns
    )  # 空df存储复合因子值
    for key, value in dic_indicator.items():
        df_composite_factor += dic_indicator[key]

    # 获取排名
    df_quantil_groups = calculate_quantile_groups(df_composite_factor.dropna(), group1)  # 对筛选因子值排序
    df_threshquantil_groups = calculate_quantile_groups(composite_factor, group2)  # 对门槛因子值排序

    # 根据阈值筛选
    df_selected_industry_no_fill = df_quantil_groups.apply(
        lambda row: select_with_threshold(row, df_threshquantil_groups, group1, 1), axis=1
    )
    df_selected_industry_no_fill = pd.DataFrame(df_selected_industry_no_fill)

    # 补齐操作
    df_selected_industry_with_filling = df_selected_industry_no_fill.apply(
        lambda row: fill_to_minimum_length(row, df_composite_factor), axis=1
    )
    return df_selected_industry_with_filling


# ETF轮动策略
class ETFRotationStrategy(bt.Strategy):
    params = (("selected_etfs", None), ("frequency", "monthly"))  # 选中的ETF列表

    def __init__(self):
        # 存储当前持仓的ETF
        self.current_etfs = []
        # 每月执行一次轮动
        self.last_rotation_date = None
        # 从参数获得threshold和filter数据
        self.selected_etfs = self.params.selected_etfs  # 选中的ETF列表

    def should_rotated(self, current_data):
        if self.params.frequency == "monthly":
            # 判断当前日期是否和上次next不同月份，不同月再执行next
            return self.last_rotation_date is None or current_data.month != self.last_rotation_date.month
        elif self.params.frequency == "weekly":
            # 判断当前日期是否和上次next不同周，不同周再执行next
            return (
                self.last_rotation_date is None
                or current_data.isocalendar()[1] != self.last_rotation_date.isocalendar()[1]
            )
        return False

    # notify
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            print(
                f"日期: {self.datetime.date()}, ETF: {order.data._name}, 订单已提交/接受, 状态: {order.getstatusname()}"
            )
            return

        if order.status == order.Completed:
            print(f"日期: {self.datetime.date()}, ETF: {order.data._name}, 订单已完成")
            print(f"-执行价格: {order.executed.price}, 执行数量: {order.executed.size}")
            print(f"-佣金: {order.executed.comm}")

            # 打印执行后的持仓和资金
            pos = self.getposition(order.data).size
            cash = self.broker.getcash()
            value = self.broker.getvalue()
            print(f"-可用资金: {cash}, 总资产: {value}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"日期: {self.datetime.date()}, ETF: {order.data._name}, 订单未执行, 状态: {order.getstatusname()}")

    def next(self):

        print("next")

        # 获取当前日期
        current_date = self.data.datetime.date(0)

        # # 判断当前日期是否和上次next不同月份，不同月再执行next
        # if self.should_rotated(current_date):
        #     # 更新策略执行日期
        #     self.last_rotation_date = current_date

        try:
            selected_etfs = self.selected_etfs.loc[current_date.strftime("%Y-%m-%d")]
        except KeyError:
            print(f"{current_date}: 无法获取ETF选择结果,跳过轮动")
            return

        # 根据 ETF 代码找到对应的 backtrader 数据对象
        selected_etfs_bt = [data for data in self.datas if data._name in selected_etfs]
        # 如果没有选中任何ETF，则不进行操作
        if not selected_etfs:
            print(f"{current_date}: 未选中任何ETF")
            return
        # log选中的ETF
        print(f"{current_date}: 选中的ETF为 {[data for data in selected_etfs]}")

        # 平仓不在新推荐列表中的 ETF
        for etf in self.current_etfs:
            if etf not in selected_etfs_bt:
                print(f"{current_date}: 平仓 ETF {etf._name}")
                self.close(etf)

        # 计算每个新推荐 ETF 的目标持仓比例
        position_percentage_per_etf = 1.0 / len(selected_etfs_bt)  # 每个 ETF 的目标仓位为 100% 除以 ETF 数量

        # 调整持仓
        # for etf in selected_etfs_bt:

        #     # 计算实际可用资金
        #     available_cash = self.broker.getvalue() * 0.98  # 保留5%的资金作为缓冲

        #     # 计算目标持仓量
        #     etf_cash_needed = position_percentage_per_etf * available_cash
        #     price = max(etf.close[0], etf.open[1])  # 使用最大值作为价格，防止margin不足
        #     target_size = math.floor(etf_cash_needed / price)

        #     # 当前持仓量
        #     current_size = self.broker.getposition(etf).size
        #     # 计算差值
        #     size_difference = target_size - current_size

        #     if size_difference > 0:
        #         order = self.buy(data=etf, size=size_difference)
        #         print("*** /n 执行买入操作,Size={}".format(size_difference))

        #     elif size_difference < 0:
        #         order = self.sell(data=etf, size=-size_difference)
        #         print("*** /n 执行卖出操作,Size={}".format(-size_difference))

        #     # 打印交易后的持仓量
        #     new_position = (self.broker.getposition(etf).size * etf.close[0]) / self.broker.getvalue()  # 当前持仓量
        #     print(f"-ETF: {etf._name}, 当前仓位占比: {new_position}")

        for etf in selected_etfs_bt:

            order = self.order_target_percent(data=etf, target=position_percentage_per_etf * 0.95)

            # 打印交易后的持仓量
            new_position = (self.broker.getposition(etf).size * etf.close[0]) / self.broker.getvalue()  # 当前持仓量
            print(f"-ETF: {etf._name}, 当前仓位占比: {new_position}")

        # 更新当前持仓
        self.current_etfs = selected_etfs_bt
