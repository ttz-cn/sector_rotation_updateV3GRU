# *_* coding : UTF-8 *_*
# Group  ：
# The Developer  ：  Boyi
# development Time  ：  2024/8/14  15:49
# Documents Name  :  src_boyi.PY
# Tools  :  PyCharm

# 自建包，供通用模块调用

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sqlalchemy import create_engine
import backtrader as bt
import pyfolio as pf
import empyrical as ep


class SignalStrategy(bt.Strategy):

    params = (("period", "monthly"),)
    """
    signal test:
    basic closing_data and signal
    """

    def __init__(self):
        self.signal = self.datas[0].signal

        if self.p.period == "monthly":
            self.days = 12
        elif self.p.period == "weekly":
            self.days = 52
        elif self.p.period == "daily":
            self.days = 252

        # 计数，计算trunover and win_rate
        self.buy_value = 0.0
        self.sell_value = 0.0
        self.winning_trades = 0
        self.total_trades = 0

    def next(self):

        dt = self.datas[0].datetime.date(0)  # 获取当前的回测时间点
        print(
            "Date:{},Close:{},Signal:{},Cash: {}, Value: {}, Position Size: {}".format(
                self.datas[0].close[0],
                dt,
                self.datas[0].signal[0],
                self.broker.getcash(),
                self.broker.getvalue(),
                self.position.size,
            )
        )  # 不会写log函数，print来凑

        if self.signal[0] == 1.0:
            if self.position.size == 0:  # 检查是否有持仓，只有在没持仓时再进行买入操作
                size = self.broker.getcash() // self.datas[0].close[0]  # 使用全部可用资金
                try:
                    print("size_to_buy:{},size_to_buy*open[1]:{}".format(size, size * self.datas[0].open[1]))
                    self.buy(size=size)
                except Exception as e:
                    print("数据不足，无法访问 open[1]，跳过操作")

        elif self.signal[0] == -1.0:
            self.close()  # 平仓
            print(self.position.size)

    # 存储订单信息
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(
                    f"BUY EXECUTED, Price: {order.executed.price}, Cost: {order.executed.value}, Comm: {order.executed.comm}"
                )
                #                 self.buy_value += size * self.datas[0].open[1]  # 缓存买入额
                self.buy_value += 1  # 缓存买入次数
            else:  # Sell
                print(
                    f"SELL EXECUTED, Price: {order.executed.price}, Cost: {order.executed.value}, Comm: {order.executed.comm}"
                )
                #             self.sell_value += self.position.size * self.datas[0].open[1]  # 缓存卖出额
                self.sell_value += 1  # 缓存卖出次数

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print("Order Canceled/Margin/Rejected:{}".format(order.status))

            # 订单处理完成后，将order置空
            self.order = None

    # 存储交易信息
    def notify_trade(self, trade):
        if trade.isclosed:
            self.total_trades += 1
            if trade.pnl > 0:
                self.winning_trades += 1

    def stop(self):

        # 换手率：交易额
        #         average_net_value = (self.broker.getvalue() + self.broker.startingcash) / 2  # 计算平均净资产
        #         turnover_rate = (self.buy_value + self.sell_value) / (2 * average_net_value)
        # 换手率：交易次数
        turnover_rate = (self.buy_value + self.sell_value) / (len(self.datas[0]) / self.days)

        # 计算胜率
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0

        print("***" * 39)
        print("turnoer（/year-wise/two-sided）: {:.2}".format(turnover_rate))
        print("win_rate: {:.2}".format(win_rate))


class MyPandasDataSignal(bt.feeds.PandasData):
    """
    data adjust:t
    get adjusted data,claim line and params
    """

    lines = ("signal",)  # 添加这行来定义新的 line
    # 设置column在line上的位置
    params = (
        ("open", "open"),  # -1 表示该列不存在
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", None),
        ("signal", "signal"),  # 指定 'signal' 列的名称
    )


class RiskAnalyzer:
    """
    risk\return analyzer:
    use empyrical
    """

    def __init__(self, returns, benchmarkreturns, risk_free=0.0, **kwargs):

        self.results = {}
        self._period = kwargs["period"]
        self._returns = returns
        self._benchmarkReturns = benchmarkreturns

        # empyrical要用的是对应period的无风险收益率,利用推导式调整
        self._risk_free = np.power(1 + risk_free, 1 / 12) - 1
        self.__risk_free = np.power(1 + risk_free, 1 / 252) - 1
        self._free_param = self.__risk_free if self._period == "weekly" else self._risk_free

        # 声明带计算指标
        self._annual_return = 0.0
        self._alpha = 0.0
        self._vola = 0.0
        self._omega = 0.0
        self._sharp = 0.0
        self._sortino = 0.0
        self._calmar = 0.0
        self._max_drawdown = 0.0
        self._cum_return = 0.0
        self._cum_return_benchmark = 0.0

    def analysis_(self, factor):

        try:

            if factor == "annual_return":
                self._annual_return = ep.annual_return(returns=self._returns, period=self._period)

            elif factor == "alpha":

                self._alpha = ep.alpha(
                    returns=self._returns,
                    factor_returns=self._benchmarkReturns,
                    risk_free=self._free_param,
                    period=self._period,
                )

            elif factor == "vola":
                self._vola = ep.annual_volatility(returns=self._returns, period=self._period)

            elif factor == "sharp":
                self._sharp = ep.sharpe_ratio(returns=self._returns, period=self._period, risk_free=self._free_param)

            elif factor == "calmar":
                self._calmar = ep.calmar_ratio(returns=self._returns, period=self._period)

            elif factor == "max_drawdown":
                self._max_drawdown = ep.max_drawdown(returns=self._returns)

            elif factor == "omega":
                self._omega = ep.omega_ratio(returns=self._returns, risk_free=self._free_param)

            elif factor == "sortino":
                self._sortino = ep.sortino_ratio(returns=self._returns, period=self._period)

            elif factor == "cum_return":
                self._returns.iloc[0] = 0
                self._benchmarkReturns.iloc[0] = 0  # 对齐净值
                self._cum_return = ep.cum_returns(returns=self._returns, starting_value=0.0)
                self._cum_return_benchmark = ep.cum_returns(returns=self._benchmarkReturns, starting_value=0.0)

        except Exception as e:

            print("错误：无法计算{}——指标:{}".format(factor, e))

    def run(self):

        factors = [
            "annual_return",
            "alpha",
            "vola",
            "sharp",
            "calmar",
            "max_drawdown",
            "omega",
            "sortino",
            "cum_return",
        ]
        for factor in factors:
            self.analysis_(factor)

        self.results = {
            "annual_return": self._annual_return,
            "alpha": self._alpha,
            "vola": self._vola,
            "sharp": self._sharp,
            "calmar": self._calmar,
            "max_drawdown": self._max_drawdown,
            "omega": self._omega,
            "sortino": self._sortino,
            "cum_return": self._cum_return,
            "cum_return_benchmark": self._cum_return_benchmark,
            "excess_return": self._cum_return - self._cum_return_benchmark,
        }

        result_str = (
            "annual_return:{:.4f}\nalpha:{:.4f}\nannual_volatility:{:.4f}\nsharpe_ratio:{:.4f}\ncalmar_ratio:{:.4f}\nmax_drawdown:{:.4f}\nomega_ratio:{:.4f}\nsortino_ratio:{:.4f}"
        ).format(
            self.results["annual_return"],
            self.results["alpha"],
            self.results["vola"],
            self.results["sharp"],
            self.results["calmar"],
            self.results["max_drawdown"],
            self.results["omega"],
            self.results["sortino"],
        )

        # get pic
        fig = plt.figure(figsize=(14, 7))
        ax1 = fig.add_subplot(2, 1, 1)

        ax1.plot(self.results["cum_return"], label="Strategy", color='blue', linewidth=2)
        ax1.plot(self.results["cum_return_benchmark"], label="Benchmark", color='gray', linewidth=1.5)
        ax1.plot(self.results["excess_return"], label="excess_return", color='green', linewidth=1.5, linestyle='-')  # 明确设置为实线
        
        ax1.legend()
        ax1.set_title("Cumulative Returns: Our Model vs Benchmark")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Cumulative Return")
        # ax.annotate(result_str,xy=(self.results['cum_return'].index[-1],self.results['cum_return'].iloc[-1]))  # 添加annotate，不好用

        sns.set()
        plt.tight_layout()  # 自动调整布局
        plt.show()
        plt.close(fig)  # close plt prevent loop

        return print(result_str)


# 获取clean_alphalens数据
def get_factor_dataframe_for_alphalens(dataframe):

    # 初始化数据格式
    dataframe.index = pd.to_datetime(dataframe.index)
    dataframe.reset_index(names=["date"], inplace=True)
    dataframe.dropna(inplace=True)

    # 调整df结构
    dataframe_: pd.DataFrame = dataframe.melt(var_name="asset", id_vars=["date"], value_name="factor_value")
    dataframe_.set_index(["date", "asset"], inplace=True)

    return dataframe_
