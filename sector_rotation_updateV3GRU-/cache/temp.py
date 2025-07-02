class BaseStrategy(bt.Strategy):
    params = (
        ('log_enabled', True),  # 添加一个参数来控制日志是否启用
    )

    def __init__(self):
        self.log_enabled = self.params.log_enabled

    def log(self, arg):
        if self.log_enabled:  # 根据参数决定是否打印日志
            print(arg)

    def getdatabyname_if_avail(self, name):
        current_dt = self.datetime.datetime(0)
        try:
            data = self.getdatabyname(name)
            data_avail_dt = data.datetime.datetime(0)
            if current_dt >= data_avail_dt:
                return data
            else:
                return None
        except KeyError:
            return None

    def get_totalvalue(self):
        positions = self.getpositions()
        portfolio_value = 0
        for data in self.datas:
            position = positions.get(data, None)
            if position is not None and position.size != 0:
                portfolio_value += position.size * data.close[0]

        total_value = self.broker.getcash() + portfolio_value
        return total_value

    def notify_order(self, order):
        dt = order.data.datetime.datetime()
        symbol = order.data._name

        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'{dt},---'
                         f'symbol:{symbol},'
                         f'buy price:{order.executed.price:.3f},'
                         f'size:{order.executed.size:.3f},'
                         f'value:{order.executed.price * order.executed.size:.3f},'
                         f'fees:{order.executed.comm:.3f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(f'{dt},---'
                         f'symbol:{symbol},'
                         f'sell price:{order.executed.price:.3f},'
                         f'size:{order.executed.size:.3f},'
                         f'value:{order.executed.price * order.executed.size:.3f},'
                         f'fees:{order.executed.comm:.3f}')
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('order failed')
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'trade pnl {trade.pnlcomm:.3f}')




class RebalanceStrategy(BaseStrategy):
    def __init__(self):
        # 记录交易日和调仓日期
        self.trade_day_counter = 0
        self.holding_symbols = {}
        #self.models_dict = models_dict
        self.features = selected_factors.copy()

    def prenext(self):
        print('prenext',self.datetime.datetime(0))
        self.next()

    def next(self):
        previous_dt = self.datetime.datetime(-1)
        current_dt = self.datetime.datetime(0)
        print(f'current_dt:{current_dt},{self.holding_symbols}')

        self.trade_day_counter += 1
        if self.trade_day_counter % 5 != 1:
            return

        data = filtered_data.loc[filtered_data['date'] <= current_date, keep_cols].copy()
        ex0 = ~(data['days_to_start'] < 15) | (data['days_to_end'] < 15)
        ex1 = ~(data['remain_balance'] < 2)
        ex2 = ~((data['days_to_start'] > 120) & (data['turn10sum'] > 1) & (
                data['redemption_met_days'] > 8) & (
                        data['close'] > 130) & (data['conversion_premium_rate_filtered'] > 30))
        ex3 = data['credict_rating'].isin(['AAA', 'AA+', 'AA', 'AA-'])
        ex4 = data['bond_style'] == 'bond'

        data = data[ex0 & ex1 & ex2 & ex3].copy()
        data[factor_names] = data[factor_names].fillna(0)

        test_set = data[data['date'] == current_date].copy()
        test_set.index = test_set['asset']
        stock_nums = 30
        test_set.loc[:, 'pred'] = -test_set.loc[:, 'double_low_factor']
        test_set = test_set.sort_values(by=['pred'], ascending=False).copy()
        buy_symbols = test_set.head(stock_nums)['asset'].tolist().copy()

        holding_symbols = list(self.holding_symbols.keys())
        for symbol in holding_symbols:
            if symbol not in buy_symbols:
                data = self.getdatabyname_if_avail(symbol)
                if data is None or data.volume[0] == 0:
                    continue
                self.close(data=data)
                self.holding_symbols.pop(symbol, None)

        portfolio_value = self.get_totalvalue()
        available_cash = self.broker.getcash() * 0.995
        print(f'portfolio_value:{portfolio_value},available_cash:{available_cash}')
        cash_per_stock = portfolio_value / min(stock_nums,len(buy_symbols))

        holding_symbols = list(self.holding_symbols.keys())
        for symbol in buy_symbols:
            if symbol not in holding_symbols:
                data = self.getdatabyname_if_avail(symbol)
                if data is None or data.volume[0] == 0:
                    continue
                size = cash_per_stock / data.close[0]

                self.buy(data=data, size=size)
                self.holding_symbols[symbol] = size

    def log(self, arg):
        pass
        #print('{}'.format(arg))

    def notify_order(self, order):
        dt = order.p.data.datetime.datetime(0)
        symbol = order.p.data._name

        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'{dt},---'
                         f'symbol:{symbol},'
                         f'buy price:{order.executed.price:.3f},'
                         f'size:{order.executed.size:.3f},'
                         f'value:{order.executed.price * order.executed.size:.3f},'
                         f'fees:{order.executed.comm:.3f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(f'{dt},---'
                         f'symbol:{symbol},'
                         f'sell price:{order.executed.price:.3f},'
                         f'size:{order.executed.size:.3f},'
                         f'value:{order.executed.price * order.executed.size:.3f},'
                         f'fees:{order.executed.comm:.3f}')
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('order failed')
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log(f'trade pnl {trade.pnlcomm:.3f}')



symbols = panel_data['asset'].unique()
cerebro = bt.Cerebro()
all_dates = panel_data['date'].unique()
start_date = min(all_dates)
end_date = max(all_dates)

for symbol in symbols[:]:
    data_ = panel_data.query(f"asset=='{symbol}'")
    if len(data_) == 0:
        continue

    date_df = pd.DataFrame(index=all_dates)
    data_ = data_.set_index('date')
    merged_data = date_df.join(data_, how='left')
    merged_data['date'] = merged_data.index
    merged_data = merged_data.sort_values(by=['date'],ascending=True)
    merged_data[['open', 'high', 'low', 'close_price']] = merged_data[['open', 'high', 'low', 'close_price']].ffill()
    merged_data = merged_data[['date','open', 'high', 'low', 'close_price', 'volume']]
    merged_data['openinterest'] = 0
    merged_data[['volume', 'openinterest']] = merged_data[['volume', 'openinterest']].fillna(0)

    datafeed = bt.feeds.PandasData(
        dataname=merged_data,
        fromdate=start_date,
        todate=end_date,
        datetime=0, open=1, high=2, low=3, close=4, volume=5, openinterest=6
    )
    cerebro.adddata(datafeed, name=symbol)
    cerebro.broker.setcommission(commission=0.0000, name=symbol)



cerebro.addstrategy(RebalanceStrategy)
cerebro.broker.setcash(1000000)
cerebro.broker.set_coc(False)
cerebro.addanalyzer(BacktraderAnalyzer, _name='daily_value')
cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
results = cerebro.run()

portfolio_stats = results[0].analyzers.getbyname('pyfolio')
daily_return, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
daily_return.index = daily_return.index.tz_localize(None)
daily_return = daily_return.to_frame(name='strategy')

df_values = results[0].analyzers.daily_value.get_analysis()
daily_returns = df_values['total_value'].pct_change()