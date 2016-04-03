# -*- coding:utf-8 -*-

class stock_object():
    def __init__(self, number, name, data):
        self.name = name
        self.number = number
        self.data = data

        self.share = 0
        self.buy_price = 0
        self.finish_flag=False

        self.profit = []
        self.profit_each_day = []
        self.buy_day = []
        self.buy_start_count = 0
        self.current_count = 0

        self.delay_sell_flag = 0
        self.delay_buy_flag = 0


    def get_name(self):
        return self.name

    def get_number(self):
        return self.number

    def get_data(self, col_name):
        return data[colname].values

    def get_share(self):
        return self.share

    def set_data(self, new_data):
        self.data = new_data  

    def get_finish_flag(self):
        return self.finish_flag

    def set_finish_flag(self, finish_flag):
        self.finish_flag = finish_flag

    def set_delay_sell_flag(self, flag):
        self.delay_sell_flag = flag

    def set_delay_buy_flag(self, flag):
        self.delay_buy_flag = flag

    def sell(self):
        price = self.data["fuquan close price"].values[-1]
        money = float(self.share)*float(price)
        print "sell buy {} sell {}".format(self.buy_price,price)
        fee = self.sell_fee(money)
        self.profit.append((price-self.buy_price)/self.buy_price)
        self.buy_day.append(self.calc_buy_day())
        self.profit_each_day.append(self.profit[-1]/self.buy_day[-1])
        self.buy_price = 0
        self.share = 0
        return money-fee

    def delay_sell(self):
        price = self.data["fuquan open price"].values[-1]
        money = float(self.share)*float(price)
        print "sell buy {} sell {}".format(self.buy_price,price)
        fee = self.sell_fee(money)
        self.profit.append((price-self.buy_price)/self.buy_price)
        self.buy_day.append(self.calc_buy_day())
        self.profit_each_day.append(self.profit[-1]/self.buy_day[-1])
        self.buy_price = 0
        self.share = 0
        self.set_delay_sell_flag(0)
        return money-fee

    def buy(self, money):
        price = self.data["fuquan close price"].values[-1]
        fee = self.buy_fee(money)
        self.share = float(money-fee)/float(price)
        self.buy_price = price
        self.buy_start_count = self.current_count

    def delay_buy(self, money):
        price = self.data["fuquan open price"].values[-1]
        fee = self.buy_fee(money)
        self.share = float(money-fee)/float(price)
        self.buy_price = price
        self.buy_start_count = self.current_count
        self.set_delay_buy_flag(0)

    def sell_fee(self, money):
        return money*0.0002+money*0.001

    def buy_fee(self, money):
        return money*0.0002

    def set_current_count(self, count):
        self.current_count = count

    def get_current_count(self):
        return self.current_count

    def calc_buy_day(self):
        return self.current_count - self.buy_start_count

    def get_profit(self):
        return self.profit

    def get_current_value(self):
        price = self.data["fuquan close price"].values[-1]
        return self.share*price
        
    def get_buy_price(self):
        return self.buy_price

    def get_current_price(self):
        return self.data["fuquan close price"].values[-1]

    def get_profit_each_day(self):
        return self.profit_each_day
