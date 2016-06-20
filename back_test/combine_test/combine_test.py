# -*- coding:utf-8 -*-
#public lib
import numpy as np
import sys
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import random


#private lib
from stock import stock_object
from load_data import load_data_from_file
from load_data import load_data_from_tushare
from load_data import load_data_from_tushare_real_time
from load_data import load_data_from_mysql
from strategy import simulate




BASE = 100
PER_INVEST = 30000

class combine_test():

    def __init__(self, global_config_file):
        self.pool = 100000
        self.pool_list = []
        with open(global_config_file) as f:
            self.global_config = yaml.load(f)

        self.stock_dict = {}
        self.stock_data_dict = {}
        self.count = 1
        self.all_finish = False

        self.total_asset = 0
        self.total_asset_list = []

    def random_fail(self):
        r = random.random()
        if r > 0.0:
            return 0
        else:
            return 1

    def prepare_data(self, stock_number, source = "file"):
        if source == "file":
            data = load_data_from_file(stock_number)
        else:
            data = load_data_from_tushare(stock_number, start_date = '2000-01-05', end_date = '2016-06-18')
            #data = load_data_from_mysql(stock_number)
            #data = load_data_from_tushare_real_time(stock_number, start_date = '2015-11-01')
        print data
        if len(data.index)>0:
            data["mp"] = data["amount"]/data["vol"]
            data["mpdiff"] = data["close price"] - data["mp"] 
            print data[-10:]
            return data
        else:
            return data
    
    def register(self):
        for stock in self.global_config["stock"]:
            stock_number = stock[0]
            stock_name = stock[1]
            each_full_data = self.prepare_data(stock_number, source = "tushare")
            if len(each_full_data.index)>0:
                self.stock_data_dict[stock_number] = each_full_data
            
                self.stock_dict[stock_number] = stock_object(stock_number, stock_name, each_full_data.head(BASE))


         

    def pre_condition(self):
        self.register()


    def buy_process(self, buy_candidate, length_candidate, fix_invest_flag=True):
        
        simulate_result = []
        for stock in buy_candidate:
            if stock.delay_buy_flag == 1:
                self.delay_buy(stock,self.pool/2)
            else:
                if BASE+self.count <= len(self.stock_data_dict[stock.get_number()].index):
                    stock.set_current_count(self.count)
                    new_data = self.stock_data_dict[stock.get_number()].head(BASE+self.count)
                    stock.set_data(new_data)
                    result, decision, position = simulate(stock)
                    
                    print "stock {} {} {}".format(stock.get_name(),decision, stock.get_current_price())
                    simulate_result.append((stock,result,decision,position,stock.get_name()))
                else:
                    stock.set_finish_flag(True)
       # sum_result = []
       # for i in simulate_result:
       #     sum_result.append(i[1])
       # if len(simulate_result)!=0:
       #     result_mean = np.mean(sum_result)
       #     result_std = np.std(sum_result)
       #     simulate_result = filter(lambda x:x[1]<result_mean, simulate_result)
        simulate_result = sorted(simulate_result, key=lambda x:x[1])#, reverse=True)
        print simulate_result


        if 1:#(len(simulate_result)/float(length_candidate))>0.5:
            if fix_invest_flag:
                for stock_pair in simulate_result:
                    if stock_pair[2] == "buy":
                        if self.buy(stock_pair[0], self.total_asset/2.0) == False:
                        #if self.buy(stock_pair[0], 10000) == False:
                            break
            else:
                for stock_pair in simulate_result:
                    if stock_pair[2] == "buy":
                        print "position {}".format(stock_pair[3])
                        if self.buy(stock_pair[0], PER_INVEST*stock_pair[3]) == False:
                            break



    def buy(self, stock, money):
        if self.random_fail()==0:
            if self.pool > money:
                stock.buy(money)
                self.pool -= money
            else:
                stock.buy(self.pool)
                self.pool = 0
            return True
        else:
            stock.set_delay_buy_flag(1)
            return False

    def delay_buy(self, stock, money):
        if self.pool > money:
            stock.delay_buy(money)
            self.pool -= money
        else:
            stock.buy(self.pool)
            self.pool = 0
       
        

    def sell_process(self, sell_candidate):

        for stock in sell_candidate:
            if BASE+self.count <= len(self.stock_data_dict[stock.get_number()].index):
                stock.set_current_count(self.count)
                new_data = self.stock_data_dict[stock.get_number()].head(BASE+self.count)
                stock.set_data(new_data)
                decision = simulate(stock)
                if stock.delay_sell_flag == 1:
                    self.delay_sell(stock)
                elif decision == "sell":
                    self.sell(stock)
                
                print "stock {} {}".format(stock.get_name(),decision)
                print "stock {} {}".format(stock.get_name(),stock.get_current_price())
            else:
                stock.set_finish_flag(True)
                self.sell(stock)

    def sell(self, stock):
        if self.random_fail()==0:
            profit = stock.sell()
            print "profit"
            print profit
            print self.pool
            self.pool += profit
            print self.pool
        else:
            stock.set_delay_sell_flag(1)

    def delay_sell(self,stock):
        print "delay sell"
        profit = stock.delay_sell()
        print "profit"
        print profit
        print self.pool
        self.pool += profit
        print self.pool
        

    def run(self):

        while self.all_finish == False:
            print "\n\n"
            print self.count
            print self.pool
            unfinish_stock_list = []
            for stock in self.stock_dict.values():
             #   print stock.get_name()
             #   print stock.get_share()
                if stock.get_finish_flag() == False:
                    unfinish_stock_list.append(stock)

            #print "unfinsih {}".format(unfinish_stock_list)
            if unfinish_stock_list != []:
                buy_candidate = []
                sell_candidate = []
                for stock in unfinish_stock_list:
                    if stock.get_share() != 0:
                        sell_candidate.append(stock)
                    else:
                        buy_candidate.append(stock)
                print self.pool
                self.sell_process(sell_candidate)
                self.buy_process(buy_candidate, len(buy_candidate))#, fix_invest_flag = False)
                self.count += 1
            else:
                self.all_finish = True
            print self.pool
            self.update_total_asset()
            self.total_asset_list.append(self.total_asset)
            self.pool_list.append(self.pool)
            print "total value {}".format(self.total_asset)

    def update_total_asset(self):
   
        hold_value = 0 
        for stock in self.stock_dict.values():
            if stock.get_share()!=0:
                hold_value += stock.get_current_value()
                print "name {} cp {} bp {}".format(stock.get_name(),stock.get_current_price(),stock.get_buy_price())
        self.total_asset = hold_value + self.pool
            

    def post_condition(self):
        total_hb = []
        for stock in self.stock_dict.values():
            print stock.get_name()
            hb = stock.get_profit_each_day()
            print hb
            total_hb.extend(hb)

        print total_hb
        plt.figure(1)
        plt.plot(self.total_asset_list)
        plt.figure(2)
        plt.plot(self.pool_list)
        plt.show()

    def start(self):
        self.pre_condition()
        self.run()
        self.post_condition()


if __name__ == "__main__":
    config_file = sys.argv[1]
    my_test = combine_test(config_file) 
    my_test.start()
