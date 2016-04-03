# -*- coding:utf-8 -*-
#public lib
import numpy as np
import sys
import os
import time
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import datetime


#private lib
from stock import stock_object
from load_data import load_data_from_file
from load_data import load_data_from_tushare
from load_data import load_data_from_tushare_real_time
from load_data import load_data_from_tushare_csv_sina
from strategy import simulate
from strategy import decision_func
from send_mail import send_mail



MAIL_TO1="thunderocean@163.com"
MAIL_TO2="skythundergogo@163.com"

class decision_maker():

    def __init__(self, global_config_file):
        with open(global_config_file) as f:
            self.global_config = yaml.load(f)

        self.stock_dict = {}
        self.stock_data_dict = {}
        self.stock_record_dict = {}

        self.stock_record_newadd_dict = {}

        self.buy_list = []
        self.sell_list = []
        self.candidate_sell_list = []

    def prepare_data(self, stock_number):
        #data = load_data_from_tushare_real_time(stock_number, start_date = '2015-11-01')
        data = load_data_from_tushare_csv_sina(stock_number)
        print data
        if len(data.index)>0:
            data["mp"] = data["amount"]/data["vol"]
            data["mpdiff"] = data["close price"] - data["mp"] 
            print data
            return data
        else:
            return data

    def prepare_record(self, stock_number):
        current_path = "/home/ly/git_repo/my_program/sds/back_test/combine_test"
        record_file = current_path+'/record/'+str(stock_number)
        if os.path.exists(record_file):
            with open(record_file) as f:
                lines = f.readlines()
                before_status = lines[-1].split(" ")[1]
                buy_price = float(lines[-1].split(" ")[2])
            return before_status, buy_price
        else:
            return "", 0

    def record_new_decision(self, new_add_content):

        abs_path = '/home/ly/git_repo/my_program/sds/back_test/combine_test/'
        for stock_number, new_content in new_add_content.items():
            record_file = abs_path+'/record/'+str(stock_number)
            print stock_number
            if os.path.exists(record_file):
                with open(record_file, 'r') as f:
                    content = f.readlines()
                legacy_date = content[-1].split(" ")[0]
                current_date = new_content.split(" ")[0]
                print "legacy date {} current date {}".format(legacy_date,current_date)
                if legacy_date != current_date:
                    content.append("\n")
                    content.append(new_content)
                    print content
                    with open(record_file, 'w') as f:
                        f.writelines(content)
            else:
                with open(record_file, 'w') as f:
                    content = new_content
                    f.write(content)

    
    def register(self):
        for stock in self.global_config["stock"]:
            stock_number = stock[0]
            stock_name = stock[1]
            each_full_data = self.prepare_data(stock_number)
            each_record_data = self.prepare_record(stock_number)
            self.stock_data_dict[stock_number] = each_full_data
            self.stock_record_dict[stock_number] = each_record_data
            self.stock_dict[stock_number] = stock_object(stock_number, stock_name, each_full_data)
        print self.stock_record_dict


    def pre_condition(self):
        self.register()


    def buy_process(self, buy_candidate):
        print "enter buy process" 
        print "buy candidate {}".format(buy_candidate) 
        simulate_result = []
        for stock_number, stock in buy_candidate:
            print "stock {} {}".format(stock.get_name(),stock.get_current_price())
            result, decision, position = simulate(stock)
            simulate_result.append((stock,result,decision,position,stock.get_number()))

        print simulate_result
        simulate_result = filter(lambda x:x[1]>11000, simulate_result)
        simulate_result = sorted(simulate_result, key=lambda x:x[1], reverse=True)
        print simulate_result
        for stock_pair in simulate_result:
            if stock_pair[2] == "buy":
                self.buy_list.append(stock_pair[4])
        for stock_number, stock in buy_candidate: 
            if stock_number in self.buy_list:
                self.stock_record_newadd_dict[stock_number] = "{} {} {} {} {}".format(self.stock_data_dict[stock_number]["date"].values[-1],"buy",self.stock_data_dict[stock_number]["close price"].values[-1],self.stock_data_dict[stock_number]["close price"].values[-1],self.stock_data_dict[stock_number]["mp"].values[-1])
            else:
                self.stock_record_newadd_dict[stock_number] = "{} {} {} {} {}".format(self.stock_data_dict[stock_number]["date"].values[-1],"don'tmove", 0, self.stock_data_dict[stock_number]["close price"].values[-1],self.stock_data_dict[stock_number]["mp"].values[-1])


        print "end buy process"


        

    def sell_process(self, sell_candidate):

        for stock_number,stock in sell_candidate:
            print stock_number
            close_price = self.stock_data_dict[stock_number]["close price"].values
            mp = self.stock_data_dict[stock_number]["mp"].values
            mpdiff = self.stock_data_dict[stock_number]["mpdiff"].values
            jede,m = decision_func(close_price, mp, mpdiff)
            buy_price = float(self.stock_record_dict[stock_number][1])

            
            if jede == 2 or ((buy_price-close_price[-1])/buy_price)>0.05:
                self.sell_list.append(stock_number)

        for stock_number, stock in sell_candidate:
            if stock_number in self.sell_list:
                self.stock_record_newadd_dict[stock_number] = "{} {} {} {} {}".format(self.stock_data_dict[stock_number]["date"].values[-1],"sell",0,self.stock_data_dict[stock_number]["close price"].values[-1],self.stock_data_dict[stock_number]["mp"].values[-1])
            else:
                self.stock_record_newadd_dict[stock_number] = "{} {} {} {} {}".format(self.stock_data_dict[stock_number]["date"].values[-1],"don'tmove", self.stock_record_dict[stock_number][1], self.stock_data_dict[stock_number]["close price"].values[-1],self.stock_data_dict[stock_number]["mp"].values[-1])

        

    def run(self):

        buy_candidate = []
        sell_candidate = []
        for stock_number, status in self.stock_record_dict.items():
            if self.stock_record_dict[stock_number][1]!=0:
                sell_candidate.append([stock_number,self.stock_dict[stock_number]])
                self.candidate_sell_list.append(stock_number)
            else:
                buy_candidate.append([stock_number,self.stock_dict[stock_number]])
        print "sell candidate {}".format(sell_candidate)
        print "buy candidate {}".format(buy_candidate)
        self.sell_process(sell_candidate)
        self.buy_process(buy_candidate)

            

    def post_condition(self):
        print "buy list {}".format(self.buy_list)
        print "sell list {}".format(self.sell_list)

        content = ""
        for i in self.buy_list:
            content += "buy {};".format(i)
        for i in self.sell_list:
            content += "sell {};".format(i)
        for i in self.candidate_sell_list:
            content += "candidatesell {};".format(i)

        ISOTIMEFORMAT='%Y-%m-%d %X'
        send_mail(MAIL_TO1, "{} decision".format(time.strftime( ISOTIMEFORMAT)),content)
        send_mail(MAIL_TO2, "{} decision".format(time.strftime( ISOTIMEFORMAT)),content)

        self.record_new_decision(self.stock_record_newadd_dict)


    def start(self):
        self.pre_condition()
        self.run()
        self.post_condition()


if __name__ == "__main__":
    starttime = datetime.datetime.now()
    config_file = sys.argv[1]
    my_test = decision_maker(config_file) 
    my_test.start()
    endtime = datetime.datetime.now()
    interval=(endtime - starttime).seconds
    print interval
