# -*- coding:utf-8 -*-
import os
from scipy import stats
import copy
import math
import sys
import matplotlib.pyplot as plt
import datetime
from scipy.signal import argrelextrema
import numpy as np
import statsmodels.tsa.stattools as ts
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from scipy.stats.stats import pearsonr
from sklearn import mixture


#sys.path.append("./emd_util")
from monte_carlo import montecarlo_simulate
from monte_carlo import montecarlo_simulate_array



class back_test():

    def __init__(self):
        
        self.money = 10000
        self.share = 0
        self.close_price = []
        self.fuquan_close_price = []
        self.mp = []
        self.mpdiff = []

        self.total_asset = self.money
        self.buy_price = 0
        self.fail_flag = 0


        self.buy_day = 0


        self.hb=[]

        self.mclist=[]
        self.mclist2=[]

    def sell(self, share, price):
        return float(share)*float(price)

    def buy(self, money, price):
        return float(money)/float(price) 


    def sell_fee(self, money):
        
        return money*0.0002+money*0.001

    def buy_fee(self, share, price):
      
        return share*price*0.0002


        
        
    def run(self, show_final_decision = False):

        final_decision = "don't move"

        current_price = self.fuquan_close_price[-1]
        trade_price = self.fuquan_close_price[-1]

#        print "current c {}".format(current_price)
#        print "current mp {}".format(self.mp[-1])
#        print "buy {}".format(self.buy_price)

        jede,m = decision_func(self.close_price, self.mp, self.mpdiff)
#        print "current m {}".format(m)
###############################################################

        if  self.buy_price!=0 and  ((jede==2)):
 
            if self.fail_flag==0:
                self.money = self.sell(self.share,trade_price)
                self.share = 0
                self.total_asset = self.money - self.sell_fee(self.money)
                self.money = self.total_asset
                if self.buy_price>=trade_price:
                    self.fail_flag=0
                else:
                    self.fail_flag=0
                self.hb.append(((trade_price-self.buy_price)/(self.buy_price)))
                self.buy_price = 0
                final_decision = "sell"
            else:
                if self.buy_price>=trade_price:
                    self.fail_flag=1
                else:
                    self.fail_flag=0
                self.buy_price=0
                
        elif   self.buy_price!=0 and ((self.buy_price-current_price)/self.buy_price>0.05 ):
            if self.fail_flag==0:
                if self.buy_price>=trade_price:
                    self.fail_flag=0
                self.money = self.sell(self.share,trade_price)
                self.share = 0
                self.buy_price = 0
                self.total_asset = self.money - self.sell_fee(self.money)
                self.hb.append(((trade_price-self.buy_price)/(self.buy_price)))
                self.money = self.total_asset
                final_decision = "sell"
            else:
                if self.buy_price>=trade_price:
                    self.fail_flag=1
                self.buy_price=0
             
        elif  self.buy_price==0   and   jede==1 :
            if self.fail_flag!=0:
                self.buy_price=trade_price
            else:
                self.share = self.buy(self.money,trade_price)
                self.money = 0
                self.buy_price = trade_price
                self.total_asset = self.share*trade_price
                final_decision = "buy"
#                print "close {} mpdiff {} m {}".format(self.close_price[-1],self.mpdiff[-1],m)


        if self.buy_price!=0:
            self.buy_day += 1


        if show_final_decision == True:
            return final_decision,m


 
    def set_data(self, close_price, mp, mpdiff, fuquan_close_price):
        self.close_price = close_price
        self.fuquan_close_price = fuquan_close_price
        self.mp = mp
        self.mpdiff = mpdiff

                 
    def get_result(self):
        return self.total_asset, self.hb#np.mean(self.hb)/np.std(self.hb)

def decision_func(close_price, mp, mpdiff):
    ###########################################################
        mean_de=0
        if np.mean(mpdiff[-3:])<0:
            mean_de=1
        elif np.mean(mpdiff[-3:])>0 :
            mean_de=2
    ############################################################

    #############################################################
        m1,s1=montecarlo_simulate_array(close_price[-6:-1],1000)
        mcde1=0
        if m1>close_price[-1]:
            mcde1 = 1

        m2,s2=montecarlo_simulate_array(close_price[-6:-1],1000)
        mcde2=0
        if m2>close_price[-1]:
            mcde2 = 1

        m3,s3=montecarlo_simulate_array(close_price[-6:-1],1000)
        mcde3=0
        if m3>close_price[-1]:
            mcde3 = 1

        mcde=0
        if mcde1 == 1 and mcde2 == 1 and mcde3==1:
            mcde = 1

    ###############################################################
#        g = mixture.GMM(n_components=2)
#        obs=close_price[-5:]
#
#        g.fit(obs)
#        m1 = np.round(g.means_, 2)[0]
#        m2 = np.round(g.means_, 2)[1]
#        if abs(obs[-1]-m1)>abs(obs[-1]-m2):
#            close=m2
#        else:
#            close=m1
##
#        gmmde=1
#        if obs[-1]>close:
#            gmmde=0
#
#    ###############################################################
#
#        def calc_recentdown_pro(datac,mp,n):
#            c=0
#            c1=0
#            for i in range(1,len(datac)-n):
#                if datac[i]<mp[i] and datac[i-1]>mp[i-1]:
#                    c+=1
#                    if datac[i+n]<datac[i]:
#                        c1+=1
#            if c!=0:
#                return float(c1)/c
#            else:
#                return 0
#
#        rdde=0
#        p=20
#        datac = close_price[-p:]
#        mp = mp[-p:]
#        if calc_recentdown_pro(datac, mp, 1)<0.5 or calc_recentdown_pro(datac, mp, 3)<0.5:
#            rdde=1
#
#    ###############################################################
        jede=0
        if  mpdiff[-1]<0 and mean_de==1 and   mcde==1 :#and gmmde==1:#  and rdde==1:
            jede=1
        if (mpdiff[-1]>0 and mean_de==2):
            jede=2

        return jede,(mcde1,mcde2,mcde3)

def simulate(stock):
    print "begin {} simulate".format(stock.get_name())
    begin = 20 #monte carlo simulate need 20 data points before to calc std
    close_price = list(stock.data["close price"].values)
    fuquan_close_price = list(stock.data["fuquan close price"].values)
    mp = list(stock.data["mp"].values)
    mpdiff = list(stock.data["mpdiff"].values)
    
    if stock.get_share() == 0: 
        back_test_process = back_test()
        for i in range(len(close_price)-30,len(close_price)):
            back_test_process.set_data(close_price[:i+1],mp[:i+1],mpdiff[:i+1],fuquan_close_price[:i+1])
            if i != len(close_price)-1:
                back_test_process.run()
            else:
   #             decision_list = []
   #             for i in range(2):
                decision,m = back_test_process.run(show_final_decision = True)
   #             decision = "buy"
   #             for i in decision_list:
   #                 if i == "sell":
   #                     decision = "sell"
   #                 elif i == "don't move":
   #                     decision = "don't move"


        asset,hb = back_test_process.get_result()
        position = kelly(hb)
        if decision == "buy":
            print "db {}".format(m)
        print "finish {} simulate".format(stock.get_name())
        return asset, decision, position
    else:
        decision = "don't move"
        jede,m = decision_func(close_price, mp, mpdiff)
        buy_price = stock.get_buy_price()
        if jede == 2 or ((buy_price-fuquan_close_price[-1])/buy_price)>0.05:
            decision = "sell"
        if decision == "sell":
            print "ds {}".format(m)

        print "finish {} simulate".format(stock.get_name())
        return decision


def kelly(data):

    if len(data)>1:
        mean = np.mean(data) 
        std = np.std(data) 
        
        if mean>0:
            return mean/std**2
        else:
            return 0

    else:
        return 0


def kelly2(data):

    c=0
    c1=0
    c2=0
    s1=[]
    s2=[]
    for i in data:
        c+=1
        if i>0:
            c1+=1
            s1.append(i)
        else:
            c2+=1
            s2.append(i)

    if c!=0 and c1!=0 and c2!=0:
        pro_p = float(c1)/c
        pro_n = float(c2)/c

        mean_p = np.mean(s1)
        mean_n = abs(np.mean(s2))
        z = pro_p*mean_p-pro_n*mean_n
        m = pro_p*pro_n
        position = z/m
        if position>1.5:
            return 1.5
        elif position < 0:
            return 0
        else:
            return position
        
    else:
        return 0

            
        
