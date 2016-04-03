import os
import sys
import libfann
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from emd import *
import datetime
from scipy.signal import argrelextrema
import numpy as np




class process_util():

    def __init__(self,open_price,close_price,macd,date,vol,high,low):
        
        self.money = 10000
        self.share = 0
        self.sell_x_index = []
        self.sell_y_index = []
        self.buy_x_index = []
        self.buy_y_index = []
        self.open_price = open_price
        self.close_price = close_price
        
        self.total_asset_list = []
        self.total_asset = 0
        self.buy_price = 0
        self.predict_list = []

        self.period_decision_buy_point = []
        self.period_decision_sell_point = []

        self.max_min_period_delta = []
        self.min_max_period_delta = []
        self.last_extrem = ""
        self.current_extrem = ""

        self.last_period_decision = 0
        self.last_last_period_decision = 0

        self.last_rise_flag = 0
        self.current_rise_flag = 0
        self.last_buy_flag = 0

        self.last_now_max = 0
        self.last_now_min = 0
        self.before_last_max = 0
        self.before_last_min = 0
        self.max_keep_flag = 0    
        self.macd = macd
        self.last_bottom = 0
        self.date = date
        self.buy_macd = 0
        self.down_macd_counter = 0 
        self.imf_pre = []
        self.vol = vol
        self.high_price = high
        self.low_price = low


    def sell(self, share, price):
        return float(share)*float(price)

    def buy(self, money, price):
        return float(money)/float(price) 
    
    def generate_all(self, data, flag, input_number):
        print "length data %s"%len(data)
        print "length flag %s"%len(flag)
        fp = open("train_data_all", 'w')
        for i in range(len(data)-input_number+1):
           for j in range(i,input_number+i):
               fp.writelines(str(data[j]))
               fp.writelines(" ")
           fp.writelines("\n")
           fp.writelines(str(flag[i]))
           fp.writelines("\n")
        fp.close()


    def generate_all_price(self,data,input_num):
        expect = 0
        fp = open("train_data_all",'w')
        for i in range(len(data)-input_num+1): 
            for j in range(i,input_num+i):
                fp.writelines(str(data[j]))
                fp.writelines(" ")
                expect = j+1
            fp.writelines("\n")
            if expect<len(data):
                fp.writelines(str(data[expect]))
                fp.writelines("\n")
        fp.writelines("0")
        fp.close()
    
    def generate_divide(self, data_raw, eachfile_group_num, input_num):
        fp = open("train_data_all",'r')
        data = fp.readlines()
        fp.close()
        total_group_num = len(data_raw)-input_num+1
        total_file_num = total_group_num - eachfile_group_num-1 
        for i in range(total_file_num):
            fp = open("../../train_data/data_%s"%i,'w')
            fp.write("%s %s 1\n"%(eachfile_group_num,input_num))
            for j in range(i*2,(eachfile_group_num+i)*2):
                fp.writelines(str(data[j]))
            fp.close()


    def file_count(self, train_data_dir):
        f = os.popen("ls %s|wc -l"%train_data_dir) 
        file_num = f.readline()
        f.close()
        return file_num
        


    def sell_fee(self, money):
        
        return money*0.0002+money*0.001

    def buy_fee(self, share, price):
      
        return share*price*0.0002

    def format_data(data):
        if isinstance(data,list):
            return [ (float(data[i])-min(data))/(max(data)-float(min(data))) for i in range(len(data))]
            
   


    def generate_emd_data(self,data,start):
        
        #for i in range(start,len(data)-2):
        for i in range(start,len(data)+1):
            fp = open("../../emd_data/emd_%s"%i,'w')
            for j in range(i):
                fp.writelines(str(data[j]))
                fp.writelines("\n")
   #         fp.writelines(str(self.open_price[i]))
   #         fp.writelines("\n")
   #         fp.writelines(str(self.close_price[i-1]))
   #         fp.writelines("\n")
            fp.close()

    def generate_emd_data_fix(self,data,start,period):

        for i in range(start,len(data)+1):
            fp = open("../../emd_data_fix/emd_%s"%i,'w')
            for j in range(i-period,i):
                print j
                fp.writelines(str(data[j]))
                fp.writelines("\n")
            fp.close()


    def macd_stat(self, index):
        macd_border = []
        macd_border2 = []
        macd_border_index = []
        macd_border_index2 = []
        all_index = []
        all_index2 = []
        num = 0
        n = 0
        for i in range(2,len(self.macd[:index])):
            if self.macd[i+1]>0 and self.macd[i]<0:            
                macd_border.append(self.macd[i])
                macd_border_index.append(i)
            if self.macd[i]>0 and self.macd[i-1]<self.macd[i] and self.macd[i-2]>self.macd[i-1] and self.macd[i+1]>self.macd[i]:
                macd_border2.append(self.macd[i])
                macd_border_index2.append(i)

        for i in range(3,len(self.macd[:index])):
            if  self.macd[i-2]>self.macd[i-3] and self.macd[i-1]>self.macd[i-2] and self.macd[i]>self.macd[i-1] and self.macd[i]>(np.mean(macd_border)) and self.macd[i]<0:
                all_index.append(i)
            if  self.macd[i-2]>self.macd[i-1] and self.macd[i]>self.macd[i-1] and self.macd[i]>(np.mean(macd_border2)+np.std(macd_border2)) and self.macd[i]>0:
                all_index2.append(i)

        for i in all_index:
            if i in macd_border_index:
                n += 1
            elif i+1 in macd_border_index:
                n += 1

        for i in all_index2:
            if i in macd_border_index2:
                num += 1
            
        print float(n)/len(all_index) 
        print float(num)/len(all_index2) 
        return (np.mean(macd_border),np.std(macd_border))

    def run_predict_price_macd(self,current_price_index,date,datafile):

        trade_price = self.open_price[current_price_index+1]
        current_price = self.close_price[current_price_index]
        print "current price%s"%current_price
        print "macd %s"% self.macd[current_price_index]
        print "macd %s"% self.macd[current_price_index-1]
        print "macd %s"% self.macd[current_price_index-2]
        print "macd %s"% self.macd[current_price_index-3]
        print "macd %s"% self.macd[current_price_index-4]
        
        print self.macd_stat(current_price_index)
        if  self.money>0 and self.macd[current_price_index]>self.macd[current_price_index-1] and self.macd[current_price_index]<0:
            print "buy 1"
            self.share = self.buy(self.money,trade_price)
            self.money = 0
            self.buy_price = trade_price
            self.total_asset = self.share*trade_price
            self.buy_x_index.append(current_price_index)
            self.buy_y_index.append(current_price)
        elif  self.money>0 and self.macd[current_price_index-4]<0 and self.macd[current_price_index-3]<0 and self.macd[current_price_index-2]<0 and self.macd[current_price_index-1]<0 and self.macd[current_price_index]>0 :
            print "buy 2"
            self.share = self.buy(self.money,trade_price)
            self.money = 0
            self.buy_price = trade_price
            self.total_asset = self.share*trade_price
            self.buy_x_index.append(current_price_index)
            self.buy_y_index.append(current_price)
        elif self.macd[current_price_index]<self.macd[current_price_index-1] and self.share>0 and float(self.buy_price)!=0 :
            print "sell 1"
            self.money = self.sell(self.share,trade_price)
            self.share = 0
            self.buy_price = 0
            self.total_asset = self.money - self.sell_fee(self.money)
            self.last_buy_flag = 0
            self.money = self.total_asset
            self.sell_x_index.append(current_price_index)
            self.sell_y_index.append(current_price)

        self.total_asset_list.append(self.total_asset)
        print "total asset is %s"%(self.total_asset)
        print "money is %s"%(self.money)
        print "share is %s"%(self.share)


    def run_predict_price_period(self,imf,current_price_index,residual,imf_next,date,datafile):

        decision = 0
        power_decision = 0
        imf_power_decision = 0
        period_decision = 0
        extrem_decision = 0
        current_rise_flag = 0

        self.imf_pre.append(imf[-1])

        data = np.array(imf)
        #data_large = np.array(self.power_ma_decision(current_price_index)[1])
        imf_max_index = argrelextrema(data,np.greater)[0]
        imf_min_index = argrelextrema(data,np.less)[0]




        ma_line = self.ma_line(5,current_price_index)
        data_large = np.array(ma_line)
        data_large = np.array(imf_next)
        imf_large_max_index = argrelextrema(data_large,np.greater)[0]
        imf_large_min_index = argrelextrema(data_large,np.less)[0]







        print "large max index %s"%imf_large_max_index
        print "large min index %s"%imf_large_min_index

        data_recent = np.array(imf[-100:])
        imf_max_index_recent = argrelextrema(data_recent,np.greater)[0]
        imf_min_index_recent = argrelextrema(data_recent,np.less)[0]
        print "len of max %s portion %s"%(len(imf_max_index_recent),float(len(imf_max_index_recent))/len(imf[-100:]))
        print "len of min %s portion %s"%(len(imf_min_index_recent),float(len(imf_min_index_recent))/len(imf[-100:]))

        print "max index %s"%imf_max_index
        print "min index %s"%imf_min_index
        max_period_delta = [imf_max_index[i+1]-imf_max_index[i] for i in range(len(imf_max_index)-1)]
        min_period_delta = [imf_min_index[i+1]-imf_min_index[i] for i in range(len(imf_min_index)-1)]

        if imf_max_index[0]>imf_min_index[0]:
            self.max_min_period_delta = [imf_max_index[i]-imf_min_index[i] for i in range(min(len(imf_max_index),len(imf_min_index)))]
            self.min_max_period_delta = [imf_min_index[i+1]-imf_max_index[i] for i in range(min(len(imf_max_index),len(imf_min_index))-1)]
        else:
            self.max_min_period_delta = [imf_max_index[i+1]-imf_min_index[i] for i in range(min(len(imf_max_index),len(imf_min_index))-1)]
            self.min_max_period_delta = [imf_min_index[i]-imf_max_index[i] for i in range(min(len(imf_max_index),len(imf_min_index)))]

        max_period_delta.remove(max(max_period_delta))
        max_period_delta.remove(min(max_period_delta))
        min_period_delta.remove(max(min_period_delta))
        min_period_delta.remove(min(min_period_delta))
        #print "mean max%s"%max_period_delta
        #print "mean min%s"%min_period_delta
        #print "mean max min%s"%max_min_period_delta
        mean_max_delta = float(sum(max_period_delta))/len(max_period_delta)
        mean_min_delta = float(sum(min_period_delta))/len(min_period_delta)



        max_min_period_delta = self.max_min_period_delta[-5:]
        max_min_period_delta.remove(max(max_min_period_delta))
        max_min_period_delta.remove(min(max_min_period_delta))
        
        min_max_period_delta = self.min_max_period_delta[-5:]
        min_max_period_delta.remove(max(min_max_period_delta))
        min_max_period_delta.remove(min(min_max_period_delta))
        mean_max_min = float(sum(max_min_period_delta))/len(max_min_period_delta)
        mean_min_max = float(sum(min_max_period_delta))/len(min_max_period_delta)


        #print "mean max delta%s"%mean_max_delta
        #print "mean min delta%s"%mean_min_delta
        #print "mean max min%s"%mean_max_min

        now_position_last_min_delta = len(imf)-1-imf_min_index[-1]
        print "now min delta%s"%now_position_last_min_delta
        now_position_last_max_delta = len(imf)-1-imf_max_index[-1]
        print "now max delta%s"%now_position_last_max_delta
         
        (max_min_low,max_min_high) = self.span_estimate(max_min_period_delta)
        (min_max_low,min_max_high) = self.span_estimate(min_max_period_delta)
        

        print "date %s"%self.date[current_price_index]

#        if abs(now_position_last_min_delta-mean_min_delta)/mean_min_delta < 0.3 and abs(now_position_last_max_delta-mean_max_min)/mean_max_min < 0.3:
        if imf_min_index[-1]<imf_max_index[-1]:
            print "now max delta%s"%now_position_last_max_delta
            print "min max%s"%min_max_period_delta
            print "mean min max%s"%mean_min_max
            self.current_extrem = "max"
            print "low high %s %s"%(min_max_low,min_max_high)
        #    if  abs(now_position_last_max_delta-mean_min_max)/mean_min_max < 0.2:
        #    if now_position_last_max_delta+1 in self.top_period(min_max_period_delta)[0]:
            if now_position_last_max_delta+1 >= min_max_low and now_position_last_max_delta+1 <= min_max_high:
                period_decision = -1
                print "period decision is -1"
                self.period_decision_buy_point.append(current_price_index)
#        if abs(now_position_last_max_delta-mean_max_delta)/mean_max_delta < 0.3 and abs(now_position_last_min_delta-mean_max_min)/mean_max_min < 0.3:
        elif imf_min_index[-1]>imf_max_index[-1]:
            print "now min delta%s"%now_position_last_min_delta
            print "max min%s"%max_min_period_delta
            print "mean max min%s"%mean_max_min
            self.current_extrem = "min"
            print "low high %s %s"%(max_min_low,max_min_high)
        #    if abs(now_position_last_min_delta-mean_max_min)/mean_max_min < 0.2:
        #    if now_position_last_min_delta+1 in self.top_period(max_min_period_delta)[0]:
            if now_position_last_min_delta+1 >= max_min_low and now_position_last_min_delta+1 <= max_min_high:  
                period_decision = 1
                print "period decision is 1"
                self.period_decision_sell_point.append(current_price_index)

        print "last extre%s"%self.last_extrem
        print "current extre %s"%self.current_extrem
        if self.last_extrem == "max" and self.current_extrem == "min":
            extrem_decision = -1
        elif self.last_extrem == "min" and self.current_extrem == "max":
            extrem_decision = 1
        elif self.last_extrem == "min" and self.current_extrem == "min":
            extrem_decision = 0.5
        elif self.last_extrem == "max" and self.current_extrem == "max":
            extrem_decision = -0.5
        print "extre decision %s"%extrem_decision        
        self.last_extrem = self.current_extrem
        current_price_index = current_price_index#-1
        current_price = self.close_price[current_price_index]
        trade_price = self.open_price[current_price_index+1]
        print "current price%s"%current_price
        print "high price%s"%self.high_price[current_price_index]
        print "low price%s"%self.low_price[current_price_index]
        print "trade price%s"%trade_price
        print "buy price%s"%self.buy_price

#####################
#large period
#####################
        if imf_large_min_index[-1]<imf_large_max_index[-1]:
            self.large_current_extrem = "max"
        elif imf_large_min_index[-1]>imf_large_max_index[-1]:
            self.large_current_extrem = "min"


        if self.large_current_extrem == "min":
            large_ex_decision1 = 1
        else:
            large_ex_decision1 = 0
            






















        current_index = len(imf)-1
        print "imf now %s "%imf[current_index]
        print "imf now %s "%imf[current_price_index]
        print "imf before %s "%imf[current_price_index-1]
        print "imf before before %s "%imf[current_price_index-2]
        print "imf delta %s "%(imf[current_price_index]-imf[current_price_index-1])
        print "imf before delta %s "%(imf[current_price_index-1]-imf[current_price_index-2])

        imf_ma5_before = (imf[current_index-5]+imf[current_index-4]+imf[current_index-3]+imf[current_index-2]+imf[current_index-1])/5.0
        imf_ma5_current = (imf[current_index-4]+imf[current_index-3]+imf[current_index-2]+imf[current_index-1]+imf[current_index])/5.0
        imf_power_before = imf[current_index-1] - imf_ma5_before
        imf_power_current = imf[current_index] - imf_ma5_current
        print "imf power before %s"%imf_power_before
        print "imf power current %s"%imf_power_current
        if (imf_power_before>0 and imf_power_current<0): #or float(imf_power_before)/imf_power_current > 2.5:
            imf_power_decision = 1
        if (imf_power_before<0 and imf_power_current>0):# or float(imf_power_current)/imf_power_before > 2.5:
            imf_power_decision = -1

        print "imf power decision %s"%imf_power_decision
        delta_imf_ma20 = self.ma(imf, 20, 0, current_index)-self.ma(imf, 20, -1, current_index)
        if delta_imf_ma20 > 0.02:
            imf_ma_decision1 = 1
        else:
            imf_ma_decision1 = 0

#        imf_ma3_before = (imf[current_index-3]+imf[current_index-2]+imf[current_index-1])/3.0
#        print "ma before%s"%imf_ma3_before
#        imf_ma3_current = (imf[current_index-2]+imf[current_index-1]+imf[current_index])/3.0
#        print "ma current%s"%imf_ma3_current
#        power_before = imf[current_index-1] - imf_ma3_before
#        print "power before%s"%power_before
#        power_current = imf[current_index] - imf_ma3_current
#        print "power current%s"%power_current

        price_ma5_before2 = (self.close_price[current_price_index-6]+self.close_price[current_price_index-5]+self.close_price[current_price_index-4]+self.close_price[current_price_index-3]+self.close_price[current_price_index-2])/5.0
        price_ma5_before = (self.close_price[current_price_index-5]+self.close_price[current_price_index-4]+self.close_price[current_price_index-3]+self.close_price[current_price_index-2]+self.close_price[current_price_index-1])/5.0
        price_ma5_current = (self.close_price[current_price_index-4]+self.close_price[current_price_index-3]+self.close_price[current_price_index-2]+self.close_price[current_price_index-1]+self.close_price[current_price_index])/5.0
        power_before2 = self.close_price[current_price_index-2] - price_ma5_before2
        power_before = self.close_price[current_price_index-1] - price_ma5_before
        power_current = self.close_price[current_price_index] - price_ma5_current
        print "power  %s"%power_current
        print "power before %s"%power_before
#
##        price_ma3_before = (self.close_price[current_price_index-3]+self.close_price[current_price_index-2]+self.close_price[current_price_index-1])/5.0
##        price_ma3_current = (self.close_price[current_price_index-2]+self.close_price[current_price_index-1]+self.close_price[current_price_index])/5.0
##        power_before = self.close_price[current_price_index-1] - price_ma3_before
##        power_current = self.close_price[current_price_index] - price_ma3_current
        if power_before>0 and power_current<0:
            power_decision = 1
        if power_before<0 and power_current>0:
            power_decision = -1 
        print "power decision %s"%power_decision


        print "ma 60 -1 %s"%self.ma(self.close_price, 60, -1, current_price_index)
        print "ma 60 %s"%self.ma(self.close_price, 60, 0, current_price_index)
        print "ma 30 -1 %s"%self.ma(self.close_price, 30, -1, current_price_index)
        print "ma 30 %s"%self.ma(self.close_price, 30, 0, current_price_index)

        print "ma 20 -1 %s"%self.ma(self.close_price, 20, -1, current_price_index)
        print "ma 20 %s"%self.ma(self.close_price, 20, 0, current_price_index)
        print "ma 10 -1 %s"%self.ma(self.close_price, 10, -1, current_price_index)
        print "ma 10 %s"%self.ma(self.close_price, 10, 0, current_price_index)
        print "ma 5 -1 %s"%self.ma(self.close_price, 5, -1, current_price_index)
        print "ma 5 %s"%self.ma(self.close_price, 5, 0, current_price_index)
        print "ma 3 %s"%self.ma(self.close_price, 3, 0, current_price_index)
        print "ma 3  -1 %s"%self.ma(self.close_price, 3, -1, current_price_index)
        print "imf ma 5 %s"%self.ma(imf, 5, 0, current_index)
        print "imf ma 10 %s"%self.ma(imf, 10, 0, current_index)
        print "imf ma 20 %s"%self.ma(imf, 20, 0, current_index)
        print "imf ma 5 -1%s"%self.ma(imf, 5, -1, current_index)
        print "imf ma 10 -1%s"%self.ma(imf, 10, -1, current_index)
        print "imf ma 20 -1%s"%self.ma(imf, 20, -1, current_index)
        #if self.close_price[current_price_index-1] > self.ma(self.close_price, 5 , -1, current_price_index) and self.close_price[current_price_index] > self.ma(self.close_price, 5 , 0, current_price_index):
        if self.close_price[current_price_index] > self.ma(self.close_price, 5 , 0, current_price_index):
            ma_decision = 1
            print "ma decision is 1"
        else:
            ma_decision = 0
            print "ma decision is 0"

        if self.ma(self.close_price, 60, 0, current_price_index) > self.ma(self.close_price, 60, -1, current_price_index):
            ma_decision4 = 1
        else:
            ma_decision4 = 0
#        if self.ma(self.close_price, 5, 0, current_price_index) > self.ma(self.close_price, 20, 0, current_price_index):
#            ma_decision3 = 1
#        else:
#            ma_decision3 = 0
        if self.ma(self.close_price, 10, -1, current_price_index) < self.ma(self.close_price, 10, 0, current_price_index):
            ma_decision5 = 1
        else:
            ma_decision5 = 0

        if self.ma(imf, 5, 0, current_price_index) > self.ma(imf, 5, -1, current_price_index):
            ma_decision2 = 1
        else:
            ma_decision2 = 0
        if self.ma(imf, 5, 0, current_price_index) > self.ma(imf, 20, 0, current_price_index):
            ma_decision3 = 1
        else:
            ma_decision3 = 0

        vol_ma5 = self.ma(self.vol,5,0,current_price_index)
        vol_ma5_before = self.ma(self.vol,5,-1,current_price_index)
        vol_ma10 = self.ma(self.vol,10,0,current_price_index)
        vol_ma20 = self.ma(self.vol,20,0,current_price_index)
        vol_ma30 = self.ma(self.vol,30,0,current_price_index)
        vol_ma60 = self.ma(self.vol,60,0,current_price_index)
        print "vol 5 %s"%vol_ma5
        print "vol 5 before %s"%vol_ma5_before
        print "vol 10 %s"%vol_ma10
        print "vol 20 %s"%vol_ma20
        print "vol 30 %s"%vol_ma30
        print "vol 60 %s"%vol_ma60
        print "vol %s"%self.vol[current_price_index]
        print "vol -1 %s"%self.vol[current_price_index-1]
        print "vol -2 %s"%self.vol[current_price_index-2]
        print "delta vol %s"%(self.vol[current_price_index]-self.vol[current_price_index-1])
        print "delta vol -1 %s"%(self.vol[current_price_index-1]-self.vol[current_price_index-2])
        print "delta vol ma5 %s"%(self.ma(self.vol,5,0,current_price_index)-self.ma(self.vol,5,-1,current_price_index))
        print "delta delta vol ma5 %s"%(self.ma(self.vol,5,0,current_price_index)-self.ma(self.vol,5,-1,current_price_index)-self.ma(self.vol,5,-1,current_price_index)+self.ma(self.vol,5,-2,current_price_index))
        print "delta vol ma10 %s"%(self.ma(self.vol,10,0,current_price_index)-self.ma(self.vol,10,-1,current_price_index))
        print "delta delta vol ma10 %s"%(self.ma(self.vol,10,0,current_price_index)-self.ma(self.vol,10,-1,current_price_index)-self.ma(self.vol,10,-1,current_price_index)+self.ma(self.vol,10,-2,current_price_index))
        
        print "delta ma 3 %s"%(self.ma(self.close_price, 3, 0, current_price_index)-self.ma(self.close_price, 3, -1, current_price_index))
        print "delta ma 5 %s"%(self.ma(self.close_price, 5, 0, current_price_index)-self.ma(self.close_price, 5, -1, current_price_index))
        print "delta ma 7 %s"%(self.ma(self.close_price, 7, 0, current_price_index)-self.ma(self.close_price, 7, -1, current_price_index))
        print "delta ma 10 %s"%(self.ma(self.close_price, 10, 0, current_price_index)-self.ma(self.close_price, 10, -1, current_price_index))
        print "delta ma 20 %s"%(self.ma(self.close_price, 20, 0, current_price_index)-self.ma(self.close_price, 20, -1, current_price_index))

        print "delta delta ma 5 %s"%(self.ma(self.close_price, 5, 0, current_price_index)-self.ma(self.close_price, 5, -1, current_price_index)-self.ma(self.close_price, 5, -1, current_price_index)+self.ma(self.close_price, 5, -2, current_price_index))
        print "delta delta ma 10 %s"%(self.ma(self.close_price, 10, 0, current_price_index)-self.ma(self.close_price, 10, -1, current_price_index)-self.ma(self.close_price, 10, -1, current_price_index)+self.ma(self.close_price, 10, -2, current_price_index))
        print "delta delta ma 20 %s"%(self.ma(self.close_price, 20, 0, current_price_index)-self.ma(self.close_price, 20, -1, current_price_index)-self.ma(self.close_price, 20, -1, current_price_index)+self.ma(self.close_price, 20, -2, current_price_index))
        delta_delta_ma5 = self.ma(self.close_price, 5, 0, current_price_index)-self.ma(self.close_price, 5, -1, current_price_index)-self.ma(self.close_price, 5, -1, current_price_index)+self.ma(self.close_price, 5, -2, current_price_index)
        last_delta_delta_ma5 = self.ma(self.close_price, 5, -1, current_price_index)-self.ma(self.close_price, 5, -2, current_price_index)-self.ma(self.close_price, 5, -2, current_price_index)+self.ma(self.close_price, 5, -3, current_price_index)
        last_last_delta_delta_ma5 = self.ma(self.close_price, 5, -2, current_price_index)-self.ma(self.close_price, 5, -3, current_price_index)-self.ma(self.close_price, 5, -3, current_price_index)+self.ma(self.close_price, 5, -4, current_price_index)
        delta_delta_ma10 = self.ma(self.close_price, 10, 0, current_price_index)-self.ma(self.close_price, 10, -1, current_price_index)-self.ma(self.close_price, 10, -1, current_price_index)+self.ma(self.close_price, 10, -2, current_price_index)
        delta_delta_ma20 = self.ma(self.close_price, 20, 0, current_price_index)-self.ma(self.close_price, 20, -1, current_price_index)-self.ma(self.close_price, 20, -1, current_price_index)+self.ma(self.close_price, 20, -2, current_price_index)
        if delta_delta_ma10 > delta_delta_ma5:
            ma_decision62 = 1
        else:
            ma_decision62 = 0

        if delta_delta_ma5 > delta_delta_ma20:
            ma_decision63 = 1
        else:
            ma_decision63 = 0
        if last_delta_delta_ma5 < delta_delta_ma5:
            ma_decision64 = 1
        else:
            ma_decision64 = 0
        if last_last_delta_delta_ma5 <0 and last_delta_delta_ma5 <0 and  delta_delta_ma5<0:
            delta_decision1 = 0
        else:
            delta_decision1 = 1



        if self.ma(self.close_price, 5, 0, current_price_index)-self.ma(self.close_price, 5, -1, current_price_index) > self.ma(self.close_price, 5, -1, current_price_index)-self.ma(self.close_price, 5, -2, current_price_index) :
            ma_decision6 = 1
        else:
            ma_decision6 = 0
        if self.ma(self.close_price, 5, 0, current_price_index)-self.ma(self.close_price, 5, -1, current_price_index) > self.ma(self.close_price, 5, -1, current_price_index)-self.ma(self.close_price, 5, -2, current_price_index) and self.ma(self.close_price, 5, 0, current_price_index)-self.ma(self.close_price, 5, -1, current_price_index) > 0:
            ma_decision61 = 1
        else:
            ma_decision61 = 0
        if self.ma(self.close_price, 20, 0, current_price_index)-self.ma(self.close_price, 20, -1, current_price_index) - self.ma(self.close_price, 20, -1, current_price_index)+self.ma(self.close_price, 20, -2, current_price_index)>-0.01:
            ma_decision9 = 1
        else:
            ma_decision9 = 0

        if self.ma(self.close_price, 10, 0, current_price_index) > self.ma(self.close_price, 10, -1, current_price_index):
            ma_decision7 = 1
        else:
            ma_decision7 = 0

        if (self.close_price[current_index]-self.close_price[current_index-1] )/ self.close_price[current_index-1] > 0.05:
            price_decision = 0
        else:
            price_decision = 1
        if (self.close_price[current_index-1]-self.close_price[current_index-2] )< (self.close_price[current_index]-self.close_price[current_index-1] ):
            price_decision2 = 1
        else:
            price_decision2 = 0

        if self.close_price[current_index]<self.close_price[current_index-1]:
            price_decision3 = 1
        else:
            price_decision3 = 0
        
        if self.ma(self.close_price, 10, -1, current_price_index) < self.ma(self.close_price, 10, 0, current_price_index):
            ma_decision8 = 1
        else:
            ma_decision8 = 0

        if self.close_price[current_index] > self.close_price[current_index-1] and self.close_price[current_index-1] > self.close_price[current_index-2]:
            conrise_decision = 1
        else:
            conrise_decision = 0
             
#        if residual[current_index-1]<residual[current_index]:
#            residual_decision = 1
#        else:
#            residual_decision = 0
           
#        if power_before2<power_before and power_before<power_current:
#            power_decision = -1
#        if power_before2>power_before and power_before>power_current:
#            power_decision = 1

#        if power_before > power_current:
#            power_decision = 1
#        elif power_before < power_current:
#            power_decision = -1

#        print "imf current%s"%imf[current_index]
#        print "imf before current%s"%imf[current_index-1]
#        if imf[current_index]>imf[current_index-1]:
#            power_decision = -1
#        if imf[current_index]<imf[current_index-1]:
#            power_decision = 1

#        power_decision = period_decision


###############################################################
#extreme decision
##############################################################


        current_last_max = imf_max_index[-1]
        current_last_min = imf_min_index[-1]
        
        if abs(self.before_last_max-current_last_max)<=2: #and abs(self.before_last_min-imf_min_index[-2]<=2):
            ex_decision1 = 1
        else:
            ex_decision1 = 0
        
        
        if abs(self.before_last_min-current_last_min)>=3:
            ex_decision2 = 1
        else:
            ex_decision2 = 0
        
        if abs(current_last_max-current_last_min)<=3:
            ex_decision3 = 0
        else:
            ex_decision3 = 1
        
        if ex_decision1 and self.buy_price>0:
            self.max_keep_flag += 1
        else:
            self.max_keep_flag = 0
        
        if self.max_keep_flag >= 5:
            ex_decision4 = 1
        else:
            ex_decision4 = 0
        
#        if abs(self.before_last_min-current_last_min)>=3 and abs(self.before_last_max-current_last_max)>=3 and self.close_price[current_price_index]<self.close_price[current_price_index-1]:
#            ex_decision5 = 1
#        else:
#            ex_decision5 = 0
        if (self.before_last_min-current_last_min>=2 and abs(self.before_last_max-current_last_max)<=2 )or  (self.before_last_max-current_last_max<=-2 and abs(self.before_last_min-current_last_min) <=2):
            ex_decision5 = 1
        else:
            ex_decision5 = 0

        if self.before_last_min-current_last_min>=2 :
            ex_decision6 = 1
        else:
            ex_decision6 = 0
        
        print "current last max %s"%current_last_max
        print "before last max %s"%self.before_last_max
        print "current last min %s"%current_last_min
        print "before last min %s"%self.before_last_min
        print "current last last min %s"%imf_min_index[-2]
        print "max keep flag %s"%self.max_keep_flag
        print "rsi 3 %s"%self.rsi(self.close_price,3,current_price_index)
        print "rsi 5 %s"%self.rsi(self.close_price,5,current_price_index)
        print "rsi 10 %s"%self.rsi(self.close_price,10,current_price_index)
        print "rsi 15 %s"%self.rsi(self.close_price,15,current_price_index)
        print "rsi 20 %s"%self.rsi(self.close_price,20,current_price_index)
        print "rsi 30 %s"%self.rsi(self.close_price,30,current_price_index)



##################################################################
#macd decision
#######################################

        
        if self.macd[current_price_index]<0 and self.macd[current_price_index-1]<self.macd[current_price_index-2] and self.macd[current_price_index-1]>self.macd[current_price_index]:
            macd_decision1 = 1
#        elif self.macd[current_price_index]<0 and self.macd[current_price_index-1]<self.macd[current_price_index-2] and self.macd[current_price_index-1]>self.macd[current_price_index]:
#            macd_decision1 = 1
      #  elif self.macd[current_price_index]<0 and self.macd[current_price_index-1]>self.macd[current_price_index-2] and self.macd[current_price_index-1]<self.macd[current_price_index]:
      #      macd_decision1 = 1
#        elif self.macd[current_price_index]<0 and self.macd[current_price_index-1]>self.macd[current_price_index-2] and self.macd[current_price_index-2]<self.macd[current_price_index-3] and self.macd[current_price_index-1]<self.macd[current_price_index]:
#            macd_decision1 = 1
        elif self.macd[current_price_index]>0 and self.macd[current_price_index-1]<self.macd[current_price_index-2] and self.macd[current_price_index]<self.macd[current_price_index-1]:
            macd_decision1 = 1
        else:
            macd_decision1 = 0

        if self.buy_macd != 0 and self.macd[current_price_index]<self.macd[current_price_index-1]:
            self.down_macd_counter += 1 



        
        rsi_5 = self.rsi(self.close_price,5,current_price_index)
        rsi_10 = self.rsi(self.close_price,10,current_price_index)
        macd_5 = self.macd[current_price_index]
        macd_5_before = self.macd[current_price_index-1]
        macd_5_before2 = self.macd[current_price_index-2]
        macd_5_before3 = self.macd[current_price_index-3]
        macd_5_before4 = self.macd[current_price_index-4]
        macd_5_before5 = self.macd[current_price_index-5]
        macd_5_before6 = self.macd[current_price_index-6]
        macd_5_before7 = self.macd[current_price_index-7]
        macd_5_before8 = self.macd[current_price_index-8]
        delta_macd_5 = macd_5 - macd_5_before
        delta_macd_5_before = macd_5_before - macd_5_before2


        if ((macd_5 < 0 and macd_5 > -0.28)and macd_decision1 == 1 and (rsi_5<45 or (rsi_5<70 and rsi_5>60))) or (macd_5>0 and (rsi_5<45 or (rsi_5<70 and rsi_5>60))) :
             mix_decision1 = 1
        elif (macd_5 < 0 and macd_5 > -0.28) and (macd_5_before>macd_5_before2 and macd_5_before2>macd_5_before3 and macd_5_before<macd_5) and (rsi_5<45 or (rsi_5<70 and rsi_5>60)):
             mix_decision1 = 1
        elif (macd_5 < 0 and macd_5 > -0.28) and (delta_macd_5>delta_macd_5_before) and (rsi_5<45 or (rsi_5<70 and rsi_5>60)):
             mix_decision1 = 1
        else:
             mix_decision1 = 0

        if self.macd[current_price_index] < 0 and self.macd[current_price_index-1]>self.macd[current_price_index-2] and self.macd[current_price_index]>self.macd[current_price_index-1]:
            macd_decision2 = 0
        elif self.macd[current_price_index] > 0:
            macd_decision2 = 0
        else:
            macd_decision2 = 1

#        if self.macd[current_price_index] < 0:
#            macd_decision3 = 1
#        elif self.macd[current_price_index] > 0 and ((self.macd[current_price_index-1]-self.macd[current_price_index-2])>(self.macd[current_price_index]-self.macd[current_price_index-1])):
        if ((self.macd[current_price_index-1]-self.macd[current_price_index-2])>(self.macd[current_price_index]-self.macd[current_price_index-1])):
            macd_decision3 = 1
        else:
            macd_decision3 = 0
        print "macd 3 last %s"%self.macd[current_price_index-3] 
        print "macd 2 last %s"%self.macd[current_price_index-2] 
        print "macd 1 last %s"%self.macd[current_price_index-1] 
        print "macd  %s"%self.macd[current_price_index] 

        if macd_5<macd_5_before and macd_5_before<macd_5_before2 and macd_5_before2<macd_5_before3 and macd_5_before3<macd_5_before4 and macd_5_before4<macd_5_before5 and macd_5_before5<macd_5_before6 and macd_5_before6<macd_5_before7 and macd_5_before7<macd_5_before8:
            macd_decision4 = 1
        else:
            macd_decision4 = 0
        if rsi_5<60 and rsi_5>50 :
            rsi_decision1 = 0
        else:
            rsi_decision1 = 1


        
        rise_num = 0
        dec_num = 0
        for p in range(10):
            if self.close_price[current_price_index-p]>self.close_price[current_price_index-p-5]:
                rise_num += 1
            else:
                dec_num += 1
        print "rise num%s"%rise_num
        print "dec num%s"%dec_num
        if rise_num > dec_num:
            rise_decision = 1
        else:
            rise_decision = 0


        if macd_5<macd_5_before and macd_5_before<macd_5_before2 and macd_5_before2<macd_5_before3 and macd_5_before3<macd_5_before4 and macd_5_before4<macd_5_before5:
            macd_decision5 = 0
        else:
            macd_decision5 = 1


        if macd_5<0 and macd_5_before<0  and macd_5_before2>0: 
            macd_decision6 = 0
        else:
            macd_decision6 = 1
        if macd_5<0 and macd_5_before<0  and macd_5_before2<0 and macd_5_before3>0: 
            macd_decision7 = 0
        else:
            macd_decision7 = 1

        
        if  period_decision == -1:
            decision = -1
        if  period_decision == 1:
            decision = 1





        
        std = []
        for i in range(current_price_index-100,current_price_index):
            std_raw = self.close_price[i-5:i]
            std.append(np.std(std_raw))

        print "std mean %s stdstd %s"%(np.mean(std),np.std(std))
       

        std_list_5 = self.close_price[current_price_index-4:current_price_index+1]
        std_list_before = self.close_price[current_price_index-4:current_price_index]
        std_list_20 = self.close_price[current_price_index-19:current_price_index+1]
        std_list_20_before = self.close_price[current_price_index-20:current_price_index]
        std_list_20_before2 = self.close_price[current_price_index-21:current_price_index-1]
        std_list_20_before3 = self.close_price[current_price_index-22:current_price_index-2]
        std_list_20_before4 = self.close_price[current_price_index-23:current_price_index-3]
        print "std 5 %s"%np.std(std_list_5)
        print "std 5 before %s"%np.std(std_list_before)
        print "std 20  %s"%np.std(std_list_20)
        print "std 20 before %s"%np.std(std_list_20_before)
        print "std 20 before2 %s"%np.std(std_list_20_before2)
        print "std 20 delta %s"%(np.std(std_list_20)-np.std(std_list_20_before))
        print "std 20 delta 2%s"%(np.std(std_list_20_before)-np.std(std_list_20_before2))

        std_decision1 = 0
        #if np.std(std_list_before2)<np.std(std_list_before) and np.std(std_list_before)<np.std(std_list) and (2*np.std(std_list)/self.ma(self.close_price, 20, 0, current_price_index))>0.05:
        if (2*np.std(std_list_20)/self.ma(self.close_price, 20, 0, current_price_index))>0.05:
        #if np.std(std_list_20)>0.5:
            std_decision1 = 1
        else:
            std_decision1 = 0


        if self.close_price[current_price_index] > self.close_price[current_price_index-1]:
            current_rise_flag = 1
        else:
            current_rise_flag = 0
        print "last last period decision%s"%self.last_last_period_decision 
        print "last period decision%s"%self.last_period_decision 
        tomorrow_will = 0
        if extrem_decision == 1 and self.buy_price < self.close_price[current_price_index] and self.share > 0 and ex_decision5 == 1 :
            print "sell 1"
            self.money = self.sell(self.share,trade_price)
            self.share = 0
            self.buy_price = 0
            self.total_asset = self.money - self.sell_fee(self.money)
            self.last_buy_flag = 0
            self.money = self.total_asset
            self.sell_x_index.append(current_price_index)
            self.sell_y_index.append(current_price)
            tomorrow_will = 1
            self.down_macd_counter = 0
            self.buy_macd = 0
#        if macd_decision4 == 1 and self.share>0 and float(self.buy_price)!=0 :
#            print "sell 2"
#            self.money = self.sell(self.share,trade_price)
#            self.share = 0
#            self.buy_price = 0
#            self.total_asset = self.money - self.sell_fee(self.money)
#            self.last_buy_flag = 0
#            self.money = self.total_asset
#            self.sell_x_index.append(current_price_index)
#            self.sell_y_index.append(current_price)
#            tomorrow_will = 1
#        elif  float(self.buy_price)!=0 and self.down_macd_counter == 2 and self.share>0:
#            print "sell 4"
#            self.money = self.sell(self.share,trade_price)
#            self.share = 0
#            self.buy_price = 0
#            self.total_asset = self.money - self.sell_fee(self.money)
#            self.money = self.total_asset
#            self.sell_x_index.append(current_price_index)
#            self.last_buy_flag = 0
#            self.sell_y_index.append(current_price)
#            self.down_macd_counter = 0
#            self.buy_macd = 0
#            tomorrow_will = 1
        elif  macd_decision2 == 1 and float(self.buy_price)!=0 and (float(self.buy_price)-float(current_price))/float(self.buy_price)>0.02 and self.share>0:
            print "sell 4"
            self.money = self.sell(self.share,trade_price)
            self.share = 0
            self.buy_price = 0
            self.total_asset = self.money - self.sell_fee(self.money)
            self.money = self.total_asset
            self.sell_x_index.append(current_price_index)
            self.last_buy_flag = 0
            self.sell_y_index.append(current_price)
            self.down_macd_counter = 0
            self.buy_macd = 0
            tomorrow_will = 1
#        elif std_decision1 == 1 and extrem_decision == -1 and self.current_extrem == "min" and  ( now_position_last_min_delta == 1 or now_position_last_min_delta == 1)  and ex_decision1 == 1 and self.money>0 :#and macd_decision1 == 1:#and power_current>power_before and (self.power_ma_decision(current_price_index)[0] == 1 ): #and (self.ma(self.close_price, 5, 0, current_price_index) > self.ma(self.close_price, 5, -1, current_price_index)) :#and imf_power_decision == -1 and power_decision == -1 and self.money>0 :
#            print "buy buy 1"
#            self.share = self.buy(self.money,trade_price)
#            self.money = 0
#            self.buy_price = trade_price
#            self.total_asset = self.share*trade_price
#            self.buy_x_index.append(current_price_index)
#            self.buy_y_index.append(current_price)
#            self.last_buy_flag = 1
#            self.buy_macd = self.macd[current_price_index]
#            tomorrow_will = -1
#        elif self.vol[current_price_index] < 2*self.vol[current_price_index-1] and power_decision != 1 and (now_position_last_min_delta == 1 or now_position_last_min_delta == 2) and (self.ma(self.close_price, 10, 0, current_price_index)-self.ma(self.close_price, 10, -1, current_price_index))>0 and self.vol[current_price_index]>vol_ma5 and self.open_price[current_price_index]<self.close_price[current_price_index] and imf_power_decision == -1  and self.money>0 :#and self.vol[current_price_index]>self.vol[current_price_index-1] and self.money>0:
        elif imf_power_decision == -1  and self.money>0 :#and self.vol[current_price_index]>self.vol[current_price_index-1] and self.money>0:
            print "buy buy 1"
            self.share = self.buy(self.money,trade_price)
            self.money = 0
            self.buy_price = trade_price
            self.total_asset = self.share*trade_price
            self.buy_x_index.append(current_price_index)
            self.buy_y_index.append(current_price)
            self.last_buy_flag = 1
            self.buy_macd = self.macd[current_price_index]
            tomorrow_will = -1
#        elif  (self.ma(self.close_price, 10, 0, current_price_index)-self.ma(self.close_price, 10, -1, current_price_index))>0 and self.vol[current_price_index]>vol_ma5 and self.current_extrem == "min" and  ( now_position_last_min_delta == 1 or now_position_last_min_delta == 2)  and ex_decision1 == 1 and self.vol[current_price_index]>self.vol[current_price_index-1] and self.money>0:
#            print "buy buy 1"
#            self.share = self.buy(self.money,trade_price)
#            self.money = 0
#            self.buy_price = trade_price
#            self.total_asset = self.share*trade_price
#            self.buy_x_index.append(current_price_index)
#            self.buy_y_index.append(current_price)
#            self.last_buy_flag = 1
#            self.buy_macd = self.macd[current_price_index]
#            tomorrow_will = -1
#        elif std_decision1 == 1 and macd_decision1 == 1 and self.current_extrem == "min"  and  ( now_position_last_min_delta == 1 or now_position_last_min_delta == 2)  and ex_decision1 == 1 and self.money>0 and (power_current-power_before)/self.close_price[current_price_index]<0.03 and power_current>0 and power_before<0: #and (self.ma(self.close_price, 5, 0, current_price_index) > self.ma(self.close_price, 5, -1, current_price_index)) :#and imf_power_decision == -1 and power_decision == -1 and self.money>0 :
#            print "buy buy 2"
#            self.share = self.buy(self.money,trade_price)
#            self.money = 0
#            self.buy_price = trade_price
#            self.total_asset = self.share*trade_price
#            self.buy_x_index.append(current_price_index)
#            self.buy_y_index.append(current_price)
#            self.last_buy_flag = 1
#            self.buy_macd = self.macd[current_price_index]
#            tomorrow_will = -1
#        elif  macd_decision1 == 1 and  std_decision1 == 1 and power_decision == -1 and imf_power_decision == -1 and (now_position_last_min_delta == 1 or now_position_last_min_delta == 2) and extrem_decision == -1 and ex_decision1 == 1 and self.money>0 : #and (self.ma(self.close_price, 5, 0, current_price_index) > self.ma(self.close_price, 5, -1, current_price_index)) :#and imf_power_decision == -1 and power_decision == -1 and self.money>0 :
#            print "buy 2"
#            self.share = self.buy(self.money,trade_price)
#            self.money = 0
#            self.buy_price = trade_price
#            self.total_asset = self.share*trade_price
#            self.buy_x_index.append(current_price_index)
#            self.buy_y_index.append(current_price)
#            self.last_buy_flag = 1
#            tomorrow_will = -1
        else:
            self.last_buy_flag = 0

        self.last_last_period_decision = self.last_period_decision
        self.last_period_decision = period_decision

        self.before_last_max = imf_max_index[-1]
        self.before_last_min = imf_min_index[-1]

        lines = []
        if os.path.exists("decision"):
            fp = open("decision",'r')
            lines = fp.readlines()
            fp.close()
            fp = open("decision",'w')
            lines.append("\n")
            lines.append(date)
            lines.append(" ")
            lines.append(datafile)
            lines.append(" ")
            lines.append(str(tomorrow_will))
            lines.append("\n")
            fp.writelines(lines)
        else:
            fp = open("decision",'w')
            lines.append(date)
            lines.append(" ")
            lines.append(datafile)
            lines.append(" ")
            lines.append(str(tomorrow_will))
            lines.append("\n")
            fp.writelines(lines)
        fp.close()

        if imf_min_index[-1]<imf_max_index[-1]:
            self.last_now_max = now_position_last_max_delta
            self.last_now_min = 0
        elif imf_min_index[-1]>imf_max_index[-1]:
            self.last_now_min = now_position_last_min_delta
            self.last_now_max = 0
        self.last_rise_flag = current_rise_flag
        self.total_asset_list.append(self.total_asset)
        print "total asset is %s"%(self.total_asset)
        print "money is %s"%(self.money)
        print "share is %s"%(self.share)

    def span_estimate(self, data):
        return (round(np.mean(data)-np.std(data),0),round(np.mean(data)+np.std(data),0))

    def format_data(self, data):
        if isinstance(data,list):
            return [ 10*((float(data[i])-min(data))/(max(data)-float(min(data)))) for i in range(len(data))]

    def run2(self,emd_data, datafile,current_index,date):
        starttime = datetime.datetime.now()
        self.run_predict_price_macd(current_index,date,datafile)
        endtime = datetime.datetime.now()
        print "run time"
        print (endtime - starttime).seconds

    def power_ma_decision(self,index):
        decision = 0
        emd_data = []
        for i in range(index):
            if i < 4:
                emd_data.append(0)
            else:
                ma5 = (self.close_price[i-4]+self.close_price[i-3]+self.close_price[i-2]+self.close_price[i-1]+self.close_price[i])/5.0
                power = self.close_price[i] - ma5
                emd_data.append(power)
        
        #emd_data = self.close_price[1000:]
        
        my_emd = one_dimension_emd(emd_data)
        (imf, residual) = my_emd.emd()
        
        imf_without0 = [emd_data[i]-imf[0][i] for i in range(len(emd_data))]
        
        imf_without1 = [imf_without0[i]-imf[1][i] for i in range(len(emd_data))]
        
        imf_without2 = [imf_without1[i]-imf[2][i] for i in range(len(emd_data))]
        imf_without3 = [imf_without2[i]-imf[3][i] for i in range(len(emd_data))]
        imf_without4 = [imf_without3[i]-imf[4][i] for i in range(len(emd_data))]
        ana_data = imf_without1
        data_large = np.array(ana_data)
        imf_large_max_index = argrelextrema(data_large,np.greater)[0]
        imf_large_min_index = argrelextrema(data_large,np.less)[0]
        print "power max %s"%imf_large_max_index
        print "power min %s"%imf_large_min_index
        
        if imf_large_max_index[-1]<imf_large_min_index[-1]:
            decision = 1
        return (decision,ana_data)



    def run(self,emd_data_raw, datafile,current_index,date):
        starttime = datetime.datetime.now()
        self.clear_emd_data()
        emd_data = self.cubicSmooth5(emd_data_raw)
        my_emd = one_dimension_emd(emd_data)
        (imf, residual) = my_emd.emd()
        for i in range(len(imf)):
            fp = open("imf%s"%i,'w')
            for j in range(len(imf[i])):
                fp.writelines(str(imf[i][j]))
                fp.writelines("\n")
            fp.close()
        print "imf num %s"%len(imf)
        fp = open("residual",'w')
        for i in range(len(residual)):
            fp.writelines(str(residual[i]))
            fp.writelines("\n")
        fp.close()

        fp = open("imf0")
        lines = fp.readlines()
        fp.close()

        #print lines

        imf_raw = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            imf_raw.append(float(eachline.split("\n")[0]))
   #     for i in range(2):
   #         imf.pop()
        imf_without = [emd_data[i]-imf_raw[i] for i in range(len(emd_data))]

        fp = open("imf1")
        lines = fp.readlines()
        fp.close()

        #print lines

        imf_raw1 = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            imf_raw1.append(float(eachline.split("\n")[0]))
        imf_without2 = [imf_without[i]-imf_raw1[i] for i in range(len(emd_data))]


        fp = open("imf2")
        lines = fp.readlines()
        fp.close()
        imf_raw2 = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            imf_raw2.append(float(eachline.split("\n")[0]))
        imf_without3 = [imf_without2[i]-imf_raw2[i] for i in range(len(emd_data))]
        fp = open("imf3")
        lines = fp.readlines()
        fp.close()
        imf_raw3 = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            imf_raw3.append(float(eachline.split("\n")[0]))
        imf_without4 = [imf_without3[i]-imf_raw3[i] for i in range(len(emd_data))]


        fp = open("residual")
        lines = fp.readlines()
        fp.close()
        residual_raw = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            residual_raw.append(float(eachline.split("\n")[0]))
        imf_without6 = [imf_without2[i]-residual_raw[i] for i in range(len(emd_data))]

        imf = imf_without
    #    imf = imf_without
        #self.clear_nn_train_data()
        #self.generate_all_price(imf,25)
        #self.generate_divide(imf,20,25)
        #self.run_predict_price_real("../../train_data","nn_file",20,25,imf)
        #self.run_predict_price_period(imf)
        print "\n\n\n"
        print "imf now %s"%imf[-1]
        self.run_predict_price_period(imf,current_index,residual,imf_without,date,datafile)
        endtime = datetime.datetime.now()
        print "run time"
        print (endtime - starttime).seconds

                 
        
    def clear_emd_data(self):
        os.popen("rm imf*")    

    def clear_nn_train_data(self):
        os.popen("rm -r ../../train_data")
        os.popen("mkdir ../../train_data")

    def top_period(self, data):
        top_period_list = []
        top_period_value = []
        data_set = set(data)
        data_pair = [(i,data.count(i)) for i in data_set]
        data_pair.sort(key = lambda k:k[1],reverse=True)
        for i in range(2):
            top_period_list.append(data_pair[i][0])
            top_period_value.append(data_pair[i][1])
        result_list = range(min(top_period_list),max(top_period_list)+1)
        portion = float(sum(top_period_value))/len(data)
        print "top period list %s"%result_list
        print "top period value %s"%portion
        return (result_list,portion)


    def ma(self, data, period, start, current_index):
        sum_period = 0
        for i in range(period):
            sum_period += data[current_index + start - i]
        return float(sum_period)/period        


    def success_portion(self):
        data = self.total_asset_list
        success_flag = 0
        fail_flag = 0
        for i in range(len(data)-1):
            if data[i+1]>data[i]:
                success_flag += 1
            elif data[i+1]<data[i]:
                fail_flag += 1
        return float(success_flag)/(success_flag+fail_flag)

    def peilv(self):
        data = self.total_asset_list
        delta = []
        delta_new = []
        positive = 0
        negative = 0
        p_flag = 0
        n_flag = 0
        for i in range(len(data)-1):
            delta.append(data[i+1]-data[i])
        for j in range(len(delta)):
            if delta[j] != 0:
                delta_new.append(delta[j])
        if len(delta_new)>0:
            delta_new.remove(delta_new[0])
            print "asset delta %s"%delta_new
            for p in range(len(delta_new)):
                if delta_new[p] > 0:
                    positive += delta_new[p]
                    p_flag += 1
                if delta_new[p] < 0:
                    negative += delta_new[p]
                    n_flag += 1
            if p_flag != 0 and n_flag != 0:
                p_mean = float(positive)/p_flag
                n_mean = float(negative)/n_flag
                print "portion corr %s"%(float(p_flag)/(p_flag+n_flag))
                print "p_mean %s"%p_mean
                print "n_mean %s"%n_mean
         
        
    def hold_time(self):
        buy_time = self.buy_x_index
        sell_time = self.sell_x_index
        if len(buy_time)>len(sell_time):
            hold_time_list = [sell_time[i]-buy_time[i] for i in range(len(buy_time)-1)]
        elif len(buy_time) == len(sell_time):
            hold_time_list = [sell_time[i]-buy_time[i] for i in range(len(buy_time))]
        print "hold time %s"%sum(hold_time_list)


    def rsi(self,data,period,start):
        down_num = 0
        down_sum = 0
        up_num = 0
        up_sum = 0
        down_mean = 0
        up_mean = 0
        for i in range(start-period,start):
            if data[i]>data[i+1]:
                down_num += 1
                down_sum += data[i]-data[i+1]
            if data[i]<data[i+1]:
                up_num += 1
                up_sum += data[i+1]-data[i] 
        if down_num != 0 and up_num != 0 :
            down_mean = float(down_sum)/down_num
            up_mean = float(up_sum)/up_num
            rs = up_mean/down_mean
        elif down_num == 0:
            rs = 100
        elif up_num == 0:
            rs = 0
            
        return (float(rs)/(rs+1))*100

    def ma_line(self,n,index):
        ma = []
        for i in range(index+1):
            sum = 0
            if i == 0 or i == 1:
                ma.append(self.close_price[i])
            else:
                for j in range(i-n+1,i+1):
                    sum += self.close_price[j]
                ma.append(sum/n)
        return ma

    def cubicSmooth5 (self,data_input):
        N = len(data_input)
        data_output = [0 for i in range(N)]
        if N<5:
            data_output = data_input
        else:
            data_output[0] = (69.0 * data_input[0] + 4.0 * data_input[1] - 6.0 * data_input[2] + 4.0 * data_input[3] - data_input[4]) / 70.0;
            data_output[1] = (2.0 * data_input[0] + 27.0 * data_input[1] + 12.0 * data_input[2] - 8.0 * data_input[3] + 2.0 * data_input[4]) / 35.0;
            for i in range(2,N-2):
                data_output[i] = (-3.0 * (data_input[i - 2] + data_input[i + 2])+ 12.0 * (data_input[i - 1] + data_input[i + 1]) + 17.0 * data_input[i] ) / 35.0;
            data_output[N - 2] = (2.0 * data_input[N - 5] - 8.0 * data_input[N - 4] + 12.0 * data_input[N - 3] + 27.0 * data_input[N - 2] + 2.0 * data_input[N - 1]) / 35.0;
            data_output[N - 1] = (- data_input[N - 5] + 4.0 * data_input[N - 4] - 6.0 * data_input[N - 3] + 4.0 * data_input[N - 2] + 69.0 * data_input[N - 1]) / 70.0;
        return data_output
        
        

    def draw_fig(self,datafile,start,save=0):
        
        data = self.close_price

        fp = open("imf0")
        lines = fp.readlines()
        fp.close()

        #print lines

        imf_raw = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            imf_raw.append(float(eachline.split("\n")[0]))
   #     for i in range(2):
   #         imf.pop()
        imf_without = [emd_data[i]-imf_raw[i] for i in range(len(emd_data))]

        fp = open("imf1")
        lines = fp.readlines()
        fp.close()
        
        #print lines
        
        imf_raw1 = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            imf_raw1.append(float(eachline.split("\n")[0]))
        imf_without2 = [imf_without[i]-imf_raw1[i] for i in range(len(emd_data))]

        fp = open("imf2")
        lines = fp.readlines()
        fp.close()
        imf_raw2 = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            imf_raw2.append(float(eachline.split("\n")[0]))
        imf_without3 = [imf_without2[i]-imf_raw2[i] for i in range(len(emd_data))]


        fp = open("imf3")
        lines = fp.readlines()
        fp.close()
        imf_raw3 = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            imf_raw3.append(float(eachline.split("\n")[0]))
        imf_without4 = [imf_without3[i]-imf_raw3[i] for i in range(len(emd_data))]

        fp = open("residual")
        lines = fp.readlines()
        fp.close()
        residual_raw = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            residual_raw.append(float(eachline.split("\n")[0]))
        imf_without6 = [imf_without2[i]-residual_raw[i] for i in range(len(emd_data))]
        imf = imf_without


        fp = open("residual",'r')
        lines = fp.readlines()
        fp.close()

        residual = []
        for eachline in lines:
            residual.append(float(eachline.split("\n")[0]))

        
        imf_next = self.power_ma_decision(len(self.close_price))[1]
        print "buy x index%s"%self.buy_x_index 
        print "sell x index%s"%self.sell_x_index 
        imf_buy_value = [imf[i] for i in self.buy_x_index]
        imf_sell_value = [imf[i] for i in self.sell_x_index]
        #imf_buy_value = [imf[i] for i in self.period_decision_buy_point]
        #imf_sell_value = [imf[i] for i in self.period_decision_sell_point]
        #self.buy_x_index = self.period_decision_buy_point
        #self.sell_x_index = self.period_decision_sell_point
        buy_x = [i-start for i in self.buy_x_index]
        sell_x = [i-start for i in self.sell_x_index]
        plt.figure(1)
        plt.subplot(311).axis([start,len(data),min(data[start:]),max(data[start:])])      
        plt.plot([i for i in range(len(data))],data,'b',self.buy_x_index,self.buy_y_index,'r*',self.sell_x_index,self.sell_y_index,'g*')             
        plt.subplot(312)
        plt.plot([i for i in range(len(imf[start:]))],imf[start:],'b',buy_x,imf_buy_value,'r*',sell_x,imf_sell_value,'g*')   
        plt.subplot(313)
        plt.plot([i for i in range(len(imf_next[start:]))],imf_next[start:],'b')
        figname = "fig1_"+datafile         
        if save==1:
            savefig(figname)
        plt.figure(2)
        plt.subplot(211)
        plt.plot([i for i in range(len(self.total_asset_list))],self.total_asset_list,'b')
        plt.subplot(212)
        plt.plot([i for i in range(len(imf[start:]))],imf[start:],'b',[i for i in range(len(self.predict_list))],self.predict_list,'r')
        figname = "fig2_"+datafile
        plt.figure(3)
        plt.subplot(211)
        plt.plot([i for i in range(len(self.imf_pre))],self.imf_pre,'b')
        plt.subplot(212)
        plt.plot([i for i in range(len(imf[start:]))],imf[start:],'b')
       
        if save==1:         
            savefig(figname)
        else:
            plt.show() 


if __name__ == "__main__":

    datafile = sys.argv[1]
    begin = int(sys.argv[2])
    fp = open(datafile)
    lines = fp.readlines()
    fp.close()

    close_price = []
    open_price = []
    high_price = []
    low_price = []
    date = [] 
    macd = []
    vol = []
    for eachline in lines:
        eachline.strip()
        close_price.append(float(eachline.split("\t")[4]))
        macd.append(float(eachline.split("\t")[6]))
        open_price.append(float(eachline.split("\t")[1]))
        high_price.append(float(eachline.split("\t")[2]))
        low_price.append(float(eachline.split("\t")[3]))
        date.append(eachline.split("\t")[0])
        vol.append(float(eachline.split("\t")[5]))
    

    process = process_util(open_price,close_price,macd,date,vol,high_price,low_price)
    os.popen("rm -r ../../emd_data")
    os.popen("mkdir ../../emd_data")
    process.generate_emd_data(close_price,begin)
    print "begin %s"%begin
    #process.generate_emd_data_fix(close_price,begin,500)
    for i in range(begin,begin+int(process.file_count("../../emd_data"))-1):
        emd_data = []
        print "emd file %s"%i
        fp = open("../../emd_data/emd_%s"%i,'r')
        lines = fp.readlines()
        fp.close()
#        
        for eachline in lines:
            eachline.strip("\n")
            emd_data.append(float(eachline))
#    #    print "emd_data %s"%emd_data
        process.run(emd_data,datafile,i-1,date[-1])
#        process.run2(emd_data,datafile,i-1,date[-1])
    process.peilv()
    process.hold_time()
    process.draw_fig(datafile,begin)

    

            
