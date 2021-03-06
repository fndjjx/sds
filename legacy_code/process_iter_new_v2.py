import os
import math
import sys
import libfann
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from emd import one_dimension_emd
from emdright2 import one_dimension_emd2
from smooth import cubicSmooth5
import datetime
from scipy.signal import argrelextrema
import numpy as np
from generate_emd_data import generateEMDdata
from generate_train_file import genTraindata
from neuro_predict import neuroprediction
from spline_predict import splinepredict




class process_util():

    def __init__(self,open_price,close_price,macd,date,vol):
        
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
        self.imf_power = []
        self.vol = vol
        self.last_imf = 0
        self.last_imf_power = 0
        self.last_imf_3_power = 0
        self.last_imf_power_decision = 0
        self.fail_flag = 0
        self.holdtime = 0
        self.last_power_decision = 0

        self.last_true_max = 0
        self.last_true_min = 0


    def sell(self, share, price):
        return float(share)*float(price)

    def buy(self, money, price):
        return float(money)/float(price) 

    def file_count(self, train_data_dir):
        f = os.popen("ls %s|wc -l"%train_data_dir) 
        file_num = f.readline()
        f.close()
        return file_num
        


    def sell_fee(self, money):
        
        return money*0.0002+money*0.001

    def buy_fee(self, share, price):
      
        return share*price*0.0002


    def run_predict(self,imf_list,current_price_index,date,datafile,train_data_dir,train_number,hidden_number,result_number):

        decision = 0
        power_decision = 0
        imf_power_decision = 0
        period_decision = 0
        extrem_decision = 0
        current_rise_flag = 0




##############################################################
#        recent_price = self.close_price[current_price_index]+self.close_price[current_price_index-2]+self.close_price[current_price_index-1]
#        before_price = self.close_price[current_price_index-7]+self.close_price[current_price_index-6]+self.close_price[current_price_index-5]
#        print "recent price %s"%recent_price
#        print "before price %s"%before_price
#
#        if (recent_price/before_price)>1.05 or (recent_price/before_price)<0.95:
#            imf_process = list(imf_list[2])
#            print "imf1 %s"%(recent_price/before_price)
#        else:
#            imf_process = list(imf_list[2])
#            print "imf2 %s"%(recent_price/before_price)
#
#
#        data = np.array(imf_process)
#        imf_max_index = argrelextrema(data,np.greater)[0]
#        imf_min_index = argrelextrema(data,np.less)[0]
#
#
#        price_list = self.close_price[current_price_index-499:current_price_index+1]
#        price_data = np.array(price_list)
#        price_max_index = argrelextrema(price_data,np.greater)[0]
#        print "price_max_index %s"%price_max_index
#        price_min_index = argrelextrema(price_data,np.less)[0]
#        print "price_min_index %s"%price_min_index
#
#
#        imf_extrem_decision = 0
#        if len(list(price_max_index)) > 0 and len(list(price_min_index)) > 0 :
#            index_list = list(price_max_index)+(list(price_min_index))
#            print "index list%s"%index_list
#            index_list.sort() 
#            last_extrem_position = index_list[-3] 
#            distance =  499 - last_extrem_position
#            true_index = 499-distance
#            print "distance %s"%distance
#            print "true index %s"%true_index
#            data = np.array(imf_process)
#            imf_max_index = argrelextrema(data,np.greater)[0]
#            print "imf max index %s"%imf_max_index
#            imf_min_index = argrelextrema(data,np.less)[0]
#            print "imf min index %s"%imf_min_index
#            
#            imf_max_true_index = list(filter(lambda n:n<=true_index,imf_max_index))
#            imf_min_true_index = list(filter(lambda n:n<=true_index,imf_min_index))
#            print "imf max true index %s"%imf_max_true_index
#            print "imf min true index %s"%imf_min_true_index
#            buyhigh = imf_process[imf_max_true_index[-1]]-imf_process[imf_min_true_index[-1]]
#            buyhigh2 = imf_process[imf_max_true_index[-1]]-imf_process[-1]
#            buyhigh6 = imf_process[imf_max_true_index[-1]]-imf_process[true_index+2]
#            buyhigh7 = imf_process[imf_max_true_index[-1]]-imf_process[true_index+1]
#            buyhigh3 = imf_process[imf_max_true_index[-1]]-imf_process[true_index]
#            buyhigh4 = imf_process[imf_max_true_index[-1]]-imf_process[true_index-1]
#            buyhigh5 = imf_process[imf_max_true_index[-1]]-imf_process[true_index-2]
#            print "buy high %s"%buyhigh
#            print "buy high2 %s"%buyhigh2
#            print "buy high6 %s"%buyhigh6
#            print "buy high7 %s"%buyhigh7
#            print "buy high3 %s"%buyhigh3
#            print "buy high4 %s"%buyhigh4
#            print "buy high5 %s"%buyhigh5
#            sellhigh = imf_process[imf_max_true_index[-1]]-imf_process[imf_min_true_index[-1]]
#            sellhigh2 = imf_process[-1]-imf_process[imf_min_true_index[-1]]
#            print "sell high %s"%sellhigh
#            print "sell high2 %s"%sellhigh2
#            distance_last_imf_max = 499-list(imf_max_true_index)[-1]
#            distance_last_imf_min = 499-list(imf_min_true_index)[-1]
#            last_min_max = list(imf_max_true_index)[-1] - list(imf_min_true_index)[-1]
#            cha = distance_last_imf_max- last_min_max
#            print "cha %s"%(distance_last_imf_max- last_min_max)
#            #if (buyhigh/buyhigh2)>0.8 and (buyhigh/buyhigh2)<1.2 and imf_max_index[-1]>imf_min_index[-1] and imf_max_true_index[-1]>imf_min_true_index[-1]: #and distance_last_imf_max<10 and distance_last_imf_max>5:
#            #    imf_extrem_decision = -1
#            #elif (sellhigh/sellhigh2)>0.8 and (sellhigh/sellhigh2)<1.2:
#            #    imf_extrem_decision = 1
#            if self.buy_price == 0:# and abs(self.last_true_max-imf_max_true_index[-1])<10 and abs(self.last_true_min-imf_min_true_index[-1])<10:#(buyhigh/buyhigh2)>0.3 and (buyhigh/buyhigh2)<3:
#                print "prepare buy"
#                if imf_max_true_index[-1]<imf_min_true_index[-1] and imf_min_true_index[-1]<485:
#                #    price_last_max_index = list(filter(lambda n:n>imf_min_true_index[-1],price_max_index))
#                #    print "price_last_max_index %s"%price_last_max_index
#                    
#                #    estimate_max = float(sum(price_last_max_index))/len(price_last_max_index)
#                #    last_min_max = math.ceil(estimate_max) - imf_min_true_index[-1]
#                #    print "last min max %s"%last_min_max
#                    estimate_max = (imf_min_true_index[-1]+imf_min_true_index[-1]-imf_max_true_index[-1])
#                    last_min_max = (estimate_max) - imf_min_true_index[-1]
#                    current_last_max = 499 - math.ceil(estimate_max)
#                    print "current_last_max %s"%current_last_max
#                    print "potion %s"%(abs(last_min_max-current_last_max)/float(last_min_max))
#                    if  abs(current_last_max-last_min_max)/float(last_min_max)<=0.25:
#                        imf_extrem_decision = -1
#                elif imf_max_true_index[-1]>imf_min_true_index[-1] and imf_max_true_index[-1]>485:
#                    last_min_max = imf_max_true_index[-1] - imf_min_true_index[-1]
#                    print "last min max %s"%last_min_max
#                    current_last_max = 499 - imf_max_true_index[-1]
#                    print "current_last_max %s"%current_last_max
#                    print "potion %s"%(abs(last_min_max-current_last_max)/float(last_min_max))
#                    #if  abs(last_min_max-current_last_max)/float(last_min_max)<=0.25:
#                    if (buyhigh/buyhigh2)>0.8 and (buyhigh/buyhigh2)<1.2: 
#                        imf_extrem_decision = -1
#                elif imf_max_true_index[-1]>imf_min_true_index[-1] and imf_max_true_index[-1]<485:
#                    price_last_min_index = list(filter(lambda n:n>imf_max_true_index[-1],price_min_index))
#                    print "price_last_min_index %s"%price_last_min_index
#
#                    estimate_min = float(sum(price_last_min_index))/len(price_last_min_index)
#                    print "estimate min  %s"%estimate_min
#                    current_min = 499 - estimate_min
#                    print "current_min %s"%current_min
#                    if current_min<7:
#                        imf_extrem_decision = -1
#                elif imf_max_true_index[-1]<imf_min_true_index[-1] and imf_min_true_index[-1]>490:
#            #        last_min_max = imf_max_true_index[-1] - imf_min_true_index[-1]
#            #        print "last min max %s"%last_min_max
#            #        current_last_max = 499 - imf_max_true_index[-1]
#            #        print "current_last_max %s"%current_last_max
#            #        print "potion %s"%(abs(last_min_max-current_last_max)/float(last_min_max))
#            #        if  abs(last_min_max-current_last_max)/float(last_min_max)<=0.3:
#            #            imf_extrem_decision = -1
#                    print "pp %s"%(buyhigh/buyhigh2)
#                    if (buyhigh/buyhigh2)>0.8 and (buyhigh/buyhigh2)<1.15:
#                        imf_extrem_decision = -1
#            if self.buy_price != 0 :#and abs(self.last_true_max-imf_max_true_index[-1])<10 and abs(self.last_true_min-imf_min_true_index[-1])<10:
#                print "prepare sell"
#                if imf_min_true_index[-1]<imf_max_true_index[-1] and imf_max_true_index[-1]<485:
#                    print "first"
#                    estimate_min = imf_max_true_index[-1]+imf_max_true_index[-1]-imf_min_true_index[-1]
#                    last_max_min = (estimate_min) - imf_max_true_index[-1]
#                    current_last_min = 499 - math.ceil(estimate_min)
#                    print "current_last_min %s"%current_last_min
#                    print "potion %s"%(abs(last_max_min-current_last_min)/float(last_max_min))
#                    if  abs(current_last_min-last_max_min)/float(last_max_min)<=0.25:
#                        imf_extrem_decision = 1
#                elif imf_min_true_index[-1]>imf_max_true_index[-1] and imf_min_true_index[-1]>=490:
#                    print "second"
#                    last_max_min = imf_min_true_index[-1] - imf_max_true_index[-1]
#                    print "last max min %s"%last_max_min
#                    current_last_min = 499 - imf_min_true_index[-1]
#                    print "current_last_min %s"%current_last_min
#                    print "potion %s"%(abs(last_max_min-current_last_min)/float(last_max_min))
#                    imf_extrem_decision = 1
#                elif imf_min_true_index[-1]>imf_max_true_index[-1] and imf_min_true_index[-1]<490:
#                    print "third"
#                    price_last_max_index = list(filter(lambda n:n>imf_min_true_index[-1],price_max_index))
#                    print "price_last_max_index %s"%price_last_max_index
#
#                    estimate_max = float(sum(price_last_max_index))/len(price_last_max_index)
#                    print "estimate max  %s"%estimate_max
#                    current_max = 499 - estimate_max
#                    print "current_max %s"%current_max
#                    if current_max<3:
#                        imf_extrem_decision = 1
#                elif imf_min_true_index[-1]<imf_max_true_index[-1] and imf_max_true_index[-1]>490:
#                    print "forth"
#                    imf_extrem_decision = -1
#
#
#
#        self.last_true_max = imf_max_true_index[-1]
#        self.last_true_min = imf_min_true_index[-1]
#
#        now_position_last_min_delta = len(imf_list[2])-1-imf_min_index[-1]
#        print "now min delta%s"%now_position_last_min_delta
#        now_position_last_max_delta = len(imf_list[2])-1-imf_max_index[-1]
#        print "now max delta%s"%now_position_last_max_delta
#         
#        
#
#        print "date %s"%self.date[current_price_index]
#        print "date %s"%date
#        print "current price %s"%self.close_price[current_price_index]
#        print "trade price %s"%self.open_price[current_price_index+1]
#        print "buy price %s"%self.buy_price
############################################################################################
#   
#        recent_price_list = self.close_price[current_price_index-4:current_price_index+1]
#
#        recent_price_format = self.format_data(recent_price_list)
#        recent_price_format2 = recent_price_format[-5:] 
#  
#
#
#        print "recent price format %s"%recent_price_format2
#
#
#        
#
#
#        imf = list(imf_list[2])  
#  
#        imf_ma5_before = (imf[-6]+imf[-5]+imf[-4]+imf[-3]+imf[-2])/5.0
#        imf_ma3_before = (imf[-4]+imf[-3]+imf[-2])/3.0
#
#        imf_ma5_current = (imf[-5]+imf[-4]+imf[-3]+imf[-2]+imf[-1])/5.0
#        imf_ma3_current = (imf[-3]+imf[-2]+imf[-1])/3.0
#        imf_power_before = imf[-2] - imf_ma5_before
#        imf_3_power_before = imf[-2] - imf_ma3_before
#        imf_power_current = imf[-1] - imf_ma5_current
#        imf_3_power_current = imf[-1] - imf_ma3_current
#        print "imf power before %s"%imf_power_before
#        print "imf power current %s"%imf_power_current
#        print "last imf power current %s"%self.last_imf_power
#        print "imf 3 power before %s"%imf_3_power_before
#        print "imf 3 power current %s"%imf_3_power_current
#        print "last imf 3 power current %s"%self.last_imf_3_power
# 
#        last_5 = self.close_price[current_price_index-5:current_price_index]
#        last_5_mean = np.mean(self.close_price[current_price_index-5:current_price_index])
#        last_5_std = np.std([last_5[i]-last_5_mean for i in range(5)])
#        print "std 5 %s"%last_5_std
#        self.imf_power.append(imf_power_current)
#
#
#        last_10 = self.close_price[current_price_index-6:current_price_index]
#
###############
        distance_decision = 1
        if len(self.buy_x_index)>1 and current_price_index-self.buy_x_index[-1]<8:
            distance_decision = 0

        distance_decision2 = 1
        if len(self.buy_x_index)>1 and current_price_index-self.buy_x_index[-1]<5:
            distance_decision2 = 0

        distance_decision3 = 1
        if len(self.sell_x_index)>1 and current_price_index-self.sell_x_index[-1]<8:
            distance_decision3 = 0



##########
        price = self.close_price[current_price_index-9:current_price_index+1]
        price_ma5_before = (price[-6]+price[-5]+price[-4]+price[-3]+price[-2])/5.0
        price_ma3_before = (price[-4]+price[-3]+price[-2])/3.0
        price_ma5_current = (price[-5]+price[-4]+price[-3]+price[-2]+price[-1])/5.0
        price_ma3_current = (price[-3]+price[-2]+price[-1])/3.0

        power_before = price[-2]-price_ma3_before
        power_ma5_before = price[-2]-price_ma5_before
        print "power before %s"%power_before
        print "power ma5 before %s"%power_ma5_before
        power_current = price[-1]-price_ma3_current
        power_ma5_current = price[-1]-price_ma5_current
        print "power current %s"%power_current
        print "power ma5 current %s"%power_ma5_current


        if power_before< power_current:
            power_decision = -1
        elif power_before>0 and power_current<0:
            power_decision = 1


        price_delta_decision = 1
        price_delta = [(price[-1-i]-price[-i-2])/price[-i-2] for i in range(3)]
        print "price delta %s"%price_delta
        price_wrong1 = filter(lambda x:x<-0.02,price_delta)
        price_wrong2 = filter(lambda x:x<-0.03,price_delta)
        if len(price_wrong1)>=2 or len(price_wrong2)>1:
            price_delta_decision = 0
        print "price delta decision %s"%price_delta_decision

###############
        price_50 = self.close_price[current_price_index-49:current_price_index+1]
        up_list = [price_50[i+1]-price_50[i] for i in range(49) if price_50[i]<price_50[i+1]]
        down_list = [price_50[i+1]-price_50[i] for i in range(49) if price_50[i]>price_50[i+1]]
        print up_list
        print down_list
        print "up number %s"%len(up_list)
        print "up potion %s"%(sum(up_list)/sum(down_list))
        
        sta_decision = 0
        if abs(sum(up_list)/sum(down_list))>0.7 and abs(sum(up_list)/sum(down_list))<1.4:
            sta_decision = 1
        
####################################################################################################################


#        imf_decision = 0 
#        data = np.array(last_5)
#        last_5_max_index = argrelextrema(data,np.greater)[0]
#        last_5_min_index = argrelextrema(data,np.less)[0]
#
#        data = np.array(last_10)
#        last_10_max_index = argrelextrema(data,np.greater)[0]
#        last_10_min_index = argrelextrema(data,np.less)[0]
#
#
#        last_10_imf = list(imf_list[2])[-10:]
#        print "last 10 price %s"%last_10
#
#        if len(last_5_max_index)>0 and len(last_5_min_index)>0: 
#            last_10_imf_array = np.array(last_10_imf)
#            last_10_imf_max_index = argrelextrema(last_10_imf_array,np.greater)[0]
#            last_10_imf_min_index = argrelextrema(last_10_imf_array,np.less)[0]
#            if len(last_10_imf_max_index)>0 and last_10_imf_max_index[-1]<4:
#                imf_decision = -1
#            if len(last_10_imf_min_index)>0 and imf_min_true_index[-1]>imf_max_true_index[-1]:
#                imf_decision = 1
#
            
#####################################################################################################################

      
#        if (imf_process[-1]>imf_process[-2] and imf_process[-2]<imf_process[-3]) : #or float(imf_power_before)/imf_power_current > 2.5:
#        #if (self.last_imf_3_power>0 and imf_3_power_current<0): #or float(imf_power_before)/imf_power_current > 2.5:
#        #if (self.last_imf_3_power>imf_3_power_current): #or float(imf_power_before)/imf_power_current > 2.5:
#            imf_power_decision = -1


        

        ma5 = self.ma(self.close_price, 5, 0, current_price_index)
        deltama5 = self.ma(self.close_price, 5, 0, current_price_index)-self.ma(self.close_price, 5, -1, current_price_index)
        ma10 = self.ma(self.close_price, 10, 0, current_price_index)
        deltama10 = self.ma(self.close_price, 10, 0, current_price_index)-self.ma(self.close_price, 10, -1, current_price_index)
        ma20 = self.ma(self.close_price, 20, 0, current_price_index)
        deltama20 = self.ma(self.close_price, 20, 0, current_price_index)-self.ma(self.close_price, 20, -1, current_price_index)
        ma30 = self.ma(self.close_price, 30, 0, current_price_index)
        deltama30 = self.ma(self.close_price, 30, 0, current_price_index)-self.ma(self.close_price, 30, -1, current_price_index)
   

        print "imf power decision %s"%imf_power_decision
        trade_price = self.open_price[current_price_index+1]
        current_price = self.close_price[current_price_index]

        imf_process = list(imf_list[2])
        data = np.array(imf_process)
        imf_max_index = list(argrelextrema(data,np.greater)[0])
        imf_min_index = list(argrelextrema(data,np.less)[0])
        print "len imf %s"%len(imf_process)
        print " imf max %s"%imf_max_index[-1]
        print " imf min %s"%imf_min_index[-1]
        print "current %s"%(float(499-imf_min_index[-1])/(imf_min_index[-1]-imf_min_index[-2]))
        #delta = (float(499-imf_min_index[-1])/(imf_min_index[-1]-imf_min_index[-2]))
        delta = (float(499-imf_min_index[-1])/12)
        #delta2 = (float(499-imf_max_index[-1])/(imf_max_index[-1]-imf_max_index[-2]))
        delta2 = (float(499-imf_max_index[-1])/12)
        if (delta>1.8 and delta <2.1 )or (delta>2.8 and delta<3.1) or (delta>3.8 and delta<4.1):
            decision = -1 
        elif (delta2>1.8 and delta2 <2.1 )or (delta2>2.8 and delta2<3.1) or (delta2>3.8 and delta2<4.1):
            decision = 1 


        if self.buy_price!=0 and distance_decision == 1 and (decision == 1 ) and  ((trade_price-self.buy_price)/self.buy_price) >0 and self.share > 0 :
            print "sell sell 1"
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
            self.holdtime = 0
            once_flag = 1
        elif self.buy_price!=0 and ((trade_price-self.buy_price)/self.buy_price) >0.045 and self.share > 0 :
            print "sell sell 2"
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
            self.holdtime = 0
            once_flag = 1
        elif self.buy_price!=0 and ((trade_price-self.buy_price)/self.buy_price) <-0.035 and self.share > 0 :
            print "sell sell 3"
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
            self.holdtime = 0
            once_flag = 1
        elif   self.buy_price == 0 and   decision == -1:# and imf_power_decision == -1:
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
            self.holdtime = 1
            print "ma5 %s"%ma5
            print "delta ma5 %s"%deltama5
            print "ma10 %s"%ma10
            print "delta ma10 %s"%deltama10
            print "ma20 %s"%ma20
            print "delta ma20 %s"%deltama20
            print "ma30 %s"%ma30
            print "delta ma30 %s"%deltama30
        else:
            self.last_buy_flag = 0
  
       # self.last_imf_power = imf_power_current
       # self.last_imf_3_power = imf_3_power_current
  
        self.total_asset_list.append(self.total_asset)
        print "total asset is %s"%(self.total_asset)
        print "money is %s"%(self.money)
        print "share is %s"%(self.share)

    def span_estimate(self, data):
        return (round(np.mean(data)-np.std(data),0),round(np.mean(data)+np.std(data),0))

    def format_data(self, data):
        if isinstance(data,list):
            return [2*((float(data[i])-min(data))/(max(data)-float(min(data))))-1 for i in range(len(data))]



    def run(self,emd_data, datafile,current_index,date):
        starttime = datetime.datetime.now()
        self.clear_emd_data()
        tmp = [i*10 for i in emd_data]
        emd_data = tmp
        my_emd = one_dimension_emd2(emd_data)
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



        self.clear_nn_train_data()
        raw_file = "imf2"
        train_data_all = "train_data_all"
        train_data_dir = "../../train_data"
        group_number = 50
        train_number = 50
        result_number = 10
        genfunc = genTraindata(raw_file,train_data_all,train_data_dir,group_number,train_number,result_number)
        genfunc.generate_all()
        genfunc.generate_divide()
 
        print "\n\n\n"
        self.run_predict(imf,current_index,date,datafile,train_data_dir,train_number,20,result_number)
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
        
        

    def draw_fig(self,datafile,start,save=0):
        
        data = self.close_price
        my_emd = one_dimension_emd(data)
        (imf, residual) = my_emd.emd()
        for i in range(len(imf)):
            fp = open("imf%s"%i,'w')
            for j in range(len(imf[i])):
                fp.writelines(str(imf[i][j]))
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

        #print lines

        imf_raw2 = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            imf_raw2.append(float(eachline.split("\n")[0]))

        fp = open("imf3")
        lines = fp.readlines()
        fp.close()

        #print lines

        imf_raw3 = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            imf_raw3.append(float(eachline.split("\n")[0]))


    #    fp = open("residual")
    #    lines = fp.readlines()
    #    fp.close()
    #    residual_raw = []
    #    for eachline in lines:
    #    #    print eachline.split(" ")[0]
    #        residual_raw.append(float(eachline.split("\n")[0]))
    #    imf_without6 = [imf_without2[i]-residual_raw[i] for i in range(len(emd_data))]
        imf = imf_raw2


        fp = open("residual",'r')
        lines = fp.readlines()
        fp.close()

        residual = []
        for eachline in lines:
            residual.append(float(eachline.split("\n")[0]))

        
        print "buy x index%s"%self.buy_x_index 
        print "sell x index%s"%self.sell_x_index 
        imf_buy_value = [imf[i] for i in self.buy_x_index]
        imf_sell_value = [imf[i] for i in self.sell_x_index]
      #  power_buy_value = [self.imf_power[i-start] for i in self.buy_x_index]
      #  power_sell_value = [self.imf_power[i-start] for i in self.sell_x_index]
      #  power_buy_index = [i-start for i in self.buy_x_index]
      #  power_sell_index = [i-start for i in self.sell_x_index]
        #imf_buy_value = [imf[i] for i in self.period_decision_buy_point]
        #imf_sell_value = [imf[i] for i in self.period_decision_sell_point]
        #self.buy_x_index = self.period_decision_buy_point
        #self.sell_x_index = self.period_decision_sell_point
      #  buy_x = [i-start for i in self.buy_x_index]
      #  sell_x = [i-start for i in self.sell_x_index]
        plt.figure(1)
        plt.subplot(211).axis([start,len(data),min(data[start:]),max(data[start:])])      
        plt.plot([i for i in range(len(data))],data,'o',[i for i in range(len(data))],data,'b',self.buy_x_index,self.buy_y_index,'r*',self.sell_x_index,self.sell_y_index,'g*')             
        plt.subplot(212).axis([start,len(imf),min(imf[start:]),max(imf[start:])])
        plt.plot([i for i in range(len(imf))],imf,'o',[i for i in range(len(imf))],imf,'b',self.buy_x_index,imf_buy_value,'r*',self.sell_x_index,imf_sell_value,'g*')             
        figname = "fig1_"+datafile         
        if save==1:
            savefig(figname)
        plt.figure(2)
        plt.plot([i for i in range(len(self.total_asset_list))],self.total_asset_list,'b')
        figname = "fig2_"+datafile
      #  plt.figure(3)
      #  plt.plot([i for i in range(len(self.imf_power))],self.imf_power,'b',power_buy_index,power_buy_value,'r*',power_sell_index,power_sell_value,'g*')
       
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
    date = [] 
    macd = []
    vol = []
    for eachline in lines:
        eachline.strip()
        close_price.append(float(eachline.split("\t")[4]))
        macd.append(float(eachline.split("\t")[6]))
        open_price.append(float(eachline.split("\t")[1]))
        date.append(eachline.split("\t")[0])
        vol.append(float(eachline.split("\t")[5]))
    

    process = process_util(open_price,close_price,macd,date,vol)
    os.popen("rm -r ../../emd_data")
    os.popen("mkdir ../../emd_data")
    generate_emd_func = generateEMDdata(datafile,begin,500)
    generate_emd_func.generate_emd_data_fix()
    print "begin %s"%begin
    for i in range(begin,begin+int(process.file_count("../../emd_data"))-1):
        emd_data = []
        print "emd file %s"%i
        fp = open("../../emd_data/emd_%s"%i,'r')
        lines = fp.readlines()
        fp.close()
        for eachline in lines:
            eachline.strip("\n")
            emd_data.append(float(eachline))
        process.run(emd_data,datafile,i-1,date[-1])
#    process.peilv()
#    process.hold_time()
    process.draw_fig(datafile,begin)

    

            
