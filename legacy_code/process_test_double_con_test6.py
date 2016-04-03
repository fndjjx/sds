import os
from svm_uti import *
import copy
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from emd import *
from smooth import cubicSmooth5
import datetime
from scipy.signal import argrelextrema
import numpy as np
from generate_emd_data import generateEMDdata
from generate_train_file import genTraindata
from spline_predict import splinepredict
from spline_predict import splinerestruct
from spline_predict import splinerestruct2
from fftfilter2 import fftfilter
from eemd import *
from analyze import *
from macd import ema





class process_util():

    def __init__(self,open_price,close_price,macd,date,vol,high_price,low_price):
        
        self.money = 10000
        self.share = 0
        self.sell_x_index = []
        self.sell_y_index = []
        self.buy_x_index = []
        self.buy_y_index = []
        self.open_price = open_price
        self.high_price = high_price
        self.low_price = low_price
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


        self.wait_flag = 0
        self.wait_count = 0

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


        self.last_small_final = 0
        self.buy_day = 0
        self.buy_decision = 0
        self.buy_count = 0
        self.jiange =0


        self.sell_decision = 0
        self.last_max_index = []
        self.last_min_index = []
        self.count = 0

        self.ma_count = 0
        self.ma_count1 = 0
        self.ma_count2 = 0
        self.ma_count3 = 0
        self.ma_count10 = 0
        self.last_buy_result=0


        self.imf_std = []
        self.ren_std = []
        self.list_std = []




        self.fluct_flag=0
        self.down_decision=0

        self.down_test = []
        self.down_test1 = []


        self.ma_record = []
        self.core=0
        self.core2=0
        self.core3=0
        self.imf3_core=0
        self.imf3_core2=0
        self.imf3_core3=0
        self.imf1_core=0
        self.imf2_core=0
        self.buy_core=0
        self.last_last=""
        self.imf1_last_last=""
        self.imf1_last=0
        self.imf2_last=0
        self.fail_flag=0

        self.qiangshi_index=[]

        self.buy_mean_price = 0
        self.result_list = []
        self.result_list2 = []
    def imf_percentage(self, imf, residual, source_data):
        sum_list = []
        source_data_without_residual = []
        for i in range(len(imf)):
            sum_list.append(sum([j*j for j in imf[i]]))
        sum_list.append(sum([j*j for j in residual]))
        source_data_without_residual = [source_data[i]-residual[i] for i in range(len(source_data))]
        source_square_sum = sum([j*j for j in source_data_without_residual])
        return [i/source_square_sum for i in sum_list]

        
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

    def generate_imf_extrem(self,residual,expect_next,first_distance):
    
    
        process_data = residual
        tmp = None
        spline_flag = 1
        raw_data_length = len(residual)
        count = 0
        print "residual  max %s"%list(argrelextrema(np.array(residual[:]),np.greater)[0])
        print "residual  min %s"%list(argrelextrema(np.array(residual[:]),np.less)[0])
        if (len(list(argrelextrema(np.array(residual[:]),np.greater)[0]))+ len(list(argrelextrema(np.array(residual[:]),np.less)[0])))<2:
            return []

        splinedata = list(np.array(splinerestruct(residual,3)))
        print "residual spline max %s"%list(argrelextrema(np.array(splinedata[:]),np.greater)[0])
        print "residual spline min %s"%list(argrelextrema(np.array(splinedata[:]),np.less)[0])
        #if len(list(argrelextrema(np.array(splinedata[:]),np.greater)[0]))==0  and len(list(argrelextrema(np.array(splinedata[:]),np.less)[0]))==0: 
        #    return []
        process_data = splinedata

        while(not tmp and count<=1 ):#and tmp != -1):
#            tmp = self.process_list(process_data,expect_next,spline_flag,raw_data_length,first_distance)
#            if tmp == []:
#                count = 10
            delta_residual = [process_data[i+1]-process_data[i] for i in range(0,len(process_data)-1)]
            process_data = delta_residual
            tmp = self.process_list(process_data,expect_next,spline_flag,raw_data_length,first_distance)
            
            #process_data = delta_residual
            spline_flag += 1
            count += 1
    
        return tmp


    def vol_cal(self,start,_from,_to):
        
        vol_list = self.vol[start+_from:start+_to]
        return sum(vol_list)

    def process_list(self,residual,expect_next,spline_flag,raw_data_length,first_distance):
    
        print residual
        print "enter process"
        okflag = None
        if spline_flag > 1:
            print "spline"
            data = list(np.array(splinerestruct(residual,3)))
        else:
            data = residual
        delta_residual_max = list(argrelextrema(np.array(data[:]),np.greater)[0])
        delta_residual_min = list(argrelextrema(np.array(data[:]),np.less)[0])
        print "process data max %s"%delta_residual_max
        print "process data min %s"%delta_residual_min
    
        tmp = delta_residual_max+delta_residual_min
        tmp.sort()
        print tmp
        
        tmp = filter(lambda x:x>9 and x<raw_data_length-2, tmp)
        #tmp = filter(lambda x:x>9, tmp)
        print tmp
   #     if tmp == []:
   #         return -1
        if len(tmp)>0:
          #  if (tmp[0] in delta_residual_min and expect_next == "max") or (tmp[0] in delta_residual_max and expect_next == "min"):
          #      tmp.pop(0)
            if (len(tmp)<=3 and len(tmp)>0) and ((tmp[0] in delta_residual_min and expect_next == "min") or (tmp[0] in delta_residual_max and expect_next == "max")):
                okflag = 1
                list_delta = [tmp[i+1]-tmp[i] for i in range(len(tmp)-1)]
                for i in list_delta:
                    if i<5:
                        okflag = 0
                if tmp[0]>15:
                    okflag = 0
        print tmp
        if okflag == 1:
            return tmp
        else:
            return None



    def run_predict1(self,imf_list,current_price_index,date,datafile,train_data_dir,train_number,hidden_number,result_number,residual):

        decision = 0
        power_decision = 0
        imf_power_decision = 0
        period_decision = 0
        extrem_decision = 0
        current_rise_flag = 0

        


      
        process_data = self.close_price[current_price_index-99:current_price_index+1]
        data = np.array(process_data)
        max_index = list(argrelextrema(data,np.greater)[0])
        min_index = list(argrelextrema(data,np.less)[0])
        tmp=[]
        tmp=max_index+min_index
        tmp.sort()
        trust_position = tmp[-3]
        trust_position1 = tmp[-3]


##############################################################

        imf_process = list(residual)
        imf_last = list(imf_list[-1])

        if len(imf_list)<9:
            big_imf = list(imf_list[-1])
        else:
            big_imf = list(imf_list[-1])
        if len(imf_list)>1:
            small_imf = list(imf_list[1])
        else:
            small_imf = list(imf_list[0])

        small_small_imf = list(imf_list[0])

        small_small_data = np.array(small_small_imf)
        small_small_imf_max_index = list(argrelextrema(small_small_data,np.greater)[0])
        print " imf0 max index %s"%small_small_imf_max_index
        small_small_imf_min_index = list(argrelextrema(small_small_data,np.less)[0])
        print " imf0 min index %s"%small_small_imf_min_index

        imf0_flag=0
        if small_small_imf_min_index[-1]>small_small_imf_max_index[-1] and small_small_imf_min_index[-1]>=96:#and self.imf1_last_last=="max":
            imf0_flag=1
        if small_small_imf_min_index[-1]<small_small_imf_max_index[-1] and small_small_imf_max_index[-1]>96:#and self.imf1_last_last=="max":
            imf0_flag=2



        big_data = np.array(big_imf[:40])
        big_data = np.array(residual)
        imf3_max_index = list(argrelextrema(big_data,np.greater)[0])
        print "big imf max index %s"%imf3_max_index
        imf3_min_index = list(argrelextrema(big_data,np.less)[0])
        print "big imf min index %s"%imf3_min_index
        print "\n"

        small_data = np.array(small_imf)
        small_imf_max_index = list(argrelextrema(small_data,np.greater)[0])
        print "small imf max index %s"%small_imf_max_index
        small_imf_min_index = list(argrelextrema(small_data,np.less)[0])
        print "small imf min index %s"%small_imf_min_index
        imf2_data = np.array(imf_process)
        imf2_max_index = list(argrelextrema(imf2_data,np.greater)[0])
        print " imf2 max index %s"%imf2_max_index
        imf2_min_index = list(argrelextrema(imf2_data,np.less)[0])
        print " imf2 min index %s"%imf2_min_index
        tmp=[]
        tmp=small_imf_max_index+small_imf_min_index
        tmp.sort()
        if tmp[-2]<90:
            trust_position = tmp[-1]
        else:
            trust_position = tmp[-2]


        if 1:#self.buy_price==0:
            if 1:#tmp[-2]>=95:
                trust_position = tmp[-2]
                trust_position = 499
                imf2_data = np.array(imf_process[:trust_position+1])
                imf2_max_index = list(argrelextrema(imf2_data,np.greater)[0])
                print " imf2 max index %s"%imf2_max_index
                imf2_min_index = list(argrelextrema(imf2_data,np.less)[0])
                print " imf2 min index %s"%imf2_min_index
            #elif tmp[-2]<95 and tmp[-1]-tmp[-2]>=5:
            #    imf2_max_index = filter(lambda n:n<=tmp[-2],small_imf_max_index)
            #    print " imf2 max index %s"%imf2_max_index
            #    imf2_min_index = filter(lambda n:n<=tmp[-2],small_imf_min_index)
            #    print " imf2 min index %s"%imf2_min_index
            else:
                trust_position = tmp[-1]
                trust_position = 95
                imf2_data = np.array(imf_process[:trust_position+1])
                imf2_max_index = list(argrelextrema(imf2_data,np.greater)[0])
                print " imf2 max index %s"%imf2_max_index
                imf2_min_index = list(argrelextrema(imf2_data,np.less)[0])
                print " imf2 min index %s"%imf2_min_index
        else:
            imf2_data = np.array(imf_process)
            imf2_max_index = list(argrelextrema(imf2_data,np.greater)[0])
            print " imf2 max index %s"%imf2_max_index
            imf2_min_index = list(argrelextrema(imf2_data,np.less)[0])
            print " imf2 min index %s"%imf2_min_index


        

        small_imf_max_true_index = filter(lambda n:n<99, small_imf_max_index)
        small_imf_min_true_index = filter(lambda n:n<99, small_imf_min_index)
        if small_imf_max_true_index[-1]>small_imf_min_true_index[-1]:
            current_imf1_core = current_price_index-(99-small_imf_max_true_index[-1])
            current_imf1_sec_core = current_price_index-(99-small_imf_min_true_index[-1])
        else:
            current_imf1_core = current_price_index-(99-small_imf_min_true_index[-1])
            current_imf1_sec_core = current_price_index-(99-small_imf_max_true_index[-1])
        print "\n\n\n"
        print "imf1 core last %s sec %s fir %s"%(self.imf1_core,current_imf1_sec_core,current_imf1_core)
        imf1_flag=0
        if small_imf_max_true_index[-1]>small_imf_min_true_index[-1]:
            imf1_flag=1
        if small_imf_max_true_index[-1]<small_imf_min_true_index[-1]:
            imf1_flag=2
        small_imf_max_true_index = filter(lambda n:n<99, small_imf_max_index)
        small_imf_min_true_index = filter(lambda n:n<99, small_imf_min_index)


        imf1_flag=0
        if small_imf_min_index[-1]>small_imf_max_index[-1] and (((46<small_imf_min_index[-1] and small_imf_max_index[-1]<45)) or (46==small_imf_min_index[-1] and small_imf_max_index[-1]<49))   :#and self.imf1_last_last=="max":
            imf1_flag=1
        if small_imf_min_index[-1]<small_imf_max_index[-1] and small_imf_max_index[-1]>=46:#and self.imf1_last_last=="max":
            imf1_flag=2
        

        residual_flag=0

        last_data = np.array(imf_list[-1][:30])
        imf_last_max_index = list(argrelextrema(last_data,np.greater)[0])
        print "last imf max index %s"%imf_last_max_index
        imf_last_min_index = list(argrelextrema(last_data,np.less)[0])
        print "last imf min index %s"%imf_last_min_index
        if imf_last_max_index!=[] and imf_last_min_index!=[]:
            tmp = []
            tmp = imf_last_max_index+imf_last_min_index
            tmp.sort()
            if tmp[-1]-tmp[-2]<5:
                tmp.pop()
            
            if (tmp[-1] in imf_last_min_index) and tmp[-1]>910:
                residual_flag=1
            if (tmp[-1] in imf_last_max_index) and 0.85<((99-tmp[-1])/float(tmp[-1]-tmp[-2]))<1.2:
                residual_flag=1


        #if small_imf_min_true_index[-1]>96 and small_imf_min_true_index[-1]>small_imf_max_true_index[-1] and abs(current_imf1_sec_core-current_imf1_core)>2:#and self.imf1_last_last=="max":
        #if abs(abs(small_imf_max_true_index[-1]-small_imf_min_true_index[-1])-(99-small_imf_max_true_index[-1]))<2 and small_imf_min_true_index[-1]<small_imf_max_true_index[-1] :#and abs(current_imf1_sec_core-current_imf1_core)>2:#and self.imf1_last_last=="max":
#        if small_imf_min_true_index[-1]>96 and small_imf_min_true_index[-1]>small_imf_max_true_index[-1] and abs(current_imf1_sec_core-current_imf1_core)>5:#and self.imf1_last_last=="max":
#            imf1_flag=1
        power_0 = self.close_price[current_price_index]-self.ma(self.close_price, 5, 0, current_price_index)
        power_1 = self.close_price[current_price_index-1]-self.ma(self.close_price, 5, -1, current_price_index)
        power_2 = self.close_price[current_price_index-2]-self.ma(self.close_price, 5, -2, current_price_index)
        power_3 = self.close_price[current_price_index-3]-self.ma(self.close_price, 5, -3, current_price_index)
        power_9 = self.close_price[current_price_index-9]-self.ma(self.close_price, 5, -9, current_price_index)
        power_5 = self.close_price[current_price_index-5]-self.ma(self.close_price, 5, -5, current_price_index)
        power_list = [power_1,power_2,power_3,power_9,power_5]
        count=0
        for i in power_list:
            if i<0:
                count+=1
        result_flag=0
        result_flag2=0
        #if len(self.result_list)>=5 and self.result_list[-1]!=2 and self.result_list[-2]!=2 and self.result_list[-3]!=2 and self.result_list[-9]!=2 and self.result_list[-5]!=2:
        if len(self.result_list)>=2 and self.result_list[-1]!=2 and self.result_list[-2]!=2 :#and self.result_list[-3]!=2:
            result_flag=1
        if len(self.result_list2)>=2 and self.result_list2[-1]!=2 and self.result_list2[-2]!=2:
            result_flag2=1
        print "result list%s"%(self.result_list[-10:])
        print "result list%s"%(self.result_list2[-10:])
        print "data %s"%str(self.close_price[current_price_index-8:current_price_index+1])
        print "mean 5 %s %s"%(self.ma(self.close_price, 5, -1, current_price_index),self.ma(self.close_price, 5, 0, current_price_index) )
        print "imf process%s"%imf_process[-20:]
        imf2_flag=0
        imf3_flag=0
        #if imf2_max_index[-1]>imf2_min_index[-1] and 985<=imf2_max_index[-1]<95 and (self.imf2_last=="max" or abs(current_imf2_sec_core-self.imf2_core)<=3) and ((power_0>0 and count>2 ) or (power_0>0 and power_1>0 and count>2) or (power_0>0 and power_1>0 and power_2>0 and power_3<0 and power_9<0)) and abs(abs(current_imf2_sec_core)-abs(current_imf2_core))>3 and result_flag==1:
        #if flag==0 and imf3_max_index[-1]>imf3_min_index[-1] and 980<=imf3_max_index[-1]<90 and  ((power_0>0 and count>2 ) or (power_0>0 and power_1>0 and count>2) or (power_0>0 and power_1>0 and power_2>0 and power_3<0 and power_9<0))  and result_flag==1:

        if imf2_max_index!=[] and imf2_min_index!=[] and imf2_max_index[-1]>imf2_min_index[-1] and 35<imf2_max_index[-1]<45 and result_flag==1 :# and result_flag==1 :#and result_flag==1 :#and self.imf1_last_last=="max":
            imf2_flag=1
        #if imf3_max_index[-1]>imf3_min_index[-1] and 975<imf3_max_index[-1]<=90 :#and result_flag2==1:# and result_flag2==1:#and result_flag2==1 :#and self.imf1_last_last=="max":
        if imf2_max_index!=[] and imf2_min_index!=[] and imf2_max_index[-1]<imf2_min_index[-1] and 35<imf2_min_index[-1]<45 :#and  self.close_price[current_price_index]<self.ma(self.close_price, 5, 0, current_price_index):#and self.imf1_last_last=="max":
            imf2_flag=2
        if imf2_max_index!=[] and imf2_min_index!=[] and imf2_max_index[-1]>imf2_min_index[-1] and imf2_max_index[-1]>45 and self.result_list[-1]!=1:#and  self.close_price[current_price_index]<self.ma(self.close_price, 5, 0, current_price_index):#and self.imf1_last_last=="max":
            imf2_flag=2
        #if imf3_max_index[-1]<imf3_min_index[-1] and 975<imf3_min_index[-1]<=90:#and self.imf1_last_last=="max":
        #if imf2_max_index[-1]>imf2_min_index[-1] and imf2_max_index[-1]>=96 and abs(current_imf2_core-self.buy_core)>3 :#and abs(imf2_max_index[-1]-imf2_min_index[-1])>=6:#and self.imf1_last_last=="max":
        #    imf1_flag=2
        #if imf2_max_index[-1]<imf2_min_index[-1] and imf2_min_index[-1]<95 :#and abs(current_imf2_core-self.buy_core)>3 :#and abs(imf2_max_index[-1]-imf2_min_index[-1])>=6:#and self.imf1_last_last=="max":
        #    imf1_flag=2
        #if small_imf_min_true_index[-1]<small_imf_max_true_index[-1] and 0.8<((small_imf_max_true_index[-1]-small_imf_min_true_index[-1])/float(99-small_imf_max_true_index[-1]))<1.2 and abs(99-current_imf1_core)>2:
        #    imf1_flag=1
        if 0:#abs(current_imf1_sec_core-current_imf1_sec_core)<2:
            imf1_flag=0

        #if small_imf_max_true_index[-1]>=96 and small_imf_max_true_index[-1]>small_imf_min_true_index[-1] :#and self.imf1_last_last=="min":
        #    imf1_flag=2
        #if small_imf_max_true_index[-1]<small_imf_min_true_index[-1] and 0.75<((small_imf_min_true_index[-1]-small_imf_max_true_index[-1])/float(99-small_imf_min_true_index[-1]))<1.3 and imf0_flag==2:
         #   imf1_flag=2
#        if small_imf_max_true_index[-1]<small_imf_min_true_index[-1] and small_imf_min_true_index[-1]<95 and imf0_flag==2:
#            imf1_flag=2

        print abs(self.imf1_core-current_imf1_sec_core)
        print "imf1 flag%s"%imf1_flag

        self.result_list.append(imf2_flag)
        self.result_list2.append(imf3_flag)

        
        return (imf2_flag,imf1_flag)
##########################



###########################
        
    def run_predict(self,imf_list,current_price_index,date,datafile,train_data_dir,train_number,hidden_number,result_number,residual,imf_sec,imf_third):
        result1 = []
        result2 = []
        (r1,r2)=self.run_predict1(imf_list,current_price_index,date,datafile,train_data_dir,train_number,hidden_number,result_number,residual)
        result1.append(r1)
        result2.append(r2)

#        (r1,r2)=self.run_predict1(imf_sec,current_price_index,date,datafile,train_data_dir,train_number,hidden_number,result_number,residual)
#        result1.append(r1)
#        result2.append(r2)

#        (r1,r2)=self.run_predict1(imf_third,current_price_index,date,datafile,train_data_dir,train_number,hidden_number,result_number,residual)
#        result1.append(r1)
#        result2.append(r2)


        imf0_flag=0
        print "result1 %s"%result1
        print "result2 %s"%result2
        result1_count=0
        result1_count2=0
        result2_count=0
        result2_count2=0
        for i in result1:
            if i==1:
                result1_count+=1

        for i in result1:
            if i==2:
                result1_count2+=1

        for i in result2:
            if i==1:
                result2_count+=1
        for i in result2:
            if i==2:
                result2_count2+=1


        #if result1[0]==1 and result1[1]==1 :
        if result1_count2<2:
            imf0_flag=1

        if result1_count<2:
            imf0_flag=2

        imf1_flag=0
        #if result2[0]==1 and result2[1]==1 :
        if result2_count>=2 and result2_count2==0:
            imf1_flag=1

        #if result2[0]==2 or result2[1]==2 or result2[2]==2 :
        if result2_count2>=2:
            imf1_flag=2

#######
        imf0_flag=0
        imf1_flag=0
        imf2_flag=0
        imf3_flag=0
      
        if result1[0]==1:
            imf2_flag=1
        if result1[0]==2:
            imf2_flag=2
        if result2[0]==1:
            imf1_flag=1
        if result2[0]==2:
            imf1_flag=2

######

        print "imf1_ flag %s"%imf1_flag
        print "imf0_ flag %s"%imf0_flag


####################
        tmp = self.close_price 

        p_mean = []
        for i in range(2):
            p_mean.append(0)

        for i in range(3,len(self.close_price)+1):
            mean = np.mean(self.close_price[i-2:i+1])
            p_mean.append(mean)

        v_mean = []
        for i in range(2):
            v_mean.append(0)

        for i in range(3,len(self.vol)+1):
            mean = np.mean(self.vol[i-2:i+1])
            v_mean.append(mean)

        self.close_price = [p_mean[i]*v_mean[i] for i in range(len(tmp))]
        self.close_price = tmp
        ma5 = self.ma(self.close_price, 5, 0, current_price_index)
        deltama5 = self.ma(self.close_price, 5, 0, current_price_index)-self.ma(self.close_price, 5, -1, current_price_index)
        deltama5_before = self.ma(self.close_price, 5, -1, current_price_index)-self.ma(self.close_price, 5, -2, current_price_index)
        deltama5_before2 = self.ma(self.close_price, 5, -2, current_price_index)-self.ma(self.close_price, 5, -3, current_price_index)
        deltama3 = self.ma(self.close_price, 3, 0, current_price_index)-self.ma(self.close_price, 3, -1, current_price_index)
        deltama3_before = self.ma(self.close_price, 3, -1, current_price_index)-self.ma(self.close_price, 3, -2, current_price_index)
        deltama3_before2 = self.ma(self.close_price, 3, -2, current_price_index)-self.ma(self.close_price, 3, -3, current_price_index)
        deltama9 = self.ma(self.close_price, 9, 0, current_price_index)-self.ma(self.close_price, 9, -1, current_price_index)
        deltama9_before = self.ma(self.close_price, 9, -1, current_price_index)-self.ma(self.close_price, 9, -2, current_price_index)
        deltama9_before2 = self.ma(self.close_price, 9, -2, current_price_index)-self.ma(self.close_price, 9, -3, current_price_index)
        ma10 = self.ma(self.close_price, 10, 0, current_price_index)
        deltama10 = self.ma(self.close_price, 10, 0, current_price_index)-self.ma(self.close_price, 10, -1, current_price_index)
        ma20 = self.ma(self.close_price, 25, 0, current_price_index)
        deltama20 = self.ma(self.close_price, 20, 0, current_price_index)-self.ma(self.close_price, 20, -1, current_price_index)
        deltama25 = self.ma(self.close_price, 25, 0, current_price_index)-self.ma(self.close_price, 25, -1, current_price_index)
        deltama25_before = self.ma(self.close_price, 25, -1, current_price_index)-self.ma(self.close_price, 25, -2, current_price_index)
        deltama25_before2 = self.ma(self.close_price, 25, -2, current_price_index)-self.ma(self.close_price, 25, -3, current_price_index)
        deltama25_before3 = self.ma(self.close_price, 25, -3, current_price_index)-self.ma(self.close_price, 25, -9, current_price_index)
        ma30 = self.ma(self.close_price, 30, 0, current_price_index)
        deltama30 = self.ma(self.close_price, 30, 0, current_price_index)-self.ma(self.close_price, 30, -1, current_price_index)

        self.close_price = tmp


###################


        last_50_price = self.close_price[current_price_index-10:current_price_index+1]
        print "50 std %s"%np.std(last_50_price)
        print "std/mean %s"%(np.std(last_50_price)/np.mean(last_50_price))
        self.imf_std.append((np.std(last_50_price)/np.mean(last_50_price)))
        std_mean = (np.std(last_50_price)/np.mean(last_50_price))
        mean_std_mean = (np.mean(self.imf_std))
        std_std_mean = (np.std(self.imf_std))
        


        current_price=self.close_price[current_price_index]
        if ma5>ma10:
            if self.ma_count<=0:
                self.ma_count = 0
                self.ma_count += 1
            else:
                self.ma_count += 1
        else:
            if self.ma_count>=0:
                self.ma_count =0
                self.ma_count -=1
            else:
                self.ma_count -=1
        print "self.ma_count %s"%self.ma_count
        print " %s"%(ma5-ma10)

        if current_price>ma10:
            if self.ma_count10<=0:
                self.ma_count10 = 0
                self.ma_count10 += 1
            else:
                self.ma_count10 += 1
        else:
            if self.ma_count10>=0:
                self.ma_count10 =0
                self.ma_count10 -=1
            else:
                self.ma_count10 -=1
        print "self.ma_count10 %s"%self.ma_count10
        print " %s"%(current_price-ma10)


        if ma10>ma20:
            if self.ma_count2<=0:
                self.ma_count2 = 0
                self.ma_count2 += 1
            else:
                self.ma_count2 += 1
        else:
            if self.ma_count2>=0:
                self.ma_count2 =0
                self.ma_count2 -=1
            else:
                self.ma_count2 -=1
        print "self.ma_count2 %s"%self.ma_count2
        print " %s"%(ma10-ma20)
        if current_price>ma5:
            if self.ma_count1<=0:
                self.ma_count1 = 0
                self.ma_count1 += 1
            else:
                self.ma_count1 += 1
        else:
            if self.ma_count1>=0:
                self.ma_count1 =0
                self.ma_count1 -=1
            else:
                self.ma_count1 -=1
        print "self.ma_count1 %s"%self.ma_count1
        print " %s"%(current_price-ma5)

        print "self.ma_count %s"%self.ma_count
        print " %s"%(ma5-ma10)


        print "self.ma_count2 %s"%self.ma_count2
        print " %s"%(ma10-ma20)

        if ma20>ma30:
            if self.ma_count3<=0:
                self.ma_count3 = 0
                self.ma_count3 += 1
            else:
                self.ma_count3 += 1
        else:
            if self.ma_count3>=0:
                self.ma_count3 =0
                self.ma_count3 -=1
            else:
                self.ma_count3 -=1
        print "self.ma_count3 %s"%self.ma_count3
        print " %s"%(ma20-ma30)
        ma_decision10=0
        if self.ma_count2>10 and self.ma_count2>self.ma_count3 and (self.ma_count1>self.ma_count ) :
            ma_decision10=1
########################
            


         
######################


        std_list_5 = self.close_price[current_price_index-9:current_price_index+1]
        std_list_10 = self.close_price[current_price_index-9:current_price_index+1]
        std_list_before = self.close_price[current_price_index-9:current_price_index]
        std_list_20 = self.close_price[current_price_index-19:current_price_index+1]
        std_list_20_before = self.close_price[current_price_index-20:current_price_index]
        std_list_20_before2 = self.close_price[current_price_index-21:current_price_index-1]
        std_list_20_before3 = self.close_price[current_price_index-22:current_price_index-2]
        std_list_20_before9 = self.close_price[current_price_index-23:current_price_index-3]
        print "std 5 %s"%np.std(std_list_5)
        print "std 5 %s"%std_list_5
        print "std 5 before %s"%np.std(std_list_before)
        print "std 20  %s"%np.std(std_list_20)
        print "std 20 before %s"%np.std(std_list_20_before)
        print "std 20 before2 %s"%np.std(std_list_20_before2)
        print "std 20 delta %s"%(np.std(std_list_20)-np.std(std_list_20_before))
        print "std 20 delta 2%s"%(np.std(std_list_20_before)-np.std(std_list_20_before2))


###################        

        print "date %s"%self.date[current_price_index]
        print "date %s"%date
        print "current price %s"%self.close_price[current_price_index]
        print "trade price %s"%self.open_price[current_price_index+1]
        print "buy price %s"%self.buy_price
###########################################################################################
   
        recent_price_list = self.close_price[current_price_index-9:current_price_index+1]

        recent_price_format = self.format_data(recent_price_list)
        recent_price_format2 = recent_price_format[-5:] 
  


        print "recent price format %s"%recent_price_format2


        


        imf = list(imf_list[-1])  
  
        imf_below_0_decision = 0
        print "last imf %s"%imf[-1]
        print "last imf period %s"%np.mean(imf[-3:])
        if imf[-1]<0:
            imf_below_0_decision = -1



        imf_delta3 = imf[-9]-imf[-3]
        imf_delta2 = imf[-2]-imf[-3]
        imf_delta1 = imf[-1]-imf[-2]
        print "imf delta %s"%imf_delta1
        print "imf delta2 %s"%imf_delta2
        print "imf delta3 %s"%imf_delta3
        imf_ma5_before_before_before = (imf[-8]+imf[-7]+imf[-6]+imf[-5]+imf[-9])/5.0
        imf_ma5_before_before = (imf[-7]+imf[-6]+imf[-5]+imf[-9]+imf[-3])/5.0
        imf_ma5_before = (imf[-6]+imf[-5]+imf[-9]+imf[-3]+imf[-2])/5.0
        imf_ma3_before = (imf[-9]+imf[-3]+imf[-2])/3.0

        imf_ma5_current = (imf[-5]+imf[-9]+imf[-3]+imf[-2]+imf[-1])/5.0
        imf_ma3_current = (imf[-3]+imf[-2]+imf[-1])/3.0
        imf_power_before_before_before = imf[-9] - imf_ma5_before_before_before
        imf_power_before_before = imf[-3] - imf_ma5_before_before
        imf_power_before = imf[-2] - imf_ma5_before
        imf_3_power_before = imf[-2] - imf_ma3_before
        imf_power_current = imf[-1] - imf_ma5_current
        imf_3_power_current = imf[-1] - imf_ma3_current
        print "imf power before before before%s"%imf_power_before_before_before
        print "imf power before before%s"%imf_power_before_before
        print "imf power before %s"%imf_power_before
        print "imf power current %s"%imf_power_current
        print "imf delta %s"%(imf_power_before_before-imf_power_before_before_before)
        print "imf delta %s"%(imf_power_before-imf_power_before_before)
        print "imf delta %s"%(imf_power_current-imf_power_before)
        print "last imf power current %s"%self.last_imf_power
        print "imf 3 power before %s"%imf_3_power_before
        print "imf 3 power current %s"%imf_3_power_current
        print "last imf 3 power current %s"%self.last_imf_3_power
 
        last_5 = self.close_price[current_price_index-5:current_price_index]
        last_5_mean = np.mean(self.close_price[current_price_index-5:current_price_index])
        last_5_std = np.std([last_5[i]-last_5_mean for i in range(5)])
        print "std 5 %s"%last_5_std
        self.imf_power.append(imf_power_current)


        last_10 = self.close_price[current_price_index-6:current_price_index]

###############
        distance_decision = 1
        if len(self.buy_x_index)>1 and current_price_index-self.buy_x_index[-1]<2:
            distance_decision = 0

        distance_decision2 = 1
        if self.buy_price<current_price and len(self.buy_x_index)>0 and current_price_index-self.buy_x_index[-1]<9:
            distance_decision2 = 0

        distance_decision3 = 1
        if len(self.sell_x_index)>1 and current_price_index-self.sell_x_index[-1]<5:
            distance_decision3 = 0



##########
        price = self.close_price[current_price_index-9:current_price_index+1]
        price_ma5_before = (price[-6]+price[-5]+price[-9]+price[-3]+price[-2])/5.0
        price_ma5_before2 = (price[-7]+price[-6]+price[-5]+price[-9]+price[-3])/5.0
        price_ma3_before = (price[-9]+price[-3]+price[-2])/3.0
        price_ma5_current = (price[-5]+price[-9]+price[-3]+price[-2]+price[-1])/5.0
        price_ma3_current = (price[-3]+price[-2]+price[-1])/3.0

        power_before = price[-2]-price_ma3_before
        power_ma5_before = price[-2]-price_ma5_before
        power_ma5_before2 = price[-3]-price_ma5_before2
        print "power before %s"%power_before
        print "power ma5 before %s"%power_ma5_before
        power_current = price[-1]-price_ma3_current
        power_ma5_current = price[-1]-price_ma5_current
        print "power current %s"%power_current
        print "power ma5 current %s"%power_ma5_current

        delta_power_before = power_ma5_before-power_ma5_before2
        delta_power_current = power_ma5_current-power_ma5_before
        print "delta power before %s"%delta_power_before
        print "delta power current %s"%delta_power_current

        delta_power_decision = 0
        if delta_power_before<0 and delta_power_current>0:
            delta_power_decision = 1

        power_decision = 0
        if power_before<0 and power_current>0:
            power_decision = -1
        elif power_before>0 and power_current<0:
            power_decision = 1

        power_ma5_decision = 0
        if power_ma5_before<0 and power_ma5_current>0:
            power_ma5_decision = -1
        elif power_ma5_before>0 and power_ma5_current<0:
            power_ma5_decision = 1

        power_ma5_decision2 = 0
        if power_ma5_before<power_ma5_current:
            power_ma5_decision2 = -1
        print "power decision2 %s"%power_ma5_decision2



        price_delta_decision = 1
        price_delta = [(price[-1-i]-price[-i-2])/price[-i-2] for i in range(3)]
        print "price delta %s"%price_delta
        price_wrong1 = filter(lambda x:x<-0.02,price_delta)
        price_wrong2 = filter(lambda x:x<-0.03,price_delta)
        if len(price_wrong1)>=2 or len(price_wrong2)>1:
            price_delta_decision = 0
        print "price delta decision %s"%price_delta_decision



###################

###############
        price_50 = self.close_price[current_price_index-59:current_price_index+1]
        up_list = [price_50[i+1]-price_50[i] for i in range(59) if price_50[i]<price_50[i+1]]
        down_list = [price_50[i+1]-price_50[i] for i in range(59) if price_50[i]>price_50[i+1]]
        print up_list
        print down_list
        up_potion = abs((sum(up_list)/sum(down_list)))
        print "up number %s"%len(up_list)
        print "up potion %s"%(sum(up_list)/sum(down_list))
        up_decision1 = 1
        if self.ma_count<0 and up_potion>1.6:
            up_decision1 = 0
         
        
        sta_decision = 0
        if abs(sum(up_list)/sum(down_list))>0.7 and abs(sum(up_list)/sum(down_list))<1.9:
            sta_decision = 1
        
####################################################################################################################


        imf_decision = 0 
        data = np.array(last_5)
        last_5_max_index = argrelextrema(data,np.greater)[0]
        last_5_min_index = argrelextrema(data,np.less)[0]

        data = np.array(last_10)
        last_10_max_index = argrelextrema(data,np.greater)[0]
        last_10_min_index = argrelextrema(data,np.less)[0]


        last_10_imf = list(imf_list[-1])[-10:]
        print "last 10 price %s"%last_10


            
#####################################################################################################################

      

   
        ma_decision2 = 1
        if deltama5<0 and deltama10<0 and deltama20<0 and deltama30<0:
            ma_decision2 = 0
        
        print "ma5 %s"%ma5
        print "delta ma5 %s %s"%(deltama5_before,deltama5)
        print "delta ma3 before %s"%deltama3_before
        print "delta ma3 %s"%deltama3
        print "delta ma9 before %s %s"%(deltama9_before,deltama9)
        print "ma10 %s"%ma10
        print "delta ma10 %s"%deltama10
        print "ma20 %s"%ma20
        print "delta ma20 %s"%deltama20
        print "delta ma25 %s"%deltama25
        print "ma30 %s"%ma30
        print "delta ma30 %s"%deltama30
        





        if self.buy_price==0:
            trade_price = self.open_price[current_price_index+1]+0.1
            if trade_price>self.high_price[current_price_index+1]:
                trade_price = self.close_price[current_price_index]
        else:
            trade_price = self.open_price[current_price_index+1]-0.1
            if trade_price<self.low_price[current_price_index+1]:
                trade_price = self.close_price[current_price_index]
        current_price = self.close_price[current_price_index]





        print "macd %s"%self.macd[current_price_index-10:current_price_index+1]
        macd_list = self.macd[current_price_index-5:current_price_index+1] 
        macd_decision1 = 1
        if macd_list[-1]>0 and macd_list[-1]>macd_list[-2]>macd_list[-3]:
            macd_decision1 = 0

        macd_decision2 = 1
        delta_macd_0 = macd_list[-1]-macd_list[-2]
        delta_macd_1 = macd_list[-2]-macd_list[-3]
        delta_macd_2 = macd_list[-3]-macd_list[-4]
        delta_macd_3 = macd_list[-4]-macd_list[-5]
        if delta_macd_0>delta_macd_1:
            macd_decision2=0

############
#########



        up_ma_decision1=1
        if (up_potion>1.5 and self.ma_count1<0) or up_potion>2 :#or up_potion<0.55:
            up_ma_decision1=0

        ma_macd_decision1=0
        if ((self.ma_count<0 and self.ma_count1<0 and self.ma_count2<0 and self.ma_count3<0) or (0>self.ma_count>self.ma_count1) ) and macd_list[-1]>macd_list[-2]:
            ma_macd_decision1=1

        ma_decision9=1
        if (self.ma_count1<self.ma_count<0 and (delta_power_current<0 or delta_power_current<delta_power_before)) or (self.ma_count<self.ma_count2<-1 and self.ma_count1<1 and (delta_power_current<0 or delta_power_current<delta_power_before) ) or (0>self.ma_count3>self.ma_count2>self.ma_count):
        #if (self.ma_count1<self.ma_count<0 ) or (self.ma_count<self.ma_count2<-1 and self.ma_count1<1 and (delta_power_current<0 or delta_power_current<delta_power_before) ) or (0>self.ma_count3>self.ma_count2>self.ma_count):
            ma_decision9=0



        macd_decision3 = 0
        if  delta_macd_0>delta_macd_1:#>delta_macd_2  :#or macd_list[-1]>macd_list[-2]:#>macd_list[-3]:
        #if  macd_list[-1]>macd_list[-2] and abs(macd_list[-1])<1 and abs(macd_list[-2])<1:#>macd_list[-3]:
            macd_decision3=1

        macd_decision9 = 1
        if macd_list[-1]<-1 or (macd_list[-1]>1 and macd_list[-1]<macd_list[-2]):
            macd_decision9=0

        macd_decision6=1
        if macd_list[-1]<0 and macd_list[-2]>0 and macd_list[-3]>0: 
            macd_decision6=0

        macd_decision8=1
        if macd_list[-1]>macd_list[-2]:
            macd_decision8=0

        ma_decision6=1
        if  (self.ma_count1<self.ma_count<0 and self.ma_count3<0) or (self.ma_count1<self.ma_count<-2) or (self.ma_count1<self.ma_count<0 and self.ma_count2<0) or (delta_power_current<0 and self.ma_count1<self.ma_count  ) or (self.ma_count1<0 and self.ma_count<0 and self.ma_count2<0 and self.ma_count3<-10) or self.ma_count<self.ma_count2<0 or (self.ma_count1<self.ma_count and self.ma_count2<self.ma_count3<0 ):
            ma_decision6=0
        ma_decision7=0
        if self.ma_count>0 and self.ma_count2>5:
            ma_decision7=1



        macd_list_10 = self.macd[current_price_index-10:current_price_index+1]
        macd_max = list(argrelextrema(np.array(macd_list_10),np.greater)[0])
        macd_min = list(argrelextrema(np.array(macd_list_10),np.less)[0])
        macd_max.insert(0,0)
        macd_min.insert(0,0)
        print "macd_max%s"%macd_max
        print "macd_min%s"%macd_min
        macd_decision5=1
        if (self.ma_count>0 ) and 10-macd_max[-1]<8:
            macd_decision5=0
            print "10-macd %s"%(10-macd_max[-1])


         
        if self.wait_flag>0:
            self.wait_count+=1
        

        if self.wait_count>5:
            self.wait_flag=0
            self.wait_count=0


        sell_decision1 = 0
        #if self.buy_decision == -1 and this_buy==0 :#and current_price>self.close_price[current_price_index-1] :#and self.buy_count>2:#  (current_price-ma5) > (self.close_price[current_price_index-1]-self.ma(self.close_price, 5, -1, current_price_index)) and ( imf_extrem_decision6 == -1):
         #   buy_decision2 = -1

        
        delta_buy_decision1=1
        if self.last_buy_result==1 or self.last_buy_result==3 or self.last_buy_result==9:
            if current_price_index-self.sell_x_index[-1]<3:
                delta_buy_decision1=0

        if self.buy_price!=0:
            self.count +=1 

        rsi_decision1=1
        if self.rsi(self.close_price,10,current_price_index)>70:
            rsi_decision1=0

        open_close_decision1=0
        #if self.open_price[current_price_index]<current_price and self.open_price[current_price_index-1]<self.close_price[current_price_index-1] and self.open_price[current_price_index]>self.close_price[current_price_index-1]:
        current_high_price = self.high_price[current_price_index]
        current_low_price = self.low_price[current_price_index]
        current_open_price = self.open_price[current_price_index]
        #if current_price>self.open_price[current_price_index] or (current_price<current_open_price and (current_high_price-current_open_price)<(current_price-current_low_price)):#and trade_price>current_price:#>self.close_price[current_price_index-1]:
        if current_price>self.open_price[current_price_index] and ((current_high_price-current_price)<(current_open_price-current_low_price)):#and trade_price>current_price:#>self.close_price[current_price_index-1]:
            open_close_decision1=1

        ma_decision5=1
        #if self.ma_count1<0 and self.ma_count<0 and self.ma_count2<0 and self.ma_count3<0 and current_price<self.close_price[current_price_index-1]:
        #    ma_decision5=0

        up_potion_decision2=1
        if up_potion>1.5 and self.ma_count1<self.ma_count<0 :
            up_potion_decision2=0



        delta_macd1=self.macd[current_price_index]-self.macd[current_price_index-1]
        delta_macd2=self.macd[current_price_index-1]-self.macd[current_price_index-2]
        delta_macd3=self.macd[current_price_index-2]-self.macd[current_price_index-3]
        delta_macd9=self.macd[current_price_index-3]-self.macd[current_price_index-9]
        print "delta macd%s %s %s %s"%(delta_macd1,delta_macd2,delta_macd3,delta_macd9)



        current_mean_price = self.ma(self.close_price, 5, 0, current_price_index)
        print "buy mean%s current mean%s"%(self.buy_mean_price,current_mean_price)
#######



        ma_decision3 = 1
 
        #if  (self.ma_count<=self.ma_count2<0 and self.ma_count<=-5) or (self.ma_count2<=self.ma_count3<0 and self.ma_count<=-5):# and (np.corrcoef(macd_line,cri_sample)[0][1])<-0.8) :#or ((np.corrcoef(ma5_line,cri_sample)[0][1])<-0.9 and (np.corrcoef(macd_line,cri_sample)[0][1])<-0.9):
        if    (self.ma_count<self.ma_count2<0 and self.ma_count1<0 ):#and (np.corrcoef(macd_line,cri_sample)[0][1])<-0.8) :#or ((np.corrcoef(ma5_line,cri_sample)[0][1])<-0.9 and (np.corrcoef(macd_line,cri_sample)[0][1])<-0.9):
            ma_decision3 = 0
        print "ma_decision3 %s"%ma_decision3
        ma_decision9=0
        if deltama5>0 or self.ma_count>0 :
            ma_decision9=1
        print "ma_decision9 %s"%ma_decision9
        print "open%s close%s"%(self.open_price[current_price_index+1],self.close_price[current_price_index+1])
        print "high%s low%s"%(self.high_price[current_price_index+1],self.low_price[current_price_index+1])


        if self.fail_flag>0 and self.fail_flag<5:
            self.fail_flag+=1
        else:
            self.fail_flag=0
        
        if  self.sell_decision!=-1 and self.buy_price!=0 and ((trade_price-self.buy_price)/self.buy_price) >0.02:
            self.sell_decision = -1
 #       if distance_decision2 ==1 and self.buy_price!=0 and (imf_extrem_decision6==1 ) and ((trade_price-self.buy_price)/self.buy_price) >0.01:#and self.sell_decision==-1 and   ((self.macd[current_price_index]>0 and self.macd[current_price_index]<self.macd[current_price_index-1]<self.macd[current_price_index-2]) or (self.macd[current_price_index]<self.macd[current_price_index-1] and self.macd[current_price_index]<0)) and self.share > 0 :
        if   (imf1_flag==2 or deltama3<deltama3_before) and self.buy_price!=0 and (imf2_flag==2) and ((current_price-self.buy_price)/self.buy_price) >0:#(imf_extrem_decision6==1 ) :#and self.sell_decision==-1 and   ((self.macd[current_price_index]>0 and self.macd[current_price_index]<self.macd[current_price_index-1]<self.macd[current_price_index-2]) or (self.macd[current_price_index]<self.macd[current_price_index-1] and self.macd[current_price_index]<0)) and self.share > 0 :
            print "sell sell 1"
            print "open%s close%s"%(self.open_price[current_price_index],current_price)
            self.money = self.sell(self.share,trade_price)
            self.share = 0
            self.total_asset = self.money - self.sell_fee(self.money)
            self.last_buy_flag = 0
            self.money = self.total_asset
            self.sell_decision = 0
            self.sell_x_index.append(current_price_index)
            self.sell_y_index.append(current_price)
            tomorrow_will = 1
            self.down_macd_counter = 0
            self.buy_macd = 0
            self.count = 0
            self.holdtime = 0
            once_flag = 1
            if trade_price>self.buy_price:
                self.last_buy_result=1
            self.buy_price = 0
            self.ma_record.append(1)
            self.buy_mean_price = 0
            self.ma_record=[]
            self.buy_core=0
        elif 0:#  ma_decision3==0 and (imf1_flag==2 or deltama5<deltama5_before or deltama3<deltama3_before) and self.buy_price!=0 and (imf2_flag==2) :#(imf_extrem_decision6==1 ) :#and self.sell_decision==-1 and   ((self.macd[current_price_index]>0 and self.macd[current_price_index]<self.macd[current_price_index-1]<self.macd[current_price_index-2]) or (self.macd[current_price_index]<self.macd[current_price_index-1] and self.macd[current_price_index]<0)) and self.share > 0 :
            print "sell sell 11"
            print "open%s close%s"%(self.open_price[current_price_index],current_price)
            self.money = self.sell(self.share,trade_price)
            self.share = 0
            self.total_asset = self.money - self.sell_fee(self.money)
            self.last_buy_flag = 0
            self.money = self.total_asset
            self.sell_decision = 0
            self.sell_x_index.append(current_price_index)
            self.sell_y_index.append(current_price)
            tomorrow_will = 1
            self.down_macd_counter = 0
            self.buy_macd = 0
            self.count = 0
            self.holdtime = 0
            once_flag = 1
            if trade_price>self.buy_price:
                self.last_buy_result=1
            self.buy_price = 0
            self.ma_record.append(1)
            self.buy_mean_price = 0
            self.ma_record=[]
            self.buy_core=0
        #elif   self.buy_price!=0 and ((((self.buy_mean_price-current_mean_price)/self.buy_mean_price)>0.02 ) or (deltama5_before<0 and deltama5<0))  and self.share > 0:
        elif   self.buy_price!=0 and ((((self.buy_mean_price-current_mean_price)/self.buy_mean_price)>0.02 ) )  and self.share > 0:
            print "sell sell 2"
            self.money = self.sell(self.share,trade_price)
            self.share = 0
            self.buy_price = 0
            self.total_asset = self.money - self.sell_fee(self.money)
            self.last_buy_flag = 0
            self.money = self.total_asset
            self.sell_x_index.append(current_price_index)
            self.sell_decision = 0
            self.sell_y_index.append(current_price)
            tomorrow_will = 1
            self.down_macd_counter = 0
            self.buy_macd = 0
            self.buy_mean_price = 0
            self.holdtime = 0
            once_flag = 1
            self.last_buy_result=2
            self.ma_record.append(-1)
            self.ma_record=[]
            self.buy_core=0
            self.fail_flag=1
     #   elif self.buy_price!=0 and self.buy_day>5 and   (down_test_decision1==1 or current_price>self.buy_price) and self.share > 0 :
     #       print "sell sell 9"
     #       self.money = self.sell(self.share,trade_price)
     #       self.share = 0
     #       self.total_asset = self.money - self.sell_fee(self.money)
     #       self.last_buy_flag = 0
     #       self.money = self.total_asset
     #       self.sell_x_index.append(current_price_index)
     #       self.sell_y_index.append(current_price)
     #       tomorrow_will = 1
     #       self.down_macd_counter = 0
     #       self.buy_macd = 0
     #       self.holdtime = 0
     #       once_flag = 1
     #       if trade_price>self.buy_price:
     #           self.last_buy_result=9
     #           self.ma_record.append(1)
     #       else:
     #           self.ma_record.append(-1)
     #       self.buy_price = 0
     #       self.write_ma_train()
     #       self.ma_record=[]
##        elif self.buy_price!=0 and ((self.buy_day>15 and  ((trade_price-self.buy_price)/self.buy_price) >0) or (self.buy_day>30) ) and self.share > 0 :
##            print "sell sell 1"
##            self.money = self.sell(self.share,trade_price)
##            self.share = 0
##            self.buy_price = 0
##            self.total_asset = self.money - self.sell_fee(self.money)
##            self.last_buy_flag = 0
##            self.money = self.total_asset
##            self.sell_x_index.append(current_price_index)
##            self.sell_y_index.append(current_price)
##            tomorrow_will = 1
##            self.down_macd_counter = 0
##            self.buy_macd = 0
##            self.holdtime = 0
##            once_flag = 1
#        elif self.buy_price!=0 and distance_decision==1 and ((current_price-self.buy_price)/self.buy_price) >0.09 and self.share > 0 :
#            print "sell sell 3"
#            print "open%s close%s"%(self.open_price[current_price_index],current_price)
#            self.money = self.sell(self.share,trade_price)
#            self.share = 0
#            self.buy_price = 0
#            self.total_asset = self.money - self.sell_fee(self.money)
#            self.last_buy_flag = 0
#            self.money = self.total_asset
#            self.sell_x_index.append(current_price_index)
#            self.sell_y_index.append(current_price)
#            tomorrow_will = 1
#            self.down_macd_counter = 0
#            self.buy_macd = 0
#            self.holdtime = 0
#            once_flag = 1
#            self.last_buy_result=3
#            self.ma_record.append(1)
#            self.write_ma_train()
#            self.ma_record=[]
  #      elif self.buy_price!=0 and ((trade_price-self.buy_price)/self.buy_price) <-0.035 and self.share > 0 :
  #          print "sell sell 3"
  #          self.money = self.sell(self.share,trade_price)
  #          self.share = 0
  #          self.buy_price = 0
  #          self.total_asset = self.money - self.sell_fee(self.money)
  #          self.last_buy_flag = 0
  #          self.money = self.total_asset
  #          self.sell_x_index.append(current_price_index)
  #          self.sell_y_index.append(current_price)
  #          tomorrow_will = 1
  #          self.down_macd_counter = 0
  #          self.buy_macd = 0
  #          self.holdtime = 0
  #          once_flag = 1
        #elif macd_decision7==1 and open_close_decision1==1 and ma_decision6==1 and delta_buy_decision1==1 and ma_decision9==1 and up_ma_decision1==1 and ma_decision5==1 and  macd_decision6==1 and  macd_decision5==1 and rsi_decision1==1 and macd_decision9==1 and  macd_decision3==1 and   self.buy_price==0 and   imf_extrem_decision6==-1  :#and self.macd[current_price_index]<self.macd[current_price_index-1]<self.macd[current_price_index-2]<self.macd[current_price_index-3]<self.macd[current_price_index-9]<self.macd[current_price_index-5] :#and small_imf_decision1 == -1:#imf_extrem_decision8 == 1 and buy_decision2 == -1:# and  self.close_price[current_price_index-1]<self.close_price[current_price_index]:# and imf_below_0_decision == -1:# and imf_power_decision == -1:
  #      elif delta_macd1>delta_macd2 and macd_ma_decision==1 and  ma_decision3==1 and  self.buy_price==0 and  up_potion_decision2==1 and    down_test_decision1==-1 and delta_buy_decision1==1 :#and macd_decision3==1  :#imf_extrem_decision6==-1  :
        elif        imf2_flag==1 and  self.buy_price==0 and  delta_buy_decision1==1 and  deltama3_before<deltama3  :#(deltama5>deltama5_before>deltama5_before2 and deltama3_before<deltama3):#( (((deltama3_before<0 and deltama3>0) or ( 0<deltama3_before<deltama3 and deltama5_before<0 and deltama5>0))) )  and delta_buy_decision1==1  :#imf_extrem_decision6==-1  :
        #elif macd_decision1==1 and self.buy_price==0 and   imf_extrem_decision6==-1  and self.macd[current_price_index]<self.macd[current_price_index-1]<self.macd[current_price_index-2]<self.macd[current_price_index-3]<self.macd[current_price_index-9] and small_imf_decision1==-1 and self.macd[current_price_index]<0.5 and  self.macd[current_price_index]>-1:#imf_extrem_decision8 == 1 and buy_decision2 == -1:# and  self.close_price[current_price_index-1]<self.close_price[current_price_index]:# and imf_below_0_decision == -1:# and imf_power_decision == -1:
            print "buy buy 1"
            print "vol %s"%self.vol[current_price_index-5:current_price_index+1]
            print "open%s close%s"%(self.open_price[current_price_index-1],self.close_price[current_price_index-1])
            print "open%s close%s"%(self.open_price[current_price_index],current_price)
            print "high%s low%s"%(self.high_price[current_price_index],self.low_price[current_price_index])
            self.ma_record.append(self.ma_count1)
            self.ma_record.append(self.ma_count)
            self.ma_record.append(self.ma_count2)
            self.ma_record.append(self.ma_count3)
  #          self.ma_record.append(delta_power_current)
            print "ma record%s"%self.ma_record
            self.share = self.buy(self.money,trade_price)
            self.money = 0
            self.buy_price = trade_price
            self.buy_mean_price = current_mean_price
            self.total_asset = self.share*trade_price
            self.buy_x_index.append(current_price_index)
            self.buy_y_index.append(current_price)
            self.last_buy_flag = 1
            self.buy_macd = self.macd[current_price_index]
            tomorrow_will = -1
            self.wait_flag=0
            self.wait_count=0
            self.holdtime = 1
            self.buy_decision = 0
            self.buy_core = current_price_index

            print "ma5 %s"%ma5
            print "delta ma5 %s"%deltama5
            print "ma10 %s"%ma10
            print "delta ma10 %s"%deltama10
            print "ma20 %s"%ma20
            print "delta ma20 %s"%deltama20
            print "ma30 %s"%ma30
            print "delta ma30 %s"%deltama30
            self.last_buy_result=0
        elif 0:# self.buy_price==0 and down_test_decision1!=1 and down_test_decision2==-1 and delta_buy_decision1==1:#and macd_decision3==1  :#imf_extrem_decision6==-1  :
        #elif macd_decision1==1 and self.buy_price==0 and   imf_extrem_decision6==-1  and self.macd[current_price_index]<self.macd[current_price_index-1]<self.macd[current_price_index-2]<self.macd[current_price_index-3]<self.macd[current_price_index-9] and small_imf_decision1==-1 and self.macd[current_price_index]<0.5 and  self.macd[current_price_index]>-1:#imf_extrem_decision8 == 1 and buy_decision2 == -1:# and  self.close_price[current_price_index-1]<self.close_price[current_price_index]:# and imf_below_0_decision == -1:# and imf_power_decision == -1:
            print "buy buy 2"
            print "vol %s"%self.vol[current_price_index-5:current_price_index+1]
            print "open%s close%s"%(self.open_price[current_price_index-1],self.close_price[current_price_index-1])
            print "open%s close%s"%(self.open_price[current_price_index],current_price)
            print "high%s low%s"%(self.high_price[current_price_index],self.low_price[current_price_index])
            self.share = self.buy(self.money,trade_price)
            self.money = 0
            self.buy_price = trade_price
            self.total_asset = self.share*trade_price
            self.buy_x_index.append(current_price_index)
            self.buy_y_index.append(current_price)
            self.last_buy_flag = 1
            self.buy_macd = self.macd[current_price_index]
            tomorrow_will = -1
            self.wait_flag=0
            self.wait_count=0
            self.holdtime = 1
            self.buy_decision = 0
            print "ma5 %s"%ma5
            print "delta ma5 %s"%deltama5
            print "ma10 %s"%ma10
            print "delta ma10 %s"%deltama10
            print "ma20 %s"%ma20
            print "delta ma20 %s"%deltama20
            print "ma30 %s"%ma30
            print "delta ma30 %s"%deltama30
            self.last_buy_result=0
            self.ma_record.append(self.ma_count1)
            self.ma_record.append(self.ma_count)
            self.ma_record.append(self.ma_count2)
            self.ma_record.append(self.ma_count3)
            self.ma_record.append(delta_power_current)
            print "ma record%s"%self.ma_record
        elif 0:#self.wait_flag==1 and   self.buy_price==0 and  power_ma5_decision==-1 and (deltama5>0 or (deltama5<0 and deltama5>deltama10)):#and power_ma5_decision==-1:#imf_extrem_decision8 == 1 and buy_decision2 == -1:# and  self.close_price[current_price_index-1]<self.close_price[current_price_index]:# and imf_below_0_decision == -1:# and imf_power_decision == -1:
            if 1:
                print "buy buy 2"
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
                self.buy_decision = 0
                self.wait_flag=0
                self.wait_count=0
                print "ma5 %s"%ma5
                print "delta ma5 %s"%deltama5
                print "ma10 %s"%ma10
                print "delta ma10 %s"%deltama10
                print "ma20 %s"%ma20
                print "delta ma20 %s"%deltama20
                print "ma30 %s"%ma30
                print "delta ma30 %s"%deltama30
            else:
                self.wait_flag=0

  
        self.last_imf_power = imf_power_current
        self.last_imf_3_power = imf_3_power_current

        if self.buy_price!=0:
            self.buy_day += 1

#        if self.buy_decision == -1 and this_buy!=-1:
#            self.jiange = 1
#        elif self.jiange>0 and this_buy!=-1:
#            self.jiange += 1
#        elif self.jiange>2 :
#            self.jiange = 0
#
#        if this_buy != -1 and self.jiange>2:
#            self.buy_count=0
#            self.buy_decision = 0


        print "jiange %s"%self.jiange
        print "buy count %s"%self.buy_count
        print "buy decision %s"%self.buy_decision
        print "buy day%s"%self.buy_day

            




        print "self buy deciison %s"%self.buy_decision

  
        self.total_asset_list.append(self.total_asset)
        print "total asset is %s"%(self.total_asset)
        print "money is %s"%(self.money)
        print "share is %s"%(self.share)


    def write_ma_train(self):
        fp=open("ma_train",'a')
        for i in self.ma_record:
            fp.write(str(i))
            fp.write(" ")
        fp.write("\n")
        fp.close()

    def span_estimate(self, data):
        return (round(np.mean(data)-np.std(data),0),round(np.mean(data)+np.std(data),0))

    def format_data(self, data):
        if isinstance(data,list):
            return [2*((float(data[i])-min(data))/(max(data)-float(min(data))))-1 for i in range(len(data))]



    def run(self,emd_data, datafile,current_index,date):
        starttime = datetime.datetime.now()
        self.clear_emd_data()
        tmp = [i for i in emd_data]
        emd_data = tmp
        print "emd data std %s"%np.std(emd_data)
        my_emd = one_dimension_emd(emd_data,9)
        (imf, residual) = my_emd.emd(0.002,0.002)
#######
     #   my_eemd = eemd(emd_data,10)
     #   (imf0,imf1,imf2,imf3)= my_eemd.eemd_process(emd_data,50,9,"multi")
     #   imf=[]
     #   imf.append(imf0)
     #   imf.append(imf1)
     #   imf.append(imf2)
     #   if imf3==[]:
     #       imf3=emd_data
     #   imf.append(imf3)
#    #        
     #   residual = emd_data
########

     #   my_eemd = eemd(emd_data,10)
     #   (imf0,imf1,imf2,imf3)= my_eemd.eemd_process(emd_data,50,9,"multi")
        imf_sec=[]
     #   imf_sec.append(imf0)
     #   imf_sec.append(imf1)
     #   imf_sec.append(imf2)
     #   if imf3==[]:
     #       imf3=emd_data
     #   imf_sec.append(imf3)

     #   my_eemd = eemd(emd_data,10)
     #   (imf0,imf1,imf2,imf3)= my_eemd.eemd_process(emd_data,50,9,"multi")
        imf_third=[]
     #   imf_third.append(imf0)
     #   imf_third.append(imf1)
     #   imf_third.append(imf2)
     #   if imf3==[]:
     #       imf3=emd_data
     #   imf_third.append(imf3)


        print "len imf %s"%len(imf)
            
        
        self.clear_nn_train_data()
        train_data_all = "train_data_all"
        train_data_dir = "../../train_data"
        group_number = 50
        train_number = 50
        result_number = 10
 
        print "\n\n\n"
        last_50_price = self.close_price[current_index-10:current_index+1]
        print "50 std %s"%np.std(last_50_price)
        print "std/mean %s"%(np.std(last_50_price)/np.mean(last_50_price))
        print "rsi 10 %s"%self.rsi(self.close_price,10,current_index)
        print "rsi 15 %s"%self.rsi(self.close_price,15,current_index)
        print "rsi 20 %s"%self.rsi(self.close_price,20,current_index)
        print "rsi 30 %s"%self.rsi(self.close_price,30,current_index)
        self.run_predict(imf,current_index,date,datafile,train_data_dir,train_number,20,result_number,residual,imf_sec,imf_third)
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


    def show_success(self):
        print success_ratio(precondition(self.total_asset_list))
        
        

    def draw_fig(self,datafile,start,save=0):
        
        data = self.close_price

        ma5 = []
        for i in range(3):
            ma5.append(0)
#
        for i in range(3,len(data)+1):
            mean_5 = np.mean(data[i-2:i+1])
            ma5.append(mean_5)
#
        data = ma5

        #data = self.close_price
        my_emd = one_dimension_emd(data)
        (imf, residual) = my_emd.emd(0.01,0.01)
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


        #print lines


        fp = open("residual")
        lines = fp.readlines()
        fp.close()
        residual_raw = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            residual_raw.append(float(eachline.split("\n")[0]))
        imf = imf_raw2


        fp = open("residual",'r')
        lines = fp.readlines()
        fp.close()

        residual = []
        for eachline in lines:
            residual.append(float(eachline.split("\n")[0]))

        
        imf_buy_value = [imf[i] for i in self.buy_x_index]
        imf_sell_value = [imf[i] for i in self.sell_x_index]


        buy_x = [i-start for i in self.buy_x_index]
        sell_x = [i-start for i in self.sell_x_index]
        data = self.close_price
        plt.figure(1)
        plt.subplot(211).axis([start,len(data),min(data[start:]),max(data[start:])])      
        plt.plot([i for i in range(len(data))],data,'o',[i for i in range(len(data))],data,'b',self.buy_x_index,self.buy_y_index,'r*',self.sell_x_index,self.sell_y_index,'g*')             
        plt.subplot(212).axis([start,len(imf),min(imf[start:]),max(imf[start:])])
        plt.plot([i for i in range(len(imf))],imf,'o',[i for i in range(len(imf))],imf,'b',self.buy_x_index,imf_buy_value,'r*',self.sell_x_index,imf_sell_value,'g*')             
#####

#####


        figname = "fig1_"+datafile         
        if save==1:
            savefig(figname)
        plt.figure(2)
        plt.plot([i for i in range(len(self.total_asset_list))],self.total_asset_list,'b')
        figname = "fig2_"+datafile



       
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
        high_price.append(float(eachline.split("\t")[2]))
        low_price.append(float(eachline.split("\t")[3]))
        macd.append(float(eachline.split("\t")[5]))
        open_price.append(float(eachline.split("\t")[1]))
        date.append(eachline.split("\t")[0])
        vol.append(float(eachline.split("\t")[5]))
    
    process = process_util(open_price,close_price,macd,date,vol,high_price,low_price)
    os.popen("rm -r ../../emd_data")
    os.popen("mkdir ../../emd_data")
    generate_emd_func = generateEMDdata(datafile,begin,50)
    #generate_emd_func.generate_emd_data_fix()
##########


#    datalist = close_price[begin-500:begin+1]
#    data_0 = [datalist[i]-np.mean(datalist) for i in range(len(datalist))]
#    selfcorre = np.correlate(data_0,data_0,"full")
#    result = selfcorre
#    result = result[len(result)//2:]
#    result /= result[0]
#    selfcorre = result
#    print selfcorre
#    data = np.array(selfcorre)
#    corr_max_index = list(argrelextrema(data,np.greater)[0])
#    corr_min_index = list(argrelextrema(data,np.less)[0])
#
#    print "corr max%s"%corr_max_index
#    print "corr min%s"%corr_min_index
#
#
#    corr_diff = [corr_max_index[i+1]-corr_max_index[i] for i in range(len(corr_max_index)-1)]
#    print corr_diff
#    ma_value = int(np.mean(corr_diff))
#    print ma_value


##############


    generate_emd_func.generate_ma_emd_data_fix(3)
    #generate_emd_func.generate_mavol_emd_data_fix(3)
    #generate_emd_func.generate_ema_emd_data_fix(5)
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
    process.show_success()
#    process.peilv()
#    process.hold_time()
    process.draw_fig(datafile,begin)

    

            
