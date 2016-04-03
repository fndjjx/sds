import os
from svm_uti import *
import copy
import math
import sys
import libfann
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from emd import *
from smooth import cubicSmooth5
import datetime
from scipy.signal import argrelextrema
import numpy as np
from generate_emd_data import generateEMDdata
from generate_train_file import genTraindata
from neuro_predict import neuroprediction
from neuro_predict import test_file
from spline_predict import splinepredict
from spline_predict import splinerestruct
from spline_predict import splinerestruct2
from fftfilter2 import fftfilter





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
        self.buy_core=0
        self.last_last=""
        self.imf1_last_last=""
    def imf_percentage(self, imf, residual, source_data):
        sum_list = []
        source_data_without_residual = []
        for i in range(len(imf)):
            sum_list.append(sum([j*j for j in imf[i]]))
        sum_list.append(sum([j*j for j in residual]))
        source_data_without_residual = [source_data[i]-residual[i] for i in range(len(source_data))]
        source_square_sum = sum([j*j for j in source_data_without_residual])
        return [i/source_square_sum for i in sum_list]

    def restruct(self,rawdata,max_point,min_point):
        data = np.array(rawdata)
        max_index = list(argrelextrema(data,np.greater)[0])
        min_index = list(argrelextrema(data,np.less)[0])
        print max_index
        print min_index
        max_value = []
        min_value = []
        for i in max_index:
            max_value.append(rawdata[i])
        for i in min_index:
            min_value.append(rawdata[i])
        
        max_value.sort()
        min_value.sort()
        
        
        b = max_value[-max_point:]+min_value[:min_point]
        index=[]
        for i in b:
            for j in range(len(rawdata)):
                if i == rawdata[j]:
                    index.append(j)
        
        print index
        if not 0 in index:
            index.insert(0,0)
        if not len(rawdata)-1 in index:
            index.insert(len(rawdata)-1,len(rawdata)-1)
        print index
        index.sort()
        d=[]
        for i in index:
            d.append(rawdata[i])
        return (index,d)
        
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
        
        tmp = filter(lambda x:x>4 and x<raw_data_length-2, tmp)
        #tmp = filter(lambda x:x>4, tmp)
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

    def double_filter(self,data):

        aa=[]
        index=[]
        for i in range(len(data)):
            if i%2==0:
                aa.append(data[i])
                index.append(i)
        b1=splinerestruct2(data,aa,index,3,8)
        aa=[]
        index=[]
        for i in range(len(data)):
            if i%2!=0:
                aa.append(data[i])
                index.append(i)
        b2=splinerestruct2(data,aa,index,3,8)
        b = [(b1[i]+b2[i])/2 for i in range(len(b1))]
        print b
        data = np.array(b)
        max_index = list(argrelextrema(data,np.greater)[0])
        min_index = list(argrelextrema(data,np.less)[0])
        print max_index
        print min_index
        if max_index!=[] or min_index!=[]:
            tmp = max_index+min_index
            tmp.append(0)
            tmp.append(len(data)-1)
            tmp.sort()
            print "tmp %s"%tmp
            for i in range(1,len(tmp)-1):
                if (tmp[i]-tmp[i-1])>(tmp[i+1]-tmp[i]):
                    close = tmp[i+1]
                else:
                    close = tmp[i-1]
                if  (abs(b[tmp[i]]-b[close])/b[tmp[i]])<0.01 :#and (abs(b[tmp[i]]-b[tmp[i-1]])/b[tmp[i]])<0.01 :
                    if tmp[i] in max_index:
                        max_index.remove(tmp[i])
                    if tmp[i] in min_index:
                        min_index.remove(tmp[i])
        #if max_index!=[] or min_index!=[]:
        #    return (max_index,min_index)
        return (max_index,min_index)
        #else:
        #    c = splinerestruct2(b,[b[0],b[len(b)-1]],[0,len(b)-1],1,0)
        #    d = [b[i]-c[i] for i in range(len(c))]
        #    print d
        #    data = np.array(d)
        #    max_index = list(argrelextrema(data,np.greater)[0])
        #    min_index = list(argrelextrema(data,np.less)[0])
        #    print max_index
        #    print min_index
        #    return (max_index,min_index)

    def spline_filter(self,data,expect):

        b=splinerestruct2(data,data,np.linspace(0,len(data)-1,len(data)),2,100)
        b1=splinerestruct2(data,data,np.linspace(0,len(data)-1,len(data)),3,100)
        data = np.array(b)
        data1 = np.array(b1)
        max_index = list(argrelextrema(data,np.greater)[0])
        min_index = list(argrelextrema(data,np.less)[0])
        print "origianl max%s"%max_index
        print "original min%s"%min_index
        max_index1 = list(argrelextrema(data1,np.greater)[0])
        min_index1 = list(argrelextrema(data1,np.less)[0])
        print "origianl max1%s"%max_index1
        print "original min1%s"%min_index1
        if expect=="max":
            oppsite=min_index
        else:
            oppsite=max_index
        tmp = max_index+min_index
        tmp.sort()
        if tmp==[] or tmp[0]<=4 or (tmp[0] in oppsite):#abs(up_potion)<1.5 and abs(up_potion)>0.6:
            c = splinerestruct2(b,b,np.linspace(0,len(b)-1,len(b)),1,100)
            d = [b[i]-c[i] for i in range(len(c))]
            b=d
        print b
        data = np.array(b)
        max_index = list(argrelextrema(data,np.greater)[0])
        min_index = list(argrelextrema(data,np.less)[0])
        print max_index
        print min_index
        return (max_index,min_index)


    def mean_distance(self,imf):

        data = np.array(imf)
        imf_max_index = list(argrelextrema(data,np.greater)[0])
        imf_min_index = list(argrelextrema(data,np.less)[0])

        imf_max = imf_max_index
        imf_min = imf_min_index
        max_dis = [imf_max[i]-imf_max[i-1] for i in range(1,len(imf_max))]
        min_dis = [imf_min[i]-imf_min[i-1] for i in range(1,len(imf_min))]
        if imf_max[0]>imf_min[0] and imf_min[-1]<imf_max[-1]:
            min_max = [imf_max[i]-imf_min[i] for i in range(len(imf_max))]
            max_min = [imf_min[i+1]-imf_max[i] for i in range(len(imf_max)-1)]
        elif imf_max[0]>imf_min[0] and imf_min[-1]>imf_max[-1]:
            min_max = [imf_max[i]-imf_min[i] for i in range(len(imf_max))]
            max_min = [imf_min[i+1]-imf_max[i] for i in range(len(imf_max))]
        elif imf_max[0]<imf_min[0] and imf_min[-1]>imf_max[-1]:
            min_max = [imf_max[i+1]-imf_min[i] for i in range(len(imf_max)-1)]
            max_min = [imf_min[i]-imf_max[i] for i in range(len(imf_max))]
        elif imf_max[0]<imf_min[0] and imf_min[-1]<imf_max[-1]:
            min_max = [imf_max[i+1]-imf_min[i] for i in range(len(imf_min))]
            max_min = [imf_min[i]-imf_max[i] for i in range(len(imf_min))]

        mean_max_min = np.mean(max_min[:])
        std_max_min = np.std(max_min[:])
        mean_min_max = np.mean(min_max[:])
        std_min_max = np.std(min_max[:])
        return ((mean_max_min,std_max_min),(mean_min_max,std_min_max))



    def run_predict(self,imf_list,current_price_index,date,datafile,train_data_dir,train_number,hidden_number,result_number,residual):

        decision = 0
        power_decision = 0
        imf_power_decision = 0
        period_decision = 0
        extrem_decision = 0
        current_rise_flag = 0

        
        power_imf =  self.imf_percentage(imf_list,residual,self.close_price[current_price_index-499:current_price_index+1])
        print power_imf
        for i in range(len(power_imf)-1):
            print max(power_imf[:-2])
            if power_imf[i]==max(power_imf[:-2]):
                imf_p = imf_list[i]


        imf_p1 = imf_list[-2]
        imf_p2 = imf_list[-3]
        imf_p3 = imf_list[-4]
        print self.mean_distance(imf_p1)
        print self.mean_distance(imf_p2)
        print self.mean_distance(imf_p3)
        estimate_period1 = int(round(self.mean_distance(imf_p1)[0][0]))
        estimate_period2 = int(round(self.mean_distance(imf_p2)[0][0]))
        estimate_period3 = int(round(self.mean_distance(imf_p3)[0][0]))
        cha_20=[]
        cha_20.append(abs(estimate_period1-20))
        cha_20.append(abs(estimate_period2-20))
        cha_20.append(abs(estimate_period3-20))
        for i in range(3):
            if cha_20[i]==min(cha_20):
                estimate_period=eval("estimate_period%s"%(i+1))
        data = self.close_price[current_price_index-10:current_price_index+1]
        cha_data = [data[i+1]-data[i] for i in range(len(data)-1)]
        print "std data %s"%(np.std(cha_data))
        if np.std(cha_data)>1.5:
            estimate_period = int(round(self.mean_distance(imf_list[2])[0][0]))
        else:
            estimate_period = int(round(self.mean_distance(imf_list[3])[0][0]))
        estimate_period = int(round(self.mean_distance(imf_list[2])[0][0]))
        #estimate_period2 = int(round(self.mean_distance(imf_list[3])[0][0]))
        print "estimate_period %s "%estimate_period
      


##############################################################

        imf_process = list(imf_list[2])


        big_imf = list(imf_list[3])
        small_imf = list(imf_list[1])
        small_small_imf = list(imf_list[0])

        small_small_data = np.array(small_small_imf)
        small_small_imf_max_index = list(argrelextrema(small_small_data,np.greater)[0])
        print " imf0 max index %s"%small_small_imf_max_index
        small_small_imf_min_index = list(argrelextrema(small_small_data,np.less)[0])
        print " imf0 min index %s"%small_small_imf_min_index

        print "\n"

        small_data = np.array(small_imf)
        small_imf_max_index = list(argrelextrema(small_data,np.greater)[0])
        print "small imf max index %s"%small_imf_max_index
        small_imf_min_index = list(argrelextrema(small_data,np.less)[0])
        print "small imf min index %s"%small_imf_min_index
        
     #   if small_imf_max_index[-1]>small_imf_min_index[-1] and small_imf_max_index[-1]>496:
     #       small_imf_max_true_index = small_imf_max_index[:-1]
     #       small_imf_min_true_index = small_imf_min_index
     #   elif small_imf_min_index[-1]>small_imf_max_index[-1] and small_imf_min_index[-1]>496:
     #       small_imf_min_true_index = small_imf_min_index[:-1]
     #       small_imf_max_true_index = small_imf_max_index
     #   else:
     #       small_imf_min_true_index = small_imf_min_index
     #       small_imf_max_true_index = small_imf_max_index

        small_imf_max_true_index = filter(lambda n:n<497, small_imf_max_index)
        small_imf_min_true_index = filter(lambda n:n<497, small_imf_min_index)
        if small_imf_max_true_index[-1]>small_imf_min_true_index[-1]:
            current_imf1_core = small_imf_max_true_index[-1]
        else:
            current_imf1_core = small_imf_min_true_index[-1]
        print "\n\n\n"
        print "imf1 core %s %s"%(self.imf1_core,current_imf1_core)
        imf1_flag=0
        if small_imf_max_true_index[-1]>small_imf_min_true_index[-1]:
            imf1_flag=1
        if small_imf_max_true_index[-1]<small_imf_min_true_index[-1]:
            imf1_flag=2
        self.imf1_core=current_imf1_core
        small_imf_max_true_index = filter(lambda n:n<499, small_imf_max_index)
        small_imf_min_true_index = filter(lambda n:n<499, small_imf_min_index)

        imf1_flag=0
        if small_imf_min_true_index[-1]>=497 and small_imf_min_true_index[-1]>small_imf_max_true_index[-1] :#and self.imf1_last_last=="max":
            imf1_flag=1
        if small_imf_max_true_index[-1]>=495 and small_imf_max_true_index[-1]>small_imf_min_true_index[-1] and self.imf1_last_last=="min":
            imf1_flag=2
        if small_imf_min_true_index[-1]>small_imf_max_true_index[-1]:
            self.imf1_last_last = "min"
        else:
            self.imf1_last_last = "max"


        
##########################



###########################
        

        data = np.array(imf_process)
        imf_max_index = argrelextrema(data,np.greater)[0]
        imf_min_index = argrelextrema(data,np.less)[0]

        imf_max = imf_max_index
        imf_min = imf_min_index
        max_dis = [imf_max[i]-imf_max[i-1] for i in range(1,len(imf_max))]
        min_dis = [imf_min[i]-imf_min[i-1] for i in range(1,len(imf_min))]
        if imf_max[0]>imf_min[0] and imf_min[-1]<imf_max[-1]:
            min_max = [imf_max[i]-imf_min[i] for i in range(len(imf_max))]
            max_min = [imf_min[i+1]-imf_max[i] for i in range(len(imf_max)-1)]
        elif imf_max[0]>imf_min[0] and imf_min[-1]>imf_max[-1]:
            min_max = [imf_max[i]-imf_min[i] for i in range(len(imf_max))]
            max_min = [imf_min[i+1]-imf_max[i] for i in range(len(imf_max))]
        elif imf_max[0]<imf_min[0] and imf_min[-1]>imf_max[-1]:
            min_max = [imf_max[i+1]-imf_min[i] for i in range(len(imf_max)-1)]
            max_min = [imf_min[i]-imf_max[i] for i in range(len(imf_max))]
        elif imf_max[0]<imf_min[0] and imf_min[-1]<imf_max[-1]:
            min_max = [imf_max[i+1]-imf_min[i] for i in range(len(imf_min))]
            max_min = [imf_min[i]-imf_max[i] for i in range(len(imf_min))]

        mean_max_dis = np.mean(max_dis)
        std_max_dis = np.std(max_dis)
        mean_min_dis = np.mean(min_dis)
        std_min_dis = np.std(min_dis)


        max_min.remove(max(max_min))
        max_min.remove(min(max_min))
        min_max.remove(max(min_max))
        min_max.remove(min(min_max))
        mean_max_min = np.mean(max_min[-3:])
        std_max_min = np.std(max_min[:])
        mean_min_max = np.mean(min_max[:])
        std_min_max = np.std(min_max[:])

        price_list = self.close_price[current_price_index-499:current_price_index+1]
        price_data = np.array(price_list)
        price_max_index = argrelextrema(price_data,np.greater)[0]
        print "price_max_index %s"%price_max_index
        price_min_index = argrelextrema(price_data,np.less)[0]
        print "price_min_index %s"%price_min_index



####################
        ma5 = self.ma(self.close_price, 5, 0, current_price_index)
        deltama5 = self.ma(self.close_price, 5, 0, current_price_index)-self.ma(self.close_price, 5, -1, current_price_index)
        ma10 = self.ma(self.close_price, 10, 0, current_price_index)
        deltama10 = self.ma(self.close_price, 10, 0, current_price_index)-self.ma(self.close_price, 10, -1, current_price_index)
        ma20 = self.ma(self.close_price, 20, 0, current_price_index)
        deltama20 = self.ma(self.close_price, 20, 0, current_price_index)-self.ma(self.close_price, 20, -1, current_price_index)
        ma30 = self.ma(self.close_price, 30, 0, current_price_index)
        deltama30 = self.ma(self.close_price, 30, 0, current_price_index)-self.ma(self.close_price, 30, -1, current_price_index)


###################


        last_50_price = self.close_price[current_price_index-10:current_price_index+1]
        print "50 std %s"%np.std(last_50_price)
        print "std/mean %s"%(np.std(last_50_price)/np.mean(last_50_price))
        self.imf_std.append((np.std(last_50_price)/np.mean(last_50_price)))
        std_mean = (np.std(last_50_price)/np.mean(last_50_price))
        mean_std_mean = (np.mean(self.imf_std))
        std_std_mean = (np.std(self.imf_std))
        
        print "std mean %s"%(np.mean(self.imf_std))
        print "std std %s"%(np.std(self.imf_std))
        choice=0
        if ((std_mean-mean_std_mean)/mean_std_mean)<-0.1:
            choice=1 
        #if (abs((np.std(last_50_price)/np.mean(last_50_price))-((np.mean(self.imf_std))))/((np.mean(self.imf_std))))>0.15:
        #    choice=1 
        if (mean_std_mean/std_mean)>1.5:
            choice=2 

        imf_extrem_decision = 0
        if len(list(price_max_index)) > 0 and len(list(price_min_index)) > 0 :
            index_list = list(price_max_index)+(list(price_min_index))
            print "index list%s"%index_list
            index_list.sort() 
            data = np.array(imf_process)
            imf_max_index = list(argrelextrema(data,np.greater)[0])
            imf_min_index = list(argrelextrema(data,np.less)[0])
            tmp = imf_max_index+imf_min_index
            tmp.sort()
            too_far = 0 
            if tmp[-1]<490 and tmp[-1]>=485:
                last_extrem_position = index_list[-1] 
            elif tmp[-1]<485:
                last_extrem_position = index_list[-1] 
                too_far = 1
            elif tmp[-1]>=490:
                for i in range(1,len(index_list)-1):
                    last_extrem_position = index_list[-i] 
                    distance =  499 - last_extrem_position
                    true_index = 499-distance
                    imf_max_true_index = list(filter(lambda n:n<=true_index,imf_max_index))
                    imf_min_true_index = list(filter(lambda n:n<=true_index,imf_min_index))
                    tmp = imf_max_true_index+imf_min_true_index
                    tmp.sort()
                    print tmp
                    if tmp[-1]>=485 and tmp[-1]<490:
                        break
                    elif tmp[-1]<485:
                        too_far = 1
                        break
                
                
            distance =  499 - last_extrem_position
            true_index = 499-distance
            print "distance %s"%distance
            print "true index %s"%true_index
            data = np.array(imf_process)
            imf_max_index = argrelextrema(data,np.greater)[0]
            print "imf max index %s"%imf_max_index
            imf_min_index = argrelextrema(data,np.less)[0]
            print "imf min index %s"%imf_min_index

            big_data = np.array(big_imf)
            big_imf_max_index = list(argrelextrema(big_data,np.greater)[0])
            print "big imf max index %s"%big_imf_max_index
            big_imf_min_index = list(argrelextrema(big_data,np.less)[0])
            print "big imf min index %s"%big_imf_min_index

            big_imf_min_true_index = []
            big_imf_max_true_index = []
            big_imf_max_true_index = filter(lambda n:n<496, big_imf_max_index)
            big_imf_min_true_index = filter(lambda n:n<496, big_imf_min_index)
            for i in big_imf_max_index:
                if i < 485:
                    big_imf_max_true_index.append(i)
            for i in big_imf_min_index:
                if i < 485:
                    big_imf_min_true_index.append(i)

            if big_imf_min_true_index[-1]>big_imf_max_true_index[-1]:
                big_last_position = big_imf_min_true_index[-1]
            else:
                big_last_position = big_imf_max_true_index[-1]

            big_true_data = price_list[big_last_position:]
            print "big true data%s"%big_true_data
            (big_max_index,big_min_index)=self.double_filter(big_true_data)
            print "big max index%s"%big_max_index
            print "big min index%s"%big_min_index

            big_imf_max_true_index = filter(lambda n:n<490, big_imf_max_index)
            big_imf_min_true_index = filter(lambda n:n<490, big_imf_min_index)

            last_50_price = self.close_price[current_price_index-20:current_price_index+1]
            print "50 std %s"%np.std(last_50_price)
            last_all_price = self.close_price[:current_price_index+1]
            print "all std %s"%np.std(last_all_price)
            print "std/mean %s"%(np.std(last_50_price)/np.mean(last_50_price)) 


            delta_macd1=self.macd[current_price_index]-self.macd[current_price_index-1]
            delta_macd2=self.macd[current_price_index-1]-self.macd[current_price_index-2]
            delta_macd3=self.macd[current_price_index-2]-self.macd[current_price_index-3]
            delta_macd4=self.macd[current_price_index-3]-self.macd[current_price_index-4]

            


            if 1:#choice==0:
                if imf_max_index[-1]>imf_min_index[-1] and imf_max_index[-1]>=495:
                    imf_max_true_index = list(filter(lambda n:n<imf_max_index[-1],imf_max_index))
                    imf_min_true_index = list(filter(lambda n:n<imf_max_index[-1],imf_min_index))
                elif imf_max_index[-1]>imf_min_index[-1] and imf_max_index[-1]<495:
                    imf_max_true_index = imf_max_index
                    imf_min_true_index = imf_min_index
                elif imf_max_index[-1]<imf_min_index[-1] and imf_min_index[-1]>=495:
                    imf_max_true_index = list(filter(lambda n:n<imf_min_index[-1],imf_max_index))
                    imf_min_true_index = list(filter(lambda n:n<imf_min_index[-1],imf_min_index))
                elif imf_max_index[-1]<imf_min_index[-1] and imf_min_index[-1]<495:
                    imf_max_true_index = imf_max_index
                    imf_min_true_index= imf_min_index

            print "imf max true index %s"%imf_max_true_index
            print "imf min true index %s"%imf_min_true_index
            if 1:#choice==0:
                if imf_max_true_index[-1]>imf_min_true_index[-1] and imf_max_true_index[-1]>=495:
                    imf_max_true_index = list(filter(lambda n:n<imf_max_index[-1],imf_max_true_index))
                    imf_min_true_index = list(filter(lambda n:n<imf_max_index[-1],imf_min_true_index))
                elif imf_max_true_index[-1]<imf_min_true_index[-1] and imf_min_true_index[-1]>=495:
                    imf_max_true_index = list(filter(lambda n:n<imf_min_index[-1],imf_max_true_index))
                    imf_min_true_index = list(filter(lambda n:n<imf_min_index[-1],imf_min_true_index))
            print "imf max true index %s"%imf_max_true_index
            print "imf min true index %s"%imf_min_true_index

            if 1:#choice==1 or choice==2:
                if imf_max_true_index[-1]>imf_min_true_index[-1] and imf_max_true_index[-1]>=490:
                    imf_max_true_index = list(filter(lambda n:n<imf_max_index[-1],imf_max_true_index))
                    imf_min_true_index = list(filter(lambda n:n<imf_max_index[-1],imf_min_true_index))
                elif imf_max_true_index[-1]<imf_min_true_index[-1] and imf_min_true_index[-1]>=490:
                    imf_max_true_index = list(filter(lambda n:n<imf_min_index[-1],imf_max_true_index))
                    imf_min_true_index = list(filter(lambda n:n<imf_min_index[-1],imf_min_true_index))

            
            imf_max_true_index = list(filter(lambda n:n<497,imf_max_index))
            imf_min_true_index = list(filter(lambda n:n<497,imf_min_index))
            if imf_max_true_index[-1]>imf_min_true_index[-1]:
                imf_max_true_index = list(filter(lambda n:n<495,imf_max_index))
                imf_min_true_index = list(filter(lambda n:n<495,imf_min_index))
            imf_max_true_index = list(filter(lambda n:n<499,imf_max_index))
            imf_min_true_index = list(filter(lambda n:n<499,imf_min_index))
            value_imf_max_true=[]
            value_imf_min_true=[]
            small_value_imf_max_true=[]
            small_value_imf_min_true=[]
            for i in imf_max_true_index:
                value_imf_max_true.append(imf_process[i])
            for i in imf_min_true_index:
                value_imf_min_true.append(imf_process[i])
            for i in small_imf_max_true_index:
                small_value_imf_max_true.append(small_imf[i])
            for i in small_imf_min_true_index:
                small_value_imf_min_true.append(small_imf[i])
            print "ha\n\n"
            print "imf max true value %s"%value_imf_max_true
            print "imf min true value %s"%value_imf_min_true
            print "last 10 %s"%imf_process[-10:]
            print "last 10 %s"%np.mean(imf_process[-10:])
            print "\n\n"
            print "small imf max true value %s"%small_value_imf_max_true
            print "small imf min true value %s"%small_value_imf_min_true
            print "small last 10 %s"%np.mean(small_imf[-10:])
            print "\n\n"
            max_min_cha_decision=0
            if imf_max_true_index[-1]<imf_min_true_index[-1]: #and imf_min_true_index[-1]>496:
                if (value_imf_max_true[-1]/abs(value_imf_min_true[-1]))<1.2:
                    max_min_cha_decision=1
            elif imf_max_true_index[-1]>imf_min_true_index[-1]:
                if (small_value_imf_max_true[-1]/abs(small_value_imf_min_true[-1]))<1.2:
                    max_min_cha_decision=1


            tmp = imf_max_true_index+imf_min_true_index
            tmp.sort()
            cha_tmp = [tmp[i+1]-tmp[i] for i in range(len(tmp)-1)]
            mean_recent_imf2_jiange = np.mean(cha_tmp[-10:])

            print "imf max true index %s"%imf_max_true_index
            print "imf min true index %s"%imf_min_true_index
            imf_extrem_decision2 = 0
            imf_extrem_decision3 = 0
            imf_extrem_decision5 = 0
            imf_extrem_decision6 = 0
            imf_extrem_decision7 = 0
            imf_extrem_decision8 = 1
            down_test_decision = 0
            down_test_decision1 = 0
            down_test_decision2 = 0
            vol_decision1 = 0
            new_imf_max_true_index = []
            new_imf_min_true_index = []
            last_min = 0
            last_max = 0
            price_50 = self.close_price[current_price_index-19:current_price_index+1]
            up_list = [price_50[i+1]-price_50[i] for i in range(19) if price_50[i]<price_50[i+1]]
            down_list = [price_50[i+1]-price_50[i] for i in range(19) if price_50[i]>price_50[i+1]]
            print up_list
            print down_list
            up_potion = abs((sum(up_list)/sum(down_list)))
            current_price = self.close_price[current_price_index]
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
            imf_tmp = imf_max_true_index+imf_min_true_index
            imf_tmp.sort()
            for i in range(0,len(imf_tmp)-1,2):
                list_tmp = price_list[imf_tmp[i]:imf_tmp[i+1]]
                self.list_std.append(np.std(list_tmp)/np.mean(list_tmp))

                 

            


###################
            
            print "choice %s"%choice                
            if 1:#self.buy_price == 0:# and abs(self.last_true_max-imf_max_true_index[-1])<10 and abs(self.last_true_min-imf_min_true_index[-1])<10:#(buyhigh/buyhigh2)>0.3 and (buyhigh/buyhigh2)<3:
                print "prepare buy"
                print "mean min max %s"%mean_min_max
                print "std min max %s"%std_min_max
                print "mean max min %s"%mean_max_min
                print "std max min %s"%std_max_min
                if imf_max_true_index[-1]<imf_min_true_index[-1]:# and 499-imf_min_true_index[-1]>(mean_min_max):# and imf_min_true_index[-1]>465:# and 499-imf_max_true_index[-1]<(mean_min_max+mean_max_min):# and (499-imf_min_true_index[-1])>(mean_max_min+std_max_min)) :
                    price_last_max_index = list(filter(lambda n:n>imf_min_true_index[-1],price_max_index))
                    print "price_last_max_index %s"%price_last_max_index
                    if len(price_last_max_index)>3:
                        price_max_data = [price_list[i] for i in price_last_max_index]
                    price_max_data = price_list[imf_min_true_index[-1]:] 
                    print "price max data %s"%price_max_data
                    print "price max %s"%list(argrelextrema(np.array(price_max_data),np.greater)[0])
                    print "len %s"%len(list(argrelextrema(np.array(price_max_data),np.greater)[0]))
                    print "price min %s"%list(argrelextrema(np.array(price_max_data),np.less)[0])
                    print "len %s"%len(list(argrelextrema(np.array(price_max_data),np.less)[0]))
                    residual = price_max_data
            
                    max_freq = 1.0/(mean_max_min+mean_min_max-2*std_min_max-2*std_max_min)
                    min_freq = 1.0/(mean_max_min+mean_min_max+2*std_min_max+2*std_max_min)
                    last_freq = 1.0/(imf_min_true_index[-1]-imf_min_true_index[-2])
                    print "max freq %s"%max_freq
                    print "min freq %s"%min_freq
                    print "last freq %s"%last_freq


                    rawdata = price_max_data
                    data = [rawdata[i]-rawdata[i+1] for i in range(len(rawdata)-1)]
                    max_index = list(argrelextrema(np.array(data),np.greater)[0])
                    min_index = list(argrelextrema(np.array(data),np.less)[0])
                            
                    data = price_max_data
                    data = self.close_price[current_price_index-10:current_price_index+1]
                    cha_data = [data[i+1]-data[i] for i in range(len(data)-1)]
                    
                    #print "std data %s"%(np.std(data)/np.mean(data))
                    print "std data %s"%(np.std(cha_data))
                    std_flag = ((np.std(data)/np.mean(data)))
                    self.ren_std.append(std_flag)
                    print "mean std %s"%np.mean(self.ren_std)
                    print "mean all std %s"%np.mean(self.list_std[-10:])
                    mean_all_std = np.mean(self.list_std[-10:])
                    print "mean all std std %s"%np.std(self.list_std[-10:])
                    std_all_std = np.std(self.list_std[-10:])
                    cd11 = list(range(len(data)))
                    cd22 = list(range(len(data)))[::-1]
                    up_test1 = []
                    down_test1 = []
                    up_test1.append((np.corrcoef(data,cd11)[0][1]))
                    down_test1.append((np.corrcoef(data,cd22)[0][1]))
                    print "\n\n\n"

                    trend = 0
                    if self.ma_count2>0 and self.ma_count>0 and self.ma_count3>0 and up_test1[0]>0.8:
                        trend = 1
                    elif self.ma_count2<0  and (self.ma_count<0 or self.ma_count3<0) and down_test1[0]>0.35:
                        trend = -1
                    elif down_test1[0]>0.65:
                        trend = -1
                    print "trend %s"%trend
                    
                    fluct = 0     
                    if std_flag>(mean_all_std+2*std_all_std):
                        fluct=1
                    #elif std_flag<mean_all_std-std_all_std:
                    elif ((mean_all_std-std_flag)/mean_all_std)>0.3:
                        fluct=-1
                    print "fluct %s"%fluct

                    if fluct == 0:
                        if len(data)>15:
                            num = len(data)/3
                            duan = 3
                        else:
                            num = len(data)/2
                            duan = 2
                    elif fluct ==1: 
                        if -0.7<up_test1[0]<0.7  :
                            num = len(data)/2
                            duan = 2
                        else:
                            num = len(data)
                            duan = 1
                    elif fluct==-1:
                        num = len(data)/3
                        duan = 3
                        
                        



                    data = self.close_price[current_price_index-9:current_price_index+1]
                    cd1 = list(range(num))
                    cd11 = list(range(len(data)))
                    cd2 = list(range(num))[::-1]
                    cd22 = list(range(len(data)))[::-1]
                    up_test1 = []
                    down_test1 = []
                    up_test1.append((np.corrcoef(data,cd11)[0][1]))
                    down_test1.append((np.corrcoef(data,cd22)[0][1]))
                    print "up_test1 %s"%up_test1
                    print "down_test1 %s"%down_test1
 
                    up_test = []
                    down_test = []
                    i = 0
        
                #    if duan>1:
                #        for i in range(1,duan):
                #            dd = data[i*num-num:i*num]
                #            print "dd %s"%dd
                #            print "%s dd %s"%(i,np.std(dd)/np.mean(dd))
                #            #print "up test %s"%(np.corrcoef(dd,cd1)[0][1])
                #            print "down test %s"%(np.corrcoef(dd,cd2)[0][1])
                #            up_test.append((np.corrcoef(dd,cd1)[0][1]))
                #            down_test.append((np.corrcoef(dd,cd2)[0][1]))
                #    dd = data[i*num:]
                #    print "dd %s"%dd
                #    cd1 = list(range(len(dd)))
                #    cd2 = list(range(len(dd)))[::-1]
                #    up_test.append((np.corrcoef(dd,cd1)[0][1]))
                #    down_test.append((np.corrcoef(dd,cd2)[0][1]))
                #    print "up test %s"%(np.corrcoef(dd,cd1)[0][1])
                #    print "down test %s"%(np.corrcoef(dd,cd2)[0][1])

               #     if (duan==3 and up_test[0]>0 and down_test[-1]>0.5) or (duan==2 and up_test[0]>0 and down_test[1]>0.5) or (up_test[0]>0 and down_test[-1]>0.5 and len(dd)>5) or (duan==3 and up_test[0]>0 and 0<up_test[-1]<up_test[-2]):
               #         down_test_decision1 = -1
               #     if up_test1[0]>0.7 :
               #         down_test_decision1 = 0

               #     if (duan==3 and down_test[0]>0 and down_test[1]>0 and down_test[2]>0) or (duan==2 and down_test[0]>0 and down_test[1]>0):
               #         down_test_decision1 = 0
               #     if (duan==3 and up_test[0]>0 and up_test[1]>0 and up_test[2]>0) or (duan==2 and up_test[0]>0 and up_test[1]>0):
               #         down_test_decision1 = 0
               #     if (len(up_test)==2 and up_test[0]>up_test[1] and up_test1[0]>0.8 ) or (len(up_test)==3 and up_test1[0]>0.8 and (up_test[0]>up_test[2] or up_test[1]>up_test[2]) ):
               #         down_test_decision = -1
                             

#######################################
                 ##   if trend>=0:
                 ##       if trend == 1:
                 ##           if fluct==1:
                 ##               if duan==1:
                 ##                   pass
                 ##               if duan==2 and up_test[0]>up_test[1]:
                 ##                   print "down 11"
                 ##                   down_test_decision = -1
                 ##           elif fluct==-1:
                 ##               if duan==3:
                 ##                   if up_test[1]>up_test[2]:
                 ##                       print "down 12"
                 ##                       down_test_decision = -1
                 ##           elif fluct ==0: 
                 ##               if duan==2:
                 ##                   if up_test[0]>up_test[1]:
                 ##                       print "down 13"
                 ##                       down_test_decision = -1
                 ##               if duan==3:
                 ##                   if up_test[1]>up_test[2] and up_test[2]<0.5:
                 ##                       print "down 14"
                 ##                       down_test_decision = -1
                 ##       elif trend == 0:
                 ##           if fluct==1:
                 ##               if duan==1:
                 ##                   pass
                 ##               if duan==2 and up_test[0]>0.5 and down_test[1]>0.5:
                 ##                   print "down 111"
                 ##                   down_test_decision = -1
                 ##           elif fluct==-1:
                 ##               if duan==3:
                 ##                   if up_test[0]>0.5 and down_test[2]>0.5:
                 ##                       print "down 122"
                 ##                       down_test_decision = -1
                 ##           elif fluct ==0:
                 ##               if duan==2:
                 ##                   if up_test[0]>0.5 and down_test[1]>0.5:
                 ##                       print "down 133"
                 ##                       down_test_decision = -1
                 ##               if duan==3:
                 ##                   if up_test[0]>0.5 and down_test[2]>0.5:
                 ##                       print "down 144"
                 ##                       down_test_decision = -1


                        



#######################################
                          

#                    last_five = data[-3:]
#                    last_five_before = data[-4:-1]
#                    last_five_before2 = data[-5:-2]
#                    last_five_before3 = data[-6:-3]
#
                    last_five = data[-5:]
                    last_five_before = data[-6:-1]
                    last_five_before2 = data[-7:-2]
                    last_five_before3 = data[-8:-3]
                    last_five_before4 = data[-9:-4]
                    other = data[:-5]

                    cd1 = list(range(5))
                    cd2 = list(range(5))[::-1]
                    cd21 = list(range(len(data)-5))
                    cd22 = list(range(len(data)-5))[::-1]

                    down_test = []
                    i = 0

                    #print "other down test %s"%(np.corrcoef(other,cd22)[0][1])
                    #down_test.append((np.corrcoef(other,cd22)[0][1]))
                    print "5 down test %s"%(np.corrcoef(last_five,cd2)[0][1])
                    down_test.append((np.corrcoef(last_five,cd2)[0][1]))

                    print "other %s"%other
                    print "last_five %s"%last_five
                    print "last_five_before %s"%last_five_before

                    down_test_before = np.corrcoef(last_five_before,cd2)[0][1]
                    down_test_before2 = np.corrcoef(last_five_before2,cd2)[0][1]
                    down_test_before3 = np.corrcoef(last_five_before3,cd2)[0][1]
                    #down_test_before2 = np.corrcoef(last_five_before2,cd2)[0][1]
                    #down_test_before3 = np.corrcoef(last_five_before3,cd2)[0][1]
                    print "before 5 down test %s"%(np.corrcoef(last_five_before,cd2)[0][1])
                    print "before 5 2down test %s"%(np.corrcoef(last_five_before2,cd2)[0][1])
                    print "before 5 3down test %s"%(np.corrcoef(last_five_before3,cd2)[0][1])


######

                    data = self.close_price[current_price_index-300:current_price_index+1]
                    i=imf_min_true_index[-1]-imf_max_true_index[-1]-2
                    if ma_decision10==1:
                        i=5
                    else:
                        i=10
                    i=estimate_period
                    #if std_flag>0.03:
                    #    i=20
                    #else:
                    #    i=5
                    #i=estimate_period
                    #i=20
                    last_five = data[-i:]
                    last_five_before = data[-(i+1):-1]
                    last_five_before2 = data[-(i+2):-2]
                    last_five_before3 = data[-(i+3):-3]
                    last_five_before4 = data[-(i+4):-4]
                    last_five_before5 = data[-(i+5):-5]

                    data_line = splinerestruct(price_max_data,1)
                    print "data line"
                    print data_line
                    data_line_arc = np.arctan((data_line[-1]-data_line[0])/(len(data_line)-1))
                    print data_line_arc
  
              
                    #last_five = data[-(499-imf_min_true_index[-1]+1):]
                    #last_five_before = data[-(499-imf_min_true_index[-1]+1):-1]
                    #last_five_before2 = data[-(499-imf_min_true_index[-1]+1):-2]
                    #last_five_before3 = data[-(499-imf_min_true_index[-1]+1):-3]
                    print "haha %s %s"%(last_five,last_five_before)
                    line = splinerestruct(last_five,1)
                    print "line"
                    print line
                    line_before = splinerestruct(last_five_before,1)
                    print "line"
                    print line_before
                    line_before2 = splinerestruct(last_five_before2,1)
                    print "line"
                    print line_before2
                    line_before3 = splinerestruct(last_five_before3,1)
                    line_before4 = splinerestruct(last_five_before4,1)
                    line_before5 = splinerestruct(last_five_before5,1)
                    print "line"
                    print line_before3


                    #line_arc = np.arctan((line[-1]-line[0])/(len(line)-1))
                    line_arc = ((line[-1]-line[0])/(len(line)-1))
                    self.down_test.append(line_arc)
                    line_arc_before = ((line_before[-1]-line_before[0])/(len(line)-1))
                    line_arc_before2 = ((line_before2[-1]-line_before2[0])/(len(line_before2)-1))
                    line_arc_before3 = ((line_before3[-1]-line_before3[0])/(len(line_before3)-1))
                    line_arc_before4 = np.arctan((line_before4[-1]-line_before4[0])/(len(line_before4)-1))
                    line_arc_before5 = np.arctan((line_before5[-1]-line_before5[0])/(len(line_before5)-1))
                    print "arc %s %s %s %s"%(line_arc,line_arc_before,line_arc_before2,line_arc_before3)
                    if 0:#len(self.down_test)>7:
                        #line_arc_re = splinerestruct(self.down_test[-5:],3)
                        #line_arc_re = cubicSmooth5(self.down_test[-8:])
                        line_arc_re = self.down_test
                        line_arc = line_arc_re[-1]
                        line_arc_before = line_arc_re[-2]
                        line_arc_before2 = line_arc_re[-3]
                        line_arc_before3 = line_arc_re[-4]
                        line_arc_before4 = line_arc_re[-5]
                        line_arc_before5 = line_arc_re[-6]
                        line_arc_before6 = line_arc_re[-7]
                        print "arc %s "%(line_arc_re)
                    print "arc %s %s %s %s"%(line_arc,line_arc_before,line_arc_before2,line_arc_before3)
                    last_big=0
                    if len(self.down_test)>3:
                        max_xielv = list(argrelextrema(np.array(self.down_test),np.greater)[0])
                        if max_xielv!=[]:
                            last_big=self.down_test[max_xielv[-1]]
#######


                  
                    line_line_arc = [line_arc_before4,line_arc_before3,line_arc_before2,line_arc_before,line_arc]
                    line_line_arc2 = [line_arc_before5,line_arc_before4,line_arc_before3,line_arc_before2,line_arc_before]
                    if 1:#self.buy_price!=0:
                        line_line_arc = [line_arc_before2,line_arc_before,line_arc]
                        #line_line_arc = [line_arc_before,line_arc]
                        line_line_arc2 = [line_arc_before3,line_arc_before2,line_arc_before]
                        line_line_arc3 = [line_arc_before4,line_arc_before3,line_arc_before2]
                        line_line_arc4 = [line_arc_before5,line_arc_before4,line_arc_before3]
                        #line_line_arc2 = [line_arc_before2,line_arc_before]
                    line_line_arc3 = [line_arc_before4,line_arc_before3,line_arc_before2]
                    line_line = ((line_line_arc[-1]-line_line_arc[0])/(len(line_line_arc)-1))
                    line_line2 = ((line_line_arc2[-1]-line_line_arc2[0])/(len(line_line_arc2)-1))
                    line_line3 = ((line_line_arc3[-1]-line_line_arc3[0])/(len(line_line_arc3)-1))
                    line_line4 = ((line_line_arc4[-1]-line_line_arc4[0])/(len(line_line_arc4)-1))
                    self.down_test1.append(line_line)
                    print "last five mean%s"%np.mean(self.down_test1[-5:])
                    print "down test1 %s"%self.down_test1[-10:]
                    down_test1_data = np.array(self.down_test1[-10:])
                    down_test1_max_index = list(argrelextrema(down_test1_data,np.greater)[0])
                    down_test1_min_index = list(argrelextrema(down_test1_data,np.greater)[0])
                    print " down test 1 std data %s"%(np.std(down_test1_data)/np.mean(down_test1_data))
                    if len(self.down_test1)>3:
                        line_line2 = self.down_test1[-2]
                        line_line3 = self.down_test1[-3]
                        line_line4 = self.down_test1[-4]


#########################
                    curve_min_index=[]
                    curve_max_index=[]
                    
                    if len(self.down_test)>3:
                        curve = splinerestruct(self.down_test[-10:],2)
                        curve_max_index = list(argrelextrema(np.array(curve),np.greater)[0])
                        curve_min_index = list(argrelextrema(np.array(curve),np.less)[0])
                        print "curve %s"%curve
                    curve_min_index.insert(0,0)
                    curve_max_index.insert(0,0)
                    print "curve max%s"%curve_max_index
                    print "curve min%s"%curve_min_index
                 
##############

                    #if line_arc>line_arc_before>line_arc_before2 and line_arc_before2<line_arc_before3:
                    #if  line_arc>0 and line_arc_before<0 :#>line_arc_before2 and line_arc_before2<line_arc_before3 :#and line_arc_before3<line_arc_before4:#and line_arc_before3<line_arc_before4 :
                    #if 0.1>line_line>line_line2>line_line3 and line_line3<line_line4:# and line_line4>line_line3:
                    #if len(self.down_test)>3 and curve_min_index[-1]>curve_max_index[-1] and curve_min_index[-1]>6:
                    #if self.core==self.core2==current_core and line_arc>line_arc_before and line_arc_before<line_arc_before2:
                    #    down_test_decision1=0 
                    current_core = current_price_index-(499-imf_min_true_index[-1])
                    print "%s core %s last core %s now core %s"%(self.core3,self.core2,self.core,current_core)
                    if 0:#abs(current_core-self.core)>2:
                        down_test_decision1=0
                    if 0:#abs(self.core-current_core)<2:
                        b2data =  self.close_price[current_core:current_price_index-1]
                        bdata =  self.close_price[current_core:current_price_index]
                        print "bdata %s"%bdata
                        print "b2data %s"%b2data
                        b3data =  self.close_price[current_core:current_price_index-2]
                        print "b3data %s"%b3data
                        print "last five %s"%last_five
                        bline = splinerestruct(bdata,1)
                        blinearc = ((bline[-1]-bline[0])/(len(bline)-1))
                        b2line = splinerestruct(b2data,1)
                        b2linearc = ((b2line[-1]-b2line[0])/(len(b2line)-1))
                        b3line = splinerestruct(b3data,1)
                        b3linearc = ((b3line[-1]-b3line[0])/(len(b3line)-1))
                        line_arc_before=blinearc
                        line_arc_before2=b2linearc
                        line_arc_before3=b3linearc
                    print "fin arc %s %s %s %s"%(line_arc,line_arc_before,line_arc_before2,line_arc_before3)
                    delta1= self.close_price[current_price_index]-self.close_price[current_price_index-1]
                    delta2= self.close_price[current_price_index-1]-self.close_price[current_price_index-2]
                    delta3= self.close_price[current_price_index-2]-self.close_price[current_price_index-3]
                    print "delta %s %s %s"%(delta1,delta2,delta3)
                    big_imf_max_true_index = filter(lambda n:n<=imf_min_true_index[-1], big_imf_max_index)
                    big_imf_min_true_index = filter(lambda n:n<=imf_min_true_index[-1], big_imf_min_index)
                    imf3_flag=0
                    if big_imf_max_true_index[-1]<big_imf_min_true_index[-1] :#and big_imf_min_true_index[-1]>485:
                        current_imf3_core = current_price_index-(499-big_imf_min_true_index[-1])
                    #    if 0.8<((499-big_imf_min_true_index[-1])/(big_imf_min_true_index[-1]-big_imf_max_true_index[-1]))<1.2:
                    #        imf3_flag=1
                        if (499-big_imf_min_true_index[-1])<5:
                            imf3_flag=1
                    else:
                        current_imf3_core = current_price_index-(499-big_imf_max_true_index[-1])
                        if 0.8<((499-big_imf_max_true_index[-1])/(big_imf_max_true_index[-1]-big_imf_min_true_index[-1])):
                            imf3_flag=1
                    print "imf3 core %s %s"%(self.imf3_core,current_imf3_core)
                    self.imf3_core=current_imf3_core


                    ###if  delta1>0  and imf3_flag==1 and abs(self.core-self.core3)<2 and abs(self.core-current_core)<2  and delta1>delta2 and abs(current_core-self.core2)<2:#line_arc>line_arc_before and line_arc_before<line_arc_before2 and line_arc_before2<line_arc_before3:
                    #if   line_arc>line_arc_before and line_arc_before<line_arc_before2 and abs(current_core-self.core)<2:#line_arc>line_arc_before and line_arc_before<line_arc_before2 and line_arc_before2<line_arc_before3:
                    #if line_arc>line_arc_before and line_arc_before<line_arc_before2<line_arc_before3 and abs(current_core-self.core2)<=2 and abs(current_core-self.core)<=2 and imf1_flag==1:
                    #if   line_arc>line_arc_before and abs(self.core-current_core)<=2  and delta1>delta2 and delta2<delta3 and abs(current_core-self.core2)<=2:#line_arc>line_arc_before and line_arc_before<line_arc_before2 and line_arc_before2<line_arc_before3:
                    last_cha = imf_min_true_index[-1]-imf_max_true_index[-1]
                    #last_cha = round(mean_max_min)
                    est_per = round((499-imf_min_true_index[-1])/float(last_cha))
                    yu = abs(((499-imf_min_true_index[-1])/float(last_cha))-est_per)
                    print "est per %s %s"%(est_per,yu) 
                    if  (imf_min_true_index[-1]>496 or self.last_last=="max")  :#and abs(((self.core+self.core2)/2.0)-current_core)>3 :#and abs(self.core-self.core3)<=2 and abs(self.core-self.core2)<=2:#est_per%2==0 and (yu<0.3 or yu>0.7):
                        down_test_decision1=-1
                    #if line_arc>0 and line_arc_before<0 and line_arc_before2<0:#and line_arc_before3<line_arc_before4:#and line_arc<0.2:#and line_arc_before3<line_arc_before4 and line_arc_before4<line_arc_before5:
                    #    down_test_decision1=0
                    if  ma_decision10==1 and ( line_arc>line_arc_before and line_arc_before<line_arc_before2) :#or ( down_test[-1]<down_test_before<down_test_before2 and down_test_before3<down_test_before2 )) and ma_decision10==1:
                        down_test_decision2=0   
                    if (imf_min_true_index[-1]<495 and imf_min_true_index[-1]-imf_max_true_index[-1]<15) or (imf_min_true_index[-1]<493 and imf_min_true_index[-1]-imf_max_true_index[-1]>15):#or small_imf_max_true_index[-1]<small_imf_min_true_index[-1]:#or abs(data_line_arc)>0.3:
                        down_test_decision1=0   
                        down_test_decision2=0   
                    #if self.buy_price!=0 and  self.close_price[current_price_index]<self.close_price[current_price_index-1]:#and line_arc_before>line_arc_before2: 
                    #if self.buy_price!=0 and line_line<0 and line_line2>0:
                    #if self.buy_price!=0 and line_line<0 and line_line2>0:
                    ##if len(self.down_test)>3 and curve_min_index[-1]<curve_max_index[-1] :
                    #if  len(self.down_test)>2 and self.buy_price!=0 and self.down_test[-1]<self.down_test[-2] and self.down_test[-2]>self.down_test[-3]:
                    #if  abs(self.core-current_core)<2 and  last_five[-1]<bdata[-1] and bdata[-1]>b2data[-1] :#line_arc>line_arc_before and line_arc_before<line_arc_before2 and line_arc_before2<line_arc_before3:
                    #if  self.buy_price!=0 and self.buy_core!=current_core and abs(self.core-current_core)<2 and line_arc<line_arc_before :#delta1<delta2<delta3 :#line_arc>line_arc_before and line_arc_before<line_arc_before2 and line_arc_before2<line_arc_before3:
                    #    down_test_decision1=1   
                    #if   self.buy_price!=0 and self.buy_core!=current_core and abs(self.core-current_core)>2 and abs(self.core-self.core2)<2 and line_arc<line_arc_before:#delta1<delta2<delta3 :#line_arc>line_arc_before and line_arc_before<line_arc_before2 and line_arc_before2<line_arc_before3:
                    #if  self.buy_price!=0 and abs(self.buy_core-current_core)>2 and abs(current_core-self.core)<=2 and est_per%2==1 and (yu<=0.4 or yu>=0.6):
                    print "imf1 flag%s"%imf1_flag
                    if imf1_flag==2 and imf_min_true_index[-1]<496:
                        down_test_decision1=1   
                    self.core3 = self.core2
                    self.core2 = self.core
                    self.core = current_core

                    self.down_decision=down_test_decision1

########################################
                    flag=0
                    small_index_max = list(filter(lambda n:n>imf_min_true_index[-1], small_imf_max_index))
                    small_index_max_remove = list(filter(lambda n:n<=imf_min_true_index[-1], small_imf_max_index))
                    if len(small_index_max_remove)>0 and len(small_index_max)>0 and imf_min_true_index[-1]-small_index_max_remove[-1]<small_index_max[0]-imf_min_true_index[-1]:
                        small_index_max.insert(0,small_index_max_remove[-1])
                        flag=1
                    small_index_min = list(filter(lambda n:n>imf_min_true_index[-1], small_imf_min_index))
                    small_index_min_remove = list(filter(lambda n:n<=imf_min_true_index[-1], small_imf_min_index))

                    small_small_index_max = list(filter(lambda n:n>imf_min_true_index[-1], small_small_imf_max_index))
                    small_small_index_max_remove = list(filter(lambda n:n<=imf_min_true_index[-1], small_small_imf_max_index))
                    small_small_index_min = list(filter(lambda n:n>imf_min_true_index[-1], small_small_imf_min_index))
                    print small_index_max

                    max_index=[]
                    min_index=[]
                    

                    if ((np.std(data)/np.mean(data)))>0.04:#max_index==[] :#or choice==0:
                        print "enter 2"
                        if 1:#max_index==[]:
                            max_index=[]
                            max_index.append(imf_max_true_index[-1])
                        if 1:#min_index==[]:
                            min_index=[]
                            min_index.append(imf_min_true_index[-1])

                    
                    tmp = max_index+min_index
                    tmp.sort()
                        
                    print "tmp %s "%tmp
                    if tmp!=[] and max_index != []:
                        #if (tmp[0] in max_index) and  499-max_index[-1]>mean_max_min-1 and 499-max_index[-1]<mean_max_min+2:#+2*std_max_min:# and len(data)-max_index[-1]<mean_max_min+2*std_max_min:
                        #    imf_extrem_decision6 = -1
                        if (tmp[-1] in max_index) and 490<tmp[-1]<496:#std_max_min and len(data)-max_index[-1]<dis+std_max_min:#+2*std_max_min:# and len(data)-max_index[-1]<mean_max_min+2*std_max_min:
                            imf_extrem_decision6 = -1
                        if (tmp[-1] in max_index) and tmp[-1]>496:# and 499-min_index[-1]<(mean_min_max)+std_min_max:#std_max_min and len(data)-max_index[-1]<dis+std_max_min:#+2*std_max_min:# and len(data)-max_index[-1]<mean_max_min+2*std_max_min:
                            imf_extrem_decision6 = 1
                    if tmp!=[] and min_index != []:
                        #if (tmp[0] in max_index) and tmp[0]>4 and (tmp[-1] in min_index)  and len(data)-min_index[-1]<6:
                        if (tmp[-1] in min_index) and 490<tmp[-1]<496:
                            imf_extrem_decision6 = 1
                        if (tmp[-1] in min_index) and tmp[-1]>496:
                            imf_extrem_decision6 = -1
                        #if self.last_min_index == [] and self.last_max_index==[] and min_index!=[]:
                        #    imf_extrem_decision6 = 0
#                    if len(tmp)>2 or too_far==1:
#                        imf_extrem_decision6 = 0
  
                    if ((data[0]-data[-1])/data[0])>0.15:
                        imf_extrem_decision6 = 0
                    #if tmp!=[]:
                    #    if tmp[0] in min_index:
                    #        imf_extrem_decision6 = -1

                    print "imf deicison 6 %s"%imf_extrem_decision6


                    buy_decision3 = 0
                    if (std_flag>0.04 and (up_test1[0]>0.85 or down_test1[0]>0.85) and imf_extrem_decision6==-1) or (std_flag<0.04 and down_test_decision2==-1):
                        buy_decision3 = -1

                    self.last_last = "min"
              
                 

                elif imf_max_true_index[-1]<imf_min_true_index[-1] and (499-imf_min_true_index[-1])<3 :
            #            imf_extrem_decision = -1
                    print "pp %s"%(buyhigh/buyhigh2)
                    imf_extrem_decision = -1
                elif imf_max_true_index[-1]>imf_min_true_index[-1] :#and 499-imf_max_true_index[-1]>mean_max_min :#and imf_max_true_index[-1]>465:#and (((499-imf_max_true_index[-1])>(mean_max_min+std_max_min) and (499-imf_max_true_index[-1])>(mean_min_max+std_min_max)) or ((499-imf_max_true_index[-1])<(mean_max_min+3) and (499-imf_max_true_index[-1])>(mean_max_min-3) and len(imf_min_remove_index)>0)):
                    price_last_min_index = list(filter(lambda n:n>imf_max_true_index[-1],price_min_index))
                    print "price_last_min_index %s"%price_last_min_index


                    price_min_data = price_list[imf_max_true_index[-1]:] 
                    print "price min data %s"%price_min_data
                    print "price max %s"%list(argrelextrema(np.array(price_min_data),np.greater)[0])
                    print "price min %s"%list(argrelextrema(np.array(price_min_data),np.less)[0])

                    residual = price_min_data

                    max_freq = 1.0/(mean_max_min+mean_min_max-2*std_min_max-2*std_max_min)
                    min_freq = 1.0/(mean_max_min+mean_min_max+2*std_min_max+2*std_max_min)
                    last_freq = 1.0/(imf_max_true_index[-1]-imf_max_true_index[-2])
                    print "max freq %s"%max_freq
                    print "min freq %s"%min_freq
                    print "last freq %s"%last_freq


                    print "\n\n\n"
                    rawdata = price_min_data


                    data = price_min_data
                    #data = self.close_price[current_price_index-20:current_price_index+1]
                    print "std data %s"%(np.std(data)/np.mean(data))
                    std_flag = (np.std(data)/np.mean(data))
                    self.ren_std.append(std_flag)
                    print "mean std %s"%np.mean(self.ren_std)
                    print "mean all std %s"%np.mean(self.list_std[-10:])
                    print "mean all std std %s"%np.std(self.list_std[-10:])
                    mean_all_std = np.mean(self.list_std[-10:])
                    std_all_std = np.std(self.list_std[-10:])


                    cd11 = list(range(len(data)))
                    cd22 = list(range(len(data)))[::-1]
                    up_test1 = []
                    down_test1 = []
                    up_test1.append((np.corrcoef(data,cd11)[0][1]))
                    down_test1.append((np.corrcoef(data,cd22)[0][1]))
                    print "up_test1 %s"%up_test1
                    print "down_test1 %s"%down_test1



                    trend = 0
                    if self.ma_count2>0 and self.ma_count>0 and self.ma_count3>0 and up_test1[0]>0.8:
                        trend = 1
                    elif self.ma_count2<0  and (self.ma_count<0 or  self.ma_count3<0) and down_test1[0]>0.35:
                        trend = -1
                    elif down_test1[0]>0.65:
                        trend = -1
                    print "trend %s"%trend

                    fluct = 0
                    #if std_flag>(mean_all_std+2*std_all_std):
                    if ((std_flag-mean_all_std)/mean_all_std)>0.5:
                        fluct=1
                    #elif std_flag<mean_all_std-std_all_std:
                    elif ((mean_all_std-std_flag)/mean_all_std)>0.8:
                        fluct=-1
                    print "fluct %s"%fluct

                    if fluct == 0:
                        if len(data)>15:
                            num = len(data)/3
                            duan = 3
                        else:
                            num = len(data)/2
                            duan = 2
                    elif fluct ==1:
                        if -0.7<up_test1[0]<0.7  :
                            num = len(data)/2
                            duan = 2
                        else:
                            num = len(data)
                            duan = 1
                    elif fluct==-1:
                        num = len(data)/3
                        duan = 3



                    print "\n\n\n"
                    down_test = []
                    up_test = []
                    i=0

            #        cd1 = list(range(num))
            #        cd2 = list(range(num))[::-1]
            #        if duan>1:
            #            for i in range(1,duan):
            #                dd = data[i*num-num:i*num]
            #                print "dd %s"%dd
            #                print "%s dd %s"%(i,np.std(dd)/np.mean(dd))
            #                print "up test %s"%(np.corrcoef(dd,cd1)[0][1])
            #                print "down test %s"%(np.corrcoef(dd,cd2)[0][1])
            #                up_test.append((np.corrcoef(dd,cd1)[0][1]))
            #                down_test.append((np.corrcoef(dd,cd2)[0][1]))
            #        dd = data[i*num:]
            #        print "dd %s"%dd
            #        cd1 = list(range(len(dd)))
            #        cd2 = list(range(len(dd)))[::-1]
            #        up_test.append((np.corrcoef(dd,cd1)[0][1]))
            #        down_test.append((np.corrcoef(dd,cd2)[0][1]))
            #        print "up test %s"%(np.corrcoef(dd,cd1)[0][1])
            #        print "down test %s"%(np.corrcoef(dd,cd2)[0][1])

               #     if (duan==3 and down_test[0]>0 and  down_test[-1]>0.5) or (down_test[0]>0 and down_test[-1]>0.5 and len(dd)>5):# or (duan==2 and down_test[0]>0.5):
               #         down_test_decision1 = -1
               #     if up_test1[0]>0.7:
               #         down_test_decision1 = 0
               #     if (duan==3 and down_test[0]>0 and down_test[1]>0 and down_test[2]>0) or (duan==2 and down_test[0]>0 and down_test[1]>0):
               #         down_test_decision1 = 0
               #     if (duan==3 and up_test[0]>0 and up_test[1]>0 and up_test[2]>0) or (duan==2 and up_test[0]>0 and up_test[1]>0):
               #         down_test_decision1 = 0
               #     

               #     if len(up_test)==3 and up_test[1]>up_test[2] and up_test1[0]>0.8 :
               #         down_test_decision2 = -1

#######################################

#                    if trend>=0:
#                        if trend == 1:
#                            if fluct==1:
#                                if duan==1 and down_test[0]>0:
#                                    print "down 211"
#                                    down_test_decision = -1
#                                if duan==2 and up_test[1]<up_test[0]:
#                                    print "down 212"
#                                    down_test_decision = -1
#                            elif fluct==-1:
#                                if duan==3:
#                                    if up_test[1]>up_test[2]:
#                                        print "down 213"
#                                        down_test_decision = -1
#                            elif fluct ==0:
#                                if duan==2:
#                                    if up_test[1]<up_test[0] and up_test[1]<0.5:
#                                        print "down 214"
#                                        down_test_decision = -1
#                                if duan==3:
#                                    if up_test[1]>up_test[2]:
#                                        print "down 215"
#                                        down_test_decision = -1
#                        elif trend == 0:
#                            if fluct==1:
#                                if duan==1 and down_test[0]>0.5:
#                                    print "down 216"
#                                    down_test_decision = -1
#                                if duan==2 and down_test[0]>0 and down_test[1]>0:
#                                    print "down 217"
#                                    down_test_decision = -1
#                            elif fluct==-1:
#                                if duan==3:
#                                    if down_test[0]>0.3 and up_test[1]>0 and down_test[2]>0.5:
#                                        print "down 218"
#                                        down_test_decision = -1
#                            elif fluct ==0:
#                                if duan==2:
#                                    if down_test[0]>0 and down_test[1]>0:
#                                        print "down 219"
#                                        down_test_decision = -1
#                                if duan==3:
#                                    if down_test[0]>0.5 and up_test[1]>0 and down_test[2]>0.5:
#                                        print "down 2200"
#                                        down_test_decision = -1
#
#



#######################################

                    small_index_max = filter(lambda n:n<497, small_imf_max_index)
                    small_index_min = filter(lambda n:n<497, small_imf_min_index)
                        
                    small_de=0
                    if small_index_max[-1]>small_index_min[-1]:
                        chalist=[]
                        chalist.append(small_index_max[-1]-small_index_min[-1]) 
                        chalist.append(small_index_min[-1]-small_index_max[-2]) 
                        chalist.append(small_index_max[-2]-small_index_min[-2]) 
                        print "chalist%s"%chalist
                        if 0.5<((499-small_index_max[-1])/np.mean(chalist))<1.5:
                            small_de=1

                    data = self.close_price[current_price_index-9:current_price_index+1]
#                    last_five = data[-3:]
#                    last_five_before = data[-4:-1]
#                    last_five_before2 = data[-5:-2]
#                    last_five_before3 = data[-6:-3]
#                    last_five_before4 = data[-7:-4]

###########
                    data = self.close_price[current_price_index-300:current_price_index+1]
                    i=imf_max_true_index[-1]-imf_min_true_index[-1]-2
                    if ma_decision10==1:
                        i=5
                    else:
                        i=10
                    i=estimate_period
                    #if std_flag>0.03:
                    #    i=20
                    #else:
                    #    i=5
                    #i=estimate_period
                    i=5
                    last_five = data[-i:]
                    last_five_before = data[-(i+1):-1]
                    last_five_before2 = data[-(i+2):-2]
                    last_five_before3 = data[-(i+3):-3]
                    last_five_before4 = data[-(i+4):-4]
                    last_five_before5 = data[-(i+5):-5]


                    data_line = splinerestruct(price_min_data,1)
                    print "data line"
                    print data_line
                    data_line_arc = ((data_line[-1]-data_line[0])/(len(data_line)-1))
                    print data_line_arc

                    #last_five = data[-(499-imf_max_true_index[-1]+1):]
                    #last_five_before = data[-(499-imf_max_true_index[-1]+1):-1]
                    #last_five_before2 = data[-(499-imf_max_true_index[-1]+1):-2]
                    #last_five_before3 = data[-(499-imf_max_true_index[-1]+1):-3]
                    print "haha %s %s"%(last_five,last_five_before)
                    line = splinerestruct(last_five,1)
                    print "line"
                    print line
                    line_before = splinerestruct(last_five_before,1)
                    print "line"
                    print line_before
                    line_before2 = splinerestruct(last_five_before2,1)
                    print "line"
                    print line_before2
                    line_before3 = splinerestruct(last_five_before3,1)
                    line_before4 = splinerestruct(last_five_before4,1)
                    line_before5 = splinerestruct(last_five_before5,1)
                    print "line"
                    print line_before3


                    line_arc = ((line[-1]-line[0])/(len(line)-1))
                    self.down_test.append(line_arc)
                    line_arc_before = ((line_before[-1]-line_before[0])/(len(line)-1))
                    line_arc_before2 = ((line_before2[-1]-line_before2[0])/(len(line_before2)-1))
                    line_arc_before3 = ((line_before3[-1]-line_before3[0])/(len(line_before3)-1))
                    line_arc_before4 = np.arctan((line_before4[-1]-line_before4[0])/(len(line_before4)-1))
                    line_arc_before5 = np.arctan((line_before5[-1]-line_before5[0])/(len(line_before5)-1))
                    print "arc %s %s %s %s"%(line_arc,line_arc_before,line_arc_before2,line_arc_before3)

                    if 0:#len(self.down_test)>5:
                        #line_arc_re = splinerestruct(self.down_test[-5:],3)
                        #line_arc_re = cubicSmooth5(self.down_test[-8:])
                        line_arc_re = self.down_test
                        line_arc = line_arc_re[-1]
                        line_arc_before = line_arc_re[-2]
                        line_arc_before2 = line_arc_re[-3]
                        line_arc_before3 = line_arc_re[-4]
                        line_arc_before4 = line_arc_re[-5]
                        line_arc_before5 = line_arc_re[-6]
                        print "arc %s "%(line_arc_re)
                    print "arc %s %s %s %s"%(line_arc,line_arc_before,line_arc_before2,line_arc_before3)
                    last_big=0
                    if len(self.down_test)>3:
                        max_xielv = list(argrelextrema(np.array(self.down_test),np.greater)[0])
                        if max_xielv!=[]:
                            last_big=self.down_test[max_xielv[-1]]
#######

                    line_line_arc = [line_arc_before2,line_arc_before,line_arc]
                    line_line_arc2 = [line_arc_before3,line_arc_before2,line_arc_before]
                    line_line_arc = [line_arc_before4,line_arc_before3,line_arc_before2,line_arc_before,line_arc]
                    line_line_arc2 = [line_arc_before5,line_arc_before4,line_arc_before3,line_arc_before2,line_arc_before]
                    if 1:#self.buy_price!=0:
                        line_line_arc = [line_arc_before2,line_arc_before,line_arc]
                        #line_line_arc = [line_arc_before,line_arc]
                        line_line_arc2 = [line_arc_before3,line_arc_before2,line_arc_before]
                        line_line_arc3 = [line_arc_before4,line_arc_before3,line_arc_before2]
                        line_line_arc4 = [line_arc_before5,line_arc_before4,line_arc_before3]
                        #line_line_arc2 = [line_arc_before2,line_arc_before]
                    line_line_arc3 = [line_arc_before4,line_arc_before3,line_arc_before2]
                    line_line = ((line_line_arc[-1]-line_line_arc[0])/(len(line_line_arc)-1))
                    line_line2 = ((line_line_arc2[-1]-line_line_arc2[0])/(len(line_line_arc2)-1))
                    line_line3 = ((line_line_arc3[-1]-line_line_arc3[0])/(len(line_line_arc3)-1))
                    line_line4 = ((line_line_arc4[-1]-line_line_arc4[0])/(len(line_line_arc4)-1))
                    self.down_test1.append(line_line)
                    print "last five mean%s"%np.mean(self.down_test1[-5:])
                    down_test1_data = np.array(self.down_test1[-10:])
                    down_test1_max_index = list(argrelextrema(down_test1_data,np.greater)[0])
                    down_test1_min_index = list(argrelextrema(down_test1_data,np.greater)[0])
                    print " down test 1 std data %s"%(np.std(down_test1_data)/np.mean(down_test1_data))
                    print "down test1 %s"%self.down_test1[-10:]

                    if len(self.down_test1)>3:
                        line_line2 = self.down_test1[-2]
                        line_line3 = self.down_test1[-3]
                        line_line4 = self.down_test1[-4]
                    curve_min_index=[]
                    curve_max_index=[]
                    if len(self.down_test)>3:
                        curve = splinerestruct(self.down_test[-10:],2)
                        curve_max_index = list(argrelextrema(np.array(curve),np.greater)[0])
                        curve_min_index = list(argrelextrema(np.array(curve),np.less)[0])
                        print "curve %s"%curve
                    curve_min_index.insert(0,0)
                    curve_max_index.insert(0,0)
                    print "curve max%s"%curve_max_index
                    print "curve min%s"%curve_min_index
#                    if line_arc>line_arc_before>line_arc_before2 and line_arc_before2<line_arc_before3:
                    #if line_arc>line_arc_before and line_arc_before>line_arc_before2 and line_arc_before2<line_arc_before3 and line_arc_before3<line_arc_before4 and line_arc_before4<line_arc_before5 :
                     #   down_test_decision1=-1   
                    #if (line_arc>line_arc_before and line_arc_before<line_arc_before2) :#and big_imf_max_true_index[-1]>big_imf_min_true_index[-1]:#or (line_arc>line_arc_before and line_arc_before>line_arc_before2 and line_arc_before2<line_arc_before3) :#and line_arc_before2<line_arc_before3 :#and line_arc_before3<line_arc_before4:#and line_arc<0.2:#and line_arc_before3<line_arc_before4 and line_arc_before4<line_arc_before5:
                    #if  line_arc>0 and line_arc_before<0 :#>line_arc_before2 and line_arc_before2<line_arc_before3 :#and line_arc_before3<line_arc_before4:#and line_arc_before3<line_arc_before4:
                    #if 0.1>line_line>line_line2>line_line3 and line_line3<line_line4:# and line_line3<line_line4:# and line_arc_before<line_arc_before2 and line_arc_before2<line_arc_before3 :
                    if len(self.down_test)>3  and curve_max_index[-1]<curve_min_index[-1] and curve_min_index[-1]>6:
                        down_test_decision1=0
                    current_core = current_price_index-(499-imf_max_true_index[-1])
                    if 0:#abs(self.core-current_core)<2:
			b2data =  self.close_price[current_core:current_price_index-1]
                        bdata =  self.close_price[current_core:current_price_index]
                        print "bdata %s"%bdata
                        print "b2data %s"%b2data
                        b3data =  self.close_price[current_core:current_price_index-2]
                        print "b3data %s"%b3data
                        print "last five %s"%last_five
                        bline = splinerestruct(bdata,1)
                        b2line = splinerestruct(b2data,1)
                        b2linearc = ((b2line[-1]-b2line[0])/(len(b2line)-1))
                        b3line = splinerestruct(b3data,1)
                        b3linearc = ((b3line[-1]-b3line[0])/(len(b3line)-1))
                        blinearc = ((bline[-1]-bline[0])/(len(bline)-1))
                        line_arc_before=blinearc
                        line_arc_before2=b2linearc
                        line_arc_before3=b3linearc
                    print "%s core %s last core %s now core %s"%(self.core3,self.core2,self.core,current_core)
                    print "fin arc %s %s %s %s"%(line_arc,line_arc_before,line_arc_before2,line_arc_before3)
                    #if abs(current_core-self.core)>3:
                    #    down_test_decision1=-1
                    delta1= self.close_price[current_price_index]-self.close_price[current_price_index-1]
                    delta2= self.close_price[current_price_index-1]-self.close_price[current_price_index-2]
                    delta3= self.close_price[current_price_index-2]-self.close_price[current_price_index-3]
                    print "delta %s %s %s"%(delta1,delta2,delta3)
                    big_imf_max_true_index = filter(lambda n:n<=imf_max_true_index[-1], big_imf_max_index)
                    big_imf_min_true_index = filter(lambda n:n<=imf_max_true_index[-1], big_imf_min_index)
                    imf3_flag=0
                    if big_imf_max_true_index[-1]<big_imf_min_true_index[-1] :#and big_imf_min_true_index[-1]>485:
                        current_imf3_core = current_price_index-(499-big_imf_min_true_index[-1])
                        if (499-big_imf_min_true_index[-1])<5:
                            imf3_flag=1
                    else:
                        current_imf3_core = current_price_index-(499-big_imf_max_true_index[-1])
                        if 0.8<((499-big_imf_max_true_index[-1])/(big_imf_max_true_index[-1]-big_imf_min_true_index[-1]))<1.2:
                            imf3_flag=1
                    print "imf3 core %s %s"%(self.imf3_core,current_imf3_core)
                    self.imf3_core=current_imf3_core
                    ###if  delta1>0 and  imf3_flag==1 and abs(current_core-self.core2)>=2 and abs(self.core-current_core)<2  and data_line_arc<-0.01 and delta1>delta2 :#and delta2>delta3:#line_arc>line_arc_before and line_arc_before<line_arc_before2 and line_arc_before2<line_arc_before3:
                    #if  line_arc>line_arc_before and line_arc_before<line_arc_before2 and line_arc_before2<line_arc_before3 and imf1_flag==1 and abs(self.core-current_core)<=2 :# and data_line_arc<-0.01 :#and delta1>delta2 :#and delta2>delta3:#line_arc>line_arc_before and line_arc_before<line_arc_before2 and line_arc_before2<line_arc_before3:
                    last_cha = imf_max_true_index[-1]-imf_min_true_index[-1]
                    #last_cha = round(mean_max_min)
                    est_per = round((499-imf_max_true_index[-1])/float(last_cha))
                    yu = abs(((499-imf_max_true_index[-1])/float(last_cha))-est_per)
                    print "est per %s %s"%(est_per,yu)
                    #if  abs(self.core3-current_core)<2 and abs(self.core2-current_core)<2 and abs(self.core-current_core)<2 and line_arc>line_arc_before:#est_per%2==1 and (yu<0.3 or yu>0.7):
                    if  delta_macd1>delta_macd2>delta_macd3 and imf_max_true_index[-1]<=495 and imf1_flag==1:
                        down_test_decision1=-1
                    #if delta1>0  and  abs(self.core-self.core3)<2 and abs(self.core-self.core2)<2 and abs(self.core-current_core)>2 and delta1>delta2 :
                    #    down_test_decision1=-1
                    #if line_arc>0 and line_arc_before<0 and line_arc_before2<0 :#and line_arc_before3<line_arc_before4:#and line_arc<0.2:#and line_arc_before3<line_arc_before4 and line_arc_before4<line_arc_before5:
                    #    down_test_decision1=0
                    if  ma_decision10==1 and ( line_arc>line_arc_before and line_arc_before<line_arc_before2) :#or ( down_test[-1]<down_test_before<down_test_before2 and down_test_before3<down_test_before2 )) and ma_de
                        down_test_decision2=0
                    if 0:#imf_max_true_index[-1]>495 :#or small_imf_max_true_index[-1]<small_imf_min_true_index[-1]:#or abs(data_line_arc)>0.3 :
                        down_test_decision1=0
                        down_test_decision2=0
                    #if self.buy_price!=0 and line_arc<line_arc_before  and self.close_price[current_price_index]<self.close_price[current_price_index-1]:        
                    #if self.buy_price!=0 and line_line<0 and line_line2>0 :
                    #if self.buy_price!=0 and line_line<0 and line_line2>0:
                    #if len(self.down_test)>3 and curve_min_index[-1]<curve_max_index[-1] :
                    #    down_test_decision1=1
                    #if  line_arc<line_arc_before and line_arc_before>line_arc_before2:
                    #oif  abs(self.core-current_core)<2 and last_five[-1]<bdata[-1] and bdata[-1]>b2data[-1]:#line_arc>line_arc_before and line_arc_before<line_arc_before2 and line_arc_before2<line_arc_before3:
                    #if  self.buy_price!=0 and len(self.down_test)>2 and  self.down_test[-1]<self.down_test[-2] and self.down_test[-2]>self.down_test[-3]:
                    #if  abs(current_core-self.core)<=2 and self.buy_price!=0 and abs(current_core-self.buy_core)>2 and est_per%2==0 and (yu<=0.4 or yu>=0.6):#delta1<delta2<delta3 :
                    if imf_max_true_index[-1]>495 and self.buy_price!=0 and (current_core-self.buy_core)>2 :#and self.last_last=="min":
                        down_test_decision1=1
                    self.core3 = self.core2
                    self.core2 = self.core
                    self.core = current_core




########

                    self.down_decision=down_test_decision1




##################################



                    flag=0
                    small_index_max = filter(lambda n:n>imf_max_true_index[-1], small_imf_max_index)
                    small_index_max_remove = filter(lambda n:n<=imf_max_true_index[-1], small_imf_max_index)
                    small_index_min = filter(lambda n:n>imf_max_true_index[-1], small_imf_min_index)
                    small_index_min_remove = filter(lambda n:n<=imf_max_true_index[-1], small_imf_min_index)
                    if len(small_index_min_remove)>0 and len(small_index_min)>0 and imf_max_true_index[-1]-small_index_min_remove[-1]<small_index_min[0]-imf_max_true_index[-1]:
                        small_index_min.insert(0,small_index_min_remove[-1])
                        flag=1

                    small_small_index_max = list(filter(lambda n:n>imf_max_true_index[-1], small_small_imf_max_index))
                    small_small_index_min = list(filter(lambda n:n>imf_max_true_index[-1], small_small_imf_min_index))
                    small_small_index_min_remove = list(filter(lambda n:n<=imf_max_true_index[-1], small_small_imf_min_index))


                    max_index=[]
                    min_index=[]


                    print max_index
                    print min_index
                    if std_flag>0.04:#min_index==[] :#or choice==0:
                        print "enter 2"
                        if 1:#max_index==[]:
                            max_index=[]
                            max_index.append(imf_max_true_index[-1])
                        if 1:#min_index==[]:
                            min_index=[]
                            min_index.append(imf_min_true_index[-1])


                    tmp = max_index+min_index
                    tmp.sort()

                    print "tmp %s "%tmp
                    #if tmp == []:
                    #    tmp.append(0)
                    #    max_index.append(0)
                    if tmp!=[] and max_index != []:
                        #if (tmp[0] in max_index) and  499-max_index[-1]>mean_max_min-1 and 499-max_index[-1]<mean_max_min+2:#+2*std_max_min:# and len(data)-max_index[-1]<mean_max_min+2*std_max_min:
                        #    imf_extrem_decision6 = -1          
                        if (tmp[-1] in max_index) and 490<tmp[-1]<496:#std_max_min and len(data)-max_index[-1]<dis+std_max_min:#+2*std_max_min:# and len(data)-max_index[-1]<mean_max_min+2*std_max_min:
                            imf_extrem_decision6 = -1
                        if (tmp[-1] in max_index) and tmp[-1]>496:# and 499-min_index[-1]<(mean_min_max)+std_min_max:#std_max_min and len(data)-max_index[-1]<dis+std_max_min:#+2*std_max_min:# and len(data)-max_index[-1]<mean_max_min+2*std_max_min:
                            imf_extrem_decision6 = 1
                    if tmp!=[] and min_index != []:
                        #if (tmp[0] in max_index) and tmp[0]>4 and (tmp[-1] in min_index)  and len(data)-min_index[-1]<6:
                        if (tmp[-1] in min_index) and 490<tmp[-1]<496:
                            imf_extrem_decision6 = 1
                        if (tmp[-1] in min_index) and tmp[-1]>496:
                            imf_extrem_decision6 = -1
                    buy_decision3 = 0
                    if (std_flag>0.04 and (up_test1[0]>0.85 or down_test1[0]>0.85) and imf_extrem_decision6==-1) or (std_flag<0.04 and down_test_decision2==-1):
                        buy_decision3 = -1
                    print "imf deicison 6 %s"%imf_extrem_decision6
                    self.last_last = "max"





         
######################


        std_list_5 = self.close_price[current_price_index-4:current_price_index+1]
        std_list_10 = self.close_price[current_price_index-9:current_price_index+1]
        std_list_before = self.close_price[current_price_index-4:current_price_index]
        std_list_20 = self.close_price[current_price_index-19:current_price_index+1]
        std_list_20_before = self.close_price[current_price_index-20:current_price_index]
        std_list_20_before2 = self.close_price[current_price_index-21:current_price_index-1]
        std_list_20_before3 = self.close_price[current_price_index-22:current_price_index-2]
        std_list_20_before4 = self.close_price[current_price_index-23:current_price_index-3]
        print "std 5 %s"%np.std(std_list_5)
        print "std 5 %s"%std_list_5
        print "std 5 before %s"%np.std(std_list_before)
        print "std 20  %s"%np.std(std_list_20)
        print "std 20 before %s"%np.std(std_list_20_before)
        print "std 20 before2 %s"%np.std(std_list_20_before2)
        print "std 20 delta %s"%(np.std(std_list_20)-np.std(std_list_20_before))
        print "std 20 delta 2%s"%(np.std(std_list_20_before)-np.std(std_list_20_before2))

        print "portion %s"%(2*np.std(std_list_10)/np.mean(std_list_10))

        std_decision1 = 0
        #if np.std(std_list_before2)<np.std(std_list_before) and np.std(std_list_before)<np.std(std_list) and (2*np.std(std_list)/self.ma(self.close_price, 20, 0, current_price_index))>0.05:
        if (2*np.std(std_list_10)/np.mean(std_list_10))>0.045:
        #if np.std(std_list_20)>0.5:
            std_decision1 = 1
        else:
            std_decision1 = 0

###################        

        print "date %s"%self.date[current_price_index]
        print "date %s"%date
        print "current price %s"%self.close_price[current_price_index]
        print "trade price %s"%self.open_price[current_price_index+1]
        print "buy price %s"%self.buy_price
###########################################################################################
   
        recent_price_list = self.close_price[current_price_index-4:current_price_index+1]

        recent_price_format = self.format_data(recent_price_list)
        recent_price_format2 = recent_price_format[-5:] 
  


        print "recent price format %s"%recent_price_format2


        


        imf = list(imf_list[2])  
  
        imf_below_0_decision = 0
        print "last imf %s"%imf[-1]
        print "last imf period %s"%np.mean(imf[-3:])
        if imf[-1]<0:
            imf_below_0_decision = -1



        imf_delta3 = imf[-4]-imf[-3]
        imf_delta2 = imf[-2]-imf[-3]
        imf_delta1 = imf[-1]-imf[-2]
        print "imf delta %s"%imf_delta1
        print "imf delta2 %s"%imf_delta2
        print "imf delta3 %s"%imf_delta3
        imf_ma5_before_before_before = (imf[-8]+imf[-7]+imf[-6]+imf[-5]+imf[-4])/5.0
        imf_ma5_before_before = (imf[-7]+imf[-6]+imf[-5]+imf[-4]+imf[-3])/5.0
        imf_ma5_before = (imf[-6]+imf[-5]+imf[-4]+imf[-3]+imf[-2])/5.0
        imf_ma3_before = (imf[-4]+imf[-3]+imf[-2])/3.0

        imf_ma5_current = (imf[-5]+imf[-4]+imf[-3]+imf[-2]+imf[-1])/5.0
        imf_ma3_current = (imf[-3]+imf[-2]+imf[-1])/3.0
        imf_power_before_before_before = imf[-4] - imf_ma5_before_before_before
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
        if self.buy_price<current_price and len(self.buy_x_index)>0 and current_price_index-self.buy_x_index[-1]<4:
            distance_decision2 = 0

        distance_decision3 = 1
        if len(self.sell_x_index)>1 and current_price_index-self.sell_x_index[-1]<5:
            distance_decision3 = 0



##########
        price = self.close_price[current_price_index-9:current_price_index+1]
        price_ma5_before = (price[-6]+price[-5]+price[-4]+price[-3]+price[-2])/5.0
        price_ma5_before2 = (price[-7]+price[-6]+price[-5]+price[-4]+price[-3])/5.0
        price_ma3_before = (price[-4]+price[-3]+price[-2])/3.0
        price_ma5_current = (price[-5]+price[-4]+price[-3]+price[-2]+price[-1])/5.0
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
        if abs(sum(up_list)/sum(down_list))>0.7 and abs(sum(up_list)/sum(down_list))<1.4:
            sta_decision = 1
        
####################################################################################################################


        imf_decision = 0 
        data = np.array(last_5)
        last_5_max_index = argrelextrema(data,np.greater)[0]
        last_5_min_index = argrelextrema(data,np.less)[0]

        data = np.array(last_10)
        last_10_max_index = argrelextrema(data,np.greater)[0]
        last_10_min_index = argrelextrema(data,np.less)[0]


        last_10_imf = list(imf_list[2])[-10:]
        print "last 10 price %s"%last_10

        if len(last_5_max_index)>0 and len(last_5_min_index)>0: 
            last_10_imf_array = np.array(last_10_imf)
            last_10_imf_max_index = argrelextrema(last_10_imf_array,np.greater)[0]
            last_10_imf_min_index = argrelextrema(last_10_imf_array,np.less)[0]
            if len(last_10_imf_max_index)>0 and last_10_imf_max_index[-1]<4:
                imf_decision = -1
            if len(last_10_imf_min_index)>0 and imf_min_true_index[-1]>imf_max_true_index[-1]:
                imf_decision = 1

            
#####################################################################################################################

      
        #if  imf_power_current>imf_power_before:# and imf_3_power_current<0): #or float(imf_power_before)/imf_power_current > 2.5:
        if (self.last_imf_power<0 and imf_power_current>0) or (imf_power_before<0 and imf_power_current>0): #or float(imf_power_before)/imf_power_current > 2.5:
            imf_power_decision = -1


        imf_decision1 = 0
        print "last imf value %s"%imf_process[-5:]
        if imf_process[-1]<0 and imf_process[-2]<0:
            imf_decision1 = -1 

        ma5 = self.ma(self.close_price, 5, 0, current_price_index)
        deltama5 = self.ma(self.close_price, 5, 0, current_price_index)-self.ma(self.close_price, 5, -1, current_price_index)
        ma10 = self.ma(self.close_price, 10, 0, current_price_index)
        deltama10 = self.ma(self.close_price, 10, 0, current_price_index)-self.ma(self.close_price, 10, -1, current_price_index)
        ma20 = self.ma(self.close_price, 20, 0, current_price_index)
        deltama20 = self.ma(self.close_price, 20, 0, current_price_index)-self.ma(self.close_price, 20, -1, current_price_index)
        ma30 = self.ma(self.close_price, 30, 0, current_price_index)
        deltama30 = self.ma(self.close_price, 30, 0, current_price_index)-self.ma(self.close_price, 30, -1, current_price_index)
   
        ma_decision2 = 1
        if deltama5<0 and deltama10<0 and deltama20<0 and deltama30<0:
            ma_decision2 = 0
        ma_decision3 = 1
        if  (self.ma_count<=self.ma_count2<0 and self.ma_count<=-5) :#or (self.ma_count1<=self.ma_count<0 and self.ma_count1>-10):#and self.ma_count1<self.ma_count2<0:
            ma_decision3 = 0
        
        print "ma5 %s"%ma5
        print "delta ma5 %s"%deltama5
        print "ma10 %s"%ma10
        print "delta ma10 %s"%deltama10
        print "ma20 %s"%ma20
        print "delta ma20 %s"%deltama20
        print "ma30 %s"%ma30
        print "delta ma30 %s"%deltama30
        

        #vol_ma5 = self.ma(self.vol,5,0,current_price_index)
        #vol_ma5_before = self.ma(self.vol,5,-1,current_price_index)
        #vol_ma10 = self.ma(self.vol,10,0,current_price_index)
        #vol_ma20 = self.ma(self.vol,20,0,current_price_index)
        #vol_ma30 = self.ma(self.vol,30,0,current_price_index)
        #vol_ma60 = self.ma(self.vol,60,0,current_price_index)
        #print "vol 5 %s"%vol_ma5
        #print "vol 5 before %s"%vol_ma5_before
        #print "vol 10 %s"%vol_ma10
        #print "vol 20 %s"%vol_ma20
        #print "vol 30 %s"%vol_ma30
        #print "vol 60 %s"%vol_ma60


        imf_extrem_decision10 = 0
        if  imf_extrem_decision6==-1 and deltama5<0 and  deltama10<0 and deltama20<0 and deltama30<0:
            imf_extrem_decision10 = -2
            self.wait_flag=1
        elif  imf_extrem_decision6==-1:
            imf_extrem_decision10 = -1
        print "wait flag%s"%self.wait_flag
        print "deciosn10 %s"%imf_extrem_decision10





        print "std decision1 %s"%std_decision1
        ma_decision = 0
        deltama_list = [deltama10,deltama20,deltama30]
        tmp_delta_list = [i for i in deltama_list if i<0]
        if len(tmp_delta_list)>=2:
            ma_decision = 1
        print "ma deiciosn%s"%ma_decision

        print "imf power decision %s"%imf_power_decision
        trade_price = self.open_price[current_price_index+1]
        current_price = self.close_price[current_price_index]

        print "std imf %s"%np.std(list(imf_process))
        print "std 10 imf %s"%np.std(list(imf_process[-20:-10]))


        imf_std_decision = 0
        print "imf std %s"%(np.std(list(imf_process))/np.std(list(imf_process[-20:-10])))
        if (np.std(list(imf_process))/np.std(list(imf_process[-20:-10])))<5:
            imf_std_decision = 1


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
        if 0:#new_imf_max_true_index!= []:
            extrem_tmp = new_imf_max_true_index+new_imf_min_true_index
            extrem_tmp.sort()
            vol_list = []
            print current_price_index-500
 
            for i in range(len(extrem_tmp)-1):
                vol_list.append(self.vol_cal(current_price_index-500,extrem_tmp[i],extrem_tmp[i+1]))
            vol_list.append(self.vol_cal(current_price_index-500,extrem_tmp[i+1],500))
           
 
            print "vol %s"%vol_list
            (index,value) = self.restruct(vol_list,10,10)
            after_re=list(splinerestruct2(vol_list,value,index))
            print list(after_re)
            vol_max_index = list(argrelextrema(np.array(after_re),np.greater)[0])
            vol_min_index = list(argrelextrema(np.array(after_re),np.less)[0])
            vol_decision1 = 0
            print after_re[-1]
            print "last max %s"%after_re[vol_max_index[-1]]
            print "last min %s"%after_re[vol_min_index[-1]]
            max_diff = abs(after_re[-1]-after_re[vol_max_index[-1]])
            min_diff = abs(after_re[-1]-after_re[vol_min_index[-1]])

############
#########



        up_ma_decision1=1
        if (up_potion>1.5 and self.ma_count1<0) or up_potion>2 :#or up_potion<0.55:
            up_ma_decision1=0

        ma_macd_decision1=0
        if ((self.ma_count<0 and self.ma_count1<0 and self.ma_count2<0 and self.ma_count3<0) or (0>self.ma_count>self.ma_count1) ) and macd_list[-1]>macd_list[-2]:
            ma_macd_decision1=1

        ma_decision4=1
        if (self.ma_count1<self.ma_count<0 and (delta_power_current<0 or delta_power_current<delta_power_before)) or (self.ma_count<self.ma_count2<-1 and self.ma_count1<1 and (delta_power_current<0 or delta_power_current<delta_power_before) ) or (0>self.ma_count3>self.ma_count2>self.ma_count):
        #if (self.ma_count1<self.ma_count<0 ) or (self.ma_count<self.ma_count2<-1 and self.ma_count1<1 and (delta_power_current<0 or delta_power_current<delta_power_before) ) or (0>self.ma_count3>self.ma_count2>self.ma_count):
            ma_decision4=0



        macd_decision3 = 0
        if  delta_macd_0>delta_macd_1:#>delta_macd_2  :#or macd_list[-1]>macd_list[-2]:#>macd_list[-3]:
        #if  macd_list[-1]>macd_list[-2] and abs(macd_list[-1])<1 and abs(macd_list[-2])<1:#>macd_list[-3]:
            macd_decision3=1

        macd_decision4 = 1
        if macd_list[-1]<-1 or (macd_list[-1]>1 and macd_list[-1]<macd_list[-2]):
            macd_decision4=0

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
        macd_decision7=1
        if max_index==[0] and min_index==[0]  and macd_list_10[-1]>macd_list_10[-2]>0:
            macd_decision7=0

        this_buy = 0
        if  imf_extrem_decision6 == -1:#  (current_price-ma5) > (self.close_price[current_price_index-1]-self.ma(self.close_price, 5, -1, current_price_index)) and ( imf_extrem_decision6 == -1):
            this_buy = -1
        if this_buy == -1:
            self.buy_decision=-1

         
        if self.wait_flag>0:
            self.wait_count+=1
        

        if self.wait_count>5:
            self.wait_flag=0
            self.wait_count=0


        sell_decision1 = 0
        #if self.buy_decision == -1 and this_buy==0 :#and current_price>self.close_price[current_price_index-1] :#and self.buy_count>2:#  (current_price-ma5) > (self.close_price[current_price_index-1]-self.ma(self.close_price, 5, -1, current_price_index)) and ( imf_extrem_decision6 == -1):
         #   buy_decision2 = -1

        
        delta_buy_decision1=1
        if self.last_buy_result==1 or self.last_buy_result==3 or self.last_buy_result==4:
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


        cd20 = list(range(20))[::-1]
        price_20 = self.close_price[current_price_index-19:current_price_index+1]

        print "20 down test %s"%(np.corrcoef(price_20,cd20)[0][1])
        corrcoef_decision1=1
        if (np.corrcoef(price_20,cd20)[0][1])>0.85:
            corrcoef_decision1=0


######
        #train_file = "ma_train_data_set"
        #my_nn = neuroprediction(train_file,"net_file",5,10,0,1)
        #ann_result=[]
        #for i in range(1):
        ##    my_nn.train_nn()
        #    train_data = [self.ma_count1,self.ma_count,self.ma_count2,self.ma_count3,delta_power_current]
        #    ann_result.append(test_file("net_file",train_data)[0])
        #    print "ann result%s"%ann_result
        #ann_final= np.mean(ann_result)
        #print "ann result%s"%ann_final
        #ann_decision=0
        #if ann_final>0.6:
        #    ann_decision=1

######

        svm_test_data = [self.ma_count1,self.ma_count,self.ma_count2,self.ma_count3]
        svm_result = int(svm_predict(svm_test_data))
        print "svm result%s"%(svm_result)
        svm_decision=0
        if svm_result>0.8:
            svm_decision=1


        delta_macd1=self.macd[current_price_index]-self.macd[current_price_index-1]
        delta_macd2=self.macd[current_price_index-1]-self.macd[current_price_index-2]
        delta_macd3=self.macd[current_price_index-2]-self.macd[current_price_index-3]
        delta_macd4=self.macd[current_price_index-3]-self.macd[current_price_index-4]
        print "delta macd%s %s %s %s"%(delta_macd1,delta_macd2,delta_macd3,delta_macd4)


#######


        print "ma_decision4%s"%ma_decision4
        print "ma_decision5%s"%ma_decision5
        print "ma_decision6%s"%ma_decision6
        print "macd_decision4%s"%macd_decision4
        print "macd_decision3%s"%macd_decision3
        print "macd_decision5%s"%macd_decision5
        print "macd_decision6%s"%macd_decision6
        print "macd_decision7%s"%macd_decision7
        print "ma_decision10%s"%ma_decision10
        print "up deicision%s"%up_potion_decision2

        print "rsi deicision%s"%rsi_decision1
        print "open close deicision%s"%open_close_decision1
        print "delta_buy_decision1 deicision%s"%delta_buy_decision1
        print "down_test_decision1 deicision%s"%down_test_decision1
        print "down_test_decision2 deicision%s"%down_test_decision2
        print "svm deicision%s"%svm_decision



        print "period"
        print self.mean_distance(imf_p1)
        print self.mean_distance(imf_p2)
        

        print "last five mean%s"%np.std(self.down_test1[-5:])
        print "mean recent imf2 %s"%mean_recent_imf2_jiange

        print "open%s close%s"%(self.open_price[current_price_index-1],self.close_price[current_price_index-1])
        print "open%s close%s"%(self.open_price[current_price_index],current_price)
        if  self.sell_decision!=-1 and self.buy_price!=0 and ((trade_price-self.buy_price)/self.buy_price) >0.02:
            self.sell_decision = -1
 #       if distance_decision2 ==1 and self.buy_price!=0 and (imf_extrem_decision6==1 ) and ((trade_price-self.buy_price)/self.buy_price) >0.01:#and self.sell_decision==-1 and   ((self.macd[current_price_index]>0 and self.macd[current_price_index]<self.macd[current_price_index-1]<self.macd[current_price_index-2]) or (self.macd[current_price_index]<self.macd[current_price_index-1] and self.macd[current_price_index]<0)) and self.share > 0 :
        if  distance_decision2 ==1 and   self.buy_price!=0 and down_test_decision1==1: #and ((current_price-self.buy_price)/self.buy_price) >0.01:#(imf_extrem_decision6==1 ) :#and self.sell_decision==-1 and   ((self.macd[current_price_index]>0 and self.macd[current_price_index]<self.macd[current_price_index-1]<self.macd[current_price_index-2]) or (self.macd[current_price_index]<self.macd[current_price_index-1] and self.macd[current_price_index]<0)) and self.share > 0 :
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
            self.write_ma_train()
            self.ma_record=[]
            self.buy_core=0
        elif   self.buy_price!=0 and (((self.buy_price-current_price)/self.buy_price)>0.03 )  and self.share > 0:
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
            self.holdtime = 0
            once_flag = 1
            self.last_buy_result=2
            self.ma_record.append(-1)
            self.write_ma_train()
            self.ma_record=[]
            self.buy_core=0
     #   elif self.buy_price!=0 and self.buy_day>5 and   (down_test_decision1==1 or current_price>self.buy_price) and self.share > 0 :
     #       print "sell sell 4"
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
     #           self.last_buy_result=4
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
#        elif self.buy_price!=0 and distance_decision==1 and ((current_price-self.buy_price)/self.buy_price) >0.04 and self.share > 0 :
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
        #elif macd_decision7==1 and open_close_decision1==1 and ma_decision6==1 and delta_buy_decision1==1 and ma_decision4==1 and up_ma_decision1==1 and ma_decision5==1 and  macd_decision6==1 and  macd_decision5==1 and rsi_decision1==1 and macd_decision4==1 and  macd_decision3==1 and   self.buy_price==0 and   imf_extrem_decision6==-1  :#and self.macd[current_price_index]<self.macd[current_price_index-1]<self.macd[current_price_index-2]<self.macd[current_price_index-3]<self.macd[current_price_index-4]<self.macd[current_price_index-5] :#and small_imf_decision1 == -1:#imf_extrem_decision8 == 1 and buy_decision2 == -1:# and  self.close_price[current_price_index-1]<self.close_price[current_price_index]:# and imf_below_0_decision == -1:# and imf_power_decision == -1:
        elif delta_macd1>delta_macd2 and ma_decision3==1 and  self.buy_price==0 and  up_potion_decision2==1 and    down_test_decision1==-1 and delta_buy_decision1==1 :#and macd_decision3==1  :#imf_extrem_decision6==-1  :
        #elif macd_decision1==1 and self.buy_price==0 and   imf_extrem_decision6==-1  and self.macd[current_price_index]<self.macd[current_price_index-1]<self.macd[current_price_index-2]<self.macd[current_price_index-3]<self.macd[current_price_index-4] and small_imf_decision1==-1 and self.macd[current_price_index]<0.5 and  self.macd[current_price_index]>-1:#imf_extrem_decision8 == 1 and buy_decision2 == -1:# and  self.close_price[current_price_index-1]<self.close_price[current_price_index]:# and imf_below_0_decision == -1:# and imf_power_decision == -1:
            print "buy buy 1"
            print "estimate_period%s"%estimate_period
            print "vol %s"%self.vol[current_price_index-5:current_price_index+1]
            print "open%s close%s"%(self.open_price[current_price_index-1],self.close_price[current_price_index-1])
            print "open%s close%s"%(self.open_price[current_price_index],current_price)
            print "high%s low%s"%(self.high_price[current_price_index],self.low_price[current_price_index])
            self.buy_core=current_core
            self.ma_record.append(self.ma_count1)
            self.ma_record.append(self.ma_count)
            self.ma_record.append(self.ma_count2)
            self.ma_record.append(self.ma_count3)
  #          self.ma_record.append(delta_power_current)
            print "ma record%s"%self.ma_record
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
        elif 0:# self.buy_price==0 and down_test_decision1!=1 and down_test_decision2==-1 and delta_buy_decision1==1:#and macd_decision3==1  :#imf_extrem_decision6==-1  :
        #elif macd_decision1==1 and self.buy_price==0 and   imf_extrem_decision6==-1  and self.macd[current_price_index]<self.macd[current_price_index-1]<self.macd[current_price_index-2]<self.macd[current_price_index-3]<self.macd[current_price_index-4] and small_imf_decision1==-1 and self.macd[current_price_index]<0.5 and  self.macd[current_price_index]>-1:#imf_extrem_decision8 == 1 and buy_decision2 == -1:# and  self.close_price[current_price_index-1]<self.close_price[current_price_index]:# and imf_below_0_decision == -1:# and imf_power_decision == -1:
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
        else:
            self.buy_day = 0

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

        if this_buy == -1:
            self.buy_decision = -1
            self.buy_count += 1
        elif this_buy!=-1 and self.buy_count>2:
            self.buy_decision = 0
            self.buy_count = 0

        print "jiange %s"%self.jiange
        print "buy count %s"%self.buy_count
        print "buy decision %s"%self.buy_decision

            




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



        for i in range(len(imf)):
            print "imf%s data coef %s"%(i,np.corrcoef(emd_data,imf[i])[0][1])


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
        last_50_price = self.close_price[current_index-10:current_index+1]
        print "50 std %s"%np.std(last_50_price)
        print "std/mean %s"%(np.std(last_50_price)/np.mean(last_50_price))
        print "rsi 10 %s"%self.rsi(self.close_price,10,current_index)
        print "rsi 15 %s"%self.rsi(self.close_price,15,current_index)
        print "rsi 20 %s"%self.rsi(self.close_price,20,current_index)
        print "rsi 30 %s"%self.rsi(self.close_price,30,current_index)
        self.run_predict(imf,current_index,date,datafile,train_data_dir,train_number,20,result_number,residual)
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

        fp = open("imf4")
        lines = fp.readlines()
        fp.close()

        #print lines

        imf_raw4 = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            imf_raw4.append(float(eachline.split("\n")[0]))

        fp = open("residual")
        lines = fp.readlines()
        fp.close()
        residual_raw = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            residual_raw.append(float(eachline.split("\n")[0]))
        imf_without6 = [imf_without2[i]-residual_raw[i] for i in range(len(emd_data))]
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
        power_buy_value = [self.imf_power[i-start] for i in self.buy_x_index]
        power_sell_value = [self.imf_power[i-start] for i in self.sell_x_index]
        power_buy_index = [i-start for i in self.buy_x_index]
        power_sell_index = [i-start for i in self.sell_x_index]
        #imf_buy_value = [imf[i] for i in self.period_decision_buy_point]
        #imf_sell_value = [imf[i] for i in self.period_decision_sell_point]
        #self.buy_x_index = self.period_decision_buy_point
        #self.sell_x_index = self.period_decision_sell_point


        buy_x = [i-start for i in self.buy_x_index]
        sell_x = [i-start for i in self.sell_x_index]
        plt.figure(1)
        plt.subplot(411).axis([start,len(data),min(data[start:]),max(data[start:])])      
        plt.plot([i for i in range(len(data))],data,'o',[i for i in range(len(data))],data,'b',self.buy_x_index,self.buy_y_index,'r*',self.sell_x_index,self.sell_y_index,'g*')             
        plt.subplot(412).axis([start,len(imf),min(imf[start:]),max(imf[start:])])
        plt.plot([i for i in range(len(imf))],imf,'o',[i for i in range(len(imf))],imf,'b',self.buy_x_index,imf_buy_value,'r*',self.sell_x_index,imf_sell_value,'g*')             
#####

        buy_x_down_test = [i-start+1 for i in self.buy_x_index]
        buy_y_down_test = [self.down_test[i] for i in buy_x_down_test]
        sell_x_down_test = [i-start+1 for i in self.sell_x_index]
        sell_y_down_test = [self.down_test[i] for i in sell_x_down_test]

        buy_x_down_test1 = [i-start+1 for i in self.buy_x_index]
        buy_y_down_test1 = [self.down_test1[i] for i in buy_x_down_test1]
        sell_x_down_test1 = [i-start+1 for i in self.sell_x_index]
        sell_y_down_test1 = [self.down_test1[i] for i in sell_x_down_test1]
#####


        plt.subplot(413)
        plt.plot([i for i in range(len(self.down_test))],self.down_test,'b',[i for i in range(len(self.down_test))],self.down_test,'o',buy_x_down_test,buy_y_down_test,'r*',sell_x_down_test,sell_y_down_test,'g*')             
        plt.subplot(414)
        plt.plot([i for i in range(len(self.down_test1))],self.down_test1,'b',[i for i in range(len(self.down_test1))],self.down_test1,'o',buy_x_down_test1,buy_y_down_test1,'r*',sell_x_down_test1,sell_y_down_test1,'g*')
        figname = "fig1_"+datafile         
        if save==1:
            savefig(figname)
        plt.figure(2)
        plt.plot([i for i in range(len(self.total_asset_list))],self.total_asset_list,'b')
        figname = "fig2_"+datafile



        plt.figure(3)
        plt.plot([i for i in range(len(self.imf_power))],self.imf_power,'b',power_buy_index,power_buy_value,'r*',power_sell_index,power_sell_value,'g*')
       
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
        macd.append(float(eachline.split("\t")[6]))
        open_price.append(float(eachline.split("\t")[1]))
        date.append(eachline.split("\t")[0])
        #vol.append(float(eachline.split("\t")[5]))
    
    process = process_util(open_price,close_price,macd,date,vol,high_price,low_price)
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
#    process.draw_fig(datafile,begin)

    

            
