import os
import copy
import math
import sys
import matplotlib.pyplot as plt
import datetime
from scipy.signal import argrelextrema
import numpy as np



sys.path.append("../emd_util")
from generate_emd_data import generateEMDdata
from analyze import *
from spline_predict import linerestruct
from emd import *
from leastsqt import leastsqt_predict
from svm_uti import *
from spline_predict import splinerestruct
from calc_SNR import calc_SNR



class process_util():

    def __init__(self,open_price,close_price,macd,date,vol,high_price,low_price,je):
        
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
        self.je = je

        self.macd = macd
        self.date = date
        self.vol = vol
        self.fail_flag = 0



        self.buy_day = 0



        self.last_buy_result=0




        self.buy_mean_price = 0
        self.one_time_day=0


        self.ma_count1=0
        self.ma_count=0
        self.ma_count2=0
        self.sell_flag=0
        self.sell_flag2=0
        self.snr=[]
        
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




    def run_predict1(self,imf_list,current_price_index,date,datafile,residual):


        
        for i in range(len(imf_list)):
            imfp = imf_list[i]
            data = np.array(imfp)
            max_index = list(argrelextrema(data,np.greater)[0])
            min_index = list(argrelextrema(data,np.less)[0])
            print "imf %s"%i
            print "max %s"%max_index
            print "min %s"%min_index

        residual_max_index = list(argrelextrema(np.array(residual),np.greater)[0])
        residual_min_index = list(argrelextrema(np.array(residual),np.less)[0])
        print "residual"
        residual_max_index.insert(0,0)
        residual_min_index.insert(0,0)
        print "max %s"%residual_max_index
        print "min %s"%residual_min_index
        residual_flag=0
        #if (residual_max_index[-1]<residual_min_index[-1] and residual_min_index[-1]<45):
        if (residual_max_index[-1]<residual_min_index[-1] and residual_min_index[-1]>46):
            residual_flag=1


        print "residual %s"%residual_flag
      
        process_data = self.close_price[current_price_index-99:current_price_index+1]
        data = np.array(process_data)
        max_index = list(argrelextrema(data,np.greater)[0])
        min_index = list(argrelextrema(data,np.less)[0])


##############################################################

        if len(imf_list)<2:
            imf_process = list(imf_list[0])
        else:
            imf_process = list(imf_list[1])


        imf_max_index = list(argrelextrema(np.array(imf_process),np.greater)[0])
        print " imf max index %s"%imf_max_index
        imf_min_index = list(argrelextrema(np.array(imf_process),np.less)[0])
        print " imf min index %s"%imf_min_index

        imf_flag=0
        print imf_min_index[-1]
        print imf_max_index[-1]
        if imf_min_index[-1]>imf_max_index[-1] and imf_min_index[-1]>=46 and imf_min_index[-1]-imf_max_index[-1]>1:
            imf_flag=1
        if imf_min_index[-1]<imf_max_index[-1] and imf_max_index[-1]>=46:
            imf_flag=2

        

        print "imf flag%s"%imf_flag
        decision=0
        if self.buy_price==0 and residual_flag==1 :
            decision=1 
        if self.buy_price!=0 and imf_flag==2:
            decision=2
        
        return imf_flag
        #return decision
##########################



###########################
        
    def run_predict(self,imf_list,current_price_index,date,datafile,residual,emd_data):
        imf_flag=self.run_predict1(imf_list,current_price_index,date,datafile,residual)

        self.snr.append(calc_SNR(emd_data,imf_list))
        print "SNR %s"%self.snr[-5:]
        

        print "imf flag %s"%imf_flag

        emd_data_line = splinerestruct(emd_data[-10:],1)
        cd=range(10)
        emdcoef=(np.corrcoef(emd_data_line,cd)[0][1])
        print "emd coef%s"%emdcoef
        emd_decision=0
        #if emd_data[-1]>emd_data[-2]>emd_data[-3] :
        if emdcoef<-0.9 and emd_data[-1]>emd_data[-2]:#>emd_data[-3]:
            emd_decision=1


####################




#######

        ma5 = self.ma(self.close_price, 5, 0, current_price_index)
        deltama5 = self.ma(self.close_price, 5, 0, current_price_index)-self.ma(self.close_price, 5, -1, current_price_index)
        deltama5_before = self.ma(self.close_price, 5, -1, current_price_index)-self.ma(self.close_price, 5, -2, current_price_index)
        deltama5_before2 = self.ma(self.close_price, 5, -2, current_price_index)-self.ma(self.close_price, 5, -3, current_price_index)
        deltama3 = self.ma(self.close_price, 3, 0, current_price_index)-self.ma(self.close_price, 3, -1, current_price_index)
        deltama3_before = self.ma(self.close_price, 3, -1, current_price_index)-self.ma(self.close_price, 3, -2, current_price_index)
        deltama3_before2 = self.ma(self.close_price, 3, -2, current_price_index)-self.ma(self.close_price, 3, -3, current_price_index)
        deltama3_before3 = self.ma(self.close_price, 3, -3, current_price_index)-self.ma(self.close_price, 3, -4, current_price_index)
        deltama9 = self.ma(self.close_price, 9, 0, current_price_index)-self.ma(self.close_price, 9, -1, current_price_index)
        deltama9_before = self.ma(self.close_price, 9, -1, current_price_index)-self.ma(self.close_price, 9, -2, current_price_index)
        deltama9_before2 = self.ma(self.close_price, 9, -2, current_price_index)-self.ma(self.close_price, 9, -3, current_price_index)
        ma3 = self.ma(self.close_price, 3, 0, current_price_index)
        ma3_b = self.ma(self.close_price, 3, -1, current_price_index)
        ma3_b2 = self.ma(self.close_price, 3, -2, current_price_index)
        ma3_b3 = self.ma(self.close_price, 3, -3, current_price_index)
        ma3_b4 = self.ma(self.close_price, 3, -4, current_price_index)
        ma3_b5 = self.ma(self.close_price, 3, -5, current_price_index)
        ma9 = self.ma(self.close_price, 9, 0, current_price_index)
        ma10 = self.ma(self.close_price, 10, 0, current_price_index)
        deltama10 = self.ma(self.close_price, 10, 0, current_price_index)-self.ma(self.close_price, 10, -1, current_price_index)
        ma20 = self.ma(self.close_price, 25, 0, current_price_index)
        deltama20 = self.ma(self.close_price, 20, 0, current_price_index)-self.ma(self.close_price, 20, -1, current_price_index)
        deltama25 = self.ma(self.close_price, 25, 0, current_price_index)-self.ma(self.close_price, 25, -1, current_price_index)
        deltama25_before = self.ma(self.close_price, 25, -1, current_price_index)-self.ma(self.close_price, 25, -2, current_price_index)
        deltama25_before2 = self.ma(self.close_price, 25, -2, current_price_index)-self.ma(self.close_price, 25, -3, current_price_index)
        deltama25_before3 = self.ma(self.close_price, 25, -3, current_price_index)-self.ma(self.close_price, 25, -9, current_price_index)
        ma30 = self.ma(self.close_price, 30, 0, current_price_index)
        ma60 = self.ma(self.close_price, 60, 0, current_price_index)
        deltama30 = self.ma(self.close_price, 30, 0, current_price_index)-self.ma(self.close_price, 30, -1, current_price_index)

        p = self.close_price[current_price_index-10:current_price_index+1]
        print "current price%s"%self.close_price[current_price_index]
        print p
        print "cal %s"%((p[-4]+3*p[-3]+6*p[-2]+17*p[-1])/27.0)
        print "mama3 %s %s"%(((ma3_b+ma3_b2+ma3_b3)/3.0),((ma3+ma3_b+ma3_b2)/3.0))
        mama_decision=0
        mama3 = (ma3+ma3_b+ma3_b2)/3.0
        mama3_b = (ma3_b+ma3_b2+ma3_b3)/3.0
        mama3_b2 = (ma3_b2+ma3_b3+ma3_b4)/3.0
        mama3_b3 = (ma3_b3+ma3_b4+ma3_b5)/3.0

        mamama3 = (mama3+mama3_b+mama3_b2)/3.0
        mamama3_b = (mama3_b+mama3_b2+mama3_b3)/3.0
        print "mamama3 %s %s"%(mamama3_b,mamama3)
        if (mamama3>mamama3_b) and mama3>mama3_b:# or (mama3>mama3_b>mama3_b2 and mama3_b2<mama3_b3):
            mama_decision=1


###################

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
        print "ma count1 %s count %s count2 %s"%(self.ma_count1,self.ma_count,self.ma_count2)


        ma_decision=1
        if self.ma_count<0 and self.ma_count2<0 and self.ma_count1<0  :
            ma_decision=0
########



        
        print "ma5 %s"%ma5
        print "delta ma5 %s %s"%(deltama5_before,deltama5)
        print "delta ma3 before %s"%deltama3_before
        print "delta ma3 %s"%deltama3
        print "delta ma9 before %s %s"%(deltama9_before,deltama9)
        print "ma3 %s"%ma3
        print "ma9 %s"%ma9
        print "ma10 %s"%ma10
        print "delta ma10 %s"%deltama10
        print "ma20 %s"%ma20
        print "delta ma20 %s"%deltama20
        print "delta ma25 %s"%deltama25
        print "ma30 %s"%ma30
        print "ma60 %s"%ma60
        print "delta ma30 %s"%deltama30
        





        if self.buy_price==0:
            trade_price = self.open_price[current_price_index+1]+0.1
            if trade_price>self.high_price[current_price_index+1]:
                trade_price = self.close_price[current_price_index+1]
        else:
            trade_price = self.open_price[current_price_index+1]-0.1
            if trade_price<self.low_price[current_price_index+1]:
                trade_price = self.close_price[current_price_index+1]




        print "date %s"%self.date[current_price_index]
        print "current price %s %s %s %s"%(self.open_price[current_price_index],self.high_price[current_price_index],self.low_price[current_price_index],self.close_price[current_price_index])
        print "next price %s %s %s %s"%(self.open_price[current_price_index+1],self.high_price[current_price_index+1],self.low_price[current_price_index+1],self.close_price[current_price_index+1])
        print "trade price %s"%trade_price
        print "buy price %s"%self.buy_price




        current_mean_price = self.ma(self.close_price, 5, 0, current_price_index)
    
        print "buy mean%s current mean%s"%(self.buy_mean_price,current_mean_price)
#######

        print "fail flag %s"%self.fail_flag

        print "ma3 SNR %s"%np.mean(self.snr[-3:])
        print "ma5 SNR %s"%np.mean(self.snr[-5:])
        print "ma10 SNR %s"%np.mean(self.snr[-10:])
        print "ma20 SNR %s"%np.mean(self.snr[-20:])
        snr_decision=0
        if np.mean(self.snr[-10:])>0 and np.mean(self.snr[-3:])>0:
            snr_decision=1

        if self.one_time_day>0:
            self.one_time_day+=1

        
        distance_decision = 1
        if len(self.sell_x_index)>1 and current_price_index-self.sell_x_index[-1]<5:
            distance_decision = 0

        if ((self.sell_flag==2 and imf_flag!=2) or (self.sell_flag==2 and imf_flag==2 and current_price<self.close_price[current_price_index-1])) and  self.buy_price!=0:
            self.sell_flag2=2

        
        print "snr %s"%np.mean(self.snr[-3:])
        print "snr %s"%np.mean(self.snr[-4:-1])
        #if  ((imf_flag==2 ) or (deltama5<deltama5_before)) and  self.buy_price!=0  and ((current_price-self.buy_price)/self.buy_price) >0:
        #if ((self.sell_flag==2 and imf_flag!=2) or (self.sell_flag==2 and imf_flag==2 and current_price<self.close_price[current_price_index-1])) and  self.buy_price!=0 :
        if self.sell_flag2==2 and np.mean(self.snr[-3:])<np.mean(self.snr[-4:-1]) and self.buy_price!=0 :
            print "sell sell 1"
            self.money = self.sell(self.share,trade_price)
            self.share = 0
            self.total_asset = self.money - self.sell_fee(self.money)
            self.money = self.total_asset
            self.sell_x_index.append(current_price_index)
            self.sell_y_index.append(current_price)
            if self.buy_price>trade_price:
                self.fail_flag=1
            self.buy_price = 0
            self.buy_mean_price = 0
            self.one_time_day=0
            self.sell_flag=0
            self.sell_flag2=0
        elif   self.buy_price!=0 and ((((self.buy_mean_price-current_mean_price)/self.buy_mean_price)>0.01 ) )  and self.share > 0 and np.mean(self.snr[-3:])<np.mean(self.snr[-4:-1]): #and (self.buy_price-current_price)<0 and self.share > 0:
            print "sell sell 2"
            if self.buy_price>trade_price:
                self.fail_flag=1
            self.money = self.sell(self.share,trade_price)
            self.share = 0
            self.buy_price = 0
            self.total_asset = self.money - self.sell_fee(self.money)
            self.money = self.total_asset
            self.sell_x_index.append(current_price_index)
            self.sell_y_index.append(current_price)
            self.buy_mean_price = 0
            self.one_time_day=0
            self.sell_flag=0
            self.sell_flag2=0
        elif   self.buy_price!=0 and ((((self.buy_price-current_price)/self.buy_price)>0.04 ) )  and self.share > 0 :
            print "sell sell 4"
            if self.buy_price>trade_price:
                self.fail_flag=1
            self.money = self.sell(self.share,trade_price)
            self.share = 0
            self.buy_price = 0
            self.total_asset = self.money - self.sell_fee(self.money)
            self.money = self.total_asset
            self.sell_x_index.append(current_price_index)
            self.sell_y_index.append(current_price)
            self.buy_mean_price = 0
            self.one_time_day=0
            self.sell_flag=0
            self.sell_flag2=0
        elif   self.buy_price!=0 and self.one_time_day>10 and  imf_flag==2 and self.share > 0 and np.mean(self.snr[-3:])<np.mean(self.snr[-4:-1]):
            print "sell sell 3"
            if self.buy_price>trade_price:
                self.fail_flag=1
            self.money = self.sell(self.share,trade_price)
            self.share = 0
            self.buy_price = 0
            self.total_asset = self.money - self.sell_fee(self.money)
            self.money = self.total_asset
            self.sell_x_index.append(current_price_index)
            self.sell_y_index.append(current_price)
            self.buy_mean_price = 0
            self.one_time_day=0
            self.sell_flag=0
            self.sell_flag2=0
        elif    imf_flag==1  and self.buy_price==0 and snr_decision==1 and distance_decision==1 and deltama5_before<deltama5 and ma_decision==1 :#and distance_decision==1:#deltama3_before<deltama3:# (deltama3_before>deltama3 or deltama3<0):
            if 0:#self.fail_flag!=0:
                self.fail_flag-=1
            else:
                print "buy buy 1"
                self.share = self.buy(self.money,trade_price)
                self.money = 0
                self.buy_price = trade_price
                self.buy_mean_price = current_mean_price
                self.total_asset = self.share*trade_price
                self.buy_x_index.append(current_price_index)
                self.buy_y_index.append(current_price)
                self.one_time_day=1
                self.last_buy_result=0

  

        if self.buy_price!=0:
            self.buy_day += 1

        if imf_flag==2:
            self.sell_flag=2
        else:
            self.sell_flag=0



        print "buy day%s"%self.buy_day

            


  
        self.total_asset_list.append(self.total_asset)
        print "total asset is %s"%(self.total_asset)
        print "money is %s"%(self.money)
        print "share is %s"%(self.share)




    def format_data(self, data):
        if isinstance(data,list):
            return [2*((float(data[i])-min(data))/(max(data)-float(min(data))))-1 for i in range(len(data))]



    def run(self,emd_data, datafile,current_index,date,emd_file):
        starttime = datetime.datetime.now()
        tmp = [i for i in emd_data]
        emd_data = tmp
        print "\n\n\n"
        print "emd data std %s"%np.std(emd_data)
        print "emd data %s"%emd_data[-10:]
#########

#######

##

#        data = self.close_price
#
#        ma5 = []
#        for i in range(3):
#            ma5.append(0)
##
#        for i in range(3,len(data)):
#            mean_5 = np.mean(data[i-2:i+1])
#            ma5.append(mean_5)
##
#        data = ma5
#        ma5 = []
#        for i in range(3):
#            ma5.append(0)
##
#        for i in range(3,len(data)):
#            mean_5 = np.mean(data[i-2:i+1])
#            ma5.append(mean_5)
##
#        traindata = ma5
####
#        

        data=self.close_price[:current_index+1]

########mamama########

        ma5 = []
        for i in range(3):
            ma5.append(0)

        for i in range(3,len(data)):
            mean_5 = np.mean(data[i-(3-1):i+1])
            ma5.append(mean_5)

        data = []
        for i in range(3):
            data.append(0)
#
        for i in range(3,len(ma5)):
            mean_5 = np.mean(ma5[i-(3-1):i+1])
            data.append(mean_5)


#        ma5=data
#
#        data = []
#        for i in range(3):
#            data.append(0)
#
#        for i in range(3,len(ma5)):
#            mean_5 = np.mean(ma5[i-(3-1):i+1])
#            data.append(mean_5)
#


#        ma5=data
#
#        data = []
#        for i in range(3):
#            data.append(0)
#
#        for i in range(3,len(ma5)):
#            mean_5 = np.mean(ma5[i-(3-1):i+1])
#            data.append(mean_5)
#
#        ma5=data
#
#        data = []
#        for i in range(3):
#            data.append(0)
#
#        for i in range(3,len(ma5)):
#            mean_5 = np.mean(ma5[i-(3-1):i+1])
#            data.append(mean_5)
#
#        ma5=data
#
#        data = []
#        for i in range(3):
#            data.append(0)
#
#        for i in range(3,len(ma5)):
#            mean_5 = np.mean(ma5[i-(3-1):i+1])
#            data.append(mean_5)

        print "traindata %s"%data[-10:]
        traindata=data[-200:]
        traindata=self.close_price[current_index-200:current_index+1]###this line smmes good
##################
        svm_train(traindata,3)

####
        
        pp=[]
        testdata1 = traindata[-3:]
        print "testdata1"
        print testdata1
        pp.append(float(svm_predict(testdata1)))
        testdata2 = traindata[-2:]
        testdata2.append(pp[0])
        print "testdata2"
        print testdata2
        pp.append(float(svm_predict(testdata2)))
#       # testdata3 = []
        #testdata3 = traindata[-1:]
        #testdata3.append(pp[0])
#       # testdata3.append(pp[1])
        #print "testdata3"
        #print testdata3
#        pp.append(float(svm_predict(testdata3)))

######

        #print "pp %s"%pp
        emd_data.pop(0)
        emd_data.pop(0)
        emd_data.append(pp[0])
        emd_data.append(pp[1])
        ##emd_data.append(pp[2])
        #print "emd data %s"%emd_data[-10:]
######################       
        emd_data = [i-np.mean(emd_data) for i in emd_data]
        my_emd = one_dimension_emd(emd_data,9)
        (imf, residual) = my_emd.emd(0.01,0.01)


        print "len imf %s"%len(imf)
            
        
        self.run_predict(imf,current_index,date,datafile,residual,emd_data)
        endtime = datetime.datetime.now()
        print "run time"
        print (endtime - starttime).seconds

                 

    def ma(self, data, period, start, current_index):
        sum_period = 0
        for i in range(period):
            sum_period += data[current_index + start - i]
        return float(sum_period)/period        





    def show_success(self):
        print success_ratio(precondition(self.total_asset_list))

    def show_stat(self):
        print income_mean_std(precondition(self.total_asset_list))

    def show_rss(self):
        profit_smooth(self.total_asset_list)
        
        

    def draw_fig(self,datafile,start,save=0):
        
        data = self.close_price

        ma5 = []
        for i in range(5):
            ma5.append(0)
#
        for i in range(5,len(data)):
            mean_5 = np.mean(data[i-4:i+1])
            ma5.append(mean_5)
#
        data = ma5
        ma5 = []
        for i in range(5):
            ma5.append(0)
#
        for i in range(5,len(data)):
            mean_5 = np.mean(data[i-4:i+1])
            ma5.append(mean_5)
#
        data = ma5


        my_emd = one_dimension_emd(data)
        (imf, residual) = my_emd.emd(0.01,0.01)
        imf = imf[1]



        
        imf_buy_value = [imf[i] for i in self.buy_x_index]
        imf_sell_value = [imf[i] for i in self.sell_x_index]


        buy_x = [i-start for i in self.buy_x_index]
        sell_x = [i-start for i in self.sell_x_index]
        data1 = data
        data = self.close_price
        plt.figure(1)
        plt.subplot(311).axis([start,len(data),min(data[start:]),max(data[start:])])      
        plt.plot([i for i in range(len(data))],data,'o',[i for i in range(len(data))],data,'b',self.buy_x_index,self.buy_y_index,'r*',self.sell_x_index,self.sell_y_index,'g*')             
        plt.subplot(312).axis([start,len(data),min(data1[start:]),max(data1[start:])])      
        plt.plot([i for i in range(len(data1))],data1,'o',[i for i in range(len(data1))],data1,'b')             
        plt.subplot(313).axis([start,len(imf),min(imf[start:]),max(imf[start:])])
        plt.plot([i for i in range(len(imf))],imf,'o',[i for i in range(len(imf))],imf,'b',self.buy_x_index,imf_buy_value,'r*',self.sell_x_index,imf_sell_value,'g*')             
#####

#####

        asset_list=[]
        x_spline_asset=[]
        for i in range(1,len(self.total_asset_list)):
            if abs(self.total_asset_list[i]-self.total_asset_list[i-1])>10:
                x_spline_asset.append(i)
                asset_list.append(self.total_asset_list[i])

        spline_asset = linerestruct(x_spline_asset,asset_list)
        plt.figure(2)
        plt.plot([i for i in range(len(self.total_asset_list))],self.total_asset_list,'b',x_spline_asset,spline_asset,'r')

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
    je = []
    for eachline in lines:
        eachline.strip()
        close_price.append(float(eachline.split("\t")[4]))
        high_price.append(float(eachline.split("\t")[2]))
        low_price.append(float(eachline.split("\t")[3]))
        macd.append(float(eachline.split("\t")[5]))
        open_price.append(float(eachline.split("\t")[1]))
        date.append(eachline.split("\t")[0])
        vol.append(float(eachline.split("\t")[5]))
        je.append(float(eachline.split("\t")[5]))
    
    process = process_util(open_price,close_price,macd,date,vol,high_price,low_price,je)


    emd_dir = os.path.abspath('.')+"/"+"emd_data"
    print emd_dir
    os.popen("rm -r %s"%emd_dir)
    os.popen("mkdir %s"%emd_dir)
    generate_emd_func = generateEMDdata(datafile,begin,50,emd_dir)


    #generate_emd_func.generate_ma_emd_data_fix(3)
    generate_emd_func.generate_ma_emd_data_fix(3)  #should be 6 ma
    print "begin %s"%begin
    for i in range(begin,begin+int(process.file_count(emd_dir))-1):
        emd_data = []
        print "emd file %s"%i
        emd_file = "%s/emd_%s"%(emd_dir,i)
        fp = open("%s/emd_%s"%(emd_dir,i),'r')
        lines = fp.readlines()
        fp.close()
        for eachline in lines:
            eachline.strip("\n")
            emd_data.append(float(eachline))
        process.run(emd_data,datafile,i-1,date[-1],emd_file)
    process.show_success()
    process.show_stat()
    process.show_rss()
#    process.draw_fig(datafile,begin)

    

            
