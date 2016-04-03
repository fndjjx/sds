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
import talib 



sys.path.append("../emd_util")
from generate_emd_data import generateEMDdata
from analyze import *
from spline_predict import linerestruct
from emd_mean_pro import *
from emd import *
from eemd import *
from getMinData import *
from leastsqt import leastsqt_predict
from svm_uti import *
from spline_predict import splinerestruct
from calc_SNR import calc_SNR2
from calc_SNR import calc_SNR
from calc_match import matchlist2
from calc_match import matchlist3
from calc_match import matchlist1
from calc_match import matchlist4
from kalman import kalman_simple
from calc_ma_imf import most_like_imf
from calc_svm_imf import calc_recent_rilling
from smooth import cubicSmooth5
from hurst import hurst
from smooth import exp_smooth
from monte_carlo import montecarlo_simulate
from monte_carlo import montecarlo_simulate2
from core_change import  trend_predict, stable_predict
from load_data import load_data

AB=0
ABL=0

class process_util():

    def __init__(self,open_price,close_price,macd,date,vol,high_price,low_price,je,kdjk,kdjd,wr,mindata):
        
        self.money = 10000
        self.share = 0
        self.sell_x_index = []
        self.ex_x_index = []
        self.ex_x_index2 = []
        self.ex_y_index = []
        self.ex_y_index2 = []
        self.sell_y_index = []
        self.buy_x_index = []
        self.buy_y_index = []
        self.open_price = open_price
        self.high_price = high_price
        self.low_price = low_price
        self.close_price = close_price

        self.kdjk=kdjk
        self.kdjd=kdjd
        self.wr=wr
        
        self.total_asset_list = []
        self.total_asset = self.money
        self.buy_price = 0
        self.je = je

        self.macd = macd
        self.date = date
        self.vol = vol
        self.fail_flag = 0

        self.stdimf=[]



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
        self.open_flag=0
        self.power_flag=0

        self.emdstd=[]
        self.matchcha=[]
        self.extenflag=[]

        self.imf_flag_list=[]

        self.close_flag=[]
        self.high_flag=[]
        self.f_d=0

        self.day_count=0
        self.last_as=10000
        self.hb=[]


        self.rihb=[]
        self.last_rihb_as=10000
        self.each_day=0
        self.boll_de=[]

        self.cccc=0

        self.imf_sugg_x_index=[]
        
        self.normal30=[]
        self.pre_cha=[]
        self.pre_cha_de=[]
        self.mindata=mindata

        self.kvcha=[]
        self.kalv=[]
        self.stable_test=[]

        self.de = 0
        self.de2 = 0
        self.judge_de=[]
        self.peak=[]

        self.buy_ma=[]
        self.last_sel=1
        self.last_fail_de=0
        self.count_buy = 0
        self.filterdata = []
        self.last_imf1_min = []
        self.sel = 1
        self.radio1=[]
        self.radio2=[]
        self.ks=[]
        self.cu=[]
        self.he=[]
        self.pimf1=[]
        self.std10=[]
        self.diffmpcha=[]
        self.tmphigh=0
        self.mclist=[]
        self.mclist2=[]
        self.mclist3=[]
        self.mclist4=[]
        self.mclist5=[]
        self.s1=[]
        self.s2=[]
        self.diffmpstd=[]
        self.mpup=[]
        self.mpdown=[]
        self.stab_de=[]
        self.rihb2=[]

        self.scount=0
        self.ct={}
        self.td=0
        self.ytx=[]
        self.ytx2=[]
        self.low=[]
        self.hursti=[]
        self.adx=[]
        self.adx2=[]
        self.mp_buy_price=0
        self.cpstd=[]
        self.mpstd=[]
        self.cpmpstd=[]
        self.tde=[]
        self.x=[]
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


    def calc_last_trust(self,current_price_index,imf_list):

        # get ma extreme
        data_until_now = self.close_price[:current_price_index+1]
        data = data_until_now[-200:]
        #data = cubicSmooth5(data)
        ma_max_index, ma_min_index , long_period, short_period, madiff = most_like_imf(data, imf_list, 1, 2)
        
        #madiff = ma_func2(madiff,2)
        #madiff = cubicSmooth5(madiff)
        ma_max_index = list(argrelextrema(np.array(madiff),np.greater)[0])
        ma_min_index = list(argrelextrema(np.array(madiff),np.less)[0])
        print "max_index"
        print ma_max_index
        print "min_index"
        print ma_min_index
        ma_de1 = 0
        if ma_max_index[-1]-ma_min_index[-1]>0 and ma_max_index[-1] >= 497:
            ma_de1 = 2
        if ma_min_index[-1]-ma_max_index[-1]>0 and ma_min_index[-1] >= 497: #and confidence == 1: #and ma_max_index[-1]<496:
            ma_de1 = 1

#        ma_max_index, ma_min_index , long_period, short_period, madiff, confidence = most_like_imf(data, imf_list, 0, 1)
#
# #       #madiff = ma_func2(madiff,2)
# #       #madiff = cubicSmooth5(madiff)
#        ma_max_index = list(argrelextrema(np.array(madiff),np.greater)[0])
#        ma_min_index = list(argrelextrema(np.array(madiff),np.less)[0])
#        print "max_index"
#        print ma_max_index
#        print "min_index"
#        print ma_min_index
#        ma_de2 = 0
#        if ma_max_index[-1]-ma_min_index[-1]>0 and ma_max_index[-1] >= 497:
#            ma_de2 = 2
#        if ma_min_index[-1]-ma_max_index[-1]>0 and ma_min_index[-1] >= 497: #and confidence == 1: #and ma_max_index[-1]<496:
#            ma_de2 = 1

        ma_de = 0
        if ma_de1 == 1 :#and ma_de2 == 1:
            ma_de = 1
        elif ma_de1 == 2 :#and ma_de2 == 2:
            ma_de = 2
        #get imf extreme
        imf_max_index = list(argrelextrema(np.array(imf_list[2][-500:]),np.greater)[0])
        imf_min_index = list(argrelextrema(np.array(imf_list[2][-500:]),np.less)[0])
        print "imf max index"
        print imf_max_index
        print "imf min index"
        print imf_min_index


        #compare last few extreme 

        last_trust = 20
        imf_index = (imf_max_index+imf_min_index)
        imf_index.sort()
        for i in range(1,len(imf_index)+1):
            if (imf_index[-i] in imf_max_index) and ((imf_index[-i] in ma_max_index) or (imf_index[-i]+1 in ma_max_index) or (imf_index[-i]-1 in ma_max_index)):
                last_trust = i
                break
            if (imf_index[-i] in imf_min_index) and ((imf_index[-i] in ma_min_index) or (imf_index[-i]+1 in ma_min_index) or (imf_index[-i]-1 in ma_min_index)):
                last_trust = i
                break

        return last_trust, long_period, short_period, ma_de
                
        



    def run_predict1(self,imf_list,current_price_index,date,datafile,residual,imf_open,imf_macd,sel_flag,emd_data,emd_data2):


        if sel_flag==1:
            datalength=len(emd_data)
            emd_data=emd_data
        else:
            datalength=len(emd_data2)
            emd_data=emd_data2

        print "\n\n begin"
        max_index = list(argrelextrema(np.array(emd_data),np.greater)[0])
        min_index = list(argrelextrema(np.array(emd_data),np.less)[0])
        print max_index
        print min_index
        print "len {}".format(len(max_index)+len(min_index))
        
        for i in range(len(imf_list)):
            imfp = imf_list[i]
            data = np.array(imfp)
            imf_max_index = list(argrelextrema(data,np.greater)[0])
            imf_min_index = list(argrelextrema(data,np.less)[0])
            if 1:#i==1:
                print "imf %s"%i
                print "max %s"%imf_max_index
                print "min %s"%imf_min_index
                print "imf in data {}".format(calc_pro(min_index,imf_min_index))
                print "data in imf {}".format(calc_pro(imf_min_index,min_index))
                self.pimf1.append((calc_pro(min_index,imf_min_index),calc_pro(imf_min_index,min_index)))

       

        #imf_list[1]=list(imf_list[1])
        #imf_list[2]=list(imf_list[2])
        #imf_list[1]=imf_list[1][-500:]
        #imf_list[2]=imf_list[2][-500:]
        #emd_data=emd_data[-500:]
        #emd_data2=emd_data2[-500:]
        

####
        imf0_max_index = list(argrelextrema(np.array(imf_list[0]),np.greater)[0])
        imf0_min_index = list(argrelextrema(np.array(imf_list[0]),np.less)[0])
        imf0_or_flag=0
        if imf0_min_index[-1]>imf0_max_index[-1] and imf0_min_index[-1]>496 :#and imf2_min_index[-1]-imf2_max_index[-1]>3:
            imf0_or_flag=1
        if imf0_min_index[-1]>imf0_max_index[-1] and imf0_max_index[-1]<495 :#and imf2_min_index[-1]-imf2_max_index[-1]>3:
            imf0_or_flag=2 

        r0 = [emd_data[i]-imf_list[0][i] for i in range(len(emd_data))]
        r1 = [r0[i]-imf_list[1][i] for i in range(len(emd_data))]
        self.filterdata.append(r0[-1])
        #r1 = [r0[i]-imf_list[1][i] for i in range(len(emd_data))]
        imf1_max_index = list(argrelextrema(np.array(imf_list[2]),np.greater)[0])
        imf1_min_index = list(argrelextrema(np.array(imf_list[2]),np.less)[0])
        #imf1_max_index = list(argrelextrema(np.array(r0),np.greater)[0])
        #imf1_min_index = list(argrelextrema(np.array(r0),np.less)[0])
        self.last_imf1_min.append(imf1_max_index[-1])
        print "residual imf1 max {}".format(imf1_max_index)
        print "residual imf1 min {}".format(imf1_min_index)
        imf1_or_flag=0
        if imf1_min_index[-1]<imf1_max_index[-1] and imf1_max_index[-1]<=195 :#and imf2_min_index[-1]-imf2_max_index[-1]>3:
            imf1_or_flag=1
        #if imf1_min_index[-1]>imf1_max_index[-1] and imf1_min_index[-1]==797 and imf1_min_index[-1]-imf1_max_index[-1]>1:
        #    imf1_or_flag=1
        #if imf1_min_index[-1]<imf1_max_index[-1] and imf1_max_index[-1]>=198 :#and imf2_min_index[-1]-imf2_max_index[-1]>3:
        #    imf1_or_flag=2

###

     #   imf2_max_index = list(argrelextrema(np.array(imf_list[2]),np.greater)[0])
     #   #print " imf max index %s"%imf2_max_index
     #   imf2_min_index = list(argrelextrema(np.array(imf_list[2]),np.less)[0])
     #   #print " imf min index %s"%imf2_min_index
     #   imf2_or_flag=0
     #   tmp=imf2_max_index+imf2_min_index
     #   tmp.sort()
     #   if (tmp[-2] in imf2_max_index and tmp[-1]>495 ) or (tmp[-1]<=495 and tmp[-1] in imf2_max_index):
     #       imf2_or_flag=1
     #   if (tmp[-2] in imf2_min_index and tmp[-1]>495) or (tmp[-1]<=495 and tmp[-1] in imf2_min_index):
     #       imf2_or_flag=2
     #   #if ((imf2_min_index[-1]<imf2_max_index[-1] and imf2_max_index[-1]>=197)):# or (imf2_min_index[-1]>imf2_max_index[-1] and imf2_min_index[-1]>=797)) and abs(imf_list[2][-1]/imf_list[2][imf2_max_index[-1]])>0.9: 
     #   #    imf2_or_flag=2
     #   #if ((imf2_min_index[-1]>imf2_max_index[-1] and imf2_min_index[-1]<=795) or (imf2_min_index[-1]<imf2_max_index[-1] and imf2_max_index[-1]>=797)) and abs(imf_list[2][-1]/imf_list[2][imf2_min_index[-1]])>0.9: 
     #   #if ((imf2_min_index[-1]>imf2_max_index[-1] ) ) and abs(imf_list[1][-1]/imf_list[1][imf2_min_index[-1]])>0.9: 
     #   #    imf2_or_flag=2
     #   

     #   #print "imf2 or flag%s"%imf2_or_flag
     #   #imf2_max_index = list(argrelextrema(np.array(imf_list[2][:-4]),np.greater)[0])
     #   #imf2_min_index = list(argrelextrema(np.array(imf_list[2][:-4]),np.less)[0])
     #   #if imf2_min_index[-1]<imf2_max_index[-1] and imf2_max_index[-1]<995 and imf2_max_index[-1]-imf2_min_index[-1]>3: 
     #   #    imf2_or_flag=11
     #   #if imf2_min_index[-1]>imf2_max_index[-1] and imf2_min_index[-1]<995 and imf2_min_index[-1]-imf2_max_index[-1]>3:
     #   #    imf2_or_flag=0
     #   #if imf2_min_index[-1]>imf2_max_index[-1] and imf2_min_index[-1]<990 :
     #   #    imf2_or_flag=1


     #   residual_max_index = list(argrelextrema(np.array(residual),np.greater)[0])
     #   residual_min_index = list(argrelextrema(np.array(residual),np.less)[0])
     #   print "residual"
     #   residual_max_index.insert(0,0)
     #   residual_min_index.insert(0,0)
     #   print "max %s"%residual_max_index
     #   print "min %s"%residual_min_index
     #   residual_flag=1
     #   #if (residual_max_index[-1]<residual_min_index[-1] and residual_min_index[-1]<45):
     #   if (residual_max_index[-1]>residual_min_index[-1] and residual_max_index[-1]>=495) or (residual_max_index[-1]<residual_min_index[-1] and residual_min_index[-1]<=492):
     #       residual_flag=0


     #   print "residual %s"%residual_flag

#######

     #   imf0_max_index = list(argrelextrema(np.array(imf_list[0]),np.greater)[0])
     #   imf0_min_index = list(argrelextrema(np.array(imf_list[0]),np.less)[0])

     #   tmp = imf0_max_index+imf0_min_index
     #   tmp.sort()
###
     #   trust1 = tmp[-2]
     #   print "trust1 %s"%trust1
###
     #   trust0=tmp[-2]
     #   #remain = [emd_data[i]-imf_list[0][i] for i in range(len(emd_data))]
     #   #remain = [remain[i]-imf_list[1][i] for i in range(len(emd_data))]

     #   imf1_max_index = list(argrelextrema(np.array(imf_list[1]),np.greater)[0])
     #   imf1_min_index = list(argrelextrema(np.array(imf_list[1]),np.less)[0])

     #   #imf1_max_index = list(argrelextrema(np.array(remain),np.greater)[0])
     #   #imf1_min_index = list(argrelextrema(np.array(remain),np.less)[0])
######

     #   tmp = imf1_max_index+imf1_min_index
     #   tmp.sort()
     #   trust2 = tmp[-2]
     #   print "trust2 %s"%trust2
########

     #   imf1_min_index=filter(lambda n:n<trust0,imf1_min_index)
     #   imf1_max_index=filter(lambda n:n<trust0,imf1_max_index)

     #   tmp = imf1_max_index+imf1_min_index
     #   tmp.sort()
     #   #trust1=tmp[-1]
     #   imf1_min_index=filter(lambda n:n<trust0,imf1_min_index)
     #   imf1_max_index=filter(lambda n:n<trust0,imf1_max_index)
     #   tmp = imf1_max_index+imf1_min_index
     #   tmp.sort()
     #   trust11=tmp[-1]
     #   #use=2
     #   ##if trust11<993 and tmp[-1]-tmp[-2]>3:
     #   ##    use=1

     #   imf2_max_index = list(argrelextrema(np.array(imf_list[2]),np.greater)[0])
     #   imf2_min_index = list(argrelextrema(np.array(imf_list[2]),np.less)[0])


     #   imf2_min_index=filter(lambda n:n<trust1,imf2_min_index)
     #   imf2_max_index=filter(lambda n:n<trust1,imf2_max_index)

     #   tmp = imf2_max_index+imf2_min_index
     #   tmp.sort()

     #   i_n=1

     #   imf2_max_index = list(argrelextrema(np.array(imf_list[i_n]),np.greater)[0])
     #   imf2_min_index = list(argrelextrema(np.array(imf_list[i_n]),np.less)[0])

     #   tmp = imf2_max_index+imf2_min_index
     #   tmp.sort()

     #   i=1

     #   while tmp[-i]>495:
     #       i+=1

     #    
     #   last_trust, long_period, short_period, ma_de = self.calc_last_trust(current_price_index,imf_list)
     #   long_period=1
     #   short_period=1
     #   print "last trust%s"%last_trust
     #   #print "imf2 %s"%imf_list[2][-20:]
     #   #last_trust_index = tmp[-last_trust]
     # # 

     #   if 0:#last_trust==1:
     #       (imf_process,extenflag1,ex_num,cha,cha2) = matchlist1(imf_list[i_n],1,emd_data)
     #   elif 1:#last_trust==2:
     #       (imf_process,extenflag1,ex_num,cha,cha2) = matchlist2(imf_list[i_n],1,emd_data)
     #   else:
     #       (imf_process,extenflag1,ex_num,cha,cha2) = matchlist3(imf_list[i_n],1,emd_data)


     #   imf_max_index = list(argrelextrema(np.array(imf_process),np.greater)[0])
     #   print " exten imf max index %s"%imf_max_index
     #   imf_min_index = list(argrelextrema(np.array(imf_process),np.less)[0])
     #   print " exten imf min index %s"%imf_min_index
#
     #   imf1_flag=0
     #   if imf_min_index[-1]>imf_max_index[-1] and imf_min_index[-1]>=199  :#and imf_min_index[-1]-imf_max_index[-1]>10:#and ex_flag==1:#and imf_process[-1]<0 :
     #       imf1_flag=1
     #   if imf_min_index[-1]<imf_max_index[-1] and imf_max_index[-1]>=199:
     #       imf1_flag=2

        decision = 0
        last_max = 0 
###########
        if 1:#last_trust_index>485:#sel_flag==2:
        #    #if ((imf2_flag==1 and imf1_flag==1 and imf2_or_flag==11) or (imf2_or_flag==1 and imf1_flag==1)) and (imf0_flag==1) :#and imf2_flag6==1:#or self.close_flag==1:#and imf2_open_flag!=2:
        #    #if ((imf2_flag==1 and imf2_or_flag!=2) or imf2_or_flag==1):# and len(self.imf_flag_list)>1 and self.imf_flag_list[-1]==1:#and  imf2_or_flag!=2:# and imf1_flag==1:#and imf2_flag6==1:#or self.close_flag==1:#and imf2_open_flag!=2:
            if     imf1_or_flag==1 :#and imf2_or_flag==1:#and residual_flag==1:#and imf1_flag!=2 :#and imf1_flag!=2:# and imf1_flag!=2:#and imf1_flag==1:# and len(self.imf_flag_list)>1 and self.imf_flag_list[-1]==1:#and  imf2_or_flag!=2:# and imf1_flag==1:#and imf2_flag6==1:#or self.close_flag==1:#and imf2_open_flag!=2:
            #if  imf2_or_flag==1:# and imf2_or_flag!=2 :#and imf2_flag!=2 and imf1_flag!=2:# and imf1_flag!=2:#and imf1_flag==1:# and len(self.imf_flag_list)>1 and self.imf_flag_list[-1]==1:#and  imf2_or_flag!=2:# and imf1_flag==1:#and imf2_flag6==1:#or self.close_flag==1:#and imf2_open_flag!=2:
        #    #if  imf2_flag==1 and imf2_flag_open_flag!=2 and imf2_macd_flag!=2 and imf2_or_flag!=2:#and imf2_open_flag==1:#and (imf_process3[-1]<0 or imf_process[-1]<0) :#and (extenflag1+extenflag3)<3:#and imf2_open_flag!=2:
#       #     if (imf2_flag==1 and self.open_flag==1) or (imf2_flag==1 and imf2_macd_flag==1) or (imf2_macd_flag==1 and self.close_flag==1) or (imf2_macd_flag==1 and imf2_flag==1):#and imf2_macd_flag==1:#and power_de==1:#and imf3_flag==1:
                decision=1 
        #    #elif ((imf2_flag==1 and  imf2_or_flag==11 ) or (imf2_or_flag==1)) and (imf0_flag==1) :#and imf2_flag6==1:#or self.close_flag==1:#and imf2_open_flag!=2:
        #    #    decision=3 
        #    #if (imf2_flag==2 or imf2_macd_flag==2) and  imf2_or_flag==2 :#and imf2_open_flag!=2:
            if imf1_or_flag==2 :#or imf1_flag==2:#and imf2_open_flag==2 :
        #    #if (imf2_flag==2 and self.open_flag==2) or (imf2_flag==2 and imf2_open_flag==2) or (imf2_open_flag==2 and self.close_flag==2) :
        #    elif (imf2_flag==2 and imf2_or_flag!=2) or imf2_or_flag==2 :#and imf1_flag==2:#imf2_flag==2 and imf2_open_flag==2:
                decision=2
        #elif 0:#sel_flag==1:
        #    if imf1_flag==1 :
        #        decision=1
        #    elif imf1_flag==2 :
        #        decision=2
        #

      
        #self.imf_flag_list.append(imf2_flag)
        #print "flaglist %s"%self.imf_flag_list[-5:]
        #print "decision%s"%decision
        long_period=1
        short_period=1
        return (decision,long_period,short_period)
        #return (imf1_flag,last_max)
        #return imf2_flag
##########################



###########################

    def normal_boll_calc(self,current_price,data):
        
        k1=1
        k2=1
        boll_high1 = k2*np.std(data)+np.mean(data)
        boll_low1 = k1*(-np.std(data))+np.mean(data)
        
        print "diffmp std {}".format(np.std(data))

        de=0
        if current_price>boll_high1 :#and current_price>boll_low2:
            de=2
        elif current_price<boll_low1 :#and np.mean(data)>0 :#and :
            de=1

        return de
        
        
    def run_predict(self,imf_list,current_price_index,date,datafile,residual,emd_data,imf_open,imf_macd,emd_std,preflag,emd_data2,prechade,kalval,predict):
        self.kalv.append(kalval)
        print "prechade%s"%prechade
        self.pre_cha_de.append(prechade)
        print "current price index%s"%current_price_index
      
        #(imf_close_flag,long_period,short_period)=self.run_predict1(imf_list,current_price_index,date,datafile,residual,imf_open,imf_macd,1,emd_data,emd_data2)
        #(imf_high_flag,last_max)=self.run_predict1(imf_open,current_price_index,date,datafile,residual,imf_open,imf_macd,2,emd_data,emd_data2)
        #(imf_5_flag,last_max)=self.run_predict1(imf_macd,current_price_index,date,datafile,residual,imf_open,imf_macd)
        imf_high_flag=1
        imf_close_flag=1
        last_max=1

        max_index = list(argrelextrema(np.array(emd_data),np.greater)[0])
        min_index = list(argrelextrema(np.array(emd_data),np.less)[0])

        print "len extre {}".format(len(max_index)+len(min_index))
        extrem_de=0
        kkkkkkk=len(max_index)+len(min_index)
        if len(max_index)+len(min_index)>=20:
            extrem_de=1
        else:
            extrem_de=2

        deltama5 = self.ma(self.close_price, 5, 0, current_price_index)-self.ma(self.close_price, 5, -1, current_price_index)
        deltama5_before = self.ma(self.close_price, 5, -1, current_price_index)-self.ma(self.close_price, 5, -2, current_price_index)
        deltama5_before2 = self.ma(self.close_price, 5, -2, current_price_index)-self.ma(self.close_price, 5, -3, current_price_index)
        deltama3 = self.ma(self.close_price, 3, 0, current_price_index)-self.ma(self.close_price, 3, -1, current_price_index)
        deltama3_before = self.ma(self.close_price, 3, -1, current_price_index)-self.ma(self.close_price, 3, -2, current_price_index)

        imf_est_list=[]
        for i in range(len(imf_open)):
            imf_est_list.append(imf_open[i][-20:])
        self.snr.append(calc_SNR(emd_data,imf_list,10))
        print "SNR%s"%self.snr[-10:]
        macdma5 = self.ma(self.macd, 5, 0, current_price_index)
        macdma5_before = self.ma(self.macd, 5, -1, current_price_index)
        macdma5_before2 = self.ma(self.macd, 5, -2, current_price_index)
        macdma5_before3 = self.ma(self.macd, 5, -3, current_price_index)
        deltamacdma5 = macdma5-macdma5_before
        deltamacdma5_before = macdma5_before-macdma5_before2
        deltamacdma5_before2 = macdma5_before2-macdma5_before3
        imf_flag=0
        #if (imf_close_flag==1):
        #if self.buy_price==0 and (imf_close_flag==1 or (len(self.close_flag)>0 and (self.close_flag[-1]==1) ) or ( len(self.close_flag)>1 and self.close_flag[-1]!=2 and self.close_flag[-2]==1) or ( len(self.close_flag)>2 and self.close_flag[-3]==1 and self.close_flag[-1]!=2 and self.close_flag[-2]!=2) or (len(self.close_flag)>3 and (self.close_flag[-4]==1) and self.close_flag[-3]!=2 and self.close_flag[-2]!=2 and self.close_flag[-1]!=2)) :#and ((len(self.high_flag)>0 and self.high_flag[-1]==1) or imf_high_flag==1):
        #if imf_close_flag==1 and (len(self.high_flag)>1 and (self.high_flag[-1]!=2) and (self.high_flag[-2]!=2) ) and imf_high_flag!=2:#or ( len(self.close_flag)>1 and self.close_flag[-1]!=2 and self.close_flag[-2]==1) or ( len(self.close_flag)>2 and self.close_flag[-3]==1 and self.close_flag[-1]!=2 and self.close_flag[-2]!=2) or (len(self.close_flag)>3 and (self.close_flag[-4]==1) and self.close_flag[-3]!=2 and self.close_flag[-2]!=2 and self.close_flag[-1]!=2)) :#and ((len(self.high_flag)>0 and self.high_flag[-1]==1) or imf_high_flag==1):
        #if (self.macd[current_price_index]>0 and imf_close_flag==1) or (imf_high_flag==1 and imf_close_flag==1 and self.macd[current_price_index]<0):#or (len(self.high_flag)>0 and (self.high_flag[-1]==1))):# and (self.high_flag[-2]!=2) ) and imf_high_flag!=2:#or ( len(self.close_flag)>1 and self.close_flag[-1]!=2 and self.close_flag[-2]==1) or ( len(self.close_flag)>2 and self.close_flag[-3]==1 and self.close_flag[-1]!=2 and self.close_flag[-2]!=2) or (len(self.close_flag)>3 and (self.close_flag[-4]==1) and self.close_flag[-3]!=2 and self.close_flag[-2]!=2 and self.close_flag[-1]!=2)) :#and ((len(self.high_flag)>0 and self.high_flag[-1]==1) or imf_high_flag==1):
        if (imf_close_flag==1 ) :#and  (len(self.close_flag)>1 and self.close_flag[-1]==2 and self.close_flag[-2]==2 ):#and self.close_flag[-3]==2):#or (imf_high_flag==1 and imf_close_flag==1 and deltama5<deltama5_before):#or (len(self.high_flag)>0 and (self.high_flag[-1]==1))):# and (self.high_flag[-2]!=2) ) and imf_high_flag!=2:#or ( len(self.close_flag)>1 and self.close_flag[-1]!=2 and self.close_flag[-2]==1) or ( len(self.close_flag)>2 and self.close_flag[-3]==1 and self.close_flag[-1]!=2 and self.close_flag[-2]!=2) or (len(self.close_flag)>3 and (self.close_flag[-4]==1) and self.close_flag[-3]!=2 and self.close_flag[-2]!=2 and self.close_flag[-1]!=2)) :#and ((len(self.high_flag)>0 and self.high_flag[-1]==1) or imf_high_flag==1):
        #if (imf_close_flag==1 and (len(self.close_flag)>1 and self.close_flag[-2]!=2)) or (imf_high_flag==1 and (len(self.high_flag)>1 and self.high_flag[-2]!=2) ):#and  (len(self.close_flag)>1 and self.close_flag[-1]==1):#or (imf_high_flag==1 and imf_close_flag==1 and deltama5<deltama5_before):#or (len(self.high_flag)>0 and (self.high_flag[-1]==1))):# and (self.high_flag[-2]!=2) ) and imf_high_flag!=2:#or ( len(self.close_flag)>1 and self.close_flag[-1]!=2 and self.close_flag[-2]==1) or ( len(self.close_flag)>2 and self.close_flag[-3]==1 and self.close_flag[-1]!=2 and self.close_flag[-2]!=2) or (len(self.close_flag)>3 and (self.close_flag[-4]==1) and self.close_flag[-3]!=2 and self.close_flag[-2]!=2 and self.close_flag[-1]!=2)) :#and ((len(self.high_flag)>0 and self.high_flag[-1]==1) or imf_high_flag==1):
        #if (imf_close_flag==1 and (len(self.close_flag)>1 and self.close_flag[-2]!=2)) or (imf_high_flag==1 and (len(self.high_flag)>1 and self.high_flag[-2]!=2) ):#and  (len(self.close_flag)>1 and self.close_flag[-1]==1):#or (imf_high_flag==1 and imf_close_flag==1 and deltama5<deltama5_before):#or (len(self.high_flag)>0 and (self.high_flag[-1]==1))):# and (self.high_flag[-2]!=2) ) and imf_high_flag!=2:#or ( len(self.close_flag)>1 and self.close_flag[-1]!=2 and self.close_flag[-2]==1) or ( len(self.close_flag)>2 and self.close_flag[-3]==1 and self.close_flag[-1]!=2 and self.close_flag[-2]!=2) or (len(self.close_flag)>3 and (self.close_flag[-4]==1) and self.close_flag[-3]!=2 and self.close_flag[-2]!=2 and self.close_flag[-1]!=2)) :#and ((len(self.high_flag)>0 and self.high_flag[-1]==1) or imf_high_flag==1):
#        if (((deltama3+deltama3_before)>(deltama5+deltama5_before)) and imf_close_flag==1) or (imf_high_flag==1 and imf_close_flag==1 and ((deltama3+deltama3_before)<(deltama5+deltama5_before))):#or (len(self.high_flag)>0 and (self.high_flag[-1]==1))):# and (self.high_flag[-2]!=2) ) and imf_high_flag!=2:#or ( len(self.close_flag)>1 and self.close_flag[-1]!=2 and self.close_flag[-2]==1) or ( len(self.close_flag)>2 and self.close_flag[-3]==1 and self.close_flag[-1]!=2 and self.close_flag[-2]!=2) or (len(self.close_flag)>3 and (self.close_flag[-4]==1) and self.close_flag[-3]!=2 and self.close_flag[-2]!=2 and self.close_flag[-1]!=2)) :#and ((len(self.high_flag)>0 and self.high_flag[-1]==1) or imf_high_flag==1):
        #if imf_close_flag==1 and  imf_high_flag!=2:#or ( len(self.close_flag)>1 and self.close_flag[-1]!=2 and self.close_flag[-2]==1) or ( len(self.close_flag)>2 and self.close_flag[-3]==1 and self.close_flag[-1]!=2 and self.close_flag[-2]!=2) or (len(self.close_flag)>3 and (self.close_flag[-4]==1) and self.close_flag[-3]!=2 and self.close_flag[-2]!=2 and self.close_flag[-1]!=2)) :#and ((len(self.high_flag)>0 and self.high_flag[-1]==1) or imf_high_flag==1):
        #if self.buy_price==0 and (imf_close_flag==1 or imf_high_flag==1) :#or (imf_high_flag==1 and imf_close_flag!=2):
            imf_flag=1
#        if imf_high_flag==1 and (len(self.close_flag)>1 and ( self.close_flag[-1]!=2 and self.close_flag[-2]!=2 and imf_close_flag!=2)):
#            imf_flag=1
        if (imf_close_flag==2 ) :#and len(self.close_flag)>1 and self.close_flag[-1]==1 and self.close_flag[-2]==1  :#or (imf_high_flag==2 ) :#or (imf_high_flag==2 and imf_close_flag!=1):#and imf_high_flag==2:
        #if  (imf_close_flag==2 ) :#or (imf_high_flag==2 and imf_close_flag!=1):#and imf_high_flag==2:
            imf_flag=2
        

        if imf_flag==1:
            self.imf_sugg_x_index.append(current_price_index)
        self.close_flag.append(imf_close_flag)
        self.high_flag.append(imf_high_flag)
        print "imf flag %s %s"%(imf_close_flag,imf_high_flag)
        print "imf flag %s"%imf_flag
        print "high list%s"%self.high_flag[-5:]
        print "close list%s"%self.close_flag[-5:]


######
        ma5 = []
        for i in range(5):
            ma5.append(0)
#
        print current_price_index
        for i in range(5,current_price_index+1):
            mean_5 = np.mean(self.close_price[i-4:i+1])
            ma5.append(mean_5)

        mama5 = []
        for i in range(3):
            mama5.append(0)
#
        for i in range(3,current_price_index+1):
            mean_5 = np.mean(ma5[i-2:i+1])
            mama5.append(mean_5)

        ma5=mama5
        print self.close_price[i]
        print "ma5 %s"%ma5[-10:]
        ma_max_index = list(argrelextrema(np.array(ma5[-60:]),np.greater)[0])
        print " ma max index %s"%ma_max_index
        ma_max_index1 = list(argrelextrema(np.array(ma5[-20:]),np.greater)[0])
        ma_min_index = list(argrelextrema(np.array(ma5[-60:]),np.less)[0])
        ma_min_index1 = list(argrelextrema(np.array(ma5[-20:]),np.less)[0])
        print " ma min index %s"%ma_min_index


        print "last max%s"%last_max
        large_num=len(ma_max_index)+len(ma_min_index)
        small_num=len(ma_max_index1)+len(ma_min_index1)

        ma_ex_de=0
        #if ma_max_index[-1]>ma_min_index[-1] and ma_max_index[-1]>496:#and len(ma5)-ma_min_index[-1]<=5 :#and ma_min_index[-1]-ma_max_index[-1]>=2:#and abs(ma_max_index[-1]-last_max)<2:
        #    ma_ex_de=2
        if small_num>0 and ((large_num/float(small_num))>3):# or ((large_num/float(small_num))<2)):
            ma_ex_de=1


        kknd=self.close_price[current_price_index-499:current_price_index+1]
        ma5 = []
        for i in range(3):
            ma5.append(0)
#
        for i in range(3,len(kknd)):
            mean_5 = np.mean(kknd[i-2:i+1])
            ma5.append(mean_5)

        maa5 = []
        for i in range(3):
            maa5.append(0)
#
        for i in range(3,len(kknd)):
            mean_5 = np.mean(ma5[i-2:i+1])
            maa5.append(mean_5)

        max_index = list(argrelextrema(np.array(maa5),np.greater)[0])
        print " max index %s"%max_index
        min_index = list(argrelextrema(np.array(maa5),np.less)[0])
        print " min index %s"%min_index
        #com_de=0
        #if max_index[-1]<min_index[-1] and min_index[-1]>495:
        #    com_de=1
#######

        ma5 = self.ma(self.close_price, 5, 0, current_price_index)
        ma5_before = self.ma(self.close_price, 5, -1, current_price_index)
        deltama5 = self.ma(self.close_price, 5, 0, current_price_index)-self.ma(self.close_price, 5, -1, current_price_index)
        deltama5_before = self.ma(self.close_price, 5, -1, current_price_index)-self.ma(self.close_price, 5, -2, current_price_index)
        deltama5_before2 = self.ma(self.close_price, 5, -2, current_price_index)-self.ma(self.close_price, 5, -3, current_price_index)
        deltama3 = self.ma(self.close_price, 3, 0, current_price_index)-self.ma(self.close_price, 3, -1, current_price_index)
        deltama3_before = self.ma(self.close_price, 3, -1, current_price_index)-self.ma(self.close_price, 3, -2, current_price_index)
        deltama3_before2 = self.ma(self.close_price, 3, -2, current_price_index)-self.ma(self.close_price, 3, -3, current_price_index)
        deltama3_before3 = self.ma(self.close_price, 3, -3, current_price_index)-self.ma(self.close_price, 3, -4, current_price_index)
        deltama9 = self.ma(self.close_price, 9, 0, current_price_index)-self.ma(self.close_price, 9, -1, current_price_index)
        deltama10 = self.ma(self.close_price, 10, 0, current_price_index)-self.ma(self.close_price, 10, -1, current_price_index)
        deltama9_before = self.ma(self.close_price, 9, -1, current_price_index)-self.ma(self.close_price, 9, -2, current_price_index)
        deltama10_before = self.ma(self.close_price, 10, -1, current_price_index)-self.ma(self.close_price, 10, -2, current_price_index)
        deltama9_before2 = self.ma(self.close_price, 9, -2, current_price_index)-self.ma(self.close_price, 9, -3, current_price_index)
        ma3 = self.ma(self.close_price, 3, 0, current_price_index)
        ma3_b = self.ma(self.close_price, 3, -1, current_price_index)
        ma3_b2 = self.ma(self.close_price, 3, -2, current_price_index)
        ma3_b3 = self.ma(self.close_price, 3, -3, current_price_index)
        ma3_b4 = self.ma(self.close_price, 3, -4, current_price_index)
        ma3_b5 = self.ma(self.close_price, 3, -5, current_price_index)
        ma9 = self.ma(self.close_price, 9, 0, current_price_index)
        ma10 = self.ma(self.close_price, 10, 0, current_price_index)
        ma10_b = self.ma(self.close_price, 10, -1, current_price_index)
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
        deltama60 = self.ma(self.close_price, 60, 0, current_price_index)-self.ma(self.close_price, 60, -1, current_price_index)
        deltama60_b = self.ma(self.close_price, 60, -1, current_price_index)-self.ma(self.close_price, 60, -2, current_price_index)
        deltama30_b = self.ma(self.close_price, 30, -1, current_price_index)-self.ma(self.close_price, 30, -2, current_price_index)

        deltamacd10 = self.ma(self.macd, 10, 0, current_price_index)-self.ma(self.macd, 10, -1, current_price_index)
        deltamacd20 = self.ma(self.macd, 20, 0, current_price_index)-self.ma(self.macd, 20, -1, current_price_index)
        deltamacd10_b = self.ma(self.macd, 10, -1, current_price_index)-self.ma(self.macd, 10, -2, current_price_index)
        deltamacd20_b = self.ma(self.macd, 20, -1, current_price_index)-self.ma(self.macd, 20, -2, current_price_index)
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

        power_ma5_de=0
        if self.close_price[current_price_index]-ma5>0 and self.close_price[current_price_index-1]-ma5_before<0:
            power_ma5_de=1
        if self.close_price[current_price_index]-ma3<0 and self.close_price[current_price_index-1]-ma3_b>0:
            power_ma5_de=2

        

        macdma5 = self.ma(self.macd, 5, 0, current_price_index)
        macdma5_before = self.ma(self.macd, 5, -1, current_price_index)
        macdma5_before2 = self.ma(self.macd, 5, -2, current_price_index)
        macdma5_before3 = self.ma(self.macd, 5, -3, current_price_index)
        deltamacdma5 = macdma5-macdma5_before
        deltamacdma5_before = macdma5_before-macdma5_before2
        deltamacdma5_before2 = macdma5_before2-macdma5_before3
        print "deltamacd %s %s"%(deltamacdma5_before,deltamacdma5)
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


        ma_de=1
        if (deltama3_before>deltama3 and deltama3<0) :
            ma_de=0
########
##############

           

############


        
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
            date = self.date[current_price_index+1].replace("/","-")
            date = date.replace(" ","")
            #trade_price=getanytime(self.mindata,date,"14:30:00")
            trade_price = self.close_price[current_price_index]
        else:
            trade_price = self.open_price[current_price_index+1]-0.1
            if trade_price<self.low_price[current_price_index+1]:
                trade_price = self.close_price[current_price_index+1]

            date = self.date[current_price_index+1].replace("/","-")
            date = date.replace(" ","")
            #trade_price=getanytime(self.mindata,date,"14:30:00")
            trade_price = self.close_price[current_price_index]
        date_b = self.date[current_price_index].replace("/","-")
        date_b = date_b.replace(" ","")
        #price_b=getanytime(self.mindata,date_b,"11:30:00")
        price_b=0



        print "date %s"%self.date[current_price_index]
        date=self.date[current_price_index].split("/")
        now_date=datetime.datetime(int(date[0]),int(date[1]),int(date[2]))
        print "current price %s %s %s %s"%(self.open_price[current_price_index],self.high_price[current_price_index],self.low_price[current_price_index],self.close_price[current_price_index])
        print "next price %s %s %s %s"%(self.open_price[current_price_index+1],self.high_price[current_price_index+1],self.low_price[current_price_index+1],self.close_price[current_price_index+1])
        print "trade price %s"%trade_price
        print "buy price %s"%self.buy_price
        if self.open_price[current_price_index]>self.close_price[current_price_index]:
            u_s = self.high_price[current_price_index]-self.open_price[current_price_index]
            d_s = self.close_price[current_price_index]-self.low_price[current_price_index]
        else:
            u_s = self.high_price[current_price_index]-self.close_price[current_price_index]
            d_s = self.open_price[current_price_index]-self.low_price[current_price_index]
        sha_de=0
        if u_s<=d_s :#or self.open_price[current_price_index]<self.close_price[current_price_index]:
            sha_de=1
        if self.buy_price!=0 and (u_s>d_s or self.open_price[current_price_index]>self.close_price[current_price_index]):
            sha_de=2



        

        current_mean_price = self.ma(self.close_price, 5, 0, current_price_index)
    
        print "buy mean%s current mean%s"%(self.buy_mean_price,current_mean_price)
#######

        print "fail flag %s"%self.fail_flag

        #if self.snr[-1]<np.mean(self.snr[-20:]) and self.snr[-1]<np.mean(self.snr[-5:]):

        if self.one_time_day>0:
            self.one_time_day+=1

        
        distance_decision = 1
        if len(self.sell_x_index)>0 and current_price_index-self.sell_x_index[-1]<2:
            distance_decision = 0

        distance_decision2 = 1
        if len(self.buy_x_index)>0 and current_price_index-self.buy_x_index[-1]<3:
            distance_decision2 = 0

        distance_decision3 = 1
        if len(self.buy_x_index)>0 and current_price_index-self.buy_x_index[-1]<1:
            distance_decision2 = 0

        if ((self.sell_flag==2 and imf_flag!=2) or (self.sell_flag==2 and imf_flag==2 and current_price<self.close_price[current_price_index-1])) and  self.buy_price!=0:
            self.sell_flag2=2




#######

        use_boll=0
        qu_de=0
        pivot=(self.high_price[current_price_index-1]+self.low_price[current_price_index-1]+self.close_price[current_price_index-1])/3.0
        res1=2*pivot-self.low_price[current_price_index-1]
        sup1=2*pivot-self.high_price[current_price_index-1]
        if sup1<current_price<res1:
            use_boll=1
        elif current_price>res1:
            qu_de=1

########

        ma5_list=[]
        for i in range(3):
            ma5_list.append(0)
        for i in range(3,len(self.close_price)):
            ma5_list.append(np.mean(self.close_price[i-2:i+1]))
            #ma5_list.append(np.mean(obv[i-2:i+1]))


        tmp_close=self.close_price
        self.close_price=ma5_list
        current_price=self.close_price[current_price_index]

        pppp=20
        recent_20=self.close_price[current_price_index-(pppp-1):current_price_index+1]
        print len(recent_20)
        ma5_list = []
        for i in range(5):
            ma5_list.append(0)
#
        for i in range(5,current_price_index+1):
            mean_5 = np.mean(self.close_price[i-4:i+1])
            ma5_list.append(mean_5)
        ma5_list=ma5_list[-pppp:] 
        testdata1 = np.array([recent_20[i]-ma5_list[i] for i in range(len(recent_20))])
        ma10_list = []
        for i in range(60):
            ma10_list.append(0)
#
        for i in range(60,current_price_index+1):
            mean_5 = np.mean(self.close_price[i-59:i+1])
            ma10_list.append(mean_5)
        ma10_list=ma10_list[-pppp:]

        testdata2 = np.array([recent_20[i]-ma10_list[i] for i in range(len(recent_20))])

        ma20_list = []
        for i in range(20):
            ma20_list.append(0)
#       
        for i in range(20,current_price_index+1):
            mean_5 = np.mean(self.close_price[i-19:i+1])
            ma20_list.append(mean_5)
        ma20_list=ma20_list[-pppp:]

        testdata3 = np.array([recent_20[i]-ma20_list[i] for i in range(len(recent_20))])


        
        ma30_list = []
        for i in range(30):
            ma30_list.append(0)
#       
        for i in range(30,current_price_index+1):
            mean_5 = np.mean(self.close_price[i-29:i+1])
            ma30_list.append(mean_5)
        ma30_list=ma30_list[-pppp:]

        testdata4 = np.array([recent_20[i]-ma30_list[i] for i in range(len(recent_20))])

#        normaltestresult=[]
#        normaltestresult.append(stats.normaltest(testdata1)[1])
#        normaltestresult.append(stats.normaltest(testdata2)[1])
#        normaltestresult.append(stats.normaltest(testdata3)[1])
#        normaltestresult.append(stats.normaltest(testdata4)[1])
#
#        self.normal30.append(stats.normaltest(testdata2)[1])

#        ma30normal_de=0
#        if len(self.normal30)>1 and (self.normal30[-2]/self.normal30[-1])>2:
#            ma30normal_de=1
#        elif len(self.normal30)>1 and (self.normal30[-2]/self.normal30[-1])<0.5:
#            ma30normal_de=2


#####
        ma30normal_de2=0
        #print "normal ma30 %s"%self.normal30
        

        maa5 = []
        for i in range(3):
            maa5.append(0)
#
        for i in range(3,len(self.normal30)):
            mean_5 = np.mean(self.normal30[i-2:i+1])
            maa5.append(mean_5)

        ma5 = []
        for i in range(3):
            ma5.append(0)
#
        for i in range(3,len(self.normal30)):
            mean_5 = np.mean(maa5[i-2:i+1])
            ma5.append(mean_5)

        dddd=ma5
        maa5 = []
        for i in range(3):
            maa5.append(0)
#
        for i in range(3,len(self.normal30)):
            mean_5 = np.mean(ma5[i-2:i+1])
            maa5.append(mean_5)

        ma5 = []
        for i in range(3):
            ma5.append(0)
#
        for i in range(3,len(self.normal30)):
            mean_5 = np.mean(maa5[2-1:i+1])
            ma5.append(mean_5)

        #ma5=maa5
        
        normal_max_index = list(argrelextrema(np.array(dddd),np.greater)[0])
        normal_min_index = list(argrelextrema(np.array(dddd),np.less)[0])

        print "normal max%s"%normal_max_index
        print "normal min%s"%normal_min_index

        if len(ma5)>3 and ma5[-2]>ma5[-3] and ma5[-1]<ma5[-2]:
            ma30normal_de2=1
        elif len(ma5)>3 and ma5[-1]<ma5[-2]:
            ma30normal_de2=2

        ma30normal_de2=0
        tmp=normal_max_index+normal_min_index
        tmp.sort()
        if tmp!=[] and (tmp[-1] in normal_max_index) and abs(tmp[-1]-len(self.normal30))<3:
            ma30normal_de2=1
        if tmp!=[] and (tmp[-1] in normal_min_index) and abs(tmp[-1]-len(self.normal30))<3:
            ma30normal_de2=2


######


        #normal_boll_de=0
        #if periodbase==0:
        #    period=5
        #    #stddata=self.close_price[current_price_index-4:current_price_index+1]
        #    stddata=testdata1
        #    #normal_boll_de=self.normal_boll_calc(current_price,stddata)
        #    normal_boll_de=self.normal_boll_calc(current_price-ma5,stddata)
        #    periodbasemean=np.mean(stddata)
        #elif periodbase==1:
        #    period=10
        #    stddata=self.close_price[current_price_index-9:current_price_index+1]
        #    stddata=testdata2
        #    #normal_boll_de=self.normal_boll_calc(current_price,stddata)
        #    normal_boll_de=self.normal_boll_calc(current_price-ma10,stddata)
        #    periodbasemean=np.mean(stddata)
        #elif periodbase==2:
        #    period=20
        #    stddata=self.close_price[current_price_index-19:current_price_index+1]
        #    stddata=testdata3
        #    #normal_boll_de=self.normal_boll_calc(current_price,stddata)
        #    normal_boll_de=self.normal_boll_calc(current_price-ma20,stddata)
        #    periodbasemean=np.mean(stddata)
        #elif periodbase==3:
        #    period=30
        #    stddata=self.close_price[current_price_index-29:current_price_index+1]
        #    stddata=testdata4
        #    #normal_boll_de=self.normal_boll_calc(current_price,stddata)
        #    normal_boll_de=self.normal_boll_calc(current_price-ma30,stddata)
        #    periodbasemean=np.mean(stddata)

        self.close_price=tmp_close
        current_price=self.close_price[current_price_index]
#####

        obv=calc_obv(self.high_price,self.low_price,self.close_price,self.vol)

        ma2_list=[]

        for i in range(2):
            ma2_list.append(0)
        for i in range(2,len(self.close_price)):
            ma2_list.append(np.mean(self.close_price[i-1:i+1]))
        ma3_list=[]

        for i in range(3):
            ma3_list.append(0)
        for i in range(3,len(self.close_price)):
            ma3_list.append(np.mean(self.close_price[i-2:i+1]))
            #ma5_list.append(np.mean(obv[i-2:i+1]))

        ma5_list=[]
        for i in range(5):
            ma5_list.append(0)
        for i in range(5,len(self.close_price)):
            ma5_list.append(np.mean(self.close_price[i-4:i+1]))

        ma10_list=[]
        for i in range(10):
            ma10_list.append(0)
        for i in range(10,len(self.close_price)):
            ma10_list.append(np.mean(self.close_price[i-9:i+1]))


        ma20_list=[]
        for i in range(20):
            ma20_list.append(0)
        for i in range(20,len(self.close_price)):
            ma20_list.append(np.mean(self.close_price[i-19:i+1]))

        ma30_list=[]
        for i in range(30):
            ma30_list.append(0)
        for i in range(30,len(self.close_price)):
            ma30_list.append(np.mean(self.close_price[i-29:i+1]))

        ma_list=[ma3_list[i]-ma10_list[i] for i in range(len(self.close_price))]
        ma_list2=[ma10_list[i]-ma20_list[i] for i in range(len(self.close_price))]
        ma_list=[ma2_list[i]-ma5_list[i] for i in range(len(self.close_price))]

        ma_list=[ma2_list[i]-ma10_list[i] for i in range(len(self.close_price))]
        ma_list=[close_price[i]-ma3_list[i] for i in range(len(close_price))]
        #ma_list=[ma2_list[i]-ma5_list[i] for i in range(len(self.close_price))]
        ma10ma20de=0
        if ma_list2[current_price_index]>ma_list2[current_price_index-1]>ma_list2[current_price_index-2]:
            ma10ma20de=1
        print "ma10ma20 %s"%ma_list2[current_price_index-10:current_price_index+1]


#        ma5_list=[]
#        for i in range(5):
#            ma5_list.append(0)
#        for i in range(5,len(ma_list)):
#            ma5_list.append(np.mean(ma_list[i-4:i+1]))

####
        dddd=ma5_list[current_price_index-50:current_price_index+1]
        dddd=ma_list[current_price_index-50:current_price_index+1]
        ma_list_diff = list(np.diff(ma_list))
        ma_list_diff.insert(0,0)
        ma5_list=[]
        for i in range(3):
            ma5_list.append(0)
        for i in range(3,len(ma_list)):
            ma5_list.append(np.mean(ma_list_diff[i-2:i+1]))
        dddd=ma5_list[current_price_index-50:current_price_index+1]
        normal_max_index = list(argrelextrema(np.array(dddd),np.greater)[0])
        normal_min_index = list(argrelextrema(np.array(dddd),np.less)[0])

        print "normal max%s"%normal_max_index
        print "normal min%s"%normal_min_index
        normal_de=0
        #if normal_max_index[-1]>normal_min_index[-1] and normal_max_index[-1]>48:
        if dddd[-1]>0:
            normal_de=2
        #if normal_min_index[-1]>normal_max_index[-1] and normal_min_index[-1]>48:
        if dddd[-2]<0 and dddd[-1]<0: #and dddd[-1]>dddd[-2]:
            normal_de=1

#####
        tmp=self.close_price
        self.close_price=ma_list
        #self.close_price=tmp
        current_price=self.close_price[current_price_index]
        
        #testdata = np.array((self.close_price[current_price_index-21:current_price_index+1]))
        
        std_30= (np.std(emd_data[-10:])/np.mean(emd_data[-10:])) 
        std_60= (np.std(emd_data[-30:])/np.mean(emd_data[-30:])) 
        print "std1 %s"%std_30
        print "std2 %s"%std_60
        m_d=0
        if deltama5>0 and ma5>ma10>ma20>ma30:
            m_d=1
        #if self.buy_price==0 and ((std_60-std_30)/std_60)>0.25:
        if 0:#self.buy_price==0 and ma_ex_de==1:
            period=10
        else:
            period=5
        period=5
        print "period%s"%period
        k1=1
        k2=1
        boll_de=0
        boll_high1 = k2*np.std(self.close_price[current_price_index-period:current_price_index+1])+np.mean(self.close_price[current_price_index-period:current_price_index+1])
        boll_high1_b = k2*np.std(self.close_price[current_price_index-period-1:current_price_index])+np.mean(self.close_price[current_price_index-period-1:current_price_index])
        boll_high1_b2 = k2*np.std(self.close_price[current_price_index-period-2:current_price_index-1])+np.mean(self.close_price[current_price_index-period-2:current_price_index-1])
        boll_low1 = k1*(-np.std(self.close_price[current_price_index-period:current_price_index+1]))+np.mean(self.close_price[current_price_index-period:current_price_index+1])
        boll_low1_b = k1*(-np.std(self.close_price[current_price_index-period-1:current_price_index]))+np.mean(self.close_price[current_price_index-period-1:current_price_index])
        boll_low1_b2 = k1*(-np.std(self.close_price[current_price_index-period-2:current_price_index-1]))+np.mean(self.close_price[current_price_index-period-2:current_price_index-1])

        period1=20
        boll_high2 = 3*(np.std(self.close_price[current_price_index-period1:current_price_index+1]))+np.mean(self.close_price[current_price_index-period1:current_price_index+1])
        boll_low2 = 2*(-np.std(self.close_price[current_price_index-period1:current_price_index+1]))+np.mean(self.close_price[current_price_index-period1:current_price_index+1])
        if current_price<boll_low1:
            if deltama20<0:
                boll_de=2
            else:
                boll_de=1
        elif current_price>boll_high1:
            if deltama20<0:
                boll_de=1
            else:
                boll_de=2

        boll_de1=0
        if current_price>boll_high1 :#and current_price>boll_low2:
            boll_de1=2
        elif current_price<boll_low1 :#and :
            boll_de1=1



        print "bollde1   %s"%boll_de1
        ma3ma5_b = self.ma(self.close_price, 5, -1, current_price_index)
        ma3ma5 = self.ma(self.close_price, 5, 0, current_price_index)

     #   boll_de1=0
     #   if self.close_price[current_price_index-1]<boll_low1_b and current_price>boll_low1:
     #       boll_de1=1
    #    ####if self.close_price[current_price_index-1]<boll_high1_b and current_price>boll_high1:
    #    ####    boll_de=1
    #    ##if self.close_price[current_price_index-1]>boll_low1_b and current_price<boll_low1:
    #    ##    boll_de=2
      #  if self.close_price[current_price_index-1]>boll_high1_b and current_price<boll_high1:
      #      boll_de1=2


        print "low1 low2%s %s"%(boll_low1,boll_low2)
        print "current %s"%current_price
        print "width %s"%(boll_high1-boll_low1)
        print "widthb %s"%(boll_high1_b-boll_low1_b)
        print "widthb2 %s"%(boll_high1_b2-boll_low1_b2)
        self.close_price=tmp
        current_price=self.close_price[current_price_index]




##
        ma5_list=[]
        for i in range(3):
            ma5_list.append(0)
        for i in range(3,len(self.close_price)):
            ma5_list.append(np.mean(self.close_price[i-2:i+1]))

        ma1=[]
        pdata2=1
        for i in range(pdata2):
            ma1.append(0)
        for i in range(pdata2,len(emd_data2)):
            ma1.append(np.mean(emd_data[i-(pdata2-1):i+1]))

        ma2=[]
        pdata2=1
        for i in range(pdata2):
            ma2.append(0)
        for i in range(pdata2,len(emd_data2)):
            ma2.append(np.mean(emd_data2[i-(pdata2-1):i+1]))

        me = [ma1[i]*ma2[i] for i in range(len(emd_data2))]

        tmp=self.close_price
        #self.close_price=me
        #current_price=me[-1]

        #testdata = np.array((self.close_price[current_price_index-21:current_price_index+1]))

        std_30= (np.std(emd_data[-10:])/np.mean(emd_data[-10:]))
        std_60= (np.std(emd_data[-30:])/np.mean(emd_data[-30:]))
        print "std1 %s"%std_30
        print "std2 %s"%std_60
        m_d=0
        if deltama5>0 and ma5>ma10>ma20>ma30:
            m_d=1
        #if self.buy_price==0 and ((std_60-std_30)/std_60)>0.25:
        if self.buy_price==0 and ma_ex_de==1:
            period=10
        else:
            period=5
        period=5
        print "period%s"%period
        k1=1
        k2=1
        boll_de=0
        boll_high1 = k2*np.std(self.close_price[-period:])+np.mean(self.close_price[-period:])
        boll_low1 = k1*(-np.std(self.close_price[-period:]))+np.mean(self.close_price[-period:])

        boll_de2=0
        #if self.close_price[current_price_index-1]<boll_low1_b and current_price>boll_low1:
        #    boll_de2=1
        ####if self.close_price[current_price_index-1]<boll_high1_b and current_price>boll_high1:
        ####    boll_de=1
        ##if self.close_price[current_price_index-1]>boll_low1_b and current_price<boll_low1:
        ##    boll_de=2
        #if self.close_price[current_price_index-1]>boll_high1_b and current_price<boll_high1:
        #    boll_de2=2

        if current_price>boll_high1 :#and current_price>boll_low2:
            boll_de2=2
        elif current_price<boll_low1 :#and :
            boll_de2=1

        print "low1 high1%s %s"%(boll_low1,boll_high1)
        print "current %s"%current_price
        print "width %s"%(boll_high1-boll_low1)
        print "widthb %s"%(boll_high1_b-boll_low1_b)
        print "widthb2 %s"%(boll_high1_b2-boll_low1_b2)
        self.close_price=tmp
        current_price=self.close_price[current_price_index]

###

        boll_de=0
        if boll_de1==1 :#and boll_de2==1:
        #if normal_boll_de==1:# and boll_de1==1:
        #if ma30normal_de2==1 :#and (stats.normaltest(testdata1)[1])>(stats.normaltest(testdata2)[1]):# and deltama3>deltama3_before:
            boll_de=1
        #elif normal_boll_de==2:# and boll_de1==2:
        #elif ma30normal_de2==2:# and deltama3<deltama3_before:
        elif boll_de1==2:# and boll_de2==2:
            boll_de=2
            self.peak.append(current_price_index)

        self.boll_de.append(boll_de)
        if boll_de==1:
        #if imf_flag==1:
            self.ex_x_index.append(current_price_index)
            self.ex_y_index.append(current_price)
        #if imf_flag==2:
        if boll_de==2:
            self.ex_x_index2.append(current_price_index)
            self.ex_y_index2.append(current_price)


        width_b2=boll_high1_b2-boll_low1_b2
        width_b1=boll_high1_b-boll_low1_b
        width=boll_high1-boll_low1


        mean1=(width+width_b1)/2.0
        mean2=(width_b1+width_b2)/2.0

        down_de=1
        if mean1<mean2 and deltama10>10 and deltama20>0 :#and (deltama3<deltama3_before and deltama3<0) and deltama30>0:
            down_de=0
               


        ma_de_kk=1
        if ma10<ma20:
            ma_de_kk=0

        print "boll %s"%boll_de
        print "boll %s"%self.boll_de[-10:]
        decision=0
        if use_boll==1 and boll_de==1:
            decision=1
        #if use_boll==1 and boll_de==2:
        #    decision=2
        #if use_boll==0 and qu_de==1:
        #    decision=1
        print "imf flag %s"%imf_flag
#####

        current_k=self.kdjk[current_price_index+1]
        before_k=self.kdjk[current_price_index]
        before_k2=self.kdjk[current_price_index-1]
        current_d=self.kdjd[current_price_index+1]
        before_d=self.kdjd[current_price_index]
        print "k %s %s %s"%(before_k2,before_k,current_k)
        print "d %s %s"%(before_d,current_d)

        kdj_de=0
        #if before_k<before_d and current_k>current_d:
        #if  ((current_k+before_k)/2.0)<35 :#and (current_k-before_k)>(before_k-before_k2):#and current_k>current_d:
        #if mean1>mean2 and current_k>before_k and deltama5>deltama5_before:
        if current_d>before_d :
            kdj_de=1

        print "kdj_de %s"%kdj_de
        print "wr%s"%self.wr[current_price_index-5:current_price_index+1]

        print "current_price_index%s"%current_price_index
######

        ma_list=[deltama20,deltama25,deltama30,deltama60]
        mac=0
        for i in ma_list:
            if i>0:
                mac+=1

        if mac>=2 :#and (current_k+before_k)>(current_d+before_d):
            ma_decision=1
        else:
            ma_decision=0 

        ma_kdj=0
        if ma_decision==1:
            ma_kdj=1
        elif ma_decision==0 and kdj_de==1:
            ma_kdj=1

####
        wrma3=(self.wr[current_price_index]+self.wr[current_price_index-1]+self.wr[current_price_index-2])/3.0
        wr_de=0
        if self.wr[current_price_index]>60 :#and self.wr[current_price_index]<self.wr[current_price_index-1]:
        #if wrma3>60:
        #if wrma3>60:
            wr_de=1
##############





####
        if wr_de==1  and self.buy_price==0 and len(self.boll_de)>1 and self.boll_de[-2]==1 and (boll_de!=1 ) and imf_flag==1:
            self.f_d=1
        elif self.buy_price!=0:
            self.f_d=0
######

        

####
        
        #if self.fail_flag==1:
        #    if deltama60>0 and deltama20>0:
        #        self.fail_flag=0
        print "macd %s"%self.macd[current_price_index-10:current_price_index+1]
        #if self.buy_price!=0  and imf_flag!=1 and (imf_flag==2 or (deltama5<0 and deltama5_before>0) or (deltama3<0 and deltama3_before>0)) and distance_decision2==1:
        print "emd %s"%np.std(emd_data[-31:-1])
        print "emd %s"%(np.std(emd_data[-30:])/np.mean(emd_data[-30:]))
        print "emd %s"%(np.std(emd_data[-60:])/np.mean(emd_data[-60:]))
        std_de=0
        if np.std(emd_data[-30:])>0.5:
            std_de=1
        #if  self.buy_price!=0   and ( (deltama5_before2>deltama5_before>deltama5 ))  and ((current_price-self.buy_price)/self.buy_price)>0.01:
        print "madeicison%s"%ma_decision
        print "selfboll%s"%self.boll_de[-10:]
        flag_fail=1

        if self.buy_price!=0:
            self.each_day+=1

        ma5 = self.ma(self.close_price, 5, 0, current_price_index)
        print "normal max%s"%normal_max_index
        print "normal min%s"%normal_min_index
        print "len %s"%len(self.normal30)

        print "emd mean%s"%np.mean(emd_data[-30:])
        print "emd std%s"%np.std(emd_data[-30:])
        imfflag=0
        #if len(self.close_flag)>3 and (sum(self.close_flag[-3:])/3.0)<1:
        if len(self.close_flag)>2 and (imf_flag==1 or self.close_flag[-2]==1 or self.close_flag[-3]==1):
            imfflag=1 

        down_count=0
        ren_100=self.close_price[current_price_index-30:current_price_index]
        diffren100=np.diff(ren_100)
        for i in diffren100:
            if i<0:
                down_count+=1

        zhi5=self.close_price[current_price_index]+self.close_price[current_price_index-1]+self.close_price[current_price_index-2]+self.close_price[current_price_index-3]
        zhi5=zhi5+kalval[0]
        zhi2=self.close_price[current_price_index]+kalval[0]
        rz=zhi2/2.0-zhi5/5.0

        zhi5=self.close_price[current_price_index]+self.close_price[current_price_index-1]+self.close_price[current_price_index-2]+self.close_price[current_price_index-3]
        zhi5=zhi5+predict
        zhi2=self.close_price[current_price_index]+predict
        pz=zhi2/2.0-zhi5/5.0

        print "kvcha %s"%(pz-rz)
        self.kvcha.append(pz-rz)

        ma5_list=[]
        for i in range(3):
            ma5_list.append(0)
        for i in range(3,len(self.close_price)):
            ma5_list.append(np.mean(self.open_price[i-2:i+1]))

        prechabig0=0
        #if len(self.pre_cha)>1 and self.pre_cha[-2]>0 and (pz-rz)<0:
        #if len(self.kvcha)>1 and self.kvcha[-2]>0 and self.kvcha[-1]<0:
        print "kalval%s"%self.kalv[-3:]
        print "trade price%s %s"%(price_b,trade_price)
        #date = self.date[current_price_index].replace("/","-")
        #date = date.replace(" ","")
        #o=getanytime(self.mindata,date,"09:31:00")
        #mc=getanytime(self.mindata,date,"11:30:00")
        #ao=getanytime(self.mindata,date,"13:01:00")
        #c=getanytime(self.mindata,date,"15:00:00")
        #print "current %s %s %s %s"%(o,mc,ao,c)

        #date = self.date[current_price_index+1].replace("/","-")
        #date = date.replace(" ","")
        #o=getanytime(self.mindata,date,"09:31:00")
        #mc=getanytime(self.mindata,date,"11:30:00")
        #ao=getanytime(self.mindata,date,"13:01:00")
        #c=getanytime(self.mindata,date,"15:00:00")
        #print "next %s %s %s %s"%(o,mc,ao,c)
        if len(self.kalv)>2 and  self.kalv[-2][0]>self.close_price[current_price_index] and kalval[0]<self.close_price[current_price_index+1] :#and self.kalv[-3][0]>self.close_price[current_price_index-1]:
        #if len(self.kalv)>1 and self.kalv[-2][0]>price_b and kalval[0]<trade_price :#and self.kalv[-3][0]>self.close_price[current_price_index-1]:
        #if len(self.pre_cha)>1 and self.pre_cha[-2]>0 and (pz-rz)<0:
        #if len(self.kalv)>1 and self.kalv[-2]>price_b and kalval<trade_price:
            prechabig0=1
        #if len(self.kvcha)>1 and self.kvcha[-2]<0 and self.kvcha[-1]>0:
        #if len(self.pre_cha)>2 and self.pre_cha[-2]>0 and self.pre_cha[-3]<0:
        #if len(self.pre_cha)>1 and self.pre_cha[-2]<0 and (pz-rz)>0:
        #if len(self.kalv)>1 and self.kalv[-2]<price_b and kalval>trade_price:
        mapp=(self.close_price[current_price_index]+self.close_price[current_price_index-1]+kalval[1])/3.0
        #if len(self.kalv)>1 and  (self.kalv[-2][0]-ma3_list[current_price_index])<0 and (kalval[0]-(mapp))>0:
        #if len(self.kalv)>1 and self.kalv[-2][1]<self.open_price[current_price_index] and kalval[1]>self.open_price[current_price_index+1]:
        #    prechabig0=2
        #if len(self.kalv)>1 and deltama3<deltama3_before and (self.kalv[-2]-ma3_list[current_price_index])< (kalval-ma3_list[current_price_index+1]):
        #if len(self.pre_cha)>1 and self.pre_cha[-2]<0 and (pz-rz)>0:
        #    prechabig0=2
        #if len(self.kalv)>1 and  (self.kalv[-2][1]-close_price[current_price_index])<0 and (kalval[1]-close_price[current_price_index+1])>0:
        #    prechabig0=2
        #if self.buy_price!=0 and len(self.kalv)>1 and self.open_price[current_price_index+1]>mc:
        #if len(self.kalv)>2 and self.kalv[-2][0]<self.close_price[current_price_index] and kalval[0]>self.close_price[current_price_index+1] :#and self.kalv[-3][0]>self.close_price[current_price_index-1]:
        if len(self.kalv)>2 and  self.kalv[-2][0]<self.close_price[current_price_index] and kalval[0]>self.close_price[current_price_index+1] :#and self.kalv[-3][0]>self.close_price[current_price_index-1]:
        #if len(self.kalv)>2 and  kalval[0]<self.close_price[current_price_index] :#and self.kalv[-3][0]>self.close_price[current_price_index-1]:
        #if len(self.kalv)>1 and self.kalv[-2][0]<price_b and kalval[0]>trade_price:
            prechabig0=2

        print "prechade%s"%prechade

        print "SNR%s"%self.snr[-10:]



        ma_list=[ma3_list[i]-ma10_list[i] for i in range(len(self.close_price))]


        #print "stable test %s"%ts.adfuller(self.close_price[current_price_index-10:current_price_index+1], 1)[1]
        #print "stable test %s"%ts.adfuller(ma_list[current_price_index-10:current_price_index+1], 1)[1]
        #self.stable_test.append(ts.adfuller(self.close_price[current_price_index-10:current_price_index+1], 1)[1])

        #stable_de=1
        #if ts.adfuller(ma_list[current_price_index-10:current_price_index+1], 1)[1]>0.5 and deltama5<0:
        #    stable_de=0


###

        ddd = self.close_price[:current_price_index+1]
        pl = range(2,5)
        rl=[]
        for i in pl:
            for j in pl:
                if j>i:
                    ma = ma_func3(ddd,i,j)
                    #print "pingwen"
                    #print "%s %s"%(i,j)
                    #print ts.adfuller(ma[-20:], 1)[1]
                    #rl.append((i,j,ts.adfuller(ma[-100:])[1],abs(np.mean(ma[-100:]))))
            
    #    rl.sort(key=lambda x:x[3])
    #    print rl
    #    print ts.adfuller(ddd[-100:], 1)[1]
    #    if rl[0][2]<0.05:#self.buy_price==0:
    #        ma_list=ma_func3(ddd,rl[0][0],rl[0][1])
    #        #ma_list=ma_func3(ddd,3,5)
    #    else:
    #        ma_list=self.buy_ma

        ma2_list=ma_func2(ddd,2)
        ma5_list=ma_func2(ddd,3)
        ma30_list=ma_func2(ddd,30)
        ma2_ma2_list=ma_func2(ma2_list,2)
        ma2_ma2_ma2_list=ma_func2(ma2_ma2_list,2)
        ma_list=[ma5_list[i]-ma30_list[i] for i in range(len(ma2_list))]


        ref = np.array(ma_list[-100:-5])
        sample = np.array(ma_list[-5:])
        ks_d,ks_p_value = stats.ks_2samp(ref,sample)
    
        print "ks %s %s"%(ks_d,ks_p_value)
        same_de=1
        if ks_p_value < 0.1:# and ks_d > 0.5:
            same_de=0
        print "same %s"%same_de

        print "mean 100 %s"%np.mean(ma_list[-100:])
        print "mean 30 %s"%np.mean(ma_list[-30:])
        print "mean 10 %s"%np.mean(ma_list[-10:])
        ma_2=ma_func2(ma_list,2)
        sample=ma_2[-30:]
        
        #un=[]
        #an=[]
        #u=0
        #a=0
        #for i in range(1,30):
        #    if ma_list[i]>ma_2[i] and ma_list[i-1]>ma_2[i-1]:
        #        a+=1
        #    if ma_list[i]>ma_2[i] and ma_list[i-1]<ma_2[i-1]:
        #        un.append(u)
        #        u=0
        #    if ma_list[i]<ma_2[i] and ma_list[i-1]>ma_2[i-1]:
        #        an.append(a)
        #        a=0
        #    if ma_list[i]<ma_2[i] and ma_list[i-1]<ma_2[i-1]:
        #        u+=1

        #print "above %s"%an
        #print "under %s"%un

        #u1_mean=[]
        #a1_mean=[]
        #for i in un:
        #    if i!=0:
        #        u1_mean.append(i)
        #for i in an:
        #    if i!=0:
        #        a1_mean.append(i)

        #u_mean=np.mean(u1_mean)
        #a_mean=np.mean(a1_mean)
        #r=0
        #for i in range(1,int(u_mean)+1):
        #    if ma_list[-i]<ma_2[-i]:
        #        r+=1

        #r1=0
        #for i in range(1,int(a_mean)+1):
        #    if ma_list[-i]>ma_2[-i]:
        #        r1+=1
        
        power_de_ma_list1=0
        pp=100
        #if  ma_list[-1]-np.mean(ma_list[-pp:])>0 and ma_list[-2]-np.mean(ma_list[-(pp+1):-1])<0:
        #if  same_de==1 and ma_list[-1]<(np.mean(ma_list[-100:])-abs(np.std(ma_list[-100:]))) :#and ma_list[-2]<(np.mean(ma_list[-6:-1])-abs(np.std(ma_list[-6:-1]))) :#and np.mean(ma_list[-100:])>np.mean(ma_list[-101:-1]):# and ma_list[-1]-ma_list[-2]>0 and ma_list[-2]-ma_list[-3]<0:
        if  self.buy_price==0 and ma_list[-1]>ma_list[-2]:#( r==u_mean or ((ma_list[-1]>ma_2[-1]) and ma_list[-2]<ma_2[-2])):#and ma_list[-2]<np.mean(ma_list[-100:]) :#and ma_list[-2]<(np.mean(ma_list[-6:-1])-abs(np.std(ma_list[-6:-1]))) :#and np.mean(ma_list[-100:])>np.mean(ma_list[-101:-1]):# and ma_list[-1]-ma_list[-2]>0 and ma_list[-2]-ma_list[-3]<0:
        #if  self.buy_price==0 and   ma_list[-1]<np.mean(ma_list[-2:]) :#and ma_list[-2]<np.mean(ma_list[-100:]) :#and ma_list[-2]<(np.mean(ma_list[-6:-1])-abs(np.std(ma_list[-6:-1]))) :#and np.mean(ma_list[-100:])>np.mean(ma_list[-101:-1]):# and ma_list[-1]-ma_list[-2]>0 and ma_list[-2]-ma_list[-3]<0:
            power_de_ma_list1=1
        #if  ma_list[-1]-np.mean(ma_list[-pp:])<0 and ma_list[-2]-np.mean(ma_list[-(pp+1):-1])>0:
        #if ma_list[-1]>np.mean(ma_list[-5:])+abs(np.std(ma_list[-5:])):#ma_list[-1]-ma_list[-2]<0 and ma_list[-2]-ma_list[-3]>0:
        #if   ma_list[-1]>(np.mean(ma_list[-100:])+abs(np.std(ma_list[-100:]))) :#and ma_list[-2]>(np.mean(ma_list[-6:-1])-abs(np.std(ma_list[-6:-1]))) :#and np.mean(ma_list[-100:])>np.mean(ma_list[-101:-1]):# and ma_list[-1]-ma_list[-2]>0 and ma_list[-2]-ma_list[-3]<0:
        #if  ma_list[-1]>np.mean(ma_list[-2:]) :#and ma_list[-2]>np.mean(ma_list[-100:]) :#and ma_list[-2]<(np.mean(ma_list[-6:-1])-abs(np.std(ma_list[-6:-1]))) :#and np.mean(ma_list[-100:])>np.mean(ma_list[-101:-1]):# and ma_list[-1]-ma_list[-2]>0 and ma_list[-2]-ma_list[-3]<0:
        if   self.buy_price!=0 and ma_list[-1]<ma_list[-2]:#(r1==a_mean or ((ma_list[-1]<ma_2[-1]) and ma_list[-2]>ma_2[-2])):#and ma_list[-2]<np.mean(ma_list[-100:]) :#and ma_list[-2]<(np.mean(ma_list[-6:-1])-abs(np.std(ma_list[-6:-1]))) :#and np.mean(ma_list[-100:])>np.mean(ma_list[-101:-1]):# and ma_list[-1]-ma_list[-2]>0 and ma_list[-2]-ma_list[-3]<0:
        #if   self.buy_price!=0 and  ma_list[-1]>np.mean(ma_list[-2:]) :#and ma_list[-2]>np.mean(ma_list[-3:-1]))):#and ma_list[-2]<np.mean(ma_list[-100:]) :#and ma_list[-2]<(np.mean(ma_list[-6:-1])-abs(np.std(ma_list[-6:-1]))) :#and np.mean(ma_list[-100:])>np.mean(ma_list[-101:-1]):# and ma_list[-1]-ma_list[-2]>0 and ma_list[-2]-ma_list[-3]<0:
            power_de_ma_list1=2

        power_de_ma_list1=0


        
        if  self.buy_price==0 and   self.count_buy%3 == 0:#ma_list[-1]<np.mean(ma_list[-2:]) :#and ma_list[-2]<np.mean(ma_list[-100:]) :#and ma_list[-2]<(np.mean(ma_list[-6:-1])-abs(np.std(ma_list[-6:-1]))) :#and np.mean(ma_list[-100:])>np.mean(ma_list[-101:-1]):# and ma_list[-1]-ma_list[-2]>0 and ma_list[-2]-ma_list[-3]<0:
            power_de_ma_list1=1


        if   self.buy_price!=0 and  self.count_buy%3 == 0:#ma_list[-1]>np.mean(ma_list[-2:]) :#and ma_list[-2]>np.mean(ma_list[-3:-1]))):#and ma_list[-2]<np.mean(ma_list[-100:]) :#and ma_list[-2]<(np.mean(ma_list[-6:-1])-abs(np.std(ma_list[-6:-1]))) :#and np.mean(ma_list[-100:])>np.mean(ma_list[-101:-1]):# and ma_list[-1]-ma_list[-2]>0 and ma_list[-2]-ma_list[-3]<0:
            power_de_ma_list1=2

        self.count_buy += 1





        if self.last_fail_de ==1 and self.last_sel==1:
            self.last_sel=2
            self.last_fail_de=0
        if self.last_fail_de==1 and self.last_sel==2:
            self.last_sel=1
            self.last_fail_de=0
        
        if self.last_sel==1:
            power_de_ma_list=power_de_ma_list1
        else:
            power_de_ma_list=power_de_ma_list1


        ma_long = ma_func2(self.close_price[:current_price_index+1],14)
        ma_short = ma_func2(self.close_price[:current_price_index+1],2)
        madiff = [ma_short[i]-ma_long[i]  for i in range(len(ma_long))]
        max_index = list(argrelextrema(np.array(madiff[-100:]),np.greater)[0])
        min_index = list(argrelextrema(np.array(madiff[-100:]),np.less)[0])

        print " max %s"%max_index
        print " min %s"%min_index
        ma_de = 0
        if max_index[-1]<min_index[-1] and min_index[-1]>97:
            ma_de = 1
        if max_index[-1]>min_index[-1] and max_index[-1]>97:
            ma_de = 2

        print "valueimf2 {}".format(imf_list[2][-20:])
        print "valueimf1 {}".format(imf_list[1][-20:])

        #lr = rfclf(current_price_index,close_price,vol,je)
        #print "lr {}".format(lr)
        rp1 = self.close_price[current_price_index-6:current_price_index]
        rp = self.close_price[current_price_index-5:current_price_index+1]

        rc = self.close_price[:current_price_index+1]
        ma3 = ma_func2(rc,3)
        diffrcma3 = [rc[i]-ma3[i] for i in range(len(ma3))]
        print "rcma3 std"
        s1= np.std(diffrcma3[-20:-10])
        s2= np.std(diffrcma3[-10:])
        print s1
        print s2


        standardxiaoyu=[0.02770432357043262, -0.07597196894029601, -0.0944268774703545, -0.08071816012150101, -0.08530360653855418, 0.1491446904197744, 0.013247005937804701, -0.0867784991131213, -0.015042537025273361, -0.06737195768896953, 0.010744421369535573, 0.01639848948286371, 0.048238765749225365, 0.021633474909549477, 0.008042314603745382, 0.1321511394113024, -0.1731737624692702, -0.00646529827404585, 0.1827007533830809, -0.14821011127329342, -0.10931833486600695, 0.2460685482654803, -0.03996267201252124, -0.0015439392162228671, 0.1582452637096594, 0.16018489762537058, -0.0535604394132676, 0.04857271406139674, 0.041974101381011764, 0.029584617566065674, -0.27297215082415605, -0.009442540694930557, 0.2411335614578256, -0.07027573090238093, -0.08029454142358716, -0.07792223791740227, 0.16093069751769207, 0.0026641489777290417, 0.06946501142610018, 0.00859604907156708, -0.4973521841836188, -0.0015987208725185553, 0.2766299915344561, 0.0924215888445552, -0.02503916516528193, -0.037038501099168286, 0.23004221164257643, 0.12437241627392837, 0.29301285964621826, -0.1016051169186376, 0.062422389463783645, 0.3081266339684632, 0.05595631940284562, -0.10493085692614201, 0.038967308941330714, -0.04486036925406722, -0.1254400252686949, -0.012509916682546418, -0.26964675920460124, -0.03866409757720035, 0.15647220862115674, 0.03866006438517289, -0.18782581071200255, 0.1068820940850781, 0.11540574606931742, -0.20934447357244146, -0.053728533408040846, 0.08062707636286248, -0.10894216299975845, -0.07876838886206272, -0.07207386048574982, -0.2464551327289506, 0.21792129998406828, 0.1913454748305874, 0.086441507332081, 0.017890976750717158, 0.11575031476543174, -0.07289900120929893, 0.008703885998219363, -0.06114011376931394]
        standarddayu1=[0.008782436190375442, 0.006622680424860583, -0.002806639326320415, 0.012593655251002644, -0.026273398795573222, -0.013138065464399773, -0.01926067129645226, -0.03859415325868998, 0.002534426361910924, 0.0067747315086093, -0.006213274470999508, 0.008405775385043057, 0.006972348646768545, -0.010833451443303943, 0.02903618861947077, -0.006939478785268349, -0.04954989020601186, -0.005461959948990014, 0.013483710609261124, -0.01079932000588446, 0.027293851312388817, 0.015306804255247997, 0.07050879796480736, 0.03891496719938292, 0.006424304488820809, 0.080409850764112, 0.0006108555774186897, 0.004040707866620252, -0.027739454931279717, 0.013299556949950109, -0.004890993044596392, 0.007757137245460122, 0.0196424517593643, -0.06199200812078054, -0.018139573060239833, 0.04032841087920591, 9.434663100815754e-05, 0.04750949555275952, 0.009330962719862335, 0.0001974302726415189, 0.009116318896164266, 0.005258094182563511, 0.059572589803781106, -0.019300017651558754, 0.008712036644983101, -0.015317313129247445, -0.03600383468895618, -0.004784314652421351, 0.057246509181766214, 0.008149729284314233, -0.059860363238112946, 0.017155536665249826, 0.014228411632466909, -0.009726977637194878, -0.03977448216693258, -0.03435701684939918, 0.0036164270313152613, 0.002650833534868724, -0.04726207401663007, 0.09520436355612372, -0.06756966037114776, 0.1823167786644273, -0.061626091229254776, 0.010654817437662167, 0.005804274816568089, -0.048691240691505655, 0.014173330764043257, -0.03560271481601074, -0.0055591767428486705, 0.013914847916384865, 0.004833204367638011, 0.004463230863837708, -0.010608465964987879, 0.005535977771154776, -0.02572628881780048, 0.05661938136116351, 0.02520423751868961, 0.06796523901415608, -0.03586005846094498, -0.01946844796220404]
        standarddayu2=[-0.07602398523985343, 0.014334511189634469, 1.2810658468254132e-05, 0.06096550127344358, 0.05103499260719424, 0.05413542926239323, 0.030016006402561857, -0.062495320211015226, -0.00040209392307133385, -0.013326521675669056, 0.073668154419547, 0.018656241717465605, 0.02930454075972655, -0.01276699029126327, -0.006131376253451393, 0.006721009729160343, -0.06940026075619166, 0.04572625853360712, 0.0423543734781795, 0.018574651182220236, 0.03269651573809185, -0.05442443333561542, -0.014740575436359649, -0.035598521955831686, -0.0020097988116347665, -0.19466470051260387, 0.0012643506783955871, 0.06311926605504681, 0.042566371681415305, -0.051680340509530964, 0.037432463634264224, 0.08354477611940325, 0.04830183223364237, 0.022864568932146057, 0.11030129634379726, -0.12470485990405322, 0.0999714275635828, -0.08491654393716352, 0.19570069157292203, -0.045761570425561615, 0.06151491297122291, -0.0944776119402988, -0.08800791370295435, -0.15917112149006662, 0.021930501930501833, -0.011406022018835671, -4.050879040207178e-06, -0.08865081509501849, 0.10073938689042272, 0.06005822416302742, -0.0912508343345344, -0.1014838035527692, 0.03568451546584406, -0.01697721385608375, -0.020477891916033997, -0.06803077999300378, 0.010192697768763637, 0.06474386569091806, -0.04522090399592393, 0.12400000000000055, 0.15125232099112118, 0.0470441379715929, -0.11552050630388067, 0.0410978767478003, 0.13435627648692083, -0.022950090070997575, 0.000689564329700687, -0.05600130756343802, -0.005381794328989997, 0.054558322734173714]
        heihei = np.random.randn(100)
        ma3=ma_func2(self.close_price,3)[current_price_index-100:current_price_index+1]
        datac = self.close_price[current_price_index-100:current_price_index+1]
        datav = self.vol[current_price_index-100:current_price_index+1]
        dataj = self.je[current_price_index-100:current_price_index+1]
        mp=[dataj[i]/datav[i] for i in range(len(datav))]
        diffmp = [datac[i]-mp[i] for i in range(len(datac))]
        #diffmp = [mp[i]-datac[i] for i in range(len(datac))]
        ma3diffmp = [ma3[i]-mp[i] for i in range(len(datac))]
        #self.diffmpstd.append(np.std(diffmp[-5:]))
        print "diff mean std {} {}".format(np.mean(diffmp[-5:]),np.std(diffmp[-5:]))
        print "diff mean std {} {}".format(np.mean(diffmp),np.std(diffmp))
        std_de=0
        if np.std(diffmp[-10:])/np.std(diffmp)>0.9 and deltama10<0:
            std_de=0
        else:
            std_de=1
        diffmp1=diffmp[-20:]

        max_diff_index = argrelextrema(np.array(diffmp1),np.greater)[0]
        min_diff_index = argrelextrema(np.array(diffmp1),np.less)[0]
        imf0_max_index = list(argrelextrema(np.array(imf_list[0][-20:]),np.greater)[0])
        imf0_min_index = list(argrelextrema(np.array(imf_list[0][-20:]),np.less)[0])
    
        print "diff max {}".format(max_diff_index)
        print "diff min {}".format(min_diff_index)
        print "min_diff in min {}".format(calc_pro2(min_diff_index,imf0_min_index,imf0_max_index))
        print "min_diff in max {}".format(calc_pro2(min_diff_index,imf0_max_index,imf0_min_index))
        print "max_diff in min {}".format(calc_pro2(max_diff_index,imf0_min_index,imf0_max_index))
        print "max_diff in max {}".format(calc_pro2(max_diff_index,imf0_max_index,imf0_min_index))
    
        pro=calc_pro2(min_diff_index,imf0_min_index,imf0_max_index)
        pro1=calc_pro2(min_diff_index,imf0_max_index,imf0_min_index)
        pro2=calc_pro2(max_diff_index,imf0_min_index,imf0_max_index)
        pro3=calc_pro2(max_diff_index,imf0_max_index,imf0_min_index)

        diffmp2=diffmp[-10:]
        #print "extrem mean {} {}".format(np.mean(calc_recent_rilling(diffmp2)),np.std(calc_recent_rilling(diffmp2)))
        cd,cu=stats_mpup(diffmp2,self.close_price[current_price_index-9:current_price_index+1])
        #if len(self.cu)>0 and cu==self.cu[-1]:
        #    pass
        #else:
        #   self.cu.append(cu)
        #print self.cu
        print "cd {} cu {}".format(cd,cu)
        ma2=ma_func2(self.close_price,2)
        ma5=ma_func2(self.close_price,5)
        cma3=[ma2[i]-ma5[i] for i in range(len(ma5))]
        cma3r=cma3[current_price_index-10:current_price_index+1]
        stab=ts.adfuller(diffmp[-20:],1)
        print "cma3 stab {}".format(stab)
        kk=stab[0]
        kkk=stab[4]['5%']
        kkkk=stab[1]
        kkkkk=stab[4]['1%']

        stab=ts.adfuller(ma3diffmp[-20:],1)
        print "cma3 stab {}".format(stab)
        ma3kk=stab[0]
        ma3kkk=stab[4]['5%']

        self.stab_de.append(kkk/kk) 
        stab_de=0
        #if ((self.fail_flag==0 and kk<kkk) or (self.fail_flag==1 and kk<kkkkk)) or kkkk>0.05:
        if  kkk/kk<ma3kkk/ma3kk:
            stab_de=1
        else:
            stab_de=2
##
        ref = np.array(diffmp[:-10])
        sample = np.array(diffmp[-10:])
        ks_d,ks_p_value = stats.ks_2samp(ref,sample)

##
        p_c=0
        n_c=0
        for i in sample:
            if i>0:
                p_c+=1
            else:
                n_c+=1
        ma5=ma_func2(self.close_price,5)
        cdcd=calc_down(ma5[current_price_index-10:current_price_index+1])
        cdcd2=calc_down(ma5[current_price_index-12:current_price_index-1])


##
        period=200
        n=10
        mp_list=[self.je[i]/self.vol[i] for i in range(len(self.close_price))]
        mpdiff_list=[self.close_price[i]-mp_list[i] for i in range(len(self.close_price))]
         
        self.mpdown.append((calc_down(mp_list[current_price_index-29:current_price_index+1])))
        self.mpup.append((calc_up(mp_list[current_price_index-29:current_price_index+1])))
        mampdown = ma_func2(list(self.mpdown),20)
        mampup = ma_func2(list(self.mpup),10)
##
        print "trend "
        lp1= trend_predict(mp_list[current_price_index-100:current_price_index+1],self.close_price[current_price_index-100:current_price_index+1],self.mpdown[-1],self.mpup[-1],30,mpdiff_list[current_price_index-100:current_price_index+1])
        print lp1
        print "stable "

        lp2= stable_predict(mp_list[current_price_index-100:current_price_index+1],self.close_price[current_price_index-100:current_price_index+1],self.mpdown[-1],self.mpup[-1],30,mpdiff_list[current_price_index-100:current_price_index+1])
        print lp2


#####

        ref = np.array(diffmp[:-5])
        sample = np.array(diffmp[-5:])
        ks_d,ks_p_value = stats.ks_2samp(ref,sample)
        mean=np.mean(ref)
        std=np.std(ref)
        ks_p=stats.kstest(diffmp2,'norm',args=(mean,std))
        print "20ks {}".format(ks_p)

        print "ks %s"%ks_p_value
        print "kd %s"%ks_d
        self.ks.append(ks_p_value)

        print "stable test {}".format(ts.adfuller(diffmp[-10:],1))
        print "hurst %s"%(hurst(self.close_price[current_price_index-100:current_price_index+1]))
        hurst_de=0
        hhhh=abs(hurst(self.close_price[current_price_index-10:current_price_index+1]))
        if hhhh<0.5 or (hhhh>0.5 and deltama5>0):
            hurst_de=1
        #print "ksksks {}".format(self.ks)
        ks_de=0
        if ks_p_value>0.5:
            ks_de=1

        p=100
        datac = self.close_price[current_price_index-p:current_price_index+1]
        datav = self.vol[current_price_index-p:current_price_index+1]
        dataj = self.je[current_price_index-p:current_price_index+1]
        datavma1 = ma_func2(self.vol,5)[current_price_index-p:current_price_index+1]
        datavma2 = ma_func2(self.vol,10)[current_price_index-p:current_price_index+1]
        mp=[dataj[i]/datav[i] for i in range(len(datav))]
        mp_diff=[datac[i]-mp[i] for i in range(len(datac))]

        s11 = calc_mp_sum(datac,mp,mp_diff)
        s12 = calc_vol_sum(datac,datavma1,datavma2)
        print "s11 {}".format(s11)
        print "s12 {}".format(s12)

        p=10
        datac = self.close_price[current_price_index-p:current_price_index+1]
        datav = self.vol[current_price_index-p:current_price_index+1]
        dataj = self.je[current_price_index-p:current_price_index+1]
        datavma1 = ma_func2(self.vol,5)[current_price_index-p:current_price_index+1]
        datavma2 = ma_func2(self.vol,10)[current_price_index-p:current_price_index+1]
        mp=[dataj[i]/datav[i] for i in range(len(datav))]
        ma3=ma_func2(self.close_price,3)[current_price_index-p:current_price_index+1] 
        mp_diff=[datac[i]-mp[i] for i in range(len(datac))]
        ma3_mp_diff=[ma3[i]-mp[i] for i in range(len(datac))]

        s21 = calc_mp_sum(datac,mp,mp_diff,1)
        #s21 = s11[-5:]
        s22 = calc_mp_sum2(datac,mp,mp_diff)
        #s22 = calc_mp_sum(datac,mp,ma3_mp_diff,0)
        sharp=lambda x:np.mean(x)/np.std(x)
        self.s1.append(sharp(s11))
        self.s2.append(sharp(s21))

        print "fs21 {} {}".format(s21,sharp(s21))
        print "fs22 {} {}".format(s22,sharp(s22))
        sharp_de=0
        if  sharp(s21)>0:
            sharp_de=1

        print "diff mean std {} {}".format(np.mean(diffmp[-5:]),np.std(diffmp[-5:]))
        print "diff mean std {} {}".format(np.mean(diffmp[-10:]),np.std(diffmp[-10:]))
        print "diff mean std {} {}".format(np.mean(diffmp[-20:]),np.std(diffmp[-20:]))
        print "diff mean std {} {}".format(np.mean(diffmp[-50:]),np.std(diffmp[-50:]))
        print "diff mean std {} {}".format(np.mean(diffmp[-100:]),np.std(diffmp[-100:]))

##
        ma3s1=ma_func2(self.s1,2)
        ma3s2=ma_func2(self.s2,2)
        print "100 s1 {} s2 {}".format(sharp(s11),sharp(s12))
        print "30 s1 {} s2 {}".format(sharp(s21),sharp(s22))
        print "fail flag %s"%self.fail_flag
        sel=0


##

        refc=load_data("test_data/ll/axxt",4)
        refv=load_data("test_data/ll/axxt",5)
        refj=load_data("test_data/ll/axxt",6)
        refmp=[refj[i]/refv[i] for i in range(len(refc))]
        refmp=refmp[280:300]

        samc = self.close_price[:current_price_index+1]
        samv = self.vol[:current_price_index+1]
        samj = self.je[:current_price_index+1]
        sammp=[samj[i]/samv[i] for i in range(len(samc))]
        sammp=sammp[-20:]
        
        p=pearsonr(refmp,sammp)
        print "person {}".format(p)
##
        p=10
        cpstd=np.std(self.close_price[current_price_index-p:current_price_index+1])
        mpstd=np.std(mp_list[current_price_index-p:current_price_index+1])

        self.cpstd.append(cpstd)
        self.mpstd.append(mpstd)
        cpstd=np.mean(self.cpstd[-10:])
        mpstd=np.mean(self.mpstd[-10:])
        speed_c = np.diff(self.close_price[current_price_index-p:current_price_index+1])
        speed_m = np.diff(mp_list[current_price_index-p:current_price_index+1])
        f=lambda x:abs(x)
        #speed_c = map(f,speed_c)
        #speed_m = map(f,speed_m)
        print "speed {} {}".format(np.std(speed_c),np.std(speed_m))


###
        def calc_t2(mp1):
            t2=[]
            for i in range(1,len(mp1)):
                t2.append(np.log(mp1[i]/mp1[i-1]))
            return np.std(t2)

        p=15
        csstd=calc_t2(self.close_price[current_price_index-p:current_price_index+1])
        msstd=calc_t2(mp_list[current_price_index-p:current_price_index+1])
##

        dd = self.close_price[current_price_index-29:current_price_index+1]
        dd2 = mp_list[current_price_index-29:current_price_index+1]


        c=c1=0
        min_index=argrelextrema(np.array(dd),np.less)[0]
        for j in min_index:
            c+=1
            if dd[j]<dd2[j] :#or dd[j+1]<mpd[j+1] :#or dd[j-1]<mpd[j-1]:
                c1+=1
        if c!=0:
            self.x.append(c1/float(c))
        else:
            self.x.append(0)

        print "x {}".format(self.x[-10:])
        print np.mean(self.x)

        
##
        if self.buy_price==0:#and len(self.mpup)>5 and self.mpup[-1]>self.mpup[-5]:
        #    if calc_down(mp_list[current_price_index-20:current_price_index+1])<1 and calc_up(mp_list[current_price_index-20:current_price_index+1])>0.5:#0.7<p_c/n_c<1.3 :#and ks_p_value>0.5:#or (np.mean(diffmp[-5:])>0 and stab_de!=1):#self.pimf1[-1][1]>0.4:#self.cu[-1]>0.6 or (len(self.cu)>1 and self.cu[-1]>self.cu[-2]):
            if  self.x[-1]>=np.mean(self.x):#-np.std(self.x):#(np.std(speed_c)-np.std(speed_m))/np.std(speed_c)>0.1  :#self.sel==2 and self.fail_flag==3:#cpstd-mpstd/cpstd>0.00 :#and not (sharp(s21)<0 and sharp(s22)<0):#len(ma3s1)>1 and ma3s1[-1]>ma3s1[-2]:#np.mean(s21)/np.std(s21)>np.mean(s22)/np.std(s22) and np.mean(s21)/np.std(s21)>0 and np.mean(s21)/np.std(s21)>np.mean(s11)/np.std(s11):#sum(s1)>sum(s2) and sum(s1)>0:#deltama10<0 :#and  s2[-1]>0:#self.mpdown[-1]<mampdown[-1] :#and lp2!=1:
                sel=1
             #   self.fail_flag=0
            #elif p_c/n_c>1.3:
            #elif calc_up(mp_list[current_price_index-20:current_price_index+1])>1 :
            #elif self.sel==1 and self.fail_flag==3:# not (sharp(s21)<0 and sharp(s22)<0):# len(ma3s2)>1 and ma3s2[-1]>ma3s2[-2]:#if np.mean(s21)/np.std(s21)<np.mean(s22)/np.std(s22) and np.mean(s22)/np.std(s22)>0 and np.mean(s22)/np.std(s22)>np.mean(s12)/np.std(s12): #sum(s1)<sum(s2) and sum(s2)>0:
            #    sel=2
              #  self.fail_flag=0
            else:
                sel=2
            
            self.sel=sel
        else:
            sel=self.sel

##
        self.diffmpcha.append(np.mean(diffmp[-5:])-np.mean(diffmp[-10:]))
        mean_de=0
        if self.buy_price==0 and np.mean(diffmp[-3:])<0:#>np.mean(diffmp[-10:]) :#and np.mean(diffmp[-5:])>np.mean(diffmp[-6:-1]):
            mean_de=1
        elif np.mean(diffmp[-3:])>0 and self.buy_price!=0:
            mean_de=2
        #sel=0
        #if self.buy_price==0:
        #    #if c2/float(c1)>c22/float(c11):
        #    if s1>s2:#len(self.radio1)>1 and len(self.radio2)>1 and self.radio1[-1]>self.radio2[-1] and self.radio1[-1]>=self.radio1[-2]:
        #        sel=1
        #    elif s2>s1:#len(self.radio2)>1 and len(self.radio1)>1 and self.radio1[-1]<self.radio2[-1] and self.radio2[-1]>=self.radio2[-2]:
        #        sel=2
        #    self.sel=sel
        #else:
        #    sel = self.sel
###
        r_diffmp=diffmp[-5:]
        zero_count=0
        zero_index=[]
        for i in range(1,len(r_diffmp)):
            if r_diffmp[i-1]*r_diffmp[i]<0:
                zero_count+=1
                zero_index.append(i)


        print "zero {} ".format(zero_count)
        print "zero {} ".format(zero_index)
        print "mean diffmp {}".format(np.mean(r_diffmp))
      
        max_diff_index = argrelextrema(np.array(r_diffmp),np.greater)[0]
        min_diff_index = argrelextrema(np.array(r_diffmp),np.less)[0]
        print "diff max {}".format(max_diff_index)
        print "diff min {}".format(min_diff_index)
        print len(list(max_diff_index)+list(min_diff_index))


###

        mp_list=[self.je[i]/self.vol[i] for i in range(len(self.close_price))]
        mp = self.je[current_price_index]/self.vol[current_price_index]
        mp1 = self.je[current_price_index-1]/self.vol[current_price_index-1]
        mp2 = self.je[current_price_index-2]/self.vol[current_price_index-2]
        print "mp {}".format(mp)
        print "cp {}".format(self.close_price[current_price_index])
        print "mp down pro {}".format(calc_down(mp_list[current_price_index-29:current_price_index+1]))
        print "mp up pro {}".format(calc_up(mp_list[current_price_index-29:current_price_index+1]))
##
        maexp=exp_smooth(self.close_price,0.2)
        ma5=ma_func2(self.close_price,5) 
        p,m,s=montecarlo_simulate(self.close_price[current_price_index-20:current_price_index+1],10)
        p,m,s=montecarlo_simulate(self.close_price[current_price_index-20:current_price_index+1],1000)
        self.mclist.append(p)
        if len(self.mclist3)>0:
            self.mclist2.append(self.mclist3[-1]-self.close_price[current_price_index])
        self.mclist3.append(m)
        print "monte {}".format(self.mclist[-10:])
        print "cp monte {} {}".format(m,s)
        print "cp  {}".format(self.close_price[current_price_index-10:current_price_index+1])
        print "cp monte {}".format(self.mclist3[-10:])

        vp,vm,vs=montecarlo_simulate(self.vol[current_price_index-20:current_price_index+1],10)
        jp,jm,js=montecarlo_simulate(self.je[current_price_index-20:current_price_index+1],10)
        mpm,mps=montecarlo_simulate2(mp_list[current_price_index-33:current_price_index-10],10,10)
        print "monte predict step {} {}".format(mpm,mps)
        print "mp {}".format(mp_list[current_price_index])
        mc_de=1
        if mpm-mps>mp_list[current_price_index]:
            mc_de=0
        p,m,s=montecarlo_simulate(ma5[current_price_index-20:current_price_index+1],10)
        if len(self.mclist4)>0:
            self.mclist5.append(self.mclist4[-1]-ma5[current_price_index])
        #self.mclist4.append(jm/vm)
        self.mclist4.append(m)
        print "monte carlo mp {}".format(self.mclist4[-5:])

##

        ref = np.array(diffmp[:-10])
        sample = np.array(diffmp[-10:])
        ks_d,ks_p_value = stats.ks_2samp(ref,sample)
        print "10ks {}".format(ks_p_value)

##
        cr=self.close_price[current_price_index-10:current_price_index+1]
        print "recent down {}".format(calc_down(cr))
        self.diffmpstd.append(calc_down(cr))

##
        print "mean stf {} {}".format(np.mean(diffmp[-10:]),np.std(diffmp[-10:]))

        print "mpdiff continue down {}".format(calc_down(mpdiff_list[current_price_index-20:current_price_index+1]))
##

        def calc_recentup_pro(datac,dataj,datav,n):
            mp = [dataj[i]/datav[i] for i in range(len(datav))]
            c=0
            c1=0
            for i in range(1,len(datac)-n):
                if datac[i]<mp[i] and datac[i-1]>mp[i-1]:
                    c+=1
                    if datac[i+n]<datac[i]:
                        c1+=1
            if c!=0:
                print "recent up pro{}".format(float(c1)/c)
                return float(c1)/c
            else:
                print "recent up pro{}".format(0)
                return 0
        rdde=0
        p=20
        datac = self.close_price[current_price_index-p:current_price_index+1]
        datav = self.vol[current_price_index-p:current_price_index+1]
        dataj = self.je[current_price_index-p:current_price_index+1]
        if calc_recentup_pro(datac,dataj,datav,1)>0.5 or calc_recentup_pro(datac,dataj,datav,3)>0.5 or calc_recentup_pro(datac,dataj,datav,3)>0.5:
            rdde=1

###
        g = mixture.GMM(n_components=2)
        obs=self.close_price[current_price_index-4:current_price_index+1]
        #obs=mp_list[current_price_index-4:current_price_index+1]
        ma = ma_func2(self.close_price,5)
        obs1=ma[current_price_index-2:current_price_index+1]
        #obs.append(self.mclist3[-1])

        print obs
        g.fit(obs)
        print np.round(g.means_, 2)
        m1 = np.round(g.means_, 2)[0]
        m2 = np.round(g.means_, 2)[1]
        if abs(obs[-1]-m1)>abs(obs[-1]-m2):
            close=m2
        else:
            close=m1

        gmmde=1
        if obs[-1]>close:# and self.fail_flag==0) or (self.fail_flag==1 and r[-1]!=r[-2] and obs[-1]>close):
            gmmde=0



        g.fit(obs1)
        print np.round(g.means_, 2)
        m1 = np.round(g.means_, 2)[0]
        m2 = np.round(g.means_, 2)[1]
        r=[]
        for i in obs1:
            r.append(g.predict([i])[0])

        a=np.array(r)*np.array(obs1)
        a1=filter(lambda x:x>0,a)
        std1=np.std(a1)
        a=(np.array(r)-1)*np.array(obs1)
        a0=filter(lambda x:x<0,a)
        a0=[abs(i) for i in a0]
        std0=np.std(a0)

        print r
        print obs1
        print "ma obs"
        if abs(obs1[-1]-m1)>abs(obs1[-1]-m2):
            close=m2
            std=std1
        else:
            close=m1
            std=std0

        gmmde2=0
        if r[-1]==r[-2]:# and self.fail_flag==0) or (self.fail_flag==1 and r[-1]!=r[-2] and obs[-1]>close):
            gmmde2=1

##3
        jede=0
        if sel==1:
            if  diffmp[-1]<0 and  mean_de==1 and gmmde==1 and   len(self.mclist2)>0 and self.mclist2[-1]>0 :#and (( self.fail_flag==0) or (self.fail_flag==1 and gmmde2==1)):
                jede=1
            if (diffmp[-1]>0 and  mean_de==2 ):#and deltama10>0) or (deltama10<0 and diffmp[-1]>0):
                jede=2
        elif 0:#sel==2:
            if ma3_mp_diff[-1]<0:#np.mean(ma3_mp_diff[-10:]) :#and np.mean(ma3_mp_diff[-3:])<0:#self.close_price[current_price_index]<mp and mean_de==1 :#and  len(self.mclist2)>0 and self.mclist2[-1]>0 :#and self.vol[current_price_index]>np.mean(self.vol[current_price_index-4:current_price_index+1]):#>np.mean(self.vol[current_price_index-9:current_price_index+1]):#and len(self.mclist4)>2 and (self.mclist4[-2]-mp)>(self.mclist4[-3]-mp1):#and self.:#len(self.mclist2)>0 and self.mclist2[-1]>0 :#and  self.mclist5[-1]>0:#self.mclist5[-1]>0:#and self.close_price[current_price_index-1]<mp1:
                jede=1
            if ma3_mp_diff[-1]>0:#np.mean(ma3_mp_diff[-10:]) :#and np.mean(ma3_mp_diff[-3:])>0:#self.close_price[current_price_index]>mp and  mean_de==2:
                jede=2
        elif 0:#sel==2 :#and not (self.vol[current_price_index]<self.vol[current_price_index-1] and self.close_price[current_price_index]<self.open_price[current_price_index]):#self.sel%2==1:#s1<s2:#np.std(rp1)/np.mean(rp1)>np.std(rp)/np.mean(rp):
            if diffmp[-2]<0 and diffmp[-1]>0 :#and len(self.mclist2)>0 and self.mclist2[-1]<0:#self.close_price[current_price_index]>mp and mean_de==1 and self.vol[current_price_index]>np.mean(self.vol[current_price_index-4:current_price_index+1]):#>np.mean(self.vol[current_price_index-9:current_price_index+1]):#and len(self.mclist4)>2 and (self.mclist4[-2]-mp)>(self.mclist4[-3]-mp1):#and self.:#len(self.mclist2)>0 and self.mclist2[-1]>0 :#and  self.mclist5[-1]>0:#self.mclist5[-1]>0:#and self.close_price[current_price_index-1]<mp1:
            #if np.mean(self.vol[current_price_index-4:current_price_index+1])>np.mean(self.vol[current_price_index-9:current_price_index+1]) and np.mean(self.vol[current_price_index-5:current_price_index])<np.mean(self.vol[current_price_index-10:current_price_index]):
                jede=1
            #if np.mean(self.vol[current_price_index-4:current_price_index+1])<np.mean(self.vol[current_price_index-9:current_price_index+1]):
            if diffmp[-1]<0:#self.close_price[current_price_index]>mp :#and  mean_de==2:
                jede=2
     #   elif sel==2:
     #       if self.close_price[current_price_index]>mp and self.close_price[current_price_index-1]<mp1 :#and np.mean(diffmp[-10:])>0:
     #           jede=1
     #       if self.close_price[current_price_index]<mp:
     #           jede=2

        mp5=np.mean(self.je[current_price_index-4:current_price_index+1])/np.mean(self.vol[current_price_index-4:current_price_index+1])
        mp5_b=np.mean(self.je[current_price_index-5:current_price_index])/np.mean(self.vol[current_price_index-5:current_price_index])
        jede1=0 
        if self.close_price[current_price_index]-mp5>self.close_price[current_price_index-1]-mp5_b:
            jede1=1
        if self.close_price[current_price_index-1]-mp5_b>self.close_price[current_price_index]-mp5:
            jede1=2
        #jede=0
        #if lr==1:
        #    jede=1
        #else:
        #    jede=2
        print "vol {}".format(self.vol[current_price_index-10:current_price_index+1])
        print "je {}".format(self.je[current_price_index-10:current_price_index+1])

        jebollde=self.normal_boll_calc(diffmp[-1],diffmp[-20:])

        m_de=0

        if diffmp[-1]<calc_recent_rilling(diffmp)[-1]:
            m_de=1
        else:
            m_de=2

        
        ma=exp_smooth(self.close_price[:current_price_index+1],0.3)
        exp_ma_de=0
        if ma[-1]>ma[-2]:
            exp_ma_de=1
        print "emddata {}".format(emd_data)
        #print "power {}".format(imf_power(emd_data,imf_list,1))
        print kkkkkkk
        self.std10.append(np.std(self.close_price[current_price_index-9:current_price_index+1]))
        print "c {} kal {}".format(self.close_price[current_price_index],kalval)

        


        hlc=(self.high_price[current_price_index]+self.low_price[current_price_index])/2-self.close_price[current_price_index]
        hlc_de=0
        if hlc<0:
            hlc_de=1

        vv=ma_func2(self.vol[current_price_index-10:current_price_index+1],10)
        #vv=(self.vol[current_price_index-10:current_price_index+1])
        jj=ma_func2(self.je[current_price_index-10:current_price_index+1],10)
        #jj=(self.je[current_price_index-10:current_price_index+1])
        vjde=0
        if jj[-1]<jj[-2] and vv[-1]>vv[-2]:
            vjde=1
        elif vv[-1]<vv[-2]:
            if vv[-2]/vv[-1]>jj[-2]/jj[-1]:
                vjde=0
        print "vjde {}".format(vjde)


        ref = np.array(diffmp[:-10])
        mean=np.mean(ref)
        std=np.std(ref)
        ks_p=stats.kstest(diffmp[-10:],'norm',args=(mean,std))
        print "10ks {}".format(ks_p)
        print "stable test {}".format(ts.adfuller(diffmp[-5:],1))
        print "stable test {}".format(ts.adfuller(diffmp[-10:],1))
        print "stable test {}".format(ts.adfuller(diffmp[-15:],1))
        print "stable test {}".format(ts.adfuller(diffmp[-20:],1))
        print "stable test {}".format(ts.adfuller(diffmp[-100:],1))
        print "mean mp {}".format(np.mean(diffmp[-5:]))
        print diffmp[-10:]
        stab=ts.adfuller(diffmp[-10:],1)
        kk=stab[0]
        kkk=stab[4]['5%']

        print "stab de {}".format(stab_de)
        print "lp1 {} lp2 {}".format(lp1,lp2)
        print "sel {}".format(sel)
        print "now date {}".format(now_date.weekday())

        if len(self.rihb)>0:
            self.rihb2.append(self.rihb[-1])
        else:
            self.rihb2.append(0)

        self.scount+=1
        self.ct[self.scount]=self.total_asset
        #print self.ct

        if len(self.ct)>30:
            nt=self.ct[self.scount]
            bt=self.ct[self.scount-30]
            if (bt-nt)/float(bt)>0.1:
                self.td=30
        if self.td>0: 
            self.td-=1


        g = mixture.GMM(n_components=2)
        obs=self.close_price[current_price_index-4:current_price_index+1]
        obs1=mp_list[current_price_index-9:current_price_index+1]
        #obs.append(self.mclist3[-1])
        
        print obs
        g.fit(obs)
        print np.round(g.means_, 2)
        m1 = np.round(g.means_, 2)[0]
        m2 = np.round(g.means_, 2)[1]
        if abs(obs[-1]-m1)>abs(obs[-1]-m2):
            close=m2
        else:
            close=m1

        r=[]
        for i in obs:
            r.append(g.predict([i])[0])
        
        print r
        for i in range(1,len(r)-1):
            if r[-i-1]==r[-i]:
                pass
            else:
               break
         
        rr = obs[-i:]
        print rr
        ref=[i for i in range(len(rr))]
        print np.corrcoef(rr,ref)[0][1]
        qushide=0
        if np.corrcoef(rr,ref)[0][1]>-0.5:
            qushide=1
        gmmde=1
        if obs[-1]>close:# and self.fail_flag==0) or (self.fail_flag==1 and r[-1]!=r[-2] and obs[-1]>close):
            gmmde=0
        #if r[-1]!=r[-2]:
        #    gmmde=2
        print obs1
        g = mixture.GMM(n_components=2)
        g.fit(obs1)
        print np.round(g.means_, 2)
        m1 = np.round(g.means_, 2)[0]
        m2 = np.round(g.means_, 2)[1]
        r=[]
        for i in obs1:
            r.append(g.predict([i])[0])

        print r

        
        for i in range(1,len(r)-1):
            if r[-i]!=r[-i-1]:
                break

        count = i

        j=self.je[current_price_index]
        v=self.vol[current_price_index]
        c=self.close_price[current_price_index]
        mp=mp_list[current_price_index]
        mp1=mp_list[current_price_index-1]
        o=self.open_price[current_price_index]
        h=self.high_price[current_price_index]
        l=self.low_price[current_price_index]
        c1=self.close_price[current_price_index-1]
        f=abs(o-c)
        f=j

        c=mp
        c1=mp1
            
        if c!=c1:
            self.ytx.append(f*c/(abs(c-c1)))
        else:
            self.ytx.append(0)
    
        print "ytx {}".format(self.ytx[-10:])
        if len(self.ytx)>1:
            print "l {}".format(self.ytx[-1]/self.ytx[-2])
            self.ytx2.append(self.ytx[-1]/self.ytx[-2])
        ytx_de=0
        if len(self.ytx2)>0 and np.mean(self.ytx2[-3:])<1:
            ytx_de=1

        print "lllll {}".format(self.ytx2[-10:])
###
        if 1:#count>2:
            data=mp_list[current_price_index-100:current_price_index+1]
            upper, middle, lower = talib.BBANDS(np.asarray(data), timeperiod=20, nbdevup=1, nbdevdn=1, matype=0)  
            print "boll "
            print upper[-1]
            print lower[-1]
            print middle[-1]
            self.low.append(lower[-1])


            yeah_de=1
            if data[-1]<lower[-1] or data[-1]>upper[-1]:
                print "yeah"
                yeah_de=0


        print "hurst %s"%(hurst(self.close_price[current_price_index-10:current_price_index+1]))
        self.hursti.append(hurst(self.close_price[current_price_index-100:current_price_index+1]))
        demo=[1,2,3,3,2,1,1,2,3,3,2,1]
        print hurst(demo)
        hurst_de=0
        if self.hursti[-1]:
            hhhh = self.hursti[-1]
        else:
            hhhh = 1
        
        if (len(self.hursti)>1 and self.hursti[-1]>self.hursti[-2] and deltama10>0) or (len(self.hursti)>1 and self.hursti[-1]<self.hursti[-2]):#hhhh<0.5 or (hhhh>0.5 and deltama10>0):
            hurst_de=1

        high=np.array(self.high_price[current_price_index-100:current_price_index+1])
        low=np.array(self.low_price[current_price_index-100:current_price_index+1])
        close=np.array(self.close_price[current_price_index-100:current_price_index+1])
        vol=np.array(self.vol[current_price_index-100:current_price_index+1])
        ADX = talib.ADX(high,low,close,14)
        PLUS_DI = talib.PLUS_DI(high,low,close,10)
        MINUS_DI = talib.MINUS_DI(high,low,close,10)
        adx_de=0

        #self.adx.append(PLUS_DI[-1])
        #self.adx2.append(MINUS_DI[-1])
        if  PLUS_DI[-1]>MINUS_DI[-1]:
            adx_de=1
        MFI = talib.MFI(high, low, close, vol, timeperiod=30)
        self.adx.append(MFI[-1])

        print "std cp {}".format(np.std(self.close_price[current_price_index-4:current_price_index+1]))
        print "std mp {}".format(np.std(mp_list[current_price_index-4:current_price_index+1]))
        print "sel {}".format(sel)
        print "gmmde2  {}".format(gmmde2)
        self.cpmpstd.append((np.std(self.close_price[current_price_index-4:current_price_index+1])-np.std(mp_list[current_price_index-4:current_price_index+1]))/(np.std(self.close_price[current_price_index-4:current_price_index+1])))

        diff = list(np.diff(self.close_price))
        diff.insert(0,0)
        t1 = diff[current_price_index-4:current_price_index+1]
        t2 = diff[current_price_index-5:current_price_index]
        t3 = diff[current_price_index-11:current_price_index-1]
        print "t test"
        print stats.ttest_ind(t1, t2)
        print stats.ttest_ind(t1, t3)
        print sum(t1)
        print sum(t2)
        print sum(t3)
        tde=0
        if sum(t2)>0 and sum(t1)<0 :#and sum(t3)>0:
            tde=1
        t4 = diff[current_price_index-29:current_price_index+1]
        t5 = diff[current_price_index-9:current_price_index+1]
        print "hah"
        print np.mean(t5)
        print np.mean(t4)
        self.tde.append(np.mean(t5))#-np.mean(t4))
        tde2=1
        if len(self.tde)>10:
            upper, middle, lower = talib.BBANDS(np.asarray(self.tde), timeperiod=10, nbdevup=1, nbdevdn=1, matype=0)
            if self.tde[-1]<lower[-1]:
                tde2=0
        #print self.cpmpstd


        cdiff = list(np.diff(self.close_price))
        cdiff.insert(0,0)

        dddde=0
        if np.mean(cdiff[current_price_index-9:current_price_index+1])>np.mean(cdiff[current_price_index-99:current_price_index+1]):
            dddde=1

        def pro_forward(datac,mp):
        
            c=0
            c1=0
            for i in range(1,len(datac)):
                c+=1
                if datac[i]<datac[i-1]:
                    if datac[i-1]-mp[i-1]>datac[i]-mp[i]:
                        c1+=1
                else:
                    if datac[i-1]-mp[i-1]<datac[i]-mp[i]:
                        c1+=1
        
        
            print c1/float(c)
            return c1/float(c)
        pro=pro_forward(self.close_price[current_price_index-4:current_price_index+1],mp_list[current_price_index-4:current_price_index+1])
        print "for pro  {}".format(pro)
        forprode=0
        if pro>0.5:
            forprode=1
###
        #if  self.buy_price!=0  and ((len(self.boll_de)>1 and self.boll_de[-2]==2 and self.boll_de[-1]!=2) or ((imf_flag==2) and deltama3_before<0 and deltama3<0))  :# ((current_price-self.buy_price)/self.buy_price)>0.01:#((deltamacd10<0 and deltamacd10_b>0)  ) and ((current_price-self.buy_price)/self.buy_price)>0.01:#(current_price<ma5 )and distance_decision2==1 and ((current_price-self.buy_price)/self.buy_price)>0.01:
        #if  self.buy_price!=0  and ((boll_de==2 and deltama3<deltama3_before) or (boll_de!=2 and len(self.boll_de)>1 and self.boll_de[-2]==2 ) or ((imf_flag==2) and deltama3_before<0 and deltama3<0))  :# ((current_price-self.buy_price)/self.buy_price)>0.01:#((deltamacd10<0 and deltamacd10_b>0)  ) and ((current_price-self.buy_price)/self.buy_price)>0.01:#(current_price<ma5 )and distance_decision2==1 and ((current_price-self.buy_price)/self.buy_price)>0.01:
        if  self.buy_price!=0 and  ((jede==2) ):# or (tde==1 and deltama5<0)):# or (self.mclist4[-2]>mp)):#and mean_de==2:#and imf_flag==2):#and sha_de==2:#and self.close_price[current_price_index-1]>self.close_price[current_price_index]:#emd_boll_de==2:#ma_flag==2:#bayes_de==2:#emd_data[-1]<emd_data[-2]:#(power_de_ma_list1==2) and imf_flag==2:# or boll_de==2):#and self.close_price[current_price_index-1]>self.close_price[current_price_index]:#  self.close_price[current_price_index-1]>self.close_price[current_price_index] :#power_de_ma_list==2:#emd_data2[-1]>emd_data2[-2] and  emd_data[-1]<emd_data[-2]:#me[-1]<me[-2] and me[-2]>me[-3]:#boll_de==2 and self.boll_de[-2]!=2:#  me[-1]<me[-2] and me[-2]>me[-3]:#power_de_ma_list==2:#((judge_extrem(emd_data)<=0 and judge_extrem(emd_data[:-1])>0) or (judge_extrem(emd_data)<0)):#self.de2==1 and emd_data[-1]<emd_data[-2]:#((judge_extrem(emd_data[:-1])>0 and judge_extrem(emd_data)<=0) ):#imf_flag==2:# (  (len(self.boll_de)>1 and self.boll_de[-2]==2 and self.boll_de[-1]!=2) or (len(self.boll_de)>1 and self.boll_de[-2]!=1 and self.boll_de[-1]==1)):#( self.pre_cha_de[-1]==1 or imf_flag==2) :#and (boll_de==2 or self.buy_day>10):#(boll_de==2 or (deltama5<0 and deltama5_before<0)):#emd_data[-1]<emd_data[-2]<emd_data[-3]  and emd_data[-3]>emd_data[-4] :# (r (  emd_data[-3]<emd_data[-2] and emd_data[-1]<emd_data[-2] and emd_data[-1]>0)):#imf_flag==2:#boll_de==2:# self.close_flag[-2]==2 and imf_flag!=2:# and self.boll_de[-1]!=2 and len(self.boll_de)>1 and self.boll_de[-2]==2:#boll_de==2:#((boll_de!=2 and self.boll_de[-2]==2) or ((imf_flag==2) and deltama3_before<0 and deltama3<0))  :# ((current_price-self.buy_price)/self.buy_price)>0.01:#((deltamacd10<0 and deltamacd10_b>0)  ) and ((current_price-self.buy_price)/self.buy_price)>0.01:#(current_price<ma5 )and distance_decision2==1 and ((current_price-self.buy_price)/self.buy_price)>0.01:
        #if  self.buy_price!=0  and deltama5<0:
        #if self.buy_price!=0  and self.macd[current_price_index]<self.macd[current_price_index-1]<self.macd[current_price_index-2] and distance_decision2==1:
        #if self.buy_price!=0 and deltama3_before>0 and deltama3<0 and distance_decision2==1 :#and current_price-self.buy_price>0:#emd_data[-1]<emd_data[-2] and emd_data[-2]>emd_data[-3] :#mama3<mama3_b and mama3_b>mama3_b2 and current_price-self.buy_price>0 and self.buy_price!=0:
 
            if 1:#self.fail_flag==0:
                print "sell sell 1"
                self.money = self.sell(self.share,trade_price)
                self.share = 0
                self.de2=0
                self.total_asset = self.money - self.sell_fee(self.money)
                self.money = self.total_asset
                self.sell_x_index.append(current_price_index)
                self.sell_y_index.append(current_price)
                if self.buy_price>=trade_price:
                #if ((((self.buy_price-trade_price)/self.buy_price)>0.05 ) ):
                #    self.fail_flag=0
                    self.last_fail_de =1#+=flag_fail
                else:
                    self.fail_flag=0
                self.rihb.append(((trade_price-self.buy_price)/(self.each_day*self.buy_price)))
                self.each_day=0
                self.buy_price = 0
                self.buy_mean_price = 0
                self.one_time_day=0
                self.sell_flag=0
                self.sell_flag2=0
                ABL=0
            else:
                if self.buy_price>=trade_price:
                    self.fail_flag=1
                else:
                    self.fail_flag=0
                self.buy_price=0
                print "sellsell 1"
                
        elif   0:#self.buy_price!=0 and ma3<boll_low1 and  distance_decision2==1:#((((self.buy_mean_price-current_mean_price)/self.buy_mean_price)>0.01 ) )  and self.share > 0 and np.mean(self.snr[-3:])<np.mean(self.snr[-4:-1]): #and (self.buy_price-current_price)<0 and self.share > 0:
            print "sell sell 2"
            if 1:#self.fail_flag==0:
                if self.buy_price>trade_price:
                    self.fail_flag=flag_fail
                else:
                    self.fail_flag=0
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
            else:
                if self.buy_price>=trade_price:
                    self.fail_flag=1
                else:
                    self.fail_flag=0
                print "sellsell 1"
                self.buy_price=0
        elif   self.buy_price!=0 and ((self.buy_price-self.close_price[current_price_index])/self.buy_price>0.05 ):# or ((self.buy_price-self.close_price[current_price_index])/self.buy_price>0.03 and tde==1)):#and self.share > 0 :
            print "sell sell 4"
            if 1:#self.fail_flag==0:
                if self.buy_price>=trade_price:
                    self.fail_flag=1
                #else:
                #    self.fail_flag=0
                self.money = self.sell(self.share,trade_price)
                self.rihb.append(((trade_price-self.buy_price)/(self.each_day*self.buy_price)))
                self.each_day=0
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
                ABL=0
            else:
                if self.buy_price>=trade_price:
                    self.fail_flag=1
                else:
                    self.fail_flag=0
                print "sellsell 1"
                self.buy_price=0
             
        elif  0:# self.buy_price!=0 and self.one_time_day>10 and  imf_flag==2 and self.share > 0 and np.mean(self.snr[-3:])<np.mean(self.snr[-4:-1]):
            print "sell sell 3"
            if self.buy_price>=trade_price:
                self.fail_flag=1
            else:
                self.fail_flag=0
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
        #elif    imf_flag==1  and self.buy_price==0 and snr_decision==1 and distance_decision==1 and deltama5_before<deltama5 :#and ma_decision==1 :#and distance_decision==1:#deltama3_before<deltama3:# (deltama3_before>deltama3 or deltama3<0):
        #elif   (  com_de==1 and  std_de==1 and  self.f_d==1 and kdj_de==1 and boll_de!=1 and distance_decision==1 and self.fail_flag==0 and (ma_decision==0 )) or (com_de==1 and std_de==1 and  imf_flag==1 and boll_de==1 and ma_decision==1  and distance_decision==1 and self.buy_price==0):#and deltamacdma5_before<deltamacdma5:#:and deltama30>0:#emd_de==1 and (deltama3>0) and deltama10>0:# and deltama30<0 and deltama10>0) :#deltama3_before :#and mamama3>mamama3_b :#and mama3_b2<mama3_b3:#and self.macd[current_price_index]>self.macd[current_price_index-1] :#and self.macd[current_price_index-3]>self.macd[current_price_index-2] :#and deltamacdma5_before<deltamacdma5 and deltamacdma5_before<deltamacdma5_before2:#and pvpower_de==1 :#emd_data[-1]>emd_data[-2] and emd_data[-2]<emd_data[-3]:#self.close_price[current_price_index]>self.close_price[current_price_index-1]:#and (deltama3_before<deltama3 and deltama3_before<0 and deltama3>0) :#and ma_ex_de==1:#and distance_decision==1 and deltama3_before<deltama3 and deltama3_before<deltama3_before2 :#and deltama3_before2<deltama3_before3 :#and ma_ex_de==1:# and distance_decision==1 :# and self.close_price[current_price_index]>self.close_price[current_price_index-1]:#deltama3_before<deltama3:#self.macd[current_price_index-1]<self.macd[current_price_index] :#and (self.macd[current_price_index-2]<0 or self.macd[current_price_index-1]<0 or self.macd[current_price_index]<0):#and deltama5>=0 :#and deltama3_before<deltama3 ) or (deltama3>0 and deltama3_before<0)):# and  mama3>mama3_b and mama3_b<mama3_b2 and distance_decision==1:#and deltama5<0:#and self.close_price[current_price_index]>self.close_price[current_price_index-1] and deltama5_before<deltama5<0: #and deltama3_before<0 and deltama3>0 and deltama5_before<deltama5<0:#and ma_decision==1 :#and distance_decision==1:#deltama3_before<deltama3:# (deltama3_before>deltama3 or deltama3<0):
        #elif  self.buy_price==0 and self.boll_de[-1]!=1 and len(self.boll_de)>1 and self.boll_de[-2]==1:#and imf_flag==1:#and deltama3<deltama3_before:#and imf_flag==1:#deltama3>0 and deltama3_before<0:#flag==1 and (stats.normaltest(testdata)[1])>0.5:#and (sum(self.boll_de[-6:-1])==0 ):
        #elif     distance_decision==1 and len(self.boll_de)>1 and self.boll_de[-2]==1 and self.boll_de[-1]!=1 and (  (imf_flag==1 ) ) and  self.buy_price==0 :#and deltama3>deltama3_before :#and deltama3>deltama3_before:#and pvpower_de==1 :#emd_data[-1]>emd_data[-2] and emd_data[-2]<emd_data[-3]:#self.close_price[current_price_index]>self.close_price[current_price_index-1]:#and (deltama3_before<deltama3 and deltama3_before<0 and deltama3>0) :#and ma_ex_de==1:#and distance_decision==1 and deltama3_before<deltama3 and deltama3_before<deltama3_before2 :#and deltama3_before2<deltama3_before3 :#and ma_ex_de==1:# and distance_decision==1 :# and self.close_price[current_price_index]>self.close_price[current_price_index-1]:#deltama3_before<deltama3:#self.macd[current_price_index-1]<self.macd[current_price_index] :#and (self.macd[current_price_index-2]<0 or self.macd[current_price_index-1]<0 or self.macd[current_price_index]<0):#and deltama5>=0 :#and deltama3_before<deltama3 ) or (deltama3>0 and deltama3_before<0)):# and  mama3>mama3_b and mama3_b<mama3_b2 and distance_decision==1:#and deltama5<0:#and self.close_price[current_price_index]>self.close_price[current_price_index-1] and deltama5_before<deltama5<0: #and deltama3_before<0 and deltama3>0 and deltama5_before<deltama5<0:#and ma_decision==1 :#and distance_decision==1:#deltama3_before<deltama3:# (deltama3_before>deltama3 or deltama3<0):
        elif  self.buy_price==0   and   jede==1 :#and ((self.fail_flag==1 and dddde==1) or self.fail_flag==0):#dddde==1:#and (deltama5>0 or (deltama5<0 and tde==0))  :#and tde2==1:#and ytx_de==1:#and yeah_de==1:#and qushide==1:#and sharp_de==1:#and self.td==0:#and hurst_de==1:#and ((self.fail_flag==1 and mp>mp2) or self.fail_flag==0):#and mp>mp1:#ks_de==1 :#and hurst_de==1 :#and self.pimf1[-1][1]>self.pimf1[-2][1]:#and (self.fail_flag==0 or (deltama5>0 and self.fail_flag==1)) and ks_de==1 :#and pro!=0:#and imf_flag==1:#and deltama5>0:#and self.close_price[current_price_index]>self.close_price[current_price_index-1]:#and imf_list[1][-1]<0:#and emd_boll_de==1 :#and emd_data[-1]>emd_data[-2]:#and deltama10>0:#and len(self.close_flag)>1 and self.close_flag[-2]!=2:#power_de_ma_list1==1 and imf_flag==1:#and self.close_price[current_price_index]<self.close_price[current_price_index-1]:#and (boll_de==1 or (len(self.boll_de)>1 and self.boll_de[-2]==1 and self.boll_de[-1]==0)) :#and distance_decision==1 and self.close_price[current_price_index-1]<self.close_price[current_price_index]:#and ma_list[current_price_index-1]<ma_list[current_price_index]:#    ((emd_data2[-1]<emd_data2[-2] and  emd_data[-1]>emd_data[-2]) or (emd_data2[-2]<emd_data2[-3] and emd_data2[-1]>emd_data2[-2] and emd_data[-2]<emd_data[-3] and emd_data[-1]>emd_data[-2])):#<me[-2] and me[-2]>me[-3]:#boll_de==2 and len(self.boll_de)>1 and self.boll_de[-2]!=2:#me[-1]<me[-2] and me[-2]>me[-3]:#me[-1]<0 and me[-2]>0:#ma_list[current_price_index]>ma_list[current_price_index-1] and  power_de_ma_list==1 :#and judge_extrem(emd_data)<=0:#judge_extrem(emd_data)>=0 and judge_extrem(emd_data[:-1])<0 :#and  len(self.stable_test)>1 and self.stable_test[-1]<self.stable_test[-2] and judge_extrem(emd_data)>=0 and judge_extrem(emd_data[:-1])<0:#emd_data[-1]>emd_data[-2] and self.de==1 :#and (not ( len(self.snr)>1 and self.snr[-1]>self.snr[-2] and deltama3<0 and deltama3_before<0)):#imf_flag==1:# boll_de==0 and len(self.boll_de)>1 and self.boll_de[-2]==1 :#and min_index[-1]>max_index[-1]  :#and   distance_decision==1:#and emd_data[-1]>emd_data[-2]:#boll_de==1:#and deltama3>deltama3_before :#and deltama3>deltama3_before:#and pvpower_de==1 :#emd_data[-1]>emd_data[-2] and emd_data[-2]<emd_data[-3]:#self.close_price[current_price_index]>self.close_price[current_price_index-1]:#and (deltama3_before<deltama3 and deltama3_before<0 and deltama3>0) :#and ma_ex_de==1:#and distance_decision==1 and deltama3_before<deltama3 and deltama3_before<deltama3_before2 :#and deltama3_before2<deltama3_before3 :#and ma_ex_de==1:# and distance_decision==1 :# and self.close_price[current_price_index]>self.close_price[current_price_index-1]:#deltama3_before<deltama3:#self.macd[current_price_index-1]<self.macd[current_price_index] :#and (self.macd[current_price_index-2]<0 or self.macd[current_price_index-1]<0 or self.macd[current_price_index]<0):#and deltama5>=0 :#and deltama3_before<deltama3 ) or (deltama3>0 and deltama3_before<0)):# and  mama3>mama3_b and mama3_b<mama3_b2 and distance_decision==1:#and deltama5<0:#and self.close_price[current_price_index]>self.close_price[current_price_index-1] and deltama5_before<deltama5<0: #and deltama3_before<0 and deltama3>0 and deltama5_before<deltama5<0:#and ma_decision==1 :#and distance_decision==1:#deltama3_before<deltama3:# (deltama3_before>deltama3 or deltama3<0):
            if 0:#self.fail_flag!=0:
                #self.fail_flag-=1
                self.buy_price=trade_price
                print "buy buy 2"
            else:
                print "emd"
                self.de=0
                self.buy_ma=ma_list
                zhi5=self.close_price[current_price_index]+self.close_price[current_price_index-1]+self.close_price[current_price_index-2]+self.close_price[current_price_index-3]+self.close_price[current_price_index-4]
                zhi2=self.close_price[current_price_index]+self.close_price[current_price_index-1]
                print ((zhi2)/2.0)-(zhi5/5.0)
                print "prechade%s"%prechade
                print "buy buy 1"
                print "downcount %s"%down_count
                self.share = self.buy(self.money,trade_price)
                self.money = 0
                self.buy_price = trade_price
                self.buy_mp_price = mp_list[current_price_index]
                self.buy_mean_price = current_mean_price
                self.total_asset = self.share*trade_price
                self.buy_x_index.append(current_price_index)
                self.buy_y_index.append(current_price)
                self.one_time_day=1
                self.last_buy_result=0
                AB=1
                if len(emd_data)==500:
                    ABL=500
                else:
                    ABL=200
        elif  0:#self.fail_flag!=0 and  (  (imf_flag==1 ) ) and  self.buy_price==0 and deltama5>deltama5_before and deltama3>deltama3_before and distance_decision==1 :#and pvpower_de==1 :#emd_data[-1]>emd_data[-2] and emd_data[-2]<emd_data[-3]:#self.close_price[current_price_index]>self.close_price[current_price_index-1]:#and (deltama3_before<deltama3 and deltama3_before<0 and deltama3>0) :#and ma_ex_de==1:#and distance_decision==1 and deltama3_before<deltama3 and deltama3_before<deltama3_before2 :#and deltama3_before2<deltama3_before3 :#and ma_ex_de==1:# and distance_decision==1 :# and self.close_price[current_price_index]>self.close_price[current_price_index-1]:#deltama3_before<deltama3:#self.macd[current_price_index-1]<self.macd[current_price_index] :#and (self.macd[current_price_index-2]<0 or self.macd[current_price_index-1]<0 or self.macd[current_price_index]<0):#and deltama5>=0 :#and deltama3_before<deltama3 ) or (deltama3>0 and deltama3_before<0)):# and  mama3>mama3_b and mama3_b<mama3_b2 and distance_decision==1:#and deltama5<0:#and self.close_price[current_price_index]>self.close_price[current_price_index-1] and deltama5_before<deltama5<0: #and deltama3_before<0 and deltama3>0 and deltama5_before<deltama5<0:#and ma_decision==1 :#and distance_decision==1:#deltama3_before<deltama3:# (deltama3_before>deltama3 or deltama3<0):
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
            if self.tmphigh==0:
                self.tmphigh=self.buy_price
            elif self.tmphigh!=0 and current_price>self.tmphigh:
                self.tmphigh=current_price
        else:
            self.tmphigh=0


        print "cccc%s"%self.cccc
##
        datadata=(self.close_price[:current_price_index+1])

        maa5 = []
        for i in range(60):
            maa5.append(0)
#
        for i in range(60,len(datadata)):
            mean_5 = np.mean(datadata[i-59:i+1])
            maa5.append(mean_5)

        ma5 = []
        for i in range(60):
            ma5.append(0)
#
        for i in range(60,len(datadata)):
            mean_5 = np.mean(maa5[i-59:i+1])
            ma5.append(mean_5)

        maa5 = []
        for i in range(60):
            maa5.append(0)
#
        for i in range(60,len(datadata)):
            mean_5 = np.mean(ma5[i-59:i+1])
            maa5.append(mean_5)
#

        ma5 = []
        for i in range(60):
            ma5.append(0)
#
        for i in range(60,len(datadata)):
            mean_5 = np.mean(maa5[i-59:i+1])
            ma5.append(mean_5)

        ma_max_index = list(argrelextrema(np.array(maa5[-500:]),np.greater)[0])
        print " ma max index %s"%ma_max_index
        ma_min_index = list(argrelextrema(np.array(maa5[-500:]),np.less)[0])
        print " ma min index %s"%ma_min_index

        print "high list%s"%self.high_flag[-10:]
        print "close list%s"%self.close_flag[-10:]
        print "length %s"%len(emd_data)
        print "ff {}".format(self.fail_flag)
#

        if self.buy_price!=0:
            self.buy_day += 1

        if imf_flag==2:
            self.sell_flag=2
        else:
            self.sell_flag=0



        print "buy day%s"%self.buy_day

#        self.day_count+=1
#        if self.day_count%100==0:
#            self.hb.append(float(self.total_asset-self.last_as)/self.last_as)
#            self.last_as=self.total_asset



        print "hb %s"%self.hb
            

        print "rihb %s"%self.rihb



  
        self.total_asset_list.append(self.total_asset)
        print "total asset is %s"%(self.total_asset)
        print "money is %s"%(self.money)
        print "share is %s"%(self.share)

        p1 = map(lambda x:x[0], self.pimf1)
        p2 = map(lambda x:x[1], self.pimf1)
        print "p1 {}".format(np.mean(p1))
        print "p2 {}".format(np.mean(p2))

    def calc_bayes(self,data,data2):
        qujian = (max(data)-min(data))/10.0
        up = data[-1]+0.5*qujian
        down = data[-1]-0.5*qujian
        total = 0
        up_count = 0
        down_count = 0
        if data[-1]<data[-2]:
            for i in range(1,len(data)-1):
                if data[i-1]>data[i] and down<data[i]<up:
                    total += 1
                    if data2[i+1]>data2[i]:
                        up_count += 1
                    if data2[i+1]<data2[i]:
                        down_count += 1
        elif data[-1]>data[-2]:
            for i in range(1,len(data)-1):
                if data[i-1]<data[i] and down<data[i]<up:
                    total += 1
                    if data2[i+1]>data2[i]:
                        up_count += 1
                    if data2[i+1]<data2[i]:
                        down_count += 1
        if total == 0:
            return 0, 0
        else:
            return up_count/float(total),down_count/float(total) 



    def format_data(self, data):
        if isinstance(data,list):
            return [2*((float(data[i])-min(data))/(max(data)-float(min(data))))-1 for i in range(len(data))]



    def run(self,emd_data, datafile,current_index,date,emd_file,emd_data2,emd_data3,preflag,chazhi,prechade,kalval,predict):
        self.pre_cha=chazhi
        starttime = datetime.datetime.now()
        tmp = [i for i in emd_data]
        emd_data = tmp
        print "\n\n\n"
        print "emd data std %s"%(np.std(emd_data[-20:])/(np.mean(emd_data[-20:])))
        
        print "emd data %s"%emd_data[-10:]
        print "emd data %s"%emd_data2[-10:]


        #my_emd = emd_mean_pro_class(emd_data,3)
        #my_emd = one_dimension_emd(emd_data,3)
        #(imf, residual) = my_emd.emd(0.005,0.005)
        #(imf, residual) = my_emd.emd(0.1,0.1)

###


#        my_eemd = eemd(emd_data,30)
#        (imf0,imf1,imf2,imf3)= my_eemd.eemd_process(emd_data,30,4,'multi')
        imf=[]
        imf.append(emd_data)
        imf.append(emd_data)
        imf.append(emd_data)
###

        #my_emd2 = one_dimension_emd(emd_data2,3)
        #my_emd2 = emd_mean_pro_class(emd_data2,3)
        #(imf_open, residual) = my_emd2.emd(0.3,0.3)
        #(imf_open, residual) = my_emd2.emd_mean_pro(0.03,0.03)

#        my_emd3 = one_dimension_emd(emd_data3,9)
#        (imf_macd, residual) = my_emd3.emd(0.03,0.03)


        print "len imf %s"%len(imf)
            
        residual=emd_data
        imf_open=imf
        imf_macd=imf_open

        std = (np.std(emd_data[-30:])/(np.mean(emd_data[-30:])))
        self.emdstd.append(std)
        print "emdstd%s"%self.emdstd[-10:]
        if len(self.emdstd)>10:
            print "stdma3 %s %s"%(np.mean(self.emdstd[-10:-5]),np.mean(self.emdstd[-5:]))
        
        print current_index
        self.run_predict(imf,current_index,date,datafile,residual,emd_data,imf_open,imf_macd,std,preflag,emd_data2,prechade,kalval,predict)
        
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
        
        

    def draw_fig(self,emd_data,datafile,start,save=0):
        
        data = self.vol

        pv2=[self.close_price[i]*self.vol[i] for i in range(len(self.close_price))]
        pv = pvpv(self.close_price,self.vol)
####
        maa5 = []
        for i in range(3):
            maa5.append(0)
#
        for i in range(3,len(self.normal30)):
            mean_5 = np.mean(self.normal30[i-2:i+1])
            maa5.append(mean_5)

        ma5 = []
        for i in range(3):
            ma5.append(0)
#
        for i in range(3,len(self.normal30)):
            mean_5 = np.mean(maa5[i-2:i+1])
            ma5.append(mean_5)

        data1=ma5
        maa5 = []
        for i in range(3):
            maa5.append(0)
#
        for i in range(3,len(self.normal30)):
            mean_5 = np.mean(ma5[i-2:i+1])
            maa5.append(mean_5)
        ma5 = []
        for i in range(3):
            ma5.append(0)
#
        for i in range(3,len(self.normal30)):
            mean_5 = np.mean(maa5[i-2:i+1])
            ma5.append(mean_5)

        maa5 = []
        for i in range(60):
            maa5.append(0)
#
        for i in range(60,len(data)):
            mean_5 = np.mean(ma5[i-59:i+1])
            maa5.append(mean_5)
#

        ma5 = []
        for i in range(60):
            ma5.append(0)
#
        for i in range(60,len(data)):
            mean_5 = np.mean(maa5[i-59:i+1])
            ma5.append(mean_5)


        ma_max_index = list(argrelextrema(np.array(ma5[:]),np.greater)[0])
        print " ma max index %s"%ma_max_index
        ma_min_index = list(argrelextrema(np.array(ma5[:]),np.less)[0])
        print " ma min index %s"%ma_min_index
#
#
        data = maa5

        maa5 = []
        for i in range(3):
            maa5.append(0)
#
        for i in range(3,len(data)):
            mean_5 = np.mean(data[i-2:i+1])
            maa5.append(mean_5)

###
        ma5 = []
        for i in range(5):
            ma5.append(0)
#
        for i in range(5,len(data)):
            mean_5 = np.mean(self.close_price[i-4:i+1])
            ma5.append(mean_5)
#


        close_price = self.close_price
        #close_price = [np.log(i) for i in close_price]
        ma3_list=[]
        for i in range(3):
            ma3_list.append(0)
        for i in range(3,len(close_price)):
            ma3_list.append(np.mean(close_price[i-2:i+1]))
            #ma5_list.append(np.mean(obv[i-2:i+1]))

        ma5_list=[]
        for i in range(5):
            ma5_list.append(0)
        for i in range(5,len(close_price)):
            ma5_list.append(np.mean(close_price[i-4:i+1]))

        ma2_list=[]
        for i in range(2):
            ma2_list.append(0)
        for i in range(2,len(close_price)):
            ma2_list.append(np.mean(close_price[i-1:i+1]))

        ma10_list=[]
        for i in range(10):
            ma10_list.append(0)
        for i in range(10,len(close_price)):
            ma10_list.append(np.mean(close_price[i-9:i+1]))
        ma20_list=[]
        for i in range(20):
            ma20_list.append(0)
        for i in range(20,len(close_price)):
            ma20_list.append(np.mean(close_price[i-19:i+1]))

        ma30_list=[]
        for i in range(30):
            ma30_list.append(0)
        for i in range(30,len(close_price)):
            ma30_list.append(np.mean(close_price[i-29:i+1]))

        ma4=ma_func2(self.close_price,4)
        ma_list=[ma3_list[i]-ma10_list[i] for i in range(len(close_price))]
        ma_list=[ma3_list[i]-ma5[i] for i in range(len(close_price))]
        ma_list=[close_price[i]-ma5[i] for i in range(len(close_price))]
        ma_list=[ma2_list[i]-ma5[i] for i in range(len(close_price))]
        ma_list=[ma2_list[i]-ma10_list[i] for i in range(len(close_price))]
        ma_list=[close_price[i]-ma3_list[i] for i in range(len(close_price))]
        ma_list=[close_price[i]-ma5[i] for i in range(len(close_price))]
        ma_list=[ma3_list[i]-ma5[i] for i in range(len(close_price))]
        ma_list=[ma3_list[i]-ma10_list[i] for i in range(len(close_price))]
        ma_list=[ma2_list[i]-ma4[i] for i in range(len(close_price))]
        #ma_list=[ma2_list[i]-ma10_list[i] for i in range(len(close_price))]
        ma_list=[close_price[i]-ma2_list[i] for i in range(len(close_price))]
        ma_list=[ma3_list[i]-ma5[i] for i in range(len(close_price))]
#
#
        data1=ma_list

        ma2_ma2_list=ma_func2(ma2_list,2)

        ma_list = [ma2_list[i]-ma2_ma2_list[i] for i in range(len(ma2_list))]

        ma20_list=[]
        for i in range(20):
            ma20_list.append(0)
        for i in range(20,len(close_price)):
            ma20_list.append(np.mean(close_price[i-19:i+1]))
        ma60_list=[]
        for i in range(60):
            ma60_list.append(0)
        for i in range(60,len(close_price)):
            ma60_list.append(np.mean(close_price[i-59:i+1]))


        clog = [np.log(i) for i in self.close_price] 
        #data2 = list(np.diff(clog))
        data2 = list(np.diff(ma_list))
        data2.insert(0,0)
        cdiff = data2
        data1=data2
        data1_ma3_list=[]
        pdata1=3
        for i in range(pdata1):
            data1_ma3_list.append(0)
        for i in range(pdata1,len(data1)):
            data1_ma3_list.append(np.mean(data1[i-(pdata1-1):i+1]))
 

        data2 = list(np.diff(data2))
        data2.insert(0,0)
        data2_ma3_list=[]
        pdata2=5
        for i in range(pdata2):
            data2_ma3_list.append(0)
        for i in range(pdata2,len(data2)):
            data2_ma3_list.append(np.mean(data1_ma3_list[i-(pdata2-1):i+1]))


        data2=cdiff
        data2_x_index =[] 
        data2_y_index =[] 
        data2_x_index2 =[] 
        data2_y_index2 =[] 
        for i in range(1,len(data2)):
            if judge_extrem(data2[:i])>0:
                data2_x_index.append(i-1)
                data2_y_index.append(data2[i-1])
            if judge_extrem(data2[:i])<0:
                data2_x_index2.append(i-1)
                data2_y_index2.append(data2[i-1])

             

        print "normal %s"%(stats.normaltest(data2)[1])

        data = self.close_price

        data1 = [ma_list[i]*cdiff[i] for i in range(len(ma_list))]
        
        ma1=[]
        pdata2=1
        for i in range(pdata2):
            ma1.append(0)
        for i in range(pdata2,len(ma_list)):
            ma1.append(np.mean(ma_list[i-(pdata2-1):i+1]))

        ma2=[]
        pdata2=1
        for i in range(pdata2):
            ma2.append(0)
        for i in range(pdata2,len(ma_list)):
            ma2.append(np.mean(cdiff[i-(pdata2-1):i+1]))

        data1 = [ma1[i]*ma2[i] for i in range(len(ma_list))]
        data1 = ma_list
        data1 = [self.close_price[i]-self.open_price[i] for i in range(len(self.close_price))]

        ma2_ma2_list=ma_func2(ma2_list,2)
        ma5_list=ma_func2(self.close_price,5)
        ma30_list=ma_func2(self.close_price,30)
        ma2_ma2_ma2_list=ma_func2(ma2_ma2_list,2)

        ma_list = [ma2_ma2_list[i]-ma2_ma2_ma2_list[i] for i in range(len(ma2_list))]
        ma_list = [ma2_list[i]-ma5_list[i] for i in range(len(ma2_list))]
        #ma_list = [close_price[i]-ma3_list[i]  for i in range(len(ma2_list))]
        #self.je=ma_func2(self.je,5)
        #self.vol=ma_func2(self.vol,5)
        mp=[]
        for i in range(len(self.close_price)):
            if self.vol[i]==0:
                mp.append(self.close_price[i])
            else:
                mp.append(self.je[i]/self.vol[i])
             
###




        p1 = map(lambda x:x[0], self.pimf1)
        p2 = map(lambda x:x[1], self.pimf1)
        print "p1 {}".format(np.mean(p1))
        print "p2 {}".format(np.mean(p2))


###
        
        data1 = ma_list
        data1 = self.close_price
        #mp=ma_func2(mp,2)
        #ma=ma_func2(self.close_price,2)
        
        #data1 = [ma[i]-mp[i] for i in range(len(self.close_price))]
        data1 = [self.close_price[i]-mp[i] for i in range(len(self.close_price))]
        #data1 = ma_func2(data1,3)
        xx=[]
        for i in range(len(self.close_price)):
            if self.high_price[i]-self.close_price[i]>self.close_price[i]-self.low_price[i]:
                xx.append(i)
        xx=[]
        ma3=ma_func2(self.close_price,3)
        for i in range(len(self.close_price)):
            if self.close_price[i]-ma3[i]<0:
                xx.append(i)
        xx=[]
        for i in range(1,len(self.close_price)):
            #if self.je[i]>self.je[i-1] and self.vol[i]>self.vol[i-1]:
            if self.close_price[i-1]>self.close_price[i]:
                xx.append(i)
        xx_data1=[data1[i] for i in xx]
        mean = calc_recent_rilling(data1)
        print "length data1"
        print len(data1)
        up,down = calc_boll(data1,3,1)
        print len(up)
        print len(down)
        #data1 = ma_func2(data1,3)
        #data1 = ma3_list
        #
        #l1 = [self.close_price[i]/self.close_price[i-1] for i in range(1,len(self.close_price))]
        #l1.insert(0,0)
        #ma_list = calc_ma(l1)
        #data1 = ma_list


        data3 = [data1[i]*data2[i] for i in range(len(data1))]
        data3_x_index=[]
        data3_x_index2=[]
        data3_y_index=[]
        data3_y_index2=[]
        for i in range(1,len(data3)):
            if data3[i-1]>0 and data3[i]<0:
                data3_x_index.append(i)
                data3_y_index.append(data3[i])
        for i in range(1,len(data3)):
            if data3[i-1]<0 and data3[i]>0:
                data3_x_index2.append(i)
                data3_y_index2.append(data3[i])

        
        data2 = ma_func2(self.close_price,10)
        data2 = self.je
        data2 = ma_func2(data1,3)
        data2 = ma_func2(data2,3)
        data2 = self.macd
        data2 = self.je
        data2 = ma_func2(data2,10)


        data2 = [(self.high_price[i]+self.low_price[i])/2-self.close_price[i] for i in range(len(self.close_price))] 
        data2 = [self.high_price[i]-self.close_price[i] for i in range(len(self.close_price))] 
        data2 = [self.high_price[i]-self.low_price[i] for i in range(len(self.close_price))] 
        data2 = self.s1
        data2 = list(np.diff(self.close_price))
        data2.insert(0,0)
        data2 = [self.open_price[i]-mp[i] for i in range(len(self.close_price))]
        peak_value=[]
        for i in self.peak:
            peak_value.append(data1[i])
        print self.peak 
        print len(self.peak) 
        print peak_value 
        print len(peak_value) 
        ma_me=[]
        pdata2=3
        for i in range(pdata2):
            ma_me.append(0)
        for i in range(pdata2,len(data2)):
            ma_me.append(np.mean(data1[i-(pdata2-1):i+1]))

        #data1 = self.je
        my_emd = one_dimension_emd(self.close_price)
        (imf, residual) = my_emd.emd(0.05,0.05)
        imf = imf[2]

#        data1 = [i[0] for i in self.snr]
#        data1 = self.filterdata
#        data10 = data[start:]
#        data11 = [data10[i]-data1[i] for i in range(len(data1))]
#        data11 = self.last_imf1_min
#        data11 = self.je


        print "stdimf%s"%self.stdimf

        #data2_buy_value = [data2[i] for i in self.buy_x_index]
        #data2_sell_value = [data2[i] for i in self.sell_x_index]
        
        imf_buy_value = [imf[i] for i in self.buy_x_index]
        imf_ex_value = [imf[i] for i in self.ex_x_index]
        imf_ex_value2 = [imf[i] for i in self.ex_x_index2]
        imf_sell_value = [imf[i] for i in self.sell_x_index]

        imf_sugg_y_index = [imf[i] for i in self.imf_sugg_x_index]


        ma2 = ma_func2(data1,2)
        ma3 = ma_func2(close_price,30)
        ma3=self.vol
        ma3 = ma_func2(ma3,5)
        ma3=self.pimf1
        ma3=exp_smooth(self.close_price,0.3)
        ma3=list(np.diff(self.close_price))
        ma3.insert(0,0)
        tmp=[self.close_price[i]-ma3[i] for i in range(len(ma3))]
        ma3=tmp
        ma3=self.vol
        ma3=self.s2
        ma3=self.diffmpstd
        ma3=mp[start:]
        ma4=self.mclist4
        ma5=[ma4[i]-ma3[i] for i in range(len(ma3))]



        data2 = ma_func2(self.vol,5)
        data3 = ma_func2(self.vol,10)
        
        data2 = ma_func2(self.s1,3)
        data2 = self.stab_de
        data2 = [i-1 for i in data2]
        data2 = self.rihb2
        ma3=ma_func2(self.close_price,3)
        data2 = [self.close_price[i]-mp[i] for i in range(len(self.close_price))]
        #data2 = ma_func2(data2,3)
        data2 = self.hursti
        data2 = self.adx
        data2 = self.cpstd
        data22 = self.adx2
        data22 = self.mpstd
        data2 = self.tde
        data22 = self.tde
        
        data3 = self.s2
        data3 = mp
        data3 = self.tde
        upper, middle, lower = talib.BBANDS(np.asarray(data3), timeperiod=10, nbdevup=1, nbdevdn=1, matype=0)
        #data3 = self.ytx2
        #ma3=self.total_asset_list
        #ma3=self.diffmpcha
        #ma3=self.mclist2


        #data1 = [data1[i]-ma2[i] for i in range(len(data1))]
        ma100 = ma_func2(data1,2)
        #ma_buy_value = [data1[i] for i in self.buy_x_index]
        #ma_sell_value = [data1[i] for i in self.sell_x_index]

        buy_x = [i-start for i in self.buy_x_index]
        sell_x = [i-start for i in self.sell_x_index]
        data1_buy = [data1[i] for i in self.buy_x_index]
        data1_sell = [data1[i] for i in self.sell_x_index]
        big0index=[i for i in range(len(data1)) if data1[i]>0 ]
        small0index=[i for i in range(len(data1)) if data1[i]<0 ]
        big0value=[i for i in data1 if i>0 ]
        small0value=[i for i in data1 if i<0 ]

        big0index2=[i for i in range(len(data2)) if data2[i]>0 ]
        small0index2=[i for i in range(len(data2)) if data2[i]<0 ]
        big0value2=[i for i in data2 if i>0 ]
        small0value2=[i for i in data2 if i<0 ]

        data = self.close_price
        plt.figure(1)
        plt.subplot(511).axis([start,len(data),min(data[start:]),max(data[start:])])      
        plt.plot([i for i in range(len(data))],data,'o',[i for i in range(len(data))],data,'b',self.buy_x_index,self.buy_y_index,'r*',self.sell_x_index,self.sell_y_index,'g*')#,self.ex_x_index,self.ex_y_index,'yo',self.ex_x_index2,self.ex_y_index2,'ko',self.buy_x_index,self.buy_y_index,'r*',self.sell_x_index,self.sell_y_index,'g*')             
        plt.subplot(512).axis([start,len(data1),min(data1[start:]),max(data1[start:])])      
        #plt.plot([i for i in range(len(data1))],data1,'o',[i for i in range(len(data1))],data1,'b',self.ex_x_index,ex_value,'yo',self.ex_x_index2,ex_value2,'ko',self.buy_x_index,data1_buy_value,'r*',self.sell_x_index,data1_sell_value,'g*')             
        #plt.plot([i for i in range(len(data1))],data1,'o',[i for i in range(len(data1))],data1,'b',data10,'ro',data10,'r')#,self.buy_x_index,ma_buy_value,'r*',self.sell_x_index,ma_sell_value,'g*')             
        plt.plot([i for i in range(len(data1))],data1,'o',[i for i in range(len(data1))],data1,big0index,big0value,'ro',small0index,small0value,'go',mean,'g',xx,xx_data1,'y*')#up,'g',down,'g')#,self.buy_x_index,ma_buy_value,'r*',self.sell_x_index,ma_sell_value,'g*')             
        #plt.subplot(413).axis([start,len(data2),min(data2[start:]),max(data2[start:])])      
        plt.subplot(513)#.axis([start,len(data2),min(data2[start:]),max(data2[start:])])      
        #plt.plot([i for i in range(len(data2))],data2,'o',[i for i in range(len(data2))],data2,'b',data2_x_index,data2_y_index,'yo',data2_x_index2,data2_y_index2,'ko')             
        plt.plot([i for i in range(len(data2))],data2,'r',[i for i in range(len(data22))],data22,'b')#,big0index2,big0value2,'ro',small0index2,small0value2,'go')#,self.buy_x_index,data2_buy_value,'yo',self.sell_x_index,data2_sell_value,'ko')             
        plt.subplot(514)#.axis([start,len(data3),min(data3[start:]),max(data3[start:])])
        #plt.subplot(514)#.axis([0,len(ma3),8000,12000])
        plt.plot([i for i in range(len(data3))],data3,'o',[i for i in range(len(data3))],data3,'b',[i for i in range(len(lower))],lower,'r')#,[i for i in range(len(data2))],data2,'r')#,data3_x_index,data3_y_index,'yo',data3_x_index2,data3_y_index2,'ko')
        #plt.plot([i for i in range(len(ma3))],ma5,'b')
        plt.subplot(515).axis([start,len(imf),min(imf[start:]),max(imf[start:])])
        #plt.plot([i for i in range(len(imf))],imf,'o',[i for i in range(len(imf))],imf,'b',self.buy_x_index,imf_buy_value,'r*',self.sell_x_index,imf_sell_value,'g*',self.ex_x_index,imf_ex_value,'yo',self.ex_x_index2,imf_ex_value2,'ko')             
        plt.plot([i for i in range(len(imf))],imf,'o',[i for i in range(len(imf))],imf,'b',self.buy_x_index,imf_buy_value,'r*',self.sell_x_index,imf_sell_value,'g*')#,self.imf_sugg_x_index,imf_sugg_y_index,'yo')             
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



def format_data(data):
    if isinstance(data,list):
        return [10*((float(data[i])-min(data))/(max(data)-float(min(data)))) for i in range(len(data))]


def pvpv(l1,l2):
    pv=[l1[0]*l2[0]]
    for i in range(1,len(l1)):
        if l1[i]>=l1[i-1]:
            pv.append(pv[i-1]+l1[i]*l2[i])
        else:
            pv.append(pv[i-1]-l1[i]*l2[i])
    return pv
    

def wr_func(c,h,l,n):
    wr=[]
    for i in range(n):
        wr.append(0)
    for i in range(n,len(c)):
        cn=c[i]
        hn=max(h[i-(n):i+1])
        ln=min(l[i-(n):i+1])
        wr.append(100*(hn-cn)/(hn-ln))

    return wr


def hurst2(ts):
	"""Returns the Hurst Exponent of the time series vector ts"""
	# Create the range of lag values
	lags = range(2, 10)

	# Calculate the array of the variances of the lagged differences
	tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

	# Use a linear fit to estimate the Hurst Exponent
	poly = polyfit(log(lags), log(tau), 1)

	# Return the Hurst exponent from the polyfit output
	return poly[0]*2.0

def kdj_func(c,h,l,n):
    k=[]
    d=[]
    for i in range(n-2):
        k.append(0)
        d.append(0)
    k.append(50)
    d.append(50)
    for i in range(n-1,len(c)):
        cn=c[i-1]
        hn=max(h[i-(n-1):i])
        ln=min(l[i-(n-1):i])

        rsv=100*(float(cn-ln)/(hn-ln))
        k.append((2/3.0)*k[-1]+(1/3.0)*rsv)
        d.append((2/3.0)*d[-1]+(1/3.0)*k[-1])

    
    return (k,d)
    
def ma_func(data, period, start, current_index):
    sum_period = 0
    for i in range(period):
        sum_period += data[current_index + start - i]
    return float(sum_period)/period

def ma_func2(data, period):
    ma_list=[]
    for i in range(period):
        ma_list.append(0)
    for i in range(period,len(data)):
        ma_list.append(np.mean(data[i-(period-1):i+1]))
    return ma_list

def ma_func3(data,p1,p2):
    ma1=ma_func2(data,p1)
    ma2=ma_func2(data,p2)
    ma=[ma1[i]-ma2[i] for i in range(len(data))]
    return ma
 
def calc_obv(h,l,c,v):
    diff_c=list(np.diff(c))
    diff_c.insert(0,0)
    obv=[(((c[0]-l[0])-(h[0]-c[0]))/float(h[0]-l[0]))*v[0]]
    #obv=[v[0]]
    for i in range(1,len(diff_c)):
        if h[i]>l[i]:#diff_c[i]>0:
            #obv.append(obv[i-1]+v[i]*c[i])
            #obv.append(obv[i-1]+v[i])
            #obv.append(v[i]*c[i])
            #print "\n"
            #print i
            #print h[i]
            #print l[i]
            #print c[i]
            #print float(h[i]-l[i])
            up=((c[i]-l[i])-(h[i]-c[i]))
            down=float(h[i]-l[i])
            obv.append((up/down)*v[i])
        else:
            #obv.append(obv[i-1]-v[i]*c[i])
            #obv.append(obv[i-1]-v[i])
            #obv.append(v[i]*c[i])
            #print "\n"
            #print i
            #print h[i]
            #print l[i]
            #print c[i]
            up=((c[i]-l[i])-(h[i]-c[i]))
            down=float(h[i]-l[i])
            #obv.append((up/down)*v[i])
            obv.append(0)

    return obv


def rfclf(index,close_price,vol,je):
    def train_data_generate(index,close_price,vol,je):
        datav = vol[:index]
        dataj = je[:index]
        datac = close_price[:index]
        mp = []
        for i in range(len(datav)):
            mp.append(dataj[i]/datav[i])
    
    
        #ma10 = ma_func(datac,10)
        #diff10 = list(np.diff(ma10))
        #diff10.insert(0,0)
    
        sample = zip(datac[:-3],mp[:-3])
        sample = [list(i) for i in sample]
        lable = []
        for i in range(len(datac)-3):
            if mp[i]>datac[i] and datac[i]<datac[i+1] :#or datac[i]<datac[i+2] or datac[i]<datac[i+3]:
                lable.append(1)
            else:
                lable.append(0)
        return np.array(sample),np.array(lable)

    sample,lable = train_data_generate(index,close_price,vol,je)
    #clf = LogisticRegression()
    clf = RandomForestClassifier(n_estimators = 100)
    clf.fit(sample,lable)
    test_data = [[close_price[index],je[index]/vol[index]]]
    print "testdata {}".format(test_data)
    r = clf.predict(test_data)[0]
    return r
    

def judge_extrem(data):

    k1=1
    k2=1
    period=5
    boll_high1 = k2*np.std(data[-period:])+np.mean(data[-period:])
    boll_high1_b = k2*np.std(data[-period-1:])+np.mean(data[-period-1:])
    boll_low1 = k1*(-np.std(data[-period:]))+np.mean(data[-period:])
    boll_low1_b = k1*(-np.std(data[-period-1:]))+np.mean(data[-period-1:])

    if data[-1]<boll_low1:
        return -1
    elif data[-1]>boll_high1:
        return 1
    else:
        return 0
    

def getanytime(file_data, date, time):
    data = getMinDataWholeday(file_data,date)

    p1 = getMinDatabyTime(data, time)
    return float(p1)

def stats_last_time(mp):
    tmp=0
    l=[]
    for i in range(len(mp)):
        if mp[i]<0:
            tmp+=1
        else:
            l.append(tmp)
            tmp=0
    l=filter(lambda x:x>0 ,l)
    rb=np.mean(l)
    srb=np.std(l)
    print "below l {} {}".format(np.mean(l),np.std(l))


    tmp=0
    l=[]
    for i in range(len(mp)):
        if mp[i]>0:
            tmp+=1
        else:
            l.append(tmp)
            tmp=0
    l=filter(lambda x:x>0 ,l)
    ru=np.mean(l)
    sru=np.std(l)
    print "up l {} {}".format(np.mean(l),np.std(l))
    return rb,srb,ru,sru

def getMinDataAllYears(datafile, years):
    datafile = datafile.split("/")[-1]
    current_path = os.getcwd()
    if datafile.startswith("sh"):
        minbasedir = current_path + "/test_data/mindata" + "/sh"
    else:
        minbasedir = current_path + "/test_data/mindata" + "/sz"


    mindatalist = []
    for i in years:
        mindir = minbasedir + "/%s"%i
        mindatalist.append(getMinDatabyFile(mindir + "/min1_%s.csv"%datafile))
    mindata = []
    for i in mindatalist:
        mindata += i

    return mindata
    

def imf_power(data,imf,index):
    imf_power=0
    print "begin calc imf power"
#    for i in range(index+1):
#        tmp=sum(map(lambda x:x**2,imf[i]))
#        imf_power+=tmp

    data_power=sum(map(lambda x:x**2,data))
    imf_power=sum(map(lambda x:x**2,imf[index]))
    return imf_power/data_power
    

def stats_mpup(mp,datac):
    c1=0
    c2=0
    l=[]
    for i in range(1,len(mp)-2):
        if mp[i]<0 and mp[i-1]>0:
            c1+=1
            if mp[i+1]>0 or mp[i+2]>0:
                c2+=1
    if c1==0:
        mpdowncdown=0
    else:
        mpdowncdown=c2/float(c1)

    c1=0
    c2=0
    l=[]
    for i in range(1,len(mp)-1):
        if mp[i]>mp[i-1]:
            c1+=1
            if datac[i]>datac[i-1] or datac[i+1]>datac[i]:
                c2+=1
    if c1==0:
        mpupcup=0
    else:
        mpupcup=c2/float(c1)
    return mpdowncdown,mpupcup

def calc_boll(data,p,k):
    upboll=[]
    downboll=[]
    for i in range(p-1,len(data)):
        d=data[i-p+1:i+1]
        upboll.append(np.mean(d)+k*np.std(d))
        downboll.append(np.mean(d)-k*np.std(d))

    for i in range(p-1):
        upboll.insert(0,0)
        downboll.insert(0,0)
    return upboll,downboll

def calc_pro(l1,l2):
    c=0
    for i in l1:
        if i in l2:
            c+=1
    return float(c)/len(l1)


def calc_down(data):
    c=[]
    tmp=0
    for i in range(1,len(data)):
        if data[i]<data[i-1]:
            tmp+=1#data[i-1]-data[i]
        else:
            if tmp!=0:
                c.append(tmp)
                tmp=0
    if tmp!=0:
        c.append(tmp)

    if c==[]:
        c=[0]
    return np.mean(c)

def calc_mp_sum(datac,mp,mp_diff,mean_flag=1):
    b_flag=0
    c=0
    s=[]
    d=[]
    for i in range(2,len(datac)):
        if mp_diff[i]<0 and b_flag==0 and ((mean_flag==1 and np.mean(mp_diff[i-2:i+1])<0) or (mean_flag==0)):
            b_flag=datac[i]
            c+=1
        elif mp_diff[i]>0 and b_flag!=0 and ((mean_flag==1 and np.mean(mp_diff[i-2:i+1])>0) or (mean_flag==0)):
            d.append(c)
            c=0
            if (b_flag-datac[i])/b_flag>0.1:
                pass
            else:
                s.append((datac[i]-b_flag)/b_flag)
            b_flag=0
        elif b_flag!=0 and (b_flag-datac[i])/b_flag>=0.05:
            s.append((datac[i]-b_flag)/b_flag)
            b_flag=0
            d.append(c)
            c=0
        else:
            c+=1
    if b_flag!=0 and datac[i]!=b_flag:
        s.append((datac[i]-b_flag)/b_flag)
        d.append(c)
      
    if s==[]:
        s.append(0)
    print "calc mp sum"
    print s
    print d
    if d!=[]:
        r = [s[i]/d[i] for i in range(len(s))]
    else:
        r=[0]
    return r

def calc_mp_sum2(datac,mp,mp_diff,mean_flag=1):
    b_flag=0
    c=0
    s=[]
    d=[]
    for i in range(2,len(datac)):
        if mp_diff[i]>0 and b_flag==0 :#and ((mean_flag==1 and np.mean(mp_diff[i-2:i+1])<0) or (mean_flag==0)):
            b_flag=datac[i]
            c+=1
        elif mp_diff[i]<0 and b_flag!=0 :#and ((mean_flag==1 and np.mean(mp_diff[i-2:i+1])>0) or (mean_flag==0)):
            d.append(c)
            c=0
            if (b_flag-datac[i])/b_flag>0.1:
                pass
            else:
                s.append((datac[i]-b_flag)/b_flag)
            b_flag=0
        elif b_flag!=0 and (b_flag-datac[i])/b_flag>=0.05:
            s.append((datac[i]-b_flag)/b_flag)
            b_flag=0
            d.append(c)
            c=0
        else:
            c+=1
    if b_flag!=0 and datac[i]!=b_flag:
        s.append((datac[i]-b_flag)/b_flag)
        d.append(c)

    if s==[]:
        s.append(0)
    print "calc mp sum"
    print s
    print d
    if d!=[]:
        r = [s[i]/d[i] for i in range(len(s))]
    else:
        r=[0]
    return r



def calc_vol_sum(datac,datavma1,datavma2):
    b_flag=0
    s=[]
    d=[]
    c=0
    
    for i in range(1,len(datac)):
        if datavma1[i]>datavma2[i] and datavma1[i-1]<datavma2[i-1] and b_flag==0:
            b_flag=datac[i]
            c+=1
        elif datavma1[i]<datavma2[i] and b_flag!=0:
            d.append(c)
            c=0
            if (b_flag-datac[i])/b_flag>0.1:
                pass
            else:
                s.append((datac[i]-b_flag)/b_flag)
            b_flag=0
        elif b_flag!=0 and (b_flag-datac[i])/b_flag>=0.05:
            s.append((datac[i]-b_flag)/b_flag)
            b_flag=0
            d.append(c)
            c=0
        else:
            c+=1
    if b_flag!=0 and datac[i]!=b_flag:
        s.append((datac[i]-b_flag)/b_flag)
        d.append(c)
    if s==[]:
        s.append(0)
    print "calc mp vol"
    print s
    print d
    if d!=[]:
        r = [s[i]/d[i] for i in range(len(s))]
    else:
        r=[0]
    return r


def calc_up(data):
    c=[]
    tmp=0
    for i in range(1,len(data)):
        if data[i]>data[i-1]:
            tmp+=1#data[i]-data[i-1]
        else:
            if tmp!=0:
                c.append(tmp)
                tmp=0
    if tmp!=0:
        c.append(tmp)
    if c==[]:
        c=[0]
    return np.mean(c)

def calc_pro2(l1,l2,l3):
    c1=0
    c2=0
    for i in l1:
        if i in l2 :#or i+1 in l2 or i-1 in l2:
            c1+=1
    for i in l1:
        if i in l3 :#or i+1 in l3 or i-1 in l3:
            c2+=1
    if c1+c2!=0:
        return float(c1)/(c1+c2)
    else:
        return 0

def calc_ma(l):
    ma_l = ma_func2(l,3)
    return [l[i]-ma_l[i] for i in range(len(l))]

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
        #macd.append(float(eachline.split("\t")[19]))
        macd.append(float(eachline.split("\t")[2]))
        open_price.append(float(eachline.split("\t")[1]))
        date.append(eachline.split("\t")[0])
        vol.append(float(eachline.split("\t")[5]))
        je.append(float(eachline.split("\t")[6]))
    
    #mindata = getMinDataAllYears(datafile,(2010,2011,2012,2013))
    mindata=[]

    #period=2
    #c = []
    #for ii in range(period):
    #    c.append(0)
    #for ii in range(period,len(close_price)):
    #    c.append(np.mean(close_price[ii-(period-1):ii+1]))

    #h = []
    #for ii in range(period):
    #    h.append(0)
    #for ii in range(period,len(close_price)):
    #    h.append(np.mean(high_price[ii-(period-1):ii+1]))

    #l = []
    #for ii in range(period):
    #    l.append(0)
    #for ii in range(period,len(close_price)):
    #    l.append(np.mean(low_price[ii-(period-1):ii+1]))
##

##
    c=close_price
    l=low_price
    h=high_price
    (k,d)=kdj_func(c,h,l,5)
    wr=wr_func(close_price,high_price,low_price,10)
#    line=np.linspace(0,100,len(c))
#    noise = np.random.standard_normal(len(line))
#    close_price=list((np.sin(line))+noise)
    process = process_util(open_price,close_price,macd,date,vol,high_price,low_price,je,k,d,wr,mindata)

##

###

    emd_dir = os.path.abspath('.')+"/"+"emd_data"
    print emd_dir
    os.popen("rm -r %s"%emd_dir)
    os.popen("mkdir %s"%emd_dir)
    generate_emd_func = generateEMDdata(datafile,begin,1000,emd_dir)


    #generate_emd_func.generate_machao_emd_data_fix(3)
#    generate_emd_func.generate_ma_emd_data_fix(5)
    #generate_emd_func.generate_emd_data_fix()
###
    period=3
 
    #close_price=[j*10 for j in macd]
    pv=[close_price[i]*vol[i] for i in range(len(close_price))]
    pv = pvpv(close_price,vol)
    ma = []
    for ii in range(period):
        ma.append(0)
    for ii in range(period,len(close_price)):
        ma.append(np.mean(close_price[ii-(period-1):ii+1]))

    dddddd=ma
    ma2 = []
    for ii in range(period):
        ma2.append(0)
    for ii in range(period,len(close_price)):
        ma2.append(np.mean(ma[ii-(period-1):ii+1]))
 #   data=ma
 #   for ii in range(period):
 #       ma.append(0)
 #   for ii in range(period,len(close_price)):
 #       ma.append(np.mean(data[ii-(period-1):ii+1]))

    period1=5
    maa = []
    for ii in range(period1):
        maa.append(0)
    for ii in range(period,len(close_price)):
        maa.append(np.mean(close_price[ii-(period1-1):ii+1]))
   
    z=[(high_price[i]+low_price[i]+2*close_price[i])/4.0 for i in range(len(close_price))]
    z=[ i*3 for i in close_price]
    z2=[ i*10 for i in close_price]
    z=[ma[i] for i in range(len(close_price))]

    m=[p*100 for p in macd]
    pv = pvpv(close_price,vol)
    pv2=[close_price[i]*vol[i] for i in range(len(close_price))]
###
    presnr=[]
    presnr2=[]
    ma5_list=[]
    for i in range(5):
        ma5_list.append(0)
    for i in range(5,len(close_price)):
        ma5_list.append(np.mean(close_price[i-4:i+1]))
        #ma5_list.append(np.mean(obv[i-2:i+1]))

    ma10_list=[]
    for i in range(10):
        ma10_list.append(0)
    for i in range(10,len(close_price)):
        ma10_list.append(np.mean(close_price[i-9:i+1]))

    ma30_list=[]
    for i in range(30):
        ma30_list.append(0)
    for i in range(30,len(close_price)):
        ma30_list.append(np.mean(close_price[i-29:i+1]))
    ma_list=[ma5_list[i]-ma10_list[i] for i in range(len(close_price))]
    ma_list=[close_price[i]-ma5_list[i] for i in range(len(close_price))]
    ma_list=[ma10_list[i]-ma30_list[i] for i in range(len(close_price))]

    ma2_list=[]
    for i in range(2):
        ma2_list.append(0)
    for i in range(2,len(close_price)):
        ma2_list.append(np.mean(close_price[i-1:i+1]))

    ma2_ma2_list=[]
    for i in range(2):
        ma2_ma2_list.append(0)
    for i in range(2,len(close_price)):
        ma2_ma2_list.append(np.mean(ma2_list[i-1:i+1]))

    #close_price = [np.log(i) for i in close_price]

    ma3_list=[]
    for i in range(3):
        ma3_list.append(0)
    for i in range(3,len(close_price)):
        ma3_list.append(np.mean(close_price[i-2:i+1]))
        #ma5_list.append(np.mean(obv[i-2:i+1]))

    ma5_list=[]
    for i in range(5):
        ma5_list.append(0)
    for i in range(5,len(close_price)):
        ma5_list.append(np.mean(close_price[i-4:i+1]))

    ma10_list=[]
    for i in range(10):
        ma10_list.append(0)
    for i in range(10,len(close_price)):
        ma10_list.append(np.mean(close_price[i-9:i+1]))

    ma20_list=[]
    for i in range(20):
        ma20_list.append(0)
    for i in range(20,len(close_price)):
        ma20_list.append(np.mean(close_price[i-19:i+1]))

    ma_list=[ma3_list[i]-ma5_list[i] for i in range(len(close_price))]
    ma_list=[close_price[i]-ma5_list[i] for i in range(len(close_price))]
    #ma_list=[ma2_list[i]-ma5_list[i] for i in range(len(close_price))]
    #ma_list=[ma2_list[i]-ma10_list[i] for i in range(len(close_price))]
    #ma_list=[close_price[i]-ma3_list[i] for i in range(len(close_price))]
    #ma_list = ma_func2(ma_list,5)
    l1 = [close_price[i]/close_price[i-1] for i in range(1,len(close_price))]
    l1.insert(0,0)
   # l1 = [np.log(i) for i in l1]
   # ma_list = calc_ma(l1)
    ma_list100 = l1
    #ma_list100 = ma_func2(ma_list100,5)
    
    #ma_list=list(np.diff(close_price))
    #ma_list.insert(0,0)
    #ma_list=[ma2_list[i]-ma2_ma2_list[i] for i in range(len(close_price))]
#    ma_list=[close_price[i]-ma3_list[i] for i in range(len(close_price))]
    #ma_list=[ma3_list[i]-ma10_list[i] for i in range(len(close_price))]
    #ma_list=[close_price[i]-ma5_list[i] for i in range(len(close_price))]
    #ma_list=[ma2_list[i]-ma5_list[i] for i in range(len(close_price))]
    #ma_list = [ ma_list[i]/ma_list[i-1] for i in range(10,len(ma_list))]
    for i in range(10): 
        ma_list.insert(0,0)
    ma_list2=[ma10_list[i]-ma20_list[i] for i in range(len(close_price))]
    ma_list3=list(np.diff(ma_list2))
    ma_list3.insert(0,0)
    #ma_list=[close_price[i]-ma3_list[i] for i in range(len(close_price))]
    #`ma_list = ma_func2(ma_list,5)

    #ma5_list=[]
    #for i in range(2):
    #    ma5_list.append(0)
    #for i in range(2,len(close_price)):
    #    ma5_list.append(np.mean(ma_list[i-1:i+1]))
        #ma5_list.append(np.mean(obv[i-2:i+1]))
    #ma_list=ma5_list

    chazhi = []

##
    halfclose_list=[]
    near_list=[]
    near_list2=[]
    near_list3=[]
    for i in range(begin,len(date)):
        cdate = date[i].replace("/","-")
        cdate = cdate.replace(" ","")
#        p=float(getanytime(mindata,cdate,"10:30:00"))
        #halfclose_list.append(p)
        #p=float(getanytime(mindata,cdate,"14:30:00"))
        #near_list.append(p)
  #      p=float(getanytime(mindata,cdate,"14:30:00"))
  #      near_list2.append(p)

    k_v_base = [0]*(begin)
    #ma3_list.pop(0)
    #ma3_list.pop(0)
    #ma3_list.pop(0)
    #ma3_list.insert(0,close_price[2])
    #ma3_list.insert(0,close_price[1])
    #ma3_list.insert(0,close_price[0])
    #k_v2 = list(kalman_simple(near_list,close_price))
  #  k_v1 = list(kalman_simple(close_price[begin:],close_price[begin:]))
  #  k_v1=k_v_base+k_v1
    k_v = list(kalman_simple(close_price[begin:],close_price[begin:]))
  #  k_v=k_v_base+k_v
    k_v = list(kalman_simple(open_price,close_price))
    #k_v=k_v_base+k_v
    k_v2 =k_v
    k_v1 =k_v
   
##

    ma_list_diff = list(np.diff(ma_list))
    ma_list_diff.insert(0,0)
    #ma3_list=[]
    #for i in range(3):
    #    ma3_list.append(0)
    #for i in range(3,len(close_price)):
    #    ma3_list.append(np.mean(ma_list_diff[i-2:i+1]))



    pv=[close_price[i]*vol[i] for i in range(len(close_price))]

    clog = [np.log(i) for i in close_price]

    cdiff=list(np.diff(ma_list))
    cdiff.insert(0,0)

    cdiff=[]
    for i in range(len(close_price)):
        cdiff.append(close_price[i]-ma3_list[i])
 #   cdiff=list(np.diff(ma_list))
 ##   cdiff.insert(0,0)
    cdiff2=list(np.diff(cdiff))
    cdiff2.insert(0,0)


    diff_ma3_list=[]
    pp=3
    for i in range(pp):
        diff_ma3_list.append(0)
    for i in range(pp,len(close_price)):
        diff_ma3_list.append(np.mean(cdiff[i-(pp-1):i+1]))

    diff2_ma3_list=[]
    pp=5
    for i in range(pp):
        diff2_ma3_list.append(0)
    for i in range(pp,len(close_price)):
        diff2_ma3_list.append(np.mean(diff_ma3_list[i-(pp-1):i+1]))

#    diff_ma3_list = diff2_ma3_list

    #for i in range(begin,begin+int(process.file_count(emd_dir))-1):
    for i in range(begin,len(close_price)):
        print "\n\nemd file %s"%i
####
        print "close price%s"%close_price[-1]
        print "close price%s"%close_price[i]

#####
        datadata=(close_price[:i])

        maa5 = []
        for kk in range(60):
            maa5.append(0)
#
        for kk in range(60,len(datadata)):
            mean_5 = np.mean(datadata[kk-59:kk+1])
            maa5.append(mean_5)

        ma5 = []
        for kk in range(60):
            ma5.append(0)
#
        for kk in range(60,len(datadata)):
            mean_5 = np.mean(maa5[kk-59:kk+1])
            ma5.append(mean_5)

        maa5 = []
        for kk in range(60):
            maa5.append(0)
#
        for kk in range(60,len(datadata)):
            mean_5 = np.mean(ma5[kk-59:kk+1])
            maa5.append(mean_5)
#

        ma5 = []
        for kk in range(60):
            ma5.append(0)
#
        for kk in range(60,len(datadata)):
            mean_5 = np.mean(maa5[kk-59:kk+1])
            ma5.append(mean_5)

        qushi=maa5
        ma_max_index = list(argrelextrema(np.array(qushi[-500:]),np.greater)[0])
        ma_max_index1 = list(argrelextrema(np.array(qushi[-200:]),np.greater)[0])
        print " ma max index %s"%ma_max_index
        ma_min_index = list(argrelextrema(np.array(qushi[-500:]),np.less)[0])
        ma_min_index1 = list(argrelextrema(np.array(qushi[-200:]),np.less)[0])
        print " ma min index %s"%ma_min_index

        large_num=len(ma_max_index)+len(ma_min_index)
        small_num=len(ma_max_index1)+len(ma_min_index1)

        
        sel=1
        #if ma_max_index[-1]>ma_min_index[-1] and ma_max_index[-1]>496:#and len(ma5)-ma_min_index[-1]<=5 :#and ma_min_index[-1]-ma_max_index[-1]>=2:#and abs(ma_max_index[-1]-last_max)<2:
        #    ma_ex_de=2
#        if small_num>0 and ((large_num/float(small_num))>3):# or ((large_num/float(small_num))<2)):
        #if len(diff_tmp)>2 and ((diff_tmp[-1]+diff_tmp[-2]+diff_tmp[-3])/3.0)<(np.mean(diff_tmp)) :#or (i-tmp[-1])>(np.mean(diff_tmp)):#-1.5*np.std(diff_tmp)):
        if small_num!=0 and (float(large_num)/small_num)<2:
            sel=0

#####
        #pre_emd_data=close_price[i-200:i]
        #print "std emd%s"%np.std(close_price[i-30:i])
        #my_emd = one_dimension_emd(pre_emd_data)
        #(imf, residual) = my_emd.emd(0.01,0.01)
        #presnr.append(calc_SNR(pre_emd_data,imf)[0])

        #my_emd2 = one_dimension_emd(z[i-200:i])
        #(imf, residual) = my_emd2.emd(0.01,0.01)
        #presnr2.append(calc_SNR(z[i-200:i],imf)[0])
        pre_flag=0
        ##if np.mean(presnr[-3:])>np.mean(presnr[-6:-3]):
        #if len(presnr)>1 and (presnr[-1])>(presnr[-2]):
        ##if presnr[-1]>10 or presnr2[-1]>10:
        #    pre_flag=1
           
        print "presnr%s"%presnr[-10:]
        print "presnr%s"%presnr2[-10:]
        emd_data = ma[i-1000:i]
        emd_data2 = open_price[i-1000:i]
        emd_data3 = macd[i-1000:i]
        emd_data = close_price[i-1000:i]
        emd_data = [j for j in emd_data]
        #emd_data = ma2[i-1000:i]
        emd_data = ma[i-1000:i]
        emd_data2 = ma2[i-1000:i]
        emd_data = close_price[i-1000:i]
        emd_data = z[i-1000:i]
        emd_data2 = close_price[i-1000:i]
        emd_data3 = z2[i-1000:i]
        emd_data = close_price[i-1000:i]
        emd_data2 = ma[i-1000:i]
        emd_data = pv2[i-1000:i]
        emd_data = ma[i-1000:i]
        emd_data2 =maa[i-100:i]
        emd_data = close_price[i-500:i]

        ma5 = ma_func(close_price, 5, 0, i-1)
        ma10 = ma_func(close_price, 10, 0, i-1)
        ma20 = ma_func(close_price, 20, 0, i-1)
        ma30 = ma_func(close_price, 30, 0, i-1)
        print "ma%s %s %s %s"%(ma5,ma10,ma20,ma30)
        if 0:#ma5<ma10<ma20<ma30:
            emd_data = ma[i-500:i]
        else:
            emd_data=(close_price[i-500:i])
            #emd_data=(close_price[flag:i])
            #emd_data2=(close_price[flag:i])
            qushidata=qushi[i-500:i]
            emd_data=[emd_data[jj]-qushidata[jj] for jj in range(len(emd_data))]
            emd_data2=emd_data
            if ABL==0:
                if sel==1:
                    emd_data=(close_price[i-500:i])
                    emd_data2=(close_price[i-500:i])
                else:
                    emd_data=(close_price[i-200:i])
                    emd_data2=(close_price[i-200:i])
            elif ABL==200:
                emd_data=(close_price[i-200:i])
                emd_data2=(close_price[i-200:i])
            elif ABL==500:
                emd_data=(close_price[i-500:i])
                emd_data2=(close_price[i-500:i])
            emd_data=(close_price[i-500:i])
            emd_data2=(close_price[i-500:i])
            emd_data=(ma_list[i-500:i])
            emd_data=(close_price[i-500:i])
           # emd_data.append(open_price[i])
            emd_data=(close_price[i-500:i])
            emd_data=(diff_ma3_list[i-500:i])
            if  1:#ts.adfuller(close_price[i-10:i], 1)[1]>0.2:
                print "diff1"
                #emd_data=(cdiff[i-100:i])
                emd_data=(ma_list[i-800:i])
                emd_data=(close_price[i-200:i])
           #     emd_data=(vol[i-200:i])
           #     emd_data=(je[i-200:i])
           #     emd_data = ma2_list[i-200:i]
            else:
                print "diff2"
                emd_data=(ma_list[i-500:i])
            


            pp2=0
            chazhi=0 
            prechade=0
            #emd_data.append(pp1)
            emd_data2=(ma_list[i-500:i])
           # emd_data2=(close_price[i-500:i])
           # emd_data=(np.diff(emd_data))
           # emd_data=(np.diff(emd_data))
           # emd_data2=(np.diff(emd_data))
           # emd_data2=(np.diff(emd_data))
        emd_file = "%s/emd_%s"%(emd_dir,i)
        print "len emddata%s"%len(emd_data)
###

        kv=(k_v[i],k_v1[i],k_v2[i])

####
        process.run(emd_data,datafile,i-1,date[-1],emd_file,emd_data2,emd_data3,pre_flag,chazhi,prechade,kv,pp2)
    #process.show_success()
    #process.show_stat()
#    process.show_rss()
#    process.draw_fig(emd_data,datafile,begin)
    

            
