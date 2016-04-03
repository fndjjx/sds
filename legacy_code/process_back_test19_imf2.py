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
from calc_SNR import calc_SNR2
from calc_SNR import calc_SNR
from calc_match import matchlist2
from calc_match import matchlist3
from calc_match import matchlist1
from calc_match import matchlist4



class process_util():

    def __init__(self,open_price,close_price,macd,date,vol,high_price,low_price,je,kdjk,kdjd,wr):
        
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

        self.kdjk=kdjk
        self.kdjd=kdjd
        self.wr=wr
        
        self.total_asset_list = []
        self.total_asset = 0
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
        
        self.ma_index=[]
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




    def run_predict1(self,imf_list,current_price_index,date,datafile,residual,imf_open,imf_macd,sel_flag,emd_data,emd_data2):


        print "\n\n begin"
        
        for i in range(len(imf_list)):
            imfp = imf_list[i]
            data = np.array(imfp)
            max_index = list(argrelextrema(data,np.greater)[0])
            min_index = list(argrelextrema(data,np.less)[0])
            print "imf %s"%i
            print "max %s"%max_index
            print "min %s"%min_index

####
        imf1_max_index = list(argrelextrema(np.array(imf_list[1]),np.greater)[0])
        imf1_min_index = list(argrelextrema(np.array(imf_list[1]),np.less)[0])
        imf1_or_flag=0
        #if imf1_min_index[-1]>imf1_max_index[-1] and imf1_min_index[-1]>996:
        #    imf1_or_flag=1
        if imf1_min_index[-1]<imf1_max_index[-1] :#and imf1_max_index[-1]>=496:
            imf1_or_flag=2

###

        imf2_max_index = list(argrelextrema(np.array(imf_list[2]),np.greater)[0])
        #print " imf max index %s"%imf2_max_index
        imf2_min_index = list(argrelextrema(np.array(imf_list[2]),np.less)[0])
        #print " imf min index %s"%imf2_min_index
        imf2_or_flag=0
        if imf2_min_index[-1]>imf2_max_index[-1] and imf2_min_index[-1]>996 and imf2_min_index[-1]-imf2_max_index[-1]>3:
            imf2_or_flag=1
        if imf2_min_index[-1]<imf2_max_index[-1] and imf2_max_index[-1]>996 and imf2_max_index[-1]-imf2_min_index[-1]>3:
            imf2_or_flag=2

        #print "imf2 or flag%s"%imf2_or_flag
        #imf2_max_index = list(argrelextrema(np.array(imf_list[2][:-4]),np.greater)[0])
        #imf2_min_index = list(argrelextrema(np.array(imf_list[2][:-4]),np.less)[0])
        #if imf2_min_index[-1]<imf2_max_index[-1] and imf2_max_index[-1]<995 and imf2_max_index[-1]-imf2_min_index[-1]>3: 
        #    imf2_or_flag=11
        #if imf2_min_index[-1]>imf2_max_index[-1] and imf2_min_index[-1]<995 and imf2_min_index[-1]-imf2_max_index[-1]>3:
        #    imf2_or_flag=0
        #if imf2_min_index[-1]>imf2_max_index[-1] and imf2_min_index[-1]<990 :
        #    imf2_or_flag=1


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

        s=[emd_data2[i]-imf_list[0][i] for i in range(len(emd_data2))]
        s=[s[i]-imf_list[1][i] for i in range(len(emd_data2))]
        #s=imf_list[2]      
        ##process_data = self.close_price[current_price_index-99:current_price_index+1]
        data = np.array(s)
        max_index = list(argrelextrema(data,np.greater)[0])
        min_index = list(argrelextrema(data,np.less)[0])
        #print "s index%s"%max_index
        #print "s index%s"%min_index
        #tmp=max_index+min_index
        #tmp.sort()

        if 0:#tmp[-1]<270:
            s=imf_list[1]
            data = np.array(s)
            max_index = list(argrelextrema(data,np.greater)[0])
            min_index = list(argrelextrema(data,np.less)[0])
        decision=0
        if max_index[-1]>min_index[-1]:
            decision=0
        #elif max_index[-1]<min_index[-1] and min_index[-1]>980:
        else:
            decision=1
######

        imf0_max_index = list(argrelextrema(np.array(imf_list[0]),np.greater)[0])
        imf0_min_index = list(argrelextrema(np.array(imf_list[0]),np.less)[0])

        tmp = imf0_max_index+imf0_min_index
        tmp.sort()
        trust0=tmp[-2]

        imf1_max_index = list(argrelextrema(np.array(imf_list[1]),np.greater)[0])
        imf1_min_index = list(argrelextrema(np.array(imf_list[1]),np.less)[0])


        imf1_min_index=filter(lambda n:n<=trust0,imf1_min_index)
        imf1_max_index=filter(lambda n:n<=trust0,imf1_max_index)

        tmp = imf1_max_index+imf1_min_index
        tmp.sort()
        trust1=tmp[-2]
        imf1_min_index=filter(lambda n:n<trust0,imf1_min_index)
        imf1_max_index=filter(lambda n:n<trust0,imf1_max_index)
        tmp = imf1_max_index+imf1_min_index
        tmp.sort()
        trust11=tmp[-1]
        #use=2
        ##if trust11<993 and tmp[-1]-tmp[-2]>3:
        ##    use=1

        imf2_max_index = list(argrelextrema(np.array(imf_list[2]),np.greater)[0])
        imf2_min_index = list(argrelextrema(np.array(imf_list[2]),np.less)[0])


        imf2_min_index=filter(lambda n:n<=trust1,imf2_min_index)
        imf2_max_index=filter(lambda n:n<=trust1,imf2_max_index)

        tmp = imf2_max_index+imf2_min_index
        tmp.sort()
        trust2=tmp[-1]
        #trust22=tmp[-2]

        #imf3_max_index = list(argrelextrema(np.array(imf_list[2]),np.greater)[0])
        #imf3_min_index = list(argrelextrema(np.array(imf_list[2]),np.less)[0])


        #imf3_min_index=filter(lambda n:n<=trust2,imf3_min_index)
        #imf3_max_index=filter(lambda n:n<=trust2,imf3_max_index)

        #tmp = imf3_max_index+imf3_min_index
        #tmp.sort()
        #trust3=tmp[-1]


###############################################################

        #if len(imf_list)<2:
        #    imf_process = list(imf_list[0])
        #else:
        #    imf_process = list(imf_list[1])
        #imf_process = list(imf_list[2])
        #imf_process3 = list(imf_list[2])




#       # print "imf process len%s"%len(imf_process)
#       # #tmp = matchlist(imf_process,10,20)
#       # print "imf process%s"%imf_process[-10:]
#       # (tmp,extenflag1) = matchlist3(imf_process)
#       # print "tmp len%s"%len(tmp)
#       # imf_process=tmp
#       # print "imf process%s"%imf_process[-10:]



#       # tmp = imf_max_index+imf_min_index
#       # tmp.sort()
#       # i=1
#       # while tmp[-i]>trust:
#       #     i+=1
#       # print "use %s "%i

#       # if i==1 :#or tmp[-2]<990:
#       #     (imf_process,extenflag1,ex_num) = matchlist1(imf_process)
#       # elif i==2:
#       #     (imf_process,extenflag1,ex_num) = matchlist2(imf_process)
#       # else:
#       #     (imf_process,extenflag1,ex_num) = matchlist3(imf_process)


        imf2_max_index = list(argrelextrema(np.array(imf_list[2]),np.greater)[0])
        imf2_min_index = list(argrelextrema(np.array(imf_list[2]),np.less)[0])

        tmp = imf2_max_index+imf2_min_index
        tmp.sort()
        i=1
   
        while tmp[-i]>trust2:
            i+=1 

        if tmp[-i]<490:
            i-=1
        if tmp[-i]<490:
            i-=1
        if tmp[-i]>495:
            i+=1
        if i<0:
            i=1

        print "haha %s"%i


        if i==1 :#or tmp[-2]<990:
            (imf_process,extenflag1,ex_num,cha,cha2) = matchlist1(imf_list[2],3,emd_data)
        elif i==2:
            (imf_process,extenflag1,ex_num,cha,cha2) = matchlist2(imf_list[2],3,emd_data)
        else:
            (imf_process,extenflag1,ex_num,cha,cha2) = matchlist3(imf_list[2],3,emd_data)


        #print "imf2 cha %s"%cha2
#####



#####

        #last_power=imf_process[-2]-(np.mean(imf_process[-4:-1]))
        #current_power=imf_process[-1]-(np.mean(imf_process[-3:]))
        #power_de=0
        #if last_power<0 and current_power>0:
        #    power_de=1
        #if last_power>0 and current_power<0:
        #    power_de=2

        #

        imf_max_index = list(argrelextrema(np.array(imf_process),np.greater)[0])
        print " imf max index %s"%imf_max_index
        imf_min_index = list(argrelextrema(np.array(imf_process),np.less)[0])
        print " imf min index %s"%imf_min_index

####
        imf_value=[]
        for i in imf_max_index:
            imf_value.append(imf_process[i])

        print "std imf value%s"%np.std(imf_value[-10:])
        self.stdimf.append(np.std(imf_value[-10:]))

####
        #print "imf process %s"%imf_process[-10:]
        #print "imf process extenflag%s "%extenflag1
        #print imf_process[-1]
        #print "-1imf_min_index %s"%imf_process[imf_min_index[-1]]
        #print "-1imf_max_index %s"%imf_process[imf_max_index[-1]]

        #ex_flag=1
        #if ex_num>2:
        #    ex_flag=0

        #print "ex flag%s num %s"%(ex_flag,ex_num)

        #self.extenflag.append(extenflag1)
        #print "extenflag %s"%self.extenflag[-10:]
        #self.matchcha.append(cha)
        #print "matchcha%s"%self.matchcha[-10:]

        # 


        imf2_flag=0
        #print imf_min_index[-1]
        #print imf_max_index[-1]
        #last_max=current_price_index-(999-imf_max_index[-1])+1
        #if imf_min_index[-1]>imf_max_index[-1]:
        #    if imf_min_index[-1]-imf_max_index[-1]<3:
        #        imf_min_index.pop()
        #if imf_max_index[-1]>imf_min_index[-1]:
        #    if imf_max_index[-1]-imf_min_index[-1]<3:
        #        imf_max_index.pop()
        if imf_min_index[-1]>imf_max_index[-1] and imf_min_index[-1]>=496  :#and imf_min_index[-1]-imf_max_index[-1]>10:#and ex_flag==1:#and imf_process[-1]<0 :
            imf2_flag=1
        if imf_min_index[-1]<imf_max_index[-1] and imf_max_index[-1]>=496:
            imf2_flag=2


        #print "pwoer de %s"%power_de
##########
        imf1_max_index = list(argrelextrema(np.array(imf_list[1]),np.greater)[0])
        imf1_min_index = list(argrelextrema(np.array(imf_list[1]),np.less)[0])

        tmp = imf1_max_index+imf1_min_index
        tmp.sort()
        i=1
        while tmp[-i]>trust1:
            i+=1


        if tmp[-i]<490:
            i-=1
        if tmp[-i]<490:
            i-=1
        if tmp[-i]>496:
            i+=1
        if i<0:
            i=1

        print "3haha %s"%i

        if i==1 :#or tmp[-2]<990:
            (imf_process,extenflag1,ex_num,cha,cha2) = matchlist1(imf_list[1],2,emd_data)
        elif i==2:
            (imf_process,extenflag1,ex_num,cha,cha2) = matchlist2(imf_list[1],2,emd_data)
        else:
            (imf_process,extenflag1,ex_num,cha,cha2) = matchlist3(imf_list[1],2,emd_data)

        imf_max_index = list(argrelextrema(np.array(imf_process),np.greater)[0])
        print " imf1 max index %s"%imf_max_index
        imf_min_index = list(argrelextrema(np.array(imf_process),np.less)[0])
        print " imf1 min index %s"%imf_min_index



        imf1_flag=0
        if (imf_min_index[-1]>imf_max_index[-1] and imf_min_index[-1]>496) :#and imf1_or_flag!=2) or (imf1_or_flag==1) :#and ex_flag==1:#and imf_process[-1]<0 :
            imf1_flag=1
        if (imf_min_index[-1]<imf_max_index[-1] and imf_max_index[-1]>=496 ) :#or (imf1_or_flag==2):
            imf1_flag=2


############
        #ma3 = self.ma(self.close_price, 3, 0, current_price_index)
        #ma5 = self.ma(self.close_price, 5, 0, current_price_index)
        #current_price = self.close_price[current_price_index]

        #imf1_max_index = list(argrelextrema(np.array(imf_list[1]),np.greater)[0])
        #imf1_min_index = list(argrelextrema(np.array(imf_list[1]),np.less)[0])

        #tmp = imf1_max_index+imf1_min_index
        #tmp.sort()
        #i=1
        #while tmp[-i]>trust11:
        #    i+=1

        #print "1haha %s"%i

        #if i==1 :#or tmp[-2]<990:
        #    (imf_process,extenflag1,ex_num,cha,cha2) = matchlist1(imf_list[1],3,emd_data)
        #elif i==2:
        #    (imf_process,extenflag1,ex_num,cha,cha2) = matchlist2(imf_list[1],3,emd_data)
        #else:
        #    (imf_process,extenflag1,ex_num,cha,cha2) = matchlist3(imf_list[1],3,emd_data)

        #imf_max_index = list(argrelextrema(np.array(imf_process),np.greater)[0])
        #print " imf1 max index %s"%imf_max_index
        #imf_min_index = list(argrelextrema(np.array(imf_process),np.less)[0])
        #print " imf1 min index %s"%imf_min_index

        #print "cha2 %s"%cha2

        #imf1_flag=0
        #if (imf_min_index[-1]>imf_max_index[-1] and imf_min_index[-1]>996 and cha2>1) or (imf1_or_flag==1):#len(imf_process)-5 and  extenflag1<2:#and imf_process[-1]<0 :
        #    imf1_flag=1
        #if imf_max_index[-1]>imf_min_index[-1] and imf_max_index[-1]<991 and cha2>1 :#len(imf_process)-5 and  extenflag1<2:#and imf_process[-1]<0 :
        #    imf1_flag=1
        #if imf_min_index[-1]<imf_max_index[-1] and imf_max_index[-1]>996:#=len(imf_process)-5:
        #    imf1_flag=2
        #if imf_min_index[-1]>imf_max_index[-1] and imf_min_index[-1]<991 and current_price<ma5:#len(imf_process)-5 and  extenflag1<2:#and imf_process[-1]<0 :
        #    imf1_flag=2


        print "imf2flag %s"%imf2_flag
##############


  
###########
        last_max=0
        decision=0
        if sel_flag==2:
        #    #if ((imf2_flag==1 and imf1_flag==1 and imf2_or_flag==11) or (imf2_or_flag==1 and imf1_flag==1)) and (imf0_flag==1) :#and imf2_flag6==1:#or self.close_flag==1:#and imf2_open_flag!=2:
        #    #if ((imf2_flag==1 and imf2_or_flag!=2) or imf2_or_flag==1):# and len(self.imf_flag_list)>1 and self.imf_flag_list[-1]==1:#and  imf2_or_flag!=2:# and imf1_flag==1:#and imf2_flag6==1:#or self.close_flag==1:#and imf2_open_flag!=2:
            if imf2_flag==1  and imf1_flag!=2:#and imf1_flag==1:# and len(self.imf_flag_list)>1 and self.imf_flag_list[-1]==1:#and  imf2_or_flag!=2:# and imf1_flag==1:#and imf2_flag6==1:#or self.close_flag==1:#and imf2_open_flag!=2:
        #    #if  imf2_flag==1 and imf2_flag_open_flag!=2 and imf2_macd_flag!=2 and imf2_or_flag!=2:#and imf2_open_flag==1:#and (imf_process3[-1]<0 or imf_process[-1]<0) :#and (extenflag1+extenflag3)<3:#and imf2_open_flag!=2:
#       #     if (imf2_flag==1 and self.open_flag==1) or (imf2_flag==1 and imf2_macd_flag==1) or (imf2_macd_flag==1 and self.close_flag==1) or (imf2_macd_flag==1 and imf2_flag==1):#and imf2_macd_flag==1:#and power_de==1:#and imf3_flag==1:
                decision=1 
        #    #elif ((imf2_flag==1 and  imf2_or_flag==11 ) or (imf2_or_flag==1)) and (imf0_flag==1) :#and imf2_flag6==1:#or self.close_flag==1:#and imf2_open_flag!=2:
        #    #    decision=3 
        #    #if (imf2_flag==2 or imf2_macd_flag==2) and  imf2_or_flag==2 :#and imf2_open_flag!=2:
            if imf2_flag==2 :#and imf2_open_flag==2 :
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
        return (decision,last_max)
        #return (imf1_flag,last_max)
        #return imf2_flag
##########################



###########################
        
    def run_predict(self,imf_list,current_price_index,date,datafile,residual,emd_data,imf_open,imf_macd,emd_std,preflag,emd_data2):
        (imf_close_flag,last_max)=self.run_predict1(imf_list,current_price_index,date,datafile,residual,imf_open,imf_macd,2,emd_data,emd_data2)
        #(imf_high_flag,last_max)=self.run_predict1(imf_open,current_price_index,date,datafile,residual,imf_open,imf_macd,2,emd_data)
        #(imf_5_flag,last_max)=self.run_predict1(imf_macd,current_price_index,date,datafile,residual,imf_open,imf_macd)
        imf_high_flag=1
#        imf_close_flag=1
#        last_max=1

        self.snr.append(calc_SNR(emd_data,imf_list))
        print "SNR%s"%self.snr[-10:]
        imf_flag=0
        #if (imf_close_flag==1):
        #if self.buy_price==0 and (imf_close_flag==1 or (len(self.close_flag)>0 and (self.close_flag[-1]==1) ) or ( len(self.close_flag)>1 and self.close_flag[-1]!=2 and self.close_flag[-2]==1) or ( len(self.close_flag)>2 and self.close_flag[-3]==1 and self.close_flag[-1]!=2 and self.close_flag[-2]!=2) or (len(self.close_flag)>3 and (self.close_flag[-4]==1) and self.close_flag[-3]!=2 and self.close_flag[-2]!=2 and self.close_flag[-1]!=2)) :#and ((len(self.high_flag)>0 and self.high_flag[-1]==1) or imf_high_flag==1):
        if imf_close_flag==1 :#and (len(self.close_flag)>0 and (self.close_flag[-1]!=2) and (self.close_flag[-2]!=2) ) :#or ( len(self.close_flag)>1 and self.close_flag[-1]!=2 and self.close_flag[-2]==1) or ( len(self.close_flag)>2 and self.close_flag[-3]==1 and self.close_flag[-1]!=2 and self.close_flag[-2]!=2) or (len(self.close_flag)>3 and (self.close_flag[-4]==1) and self.close_flag[-3]!=2 and self.close_flag[-2]!=2 and self.close_flag[-1]!=2)) :#and ((len(self.high_flag)>0 and self.high_flag[-1]==1) or imf_high_flag==1):
        #if self.buy_price==0 and (imf_close_flag==1 or imf_high_flag==1) :#or (imf_high_flag==1 and imf_close_flag!=2):
            imf_flag=1
#        if imf_high_flag==1 and (len(self.close_flag)>1 and ( self.close_flag[-1]!=2 and self.close_flag[-2]!=2 and imf_close_flag!=2)):
#            imf_flag=1
        if (imf_close_flag==2):# and imf_high_flag!=1) or (imf_high_flag==2 and imf_close_flag!=1):#and imf_high_flag==2:
        #if  (imf_close_flag==2 ) :#or (imf_high_flag==2 and imf_close_flag!=1):#and imf_high_flag==2:
            imf_flag=2
        

        self.close_flag.append(imf_close_flag)
        self.high_flag.append(imf_high_flag)
        print "imf flag %s %s"%(imf_close_flag,imf_high_flag)
        print "imf flag %s"%imf_flag
        print "high list%s"%self.high_flag[-5:]
        print "close list%s"%self.close_flag[-5:]


######
        ma5 = []
        kkk=5
        for i in range(kkk):
            ma5.append(0)
#
        for i in range(kkk,current_price_index+1):
            mean_5 = np.mean(self.close_price[i-kkk+1:i+1])
            ma5.append(mean_5)

        print self.close_price[i]
        print "ma5 %s"%ma5[-10:]
        ma_max_index = list(argrelextrema(np.array(ma5[-20:]),np.greater)[0])
        print " ma max index %s"%ma_max_index
        ma_min_index = list(argrelextrema(np.array(ma5[-20:]),np.less)[0])
        print " ma min index %s"%ma_min_index
        print len(ma_max_index)+len(ma_min_index)
        print "len max%s"%len(ma_max_index)


        print "last max%s"%last_max

        ma_ex_de=0


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


        pvma3=(emd_data[-1]+emd_data[-2]+emd_data[-3])/3.0
        pvma5=(emd_data[-1]+emd_data[-2]+emd_data[-3]+emd_data[-4]+emd_data[-5])/5.0
        pvma5_before=(emd_data[-6]+emd_data[-2]+emd_data[-3]+emd_data[-4]+emd_data[-5])/5.0
        pvma5_before2=(emd_data[-7]+emd_data[-6]+emd_data[-3]+emd_data[-4]+emd_data[-5])/5.0
        pvma3_before=(emd_data[-2]+emd_data[-3]+emd_data[-4])/3.0
        pvma3_before2=(emd_data[-3]+emd_data[-4]+emd_data[-5])/3.0

        pvma10=np.mean(emd_data[-10:])
        pvma10_before=np.mean(emd_data[-11:-1])
        pvma10_before2=np.mean(emd_data[-12:-2])
        pvdeltama10=pvma10-pvma10_before
        pvdeltama10_before=pvma10_before-pvma10_before2
        pvdeltama3=pvma3-pvma3_before
        pvdeltama3_before=pvma3_before-pvma3_before2
        pvpower_de=0
        if emd_data[-1]-pvma3>0 and emd_data[-2]-pvma3_before<0 and pvdeltama3<0:
            pvpower_de=1

        print "pv ma3 %s %s %s"%(pvma3_before2,pvma3_before,pvma3)
        print "pv ma5 %s %s %s"%(pvma5_before2,pvma5_before,pvma5)
        print "pv ma10 %s %s %s"%(pvma10_before2,pvma10_before,pvma10)
        print "pv deltama10 %s %s"%(pvdeltama10_before,pvdeltama10)
        print "pv deltama3 %s %s"%(pvdeltama3_before,pvdeltama3)
        print "pv %s"%emd_data[-10:]
           

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

        #if self.snr[-1]<np.mean(self.snr[-20:]) and self.snr[-1]<np.mean(self.snr[-5:]):

        if self.one_time_day>0:
            self.one_time_day+=1

        
        distance_decision = 1
        if len(self.sell_x_index)>0 and current_price_index-self.sell_x_index[-1]<3:
            distance_decision = 0

        distance_decision2 = 1
        if len(self.buy_x_index)>0 and current_price_index-self.buy_x_index[-1]<3:
            distance_decision2 = 0

        distance_decision3 = 1
        if len(self.buy_x_index)>0 and current_price_index-self.buy_x_index[-1]<1:
            distance_decision2 = 0

        if ((self.sell_flag==2 and imf_flag!=2) or (self.sell_flag==2 and imf_flag==2 and current_price<self.close_price[current_price_index-1])) and  self.buy_price!=0:
            self.sell_flag2=2

        emd_de=0
        if emd_data[-1]-emd_data[-2]>0 and (emd_data[-2]-emd_data[-3])<0:
            emd_de=1
        if emd_data[-1]-emd_data[-2]>0 and (emd_data[-2]-emd_data[-3])>0 and (emd_data[-3]-emd_data[-4])<0:
            emd_de=1
        if emd_data[-1]-emd_data[-2]>0 and (emd_data[-2]-emd_data[-3])>0 and (emd_data[-3]-emd_data[-4])>0 and (emd_data[-4]-emd_data[-5])<0:
            emd_de=1
        if emd_data[-1]-emd_data[-2]>0 and (emd_data[-2]-emd_data[-3])>0 and (emd_data[-3]-emd_data[-4])>0 and (emd_data[-4]-emd_data[-5])>0 and (emd_data[-5]-emd_data[-6])>0:
            emd_de=1

        if min(emd_data[-5:])>0:
            emd_de=0


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

        tmp=self.close_price
        self.close_price=ma5_list
        current_price=self.close_price[current_price_index]
        

        
        period=5
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

        boll_de=0
        if current_price>boll_high1 :#and current_price>boll_low2:
            boll_de=2
        elif current_price<boll_low1 :#and :
            boll_de=1

        #boll_de=0
        #if self.close_price[current_price_index-1]<boll_low1_b and current_price>boll_low1:
        #    boll_de=1
        ####if self.close_price[current_price_index-1]<boll_high1_b and current_price>boll_high1:
        ####    boll_de=1
        ##if self.close_price[current_price_index-1]>boll_low1_b and current_price<boll_low1:
        ##    boll_de=2
        #if self.close_price[current_price_index-1]>boll_high1_b and current_price<boll_high1:
        #    boll_de=2

        print "low1 low2%s %s"%(boll_low1,boll_low2)
        print "current %s"%current_price
        print "width %s"%(boll_high1-boll_low1)
        print "widthb %s"%(boll_high1_b-boll_low1_b)
        print "widthb2 %s"%(boll_high1_b2-boll_low1_b2)
        self.close_price=tmp
        current_price=self.close_price[current_price_index]


        width_b2=boll_high1_b2-boll_low1_b2
        width_b1=boll_high1_b-boll_low1_b
        width=boll_high1-boll_low1


        mean1=(width+width_b1)/2.0
        mean2=(width_b1+width_b2)/2.0

        down_de=1
        if mean1<mean2 and (deltama3<deltama3_before and deltama3<0) and deltama30>0:
            down_de=0
               


        ma_de=1
        if ma5<ma10<ma20<ma30:
            ma_de=0

        print "boll %s"%boll_de
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
        if  ((current_k+before_k)/2.0)<35 :#and (current_k-before_k)>(before_k-before_k2):#and current_k>current_d:
        #if current_k>before_k:
            kdj_de=1
        print "kdj_de %s"%kdj_de
        print "wr%s"%self.wr[current_price_index-5:current_price_index+1]

        print "current_price_index%s"%current_price_index

####
        wrma3=(self.wr[current_price_index]+self.wr[current_price_index-1]+self.wr[current_price_index-2])/3.0
        wr_de=0
        #if self.wr[current_price_index]>50 and self.wr[current_price_index]<self.wr[current_price_index-1]:
        #if wrma3>60:
        if wrma3>60:
            wr_de=1
####
        
        print "macd %s"%self.macd[current_price_index-10:current_price_index+1]
        #if self.buy_price!=0  and imf_flag!=1 and (imf_flag==2 or (deltama5<0 and deltama5_before>0) or (deltama3<0 and deltama3_before>0)) and distance_decision2==1:
        print "emd %s"%emd_data[-10:]
        #if  self.buy_price!=0   and ( (deltama5_before2>deltama5_before>deltama5 ))  and ((current_price-self.buy_price)/self.buy_price)>0.01:
        flag_fail=0
        if  self.buy_price!=0  and (boll_de==2 ) :#(imf_flag==2) and deltama3<deltama3_before  :# ((current_price-self.buy_price)/self.buy_price)>0.01:#((deltamacd10<0 and deltamacd10_b>0)  ) and ((current_price-self.buy_price)/self.buy_price)>0.01:#(current_price<ma5 )and distance_decision2==1 and ((current_price-self.buy_price)/self.buy_price)>0.01:
        #if  self.buy_price!=0  and deltama5<0:
        #if self.buy_price!=0  and self.macd[current_price_index]<self.macd[current_price_index-1]<self.macd[current_price_index-2] and distance_decision2==1:
        #if self.buy_price!=0 and deltama3_before>0 and deltama3<0 and distance_decision2==1 :#and current_price-self.buy_price>0:#emd_data[-1]<emd_data[-2] and emd_data[-2]>emd_data[-3] :#mama3<mama3_b and mama3_b>mama3_b2 and current_price-self.buy_price>0 and self.buy_price!=0:
 
            if self.fail_flag==0:
                print "sell sell 1"
                self.money = self.sell(self.share,trade_price)
                self.share = 0
                self.total_asset = self.money - self.sell_fee(self.money)
                self.money = self.total_asset
                self.sell_x_index.append(current_price_index)
                self.sell_y_index.append(current_price)
                if self.buy_price>=trade_price:
                    self.fail_flag=flag_fail
                else:
                    self.fail_flag=0
                self.buy_price = 0
                self.buy_mean_price = 0
                self.one_time_day=0
                self.sell_flag=0
                self.sell_flag2=0
            else:
                if self.buy_price>=trade_price:
                    self.fail_flag=1
                else:
                    self.fail_flag=0
                self.buy_price=0
                print "sellsell 1"
                
        elif   0:#self.buy_price!=0 and ma3<boll_low1 and  distance_decision2==1:#((((self.buy_mean_price-current_mean_price)/self.buy_mean_price)>0.01 ) )  and self.share > 0 and np.mean(self.snr[-3:])<np.mean(self.snr[-4:-1]): #and (self.buy_price-current_price)<0 and self.share > 0:
            print "sell sell 2"
            if self.fail_flag==0:
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
        elif self.buy_price!=0 and ((((self.buy_price-current_price)/self.buy_price)>0.05 ) )  :#and self.share > 0 :
            print "sell sell 4"
            if self.fail_flag==0:
                if self.buy_price>=trade_price:
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
        elif  wr_de==1  and self.buy_price==0 and (boll_de==1 ) and imf_flag==1 and distance_decision==1:#and deltamacdma5_before<deltamacdma5:#:and deltama30>0:#emd_de==1 and (deltama3>0) and deltama10>0:# and deltama30<0 and deltama10>0) :#deltama3_before :#and mamama3>mamama3_b :#and mama3_b2<mama3_b3:#and self.macd[current_price_index]>self.macd[current_price_index-1] :#and self.macd[current_price_index-3]>self.macd[current_price_index-2] :#and deltamacdma5_before<deltamacdma5 and deltamacdma5_before<deltamacdma5_before2:#and pvpower_de==1 :#emd_data[-1]>emd_data[-2] and emd_data[-2]<emd_data[-3]:#self.close_price[current_price_index]>self.close_price[current_price_index-1]:#and (deltama3_before<deltama3 and deltama3_before<0 and deltama3>0) :#and ma_ex_de==1:#and distance_decision==1 and deltama3_before<deltama3 and deltama3_before<deltama3_before2 :#and deltama3_before2<deltama3_before3 :#and ma_ex_de==1:# and distance_decision==1 :# and self.close_price[current_price_index]>self.close_price[current_price_index-1]:#deltama3_before<deltama3:#self.macd[current_price_index-1]<self.macd[current_price_index] :#and (self.macd[current_price_index-2]<0 or self.macd[current_price_index-1]<0 or self.macd[current_price_index]<0):#and deltama5>=0 :#and deltama3_before<deltama3 ) or (deltama3>0 and deltama3_before<0)):# and  mama3>mama3_b and mama3_b<mama3_b2 and distance_decision==1:#and deltama5<0:#and self.close_price[current_price_index]>self.close_price[current_price_index-1] and deltama5_before<deltama5<0: #and deltama3_before<0 and deltama3>0 and deltama5_before<deltama5<0:#and ma_decision==1 :#and distance_decision==1:#deltama3_before<deltama3:# (deltama3_before>deltama3 or deltama3<0):
        #elif    (  (imf_flag==1 ) ) :#and  self.buy_price==0 and deltama3>deltama3_before :#and deltama3>deltama3_before:#and pvpower_de==1 :#emd_data[-1]>emd_data[-2] and emd_data[-2]<emd_data[-3]:#self.close_price[current_price_index]>self.close_price[current_price_index-1]:#and (deltama3_before<deltama3 and deltama3_before<0 and deltama3>0) :#and ma_ex_de==1:#and distance_decision==1 and deltama3_before<deltama3 and deltama3_before<deltama3_before2 :#and deltama3_before2<deltama3_before3 :#and ma_ex_de==1:# and distance_decision==1 :# and self.close_price[current_price_index]>self.close_price[current_price_index-1]:#deltama3_before<deltama3:#self.macd[current_price_index-1]<self.macd[current_price_index] :#and (self.macd[current_price_index-2]<0 or self.macd[current_price_index-1]<0 or self.macd[current_price_index]<0):#and deltama5>=0 :#and deltama3_before<deltama3 ) or (deltama3>0 and deltama3_before<0)):# and  mama3>mama3_b and mama3_b<mama3_b2 and distance_decision==1:#and deltama5<0:#and self.close_price[current_price_index]>self.close_price[current_price_index-1] and deltama5_before<deltama5<0: #and deltama3_before<0 and deltama3>0 and deltama5_before<deltama5<0:#and ma_decision==1 :#and distance_decision==1:#deltama3_before<deltama3:# (deltama3_before>deltama3 or deltama3<0):
            if self.fail_flag!=0:
                #self.fail_flag-=1
                self.buy_price=trade_price
                print "buy buy 2"
            else:
                print "buy buy 1"
                self.ma_index.append(len(ma_max_index))
                self.share = self.buy(self.money,trade_price)
                self.money = 0
                self.buy_price = trade_price
                self.buy_mean_price = current_mean_price
                self.total_asset = self.share*trade_price
                self.buy_x_index.append(current_price_index)
                self.buy_y_index.append(current_price)
                self.one_time_day=1
                self.last_buy_result=0
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
            self.buy_day += 1

        if imf_flag==2:
            self.sell_flag=2
        else:
            self.sell_flag=0



        print "buy day%s"%self.buy_day

            

        print "maindex%s"%self.ma_index

  
        self.total_asset_list.append(self.total_asset)
        print "total asset is %s"%(self.total_asset)
        print "money is %s"%(self.money)
        print "share is %s"%(self.share)




    def format_data(self, data):
        if isinstance(data,list):
            return [2*((float(data[i])-min(data))/(max(data)-float(min(data))))-1 for i in range(len(data))]



    def run(self,emd_data, datafile,current_index,date,emd_file,emd_data2,emd_data3,preflag):
        starttime = datetime.datetime.now()
        tmp = [i for i in emd_data]
        emd_data = tmp
        print "\n\n\n"
        print "emd data std %s"%(np.std(emd_data[-20:])/(np.mean(emd_data[-20:])))
        
        print "emd data %s"%emd_data[-10:]


        my_emd = one_dimension_emd(emd_data,4)
        (imf, residual) = my_emd.emd(0.03,0.03)

        #my_emd2 = one_dimension_emd(emd_data,3)
        #(imf_open, residual) = my_emd2.emd(0.03,0.03)

#        my_emd3 = one_dimension_emd(emd_data3,9)
#        (imf_macd, residual) = my_emd3.emd(0.03,0.03)


        print "len imf %s"%len(imf)
            
        imf_open=imf
        imf_macd=imf_open

        std = (np.std(emd_data[-30:])/(np.mean(emd_data[-30:])))
        self.emdstd.append(std)
        print "emdstd%s"%self.emdstd[-10:]
        if len(self.emdstd)>10:
            print "stdma3 %s %s"%(np.mean(self.emdstd[-10:-5]),np.mean(self.emdstd[-5:]))
        
        self.run_predict(imf,current_index,date,datafile,residual,emd_data,imf_open,imf_macd,std,preflag,emd_data2)
        
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
        
        data = self.vol

        pv2=[self.close_price[i]*self.vol[i] for i in range(len(self.close_price))]
        pv = pvpv(self.close_price,self.vol)
####
        maa5 = []
        for i in range(60):
            maa5.append(0)
#
        for i in range(60,len(data)):
            mean_5 = np.mean(self.close_price[i-59:i+1])
            maa5.append(mean_5)
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
#
#
        data = ma5

        data = self.close_price

        my_emd = one_dimension_emd(data)
        (imf, residual) = my_emd.emd(0.1,0.1)
        imf = imf[2]



        print "stdimf%s"%self.stdimf

        
        imf_buy_value = [imf[i] for i in self.buy_x_index]
        imf_sell_value = [imf[i] for i in self.sell_x_index]


        buy_x = [i-start for i in self.buy_x_index]
        sell_x = [i-start for i in self.sell_x_index]
        data1 = ma5
        data = self.close_price
        data2= maa5
        plt.figure(1)
        plt.subplot(411).axis([start,len(data),min(data[start:]),max(data[start:])])      
        plt.plot([i for i in range(len(data))],data,'o',[i for i in range(len(data))],data,'b',self.buy_x_index,self.buy_y_index,'r*',self.sell_x_index,self.sell_y_index,'g*')             
        plt.subplot(412).axis([start,len(data),min(data1[start:]),max(data1[start:])])      
        plt.plot([i for i in range(len(data1))],data1,'o',[i for i in range(len(data1))],data1,'b')             
        plt.subplot(413).axis([start,len(data),min(data2[start:]),max(data2[start:])])      
        plt.plot([i for i in range(len(data2))],data2,'o',[i for i in range(len(data2))],data2,'b')             
        plt.subplot(414).axis([start,len(imf),min(imf[start:]),max(imf[start:])])
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



def format_data(data):
    if isinstance(data,list):
        return [100*((float(data[i])-min(data))/(max(data)-float(min(data)))) for i in range(len(data))]


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
        print "wri %s"%i
        print cn
        hn=max(h[i-(n):i+1])
        ln=min(l[i-(n):i+1])
        wr.append(100*(hn-cn)/(hn-ln))

    return wr

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
        print "i%s"%i
        print "cn%s"%cn
        print i-n+1
        print i
        hn=max(h[i-(n-1):i])
        print h[i-(n-1):i]
        print l[i-(n-1):i]
        print hn
        ln=min(l[i-(n-1):i])
        print ln
        print (float(cn-ln)/(hn-ln))
        print (2/3.0)*k[-1]

        rsv=100*(float(cn-ln)/(hn-ln))
        print rsv
        print (1/3.0)*rsv
        print "k%s"%k[-5:]
        print "d%s"%d[-5:]
        k.append((2/3.0)*k[-1]+(1/3.0)*rsv)
        d.append((2/3.0)*d[-1]+(1/3.0)*k[-1])

    
    return (k,d)
    
def ma_func(data, period, start, current_index):
    sum_period = 0
    for i in range(period):
        sum_period += data[current_index + start - i]
    return float(sum_period)/period


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
    
    (k,d)=kdj_func(close_price,high_price,low_price,9)
    wr=wr_func(close_price,high_price,low_price,10)
    process = process_util(open_price,close_price,macd,date,vol,high_price,low_price,je,k,d,wr)


    emd_dir = os.path.abspath('.')+"/"+"emd_data"
    print emd_dir
    os.popen("rm -r %s"%emd_dir)
    os.popen("mkdir %s"%emd_dir)
    generate_emd_func = generateEMDdata(datafile,begin,1000,emd_dir)


    #generate_emd_func.generate_machao_emd_data_fix(3)
    generate_emd_func.generate_ma_emd_data_fix(5)
    #generate_emd_func.generate_emd_data_fix()
###
    period=5
 
    #close_price=[j*10 for j in macd]
    pv=[close_price[i]*vol[i] for i in range(len(close_price))]
    pv = pvpv(close_price,vol)
    ma = []
    for ii in range(period):
        ma.append(0)
    for ii in range(period,len(close_price)):
        ma.append(np.mean(close_price[ii-(period-1):ii+1]))

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
    f=format_data(close_price)
    for i in range(begin,begin+int(process.file_count(emd_dir))-1):
        print "\n\nemd file %s"%i
####
        print "close price%s"%close_price[-1]
        print "close price%s"%close_price[i]



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
            #emd_data=close_price[i-500:i]
            emd_data=f[i-500:i]
        emd_file = "%s/emd_%s"%(emd_dir,i)
        print "len emddata%s"%len(emd_data)
###


####
        process.run(emd_data,datafile,i-1,date[-1],emd_file,emd_data2,emd_data3,pre_flag)
    process.show_success()
    process.show_stat()
#    process.show_rss()
    process.draw_fig(datafile,begin)
    

            
