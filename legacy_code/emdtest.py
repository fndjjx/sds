
import numpy as np
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


def ma(data, period, start, current_index):
    sum_period = 0
    for i in range(period):
        sum_period += data[current_index + start - i]
    return float(sum_period)/period

if __name__ == "__main__":

    datafile = sys.argv[1]
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
    #    macd.append(float(eachline.split("\t")[7]))
    #    open_price.append(float(eachline.split("\t")[1]))
    #    date.append(eachline.split("\t")[0])
    #    vol.append(float(eachline.split("\t")[5]))
    #    je.append(float(eachline.split("\t")[6]))

    
    data = close_price
    data1 = open_price
    data3 = high_price
    data4 = low_price
    data2 = data1
    data7 = [(data3[i]+data4[i]+2*data[i])/4.0 for i in range(len(data))]
    ma5=[]
    for i in range(2):
        ma5.append(0)
    for i in range(2,len(data)):
        ma5.append(np.mean(data[i-1:i+1]))

    data3=ma5[-50:] 
    data2=close_price[-100:]
    #data4 = [data2[i]-data3[i] for i in range(len(data2))] 
    #data2 = [data[i]*vol[i] for i in range(len(data))]
    my_emd = one_dimension_emd(data2,3)
    (imf, residual) = my_emd.emd(0.002,0.002)

    imf_max_index = list(argrelextrema(np.array(imf[2]),np.greater)[0])
    imf_min_index = list(argrelextrema(np.array(imf[2]),np.less)[0])

    ex_index = imf_max_index + imf_min_index
    ex_index.sort()
    period_index = np.diff(ex_index)
    print "ex index%s"%np.diff(ex_index)
    print np.mean(period_index)
    print np.std(period_index)

    imf1_max_index = list(argrelextrema(np.array(imf[1]),np.greater)[0])
    imf1_min_index = list(argrelextrema(np.array(imf[1]),np.less)[0])
    f_num=0
    f_nt_num=0
    t_nf_num=0
    t_num=0
    both_num=0
    exten_min=[]
    exten1_min=[]
    for i in imf_min_index:
        exten_min.append(i)
        exten_min.append(i-1)
#        exten_min.append(i-2)
        if i<len(data)-2:
            exten_min.append(i+1)
#            exten_min.append(i+2)
    for i in imf1_min_index:
        exten1_min.append(i)
        exten1_min.append(i-1)
        exten1_min.append(i+1)
    print len(imf_min_index)

    #imf_min_index=exten_min
    for i in (imf_min_index ):
        deltama5 = ma(data, 5, 0, i)-ma(data, 5, -1, i)
        deltama5_before = ma(data, 5, -1, i)-ma(data, 5, -2, i)

        deltama3 = ma(data, 3, 0, i)-ma(data, 3, -1, i)
        deltama3_before = ma(data, 3, -1, i)-ma(data, 3, -2, i)
        deltama3_before2 = ma(data, 3, -2, i)-ma(data, 3, -3, i)
        deltama3_before3 = ma(data, 3, -3, i)-ma(data, 3, -4, i)

        if deltama5>deltama5_before:
            f_num+=1
        #if deltama5>0 and deltama5_before<0 :#and deltama3_before2<deltama3_before3:
        #if (data[i]-ma(data, 3, 0, i))>0 and (data[i-1]-ma(data, 3, -1, i))<0:
        #if ((data[i]-ma(data, 3, 0, i))>0 and (data[i-1]-ma(data, 3, -1, i))<0) or (data[i]>data[i-1]) or  ((data[i]-ma(data, 5, 0, i))>0 and (data[i-1]-ma(data, 5, -1, i))<0):
        #if (((data[i]-ma(data, 3, 0, i))>0 and (data[i-1]-ma(data, 3, -1, i))<0) or (data[i]>data[i-1])) and ((((data2[i]-ma(data2, 3, 0, i))>0 and (data2[i-1]-ma(data2, 3, -1, i))<0) or (data2[i]>data2[i-1]))):
        #if ((((data2[i]-ma(data2, 3, 0, i))>0 and (data2[i-1]-ma(data2, 3, -1, i))<0))) :#or (data2[i]>data2[i-1]))):
        #if  (data2[i]>data2[i-1]) and   data[i]>data[i-1]:
        if (data2[i]-ma(data2, 3, 0, i))>0 and (data2[i-1]-ma(data2, 3, -1, i))<0 and (ma(data2, 3, 0, i)-ma(data2, 3, -1, i)<0):
            t_num+=1
        if deltama5>deltama5_before and deltama3>deltama3_before:
            both_num+=1
        if deltama5>deltama5_before and deltama3<deltama3_before:
            f_nt_num+=1
        if deltama5<deltama5_before and deltama3>deltama3_before:
            t_nf_num+=1

    print "ma3 result%s"%(float(t_num)/len(imf_min_index))
    print "ma5 result%s"%(float(f_num)/len(imf_min_index))
    print "both result%s"%(float(both_num)/len(imf_min_index))

    imf_min_index=exten_min
    f_num=0
    f_nt_num=0
    t_nf_num=0
    t_num=0
    both_num=0
    for i in imf_max_index:
        deltama5 = ma(data, 5, 0, i)-ma(data, 5, -1, i)
        deltama5_before = ma(data, 5, -1, i)-ma(data, 5, -2, i)

        deltama3 = ma(data, 3, 0, i)-ma(data, 3, -1, i)
        deltama3_before = ma(data, 3, -1, i)-ma(data, 3, -2, i)
        deltama3_before2 = ma(data, 3, -2, i)-ma(data, 3, -3, i)
        deltama3_before3 = ma(data, 3, -3, i)-ma(data, 3, -4, i)

        if deltama5<deltama5_before:
            f_num+=1
        #if deltama3<deltama3_before or deltama3_before<deltama3_before2:
        #if ((data[i]-ma(data, 3, 0, i))<0 and (data[i-1]-ma(data, 3, -1, i))>0) :
        if data2[i]<data2[i-1] :
            t_num+=1
        if deltama5<deltama5_before and deltama3<deltama3_before:
            both_num+=1

    print "ma3 result%s"%(float(t_num)/len(imf_max_index))
    print "ma5 result%s"%(float(f_num)/len(imf_max_index))
    print "both result%s"%(float(both_num)/len(imf_max_index))

    plt.figure(1)
    num_fig = len(imf)
    plt.subplot(511)
    plt.plot(close_price[-100:],'r')
    for i in range(num_fig):
        x = "%d1%d"%(num_fig+1,i+2)
        print x
        plt.subplot(x)
        plt.plot(imf[i][20:],'r')


    plt.figure(2)
    plt.hist(period_index)
    plt.show()


