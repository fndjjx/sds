import numpy as np
import sys
import os
import sys
import libfann
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from emd import *
import datetime
from scipy.signal import argrelextrema

class judge():
    
    def __init__(self, data):
        self.data = data
        self.max_ma5_list = []
        self.min_ma5_list = []
        self.min_ma10_list = []
        self.min_ma30_list = []
        self.max_ma10_list = []
        self.max_ma30_list = []

        self.max_min_period_delta = []
        self.min_max_period_delta = []
    
    def var(self):
        return np.var(self.data)

    def std(self):
        return np.std(self.data)

    def ma_fig(self):
        data = np.array(self.data) 
        max_index = argrelextrema(data,np.greater)[0]
        min_index = argrelextrema(data,np.less)[0]
        for i in max_index:
            self.max_ma5_list.append(self.ma(self.data,5,0,i))
            self.max_ma10_list.append(self.ma(self.data,10,0,i))
            self.max_ma30_list.append(self.ma(self.data,30,0,i))
        for i in min_index:
            self.min_ma5_list.append(self.ma(self.data,5,0,i))
            self.min_ma10_list.append(self.ma(self.data,10,0,i))
            self.min_ma30_list.append(self.ma(self.data,30,0,i))
    
    def extrem_period(self):
        my_emd = one_dimension_emd(self.data)            
        (imf, residual) = my_emd.emd()
        for i in range(len(imf)):
            (max_min,min_max) = self.extrem_interval(imf[i])
            print "imf%s mean %s"%(i,np.mean(imf[i]))
            print "imf%s max_min %s"%(i,np.std(max_min))
            print "imf%s min_max %s"%(i,np.std(min_max))
            print "imf%s max_min mean %s"%(i,np.mean(max_min))
            print "imf%s min_max mean %s"%(i,np.mean(min_max))

    def extrem_interval(self, rawdata):
        data = np.array(rawdata)
        max_index = argrelextrema(data,np.greater)[0]
        min_index = argrelextrema(data,np.less)[0]
      
        if max_index[0]>min_index[0]:
            max_min_period_delta = [max_index[i]-min_index[i] for i in range(min(len(max_index),len(min_index)))]
            min_max_period_delta = [min_index[i+1]-max_index[i] for i in range(min(len(max_index),len(min_index))-1)]
        else:
            max_min_period_delta = [max_index[i+1]-min_index[i] for i in range(min(len(max_index),len(min_index))-1)]
            min_max_period_delta = [min_index[i]-max_index[i] for i in range(min(len(max_index),len(min_index)))]
        return (max_min_period_delta,min_max_period_delta)

    def ma(self, data, period, start, current_index):
        sum_period = 0
        for i in range(period):
            sum_period += data[current_index + start - i]
        return float(sum_period)/period

    def draw_fig(self):
        plt.subplot(311).axis([260,400,0,50])
        plt.plot([i for i in range(len(self.min_ma5_list))],self.min_ma5_list,'r',[i for i in range(len(self.min_ma10_list))],self.min_ma10_list,'y',[i for i in range(len(self.min_ma30_list))],self.min_ma30_list,'b') 
        plt.subplot(312).axis([260,400,0,50])
        plt.plot([i for i in range(len(self.max_ma5_list))],self.max_ma5_list,'r',[i for i in range(len(self.max_ma10_list))],self.max_ma10_list,'y',[i for i in range(len(self.max_ma30_list))],self.max_ma30_list,'b') 
        plt.subplot(313).axis([1200,1500,0,50])
        plt.plot([i for i in range(len(self.data))],self.data,'b') 
        plt.show()

if __name__=="__main__":
    #fp = open("data_zhaoshang")
    #fp = open("data_zgpa")
    #fp = open("data_sh")
    #fp = open("data_trt")
    #fp = open("data_shjh")
    datafile = sys.argv[1]
    start = int(sys.argv[2])
    fp = open(datafile)
    lines = fp.readlines()
    fp.close()


    price = []
    for eachline in lines:
        price.append(float(eachline.split("\t")[2]))
    my_emd = one_dimension_emd(price)
    (imf, residual) = my_emd.emd()
    imf_without = [price[i]-imf[0][i] for i in range(len(price))]
    my_judge = judge(imf_without[start:])
    print my_judge.var() 
    print my_judge.std() 
    my_judge.extrem_period()
