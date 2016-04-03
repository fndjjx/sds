import numpy as np
import os
import copy
import math
import sys
import matplotlib.pyplot as plt
from emd import *

def format_data(data):
    if isinstance(data,list):
        return [10*((float(data[i])-min(data))/(max(data)-float(min(data)))) for i in range(len(data))]


if __name__=="__main__":


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
        macd.append(float(eachline.split("\t")[5]))
        open_price.append(float(eachline.split("\t")[1]))
        date.append(eachline.split("\t")[0])
        vol.append(float(eachline.split("\t")[5]))
        je.append(float(eachline.split("\t")[5]))


    emd_data = close_price
    emd_data=[]

    for i in range(3):
        emd_data.append(0)

    for i in range(3,len(close_price)):
        emd_data.append(np.mean(close_price[i-2:i+1]))

    
    std=[(np.std(emd_data[i-2:i+1])/(np.mean(emd_data[i-2:i+1]))) for i in range(2,len(emd_data))]

    emd_data = close_price

#####
    std=[]
    for i in range(20):
        std.append(0)
    for i in range(20,(len(close_price))):
        ten = format_data(emd_data[i-19:i+1])
        std.append((np.std(ten)/(np.mean(ten))))

######

    meanstd=[]

    for i in range(5):
        meanstd.append(0)

    for i in range(5,len(emd_data)):
        meanstd.append(np.mean(std[i-4:i+1]))
    meanstd2=[]

    for i in range(5):
        meanstd2.append(0)

    for i in range(5,len(emd_data)):
        meanstd2.append(np.mean(meanstd[i-4:i+1]))

    meanstd3=[]

    for i in range(5):
        meanstd3.append(0)

    for i in range(5,len(emd_data)):
        meanstd3.append(np.mean(meanstd2[i-4:i+1]))
#    my_emd = one_dimension_emd(emd_data,9)
#    (imf, residual) = my_emd.emd(0.05,0.05)
#
#    my_emd2 = one_dimension_emd(std,9)
#    (stdimf, residual) = my_emd2.emd(0.05,0.05)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(emd_data[-50:])
    plt.subplot(212)
    plt.plot(meanstd[-50:])

#    plt.figure(2)
#    plt.subplot(211)
#    plt.plot(imf[2][-100:])
#    plt.subplot(212)
#    plt.plot(stdimf[2][-100:])
    #plt.plot(std[-100:])
    plt.show()
