
import os
from scipy import stats
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
        macd.append(float(eachline.split("\t")[5]))
        open_price.append(float(eachline.split("\t")[1]))
        date.append(eachline.split("\t")[0])
        vol.append(float(eachline.split("\t")[5]))
        je.append(float(eachline.split("\t")[5]))

    ma2_list=[]
    for i in range(2):
        ma2_list.append(0)
    for i in range(2,len(close_price)):
        ma2_list.append(np.mean(close_price[i-1:i+1]))
    
    
    ma5_list=[]
    for i in range(5):
        ma5_list.append(0)
    for i in range(5,len(close_price)):
        ma5_list.append(np.mean(close_price[i-4:i+1]))
    
    testdata = np.array([ma2_list[i]-ma5_list[i] for i in range(len(close_price))])
    testdata1 = np.diff(testdata)
    testdata2 = np.diff(testdata1)
    
    print (stats.normaltest(testdata)[1])
    print (stats.normaltest(testdata1)[1])
    print (stats.normaltest(testdata2)[1])
