
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
        macd.append(float(eachline.split("\t")[5]))
        open_price.append(float(eachline.split("\t")[1]))
        date.append(eachline.split("\t")[0])
        vol.append(float(eachline.split("\t")[5]))
        je.append(float(eachline.split("\t")[5]))

    data = close_price
    for i in range(30):
        data.append(0)
    for i in range(30,(len(close_price))):
        data.append(np.mean(close_price[i-29:i+1]))
    my_emd = one_dimension_emd(data[30:],9)
    (imf, residual) = my_emd.emd(0.03,0.03)

    imf_max_index = list(argrelextrema(np.array(imf[1]),np.greater)[0])
    imf_min_index = list(argrelextrema(np.array(imf[1]),np.less)[0])
    print imf_max_index
    print imf_min_index

    imf_max_index = list(argrelextrema(np.array(imf[2]),np.greater)[0])
    imf_min_index = list(argrelextrema(np.array(imf[2]),np.less)[0])
    print imf_max_index
    print imf_min_index

    

