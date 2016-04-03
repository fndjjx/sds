from emd import one_dimension_emd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import sys
from smooth import gaussian_smooth
from smooth import spline_smooth



def load_data(filename):
    with open(filename,'r') as f:
        lines = f.readlines()
        datalist = []
        for eachline in lines:
            eachline.strip("\n")
            datalist.append(float(eachline.split("\t")[4]))
    return datalist


def ma_func(data, period):
    ma_list=[]
    for i in range(period):
        ma_list.append(0)
    for i in range(period,len(data)):
        ma_list.append(np.mean(data[i-(period-1):i+1]))
    return ma_list


def emd_decom(data,imf_number):
    my_emd = one_dimension_emd(data,imf_number)
    (imf, residual) = my_emd.emd(0.02,0.02)
    return imf


def data_smooth(data,imf,imf_index):
    print len(imf)
    print imf_index
    for i in range(imf_index+1):
        residual = [data[j]-imf[i][j] for j in range(len(data))]
        data = residual

    return residual


def most_like_ma(data, imf, imf_index, max_period,need_ma_index=0):
    residual_imf = data_smooth(data,imf,imf_index)
    imf_sample = residual_imf[-25:-5]

    # get residual_imf most like ma
    result = []
    for period in range(2,max_period): 
        ma_test = ma_func(data,period)[-25:-5]
        for shift in range(1,4):
            result.append((period, shift, sum([abs(ma_test[i]-imf_sample[i-shift]) for i in range(shift,len(ma_test))])/(len(ma_test)-shift)))
    result.sort(key = lambda x: x[2])
    print result
    ma_test = ma_func(data,result[0][0])
    #plt.plot(ma_test[-100:],'r')
    #plt.plot(residual_imf[-100:])
    #plt.show()
    #result = result[:10]
    #result.sort(key = lambda x: x[1])
    
    print need_ma_index
    return result[need_ma_index]


def most_like_imf(data, imf, imf_index1, imf_index2): 


    residual_short_ma_period, residual_short_ma_shift, _ = most_like_ma(data,imf,imf_index1,8)
    print "short ma period %s"%residual_short_ma_period
    residual_long_ma_period, residual_long_ma_shift, _ = most_like_ma(data,imf,imf_index2,15)
    i = 1
    while residual_long_ma_period - residual_short_ma_period < 1:
        residual_long_ma_period, residual_long_ma_shift, _ = most_like_ma(data,imf,imf_index2,15,i)
        i += 1
    

    print "short ma period %s"%residual_short_ma_period
    print "short ma shift %s"%residual_short_ma_shift
    print "long ma period %s"%residual_long_ma_period
    print "long ma shift %s"%residual_long_ma_shift

    ma_long = ma_func(data, residual_long_ma_period)
    ma_short = ma_func(data, residual_short_ma_period)

    ma_short_shift = residual_short_ma_shift
    ma_long_shift = residual_long_ma_shift
   # for i in range(ma_short_shift):
   #     ma_short.pop(0)
   # for i in range(ma_long_shift):
   #     ma_long.pop(0)
    ma_diff = [ma_short[i]-ma_long[i] for i in range(len(ma_long))]
   # for i in range(ma_long_shift):
   #     ma_diff.insert(0,0)
   # print "ma diff value %s"%ma_diff
    
    data = np.array(ma_diff)
    max_index = list(argrelextrema(data,np.greater)[0])
    min_index = list(argrelextrema(data,np.less)[0])

    return max_index, min_index, residual_long_ma_period, residual_short_ma_period, ma_diff
    
    
    
            


if __name__ == "__main__":


    start = int(sys.argv[1])
    end = int(sys.argv[2])
    imf_number = int(sys.argv[3])

    data = load_data("../back_test/test_data/zgtb")
    print data[end]
    imf_base = emd_decom(data,6)
    imf_component_base = np.array(imf_base[imf_number][:end+1])
    imf_max_index_base = list(argrelextrema(imf_component_base,np.greater)[0])
    imf_min_index_base = list(argrelextrema(imf_component_base,np.less)[0])
    print "standard"
    print imf_max_index_base
    print imf_min_index_base

    data1 = data[start:end+1]
    imf = emd_decom(data1,6)
    imf_component = np.array(imf[imf_number])
    imf_component = np.array(imf[imf_number])
    imf_max_index = list(argrelextrema(imf_component,np.greater)[0])
    imf_min_index = list(argrelextrema(imf_component,np.less)[0])
    imf_max_index = [i+start for i in imf_max_index]
    imf_min_index = [i+start for i in imf_min_index]
    print "partial"
    print imf_max_index
    print imf_min_index


    max_index, min_index, lp, sp, madiff = most_like_imf(data1, imf, imf_number-1, imf_number) 
    madiff = spline_smooth(madiff,1,0.5,3)
    max_index = list(argrelextrema(np.array(madiff),np.greater)[0])
    min_index = list(argrelextrema(np.array(madiff),np.less)[0])
    max_index = [i+start for i in max_index]
    min_index = [i+start for i in min_index]

    imf = emd_decom(madiff,3)
    print "ma"
    print max_index
    print min_index
    plt.plot(imf_base[imf_number][end-99:end+1])
    plt.plot(madiff[-100:],'r')
    plt.show()
