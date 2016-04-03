from emd import one_dimension_emd
import numpy as np
from scipy.signal import argrelextrema
from scipy.signal import spline_filter
import matplotlib.pyplot as plt
import sys
from smooth import exp_smooth
from smooth import cubicSmooth5
from smooth import spline_smooth
from smooth import gaussian_smooth
from calc_match import calc_para
from calc_match import calc_SMC



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
    for i in range(imf_index+1):
        residual = [data[j]-imf[i][j] for j in range(len(data))]
        data = residual

    return residual


def most_like_gaussian(data, imf, imf_index, need_index=0):
    residual_imf = data_smooth(data,imf,imf_index)
    print "length data {}".format(len(data))
    imf_sample = residual_imf[-55:]

    result = []
    for width in range(5,40,2): 
        std = 1
        while std < 5:
            smooth_data = gaussian_smooth(data, width, std)
            smooth_data_sample = smooth_data[-55:]
            result.append((width,std, sum([abs(smooth_data_sample[i]-imf_sample[i]) for i in range(len(smooth_data_sample))])/(len(smooth_data_sample))))
            #result.append((sample_step,s, calc_para(residual_imf,smooth_data)))
            #result.append((sample_step,s, calc_SMC(residual_imf,smooth_data)))
            std += 1
    result.sort(key = lambda x: x[2])
   # result.reverse()
    print result
    smooth_data = gaussian_smooth(data, result[0][0], result[0][1])
    print "length smooth data {}".format(len(smooth_data))
    
   # print "haha"
   # print len(smooth_data)
   # print len(residual_imf)
    plt.plot(smooth_data[-70:],'r')
    plt.plot(residual_imf[-70:])
    plt.show()
    #result = result[:10]
    #result.sort(key = lambda x: x[1])
    print need_index
    
    return result[need_index]


def most_like_imf(data, imf, imf_index1, imf_index2): 


    residual_short_sample_width, residual_short_std, short_distance = most_like_gaussian(data,imf,imf_index1)
    print "short sample width %s"%residual_short_sample_width
    print "short std %s"%residual_short_std
    residual_long_sample_width, residual_long_std, long_distance = most_like_gaussian(data,imf,imf_index2)
    i = 1
    while residual_short_sample_width == residual_long_sample_width:# and residual_short_factor == residual_long_factor:
        residual_long_sample_width, residual_long_std, _ = most_like_gaussian(data,imf,imf_index2,i)
        print "long sample width %s"%residual_long_sample_width
        print "long std %s"%residual_long_std
        i += 1
    

    print "long sample width %s"%residual_long_sample_width
    print "long std %s"%residual_long_std

    confidence = 0
    if (short_distance/np.mean(data[-20:]))<0.01 and (long_distance/np.mean(data[-20:]))<0.01:
        confidence = 1
    long_period = gaussian_smooth(data, residual_long_sample_width, residual_long_std)
    short_period = gaussian_smooth(data, residual_short_sample_width, residual_short_std)
 
    diff = [short_period[i]-long_period[i] for i in range(len(long_period))]
    #diff = gaussian_smooth(diff, 21, 4)
    
    data = np.array(diff)
    max_index = list(argrelextrema(data,np.greater)[0])
    min_index = list(argrelextrema(data,np.less)[0])

    return max_index, min_index, residual_long_sample_width, residual_short_sample_width, diff, confidence
    
    
    
            


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


    max_index, min_index, la, sa, splinediff, confidence = most_like_imf(data1, imf, imf_number-1, imf_number) 
    #splinediff = gaussian_smooth(splinediff,21,4)
    max_index = list(argrelextrema(np.array(splinediff),np.greater)[0])
    min_index = list(argrelextrema(np.array(splinediff),np.less)[0])
    max_index = [i+start for i in max_index]
    min_index = [i+start for i in min_index]

    print max_index
    print min_index
    plt.plot(imf_base[imf_number][end-99:end+1])
    plt.plot(splinediff[-100:],'r')
 #   plt.figure(2)
 #   data1 = data_smooth(expdiff,imf,0)
 #   plt.plot(imf[2][-100:])
    plt.show()
