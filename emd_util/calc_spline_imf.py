from emd import one_dimension_emd
import numpy as np
from scipy.signal import argrelextrema
from scipy.signal import spline_filter
import matplotlib.pyplot as plt
import sys
from smooth import exp_smooth
from smooth import cubicSmooth5
from smooth import spline_smooth
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

def normalize(l):
    max_n = max(l)
    min_n = min(l)
    return map(lambda x: (x-min_n)/float(max_n-min_n), l)

def emd_decom(data,imf_number):
    my_emd = one_dimension_emd(data,imf_number)
    (imf, residual) = my_emd.emd(0.001,0.001)
    return imf


def data_smooth(data,imf,imf_index):
    for i in range(imf_index+1):
        residual = [data[j]-imf[i][j] for j in range(len(data))]
        data = residual

    return residual


def calc_distance(l1,l2):
    l1 = normalize(l1)
    l2 = normalize(l2)
    return sum([abs(l1[i]-l2[i]) for i in range(len(l1))])/(len(l1))

    

def most_like_spline(data, imf, imf_index, longshort, need_index=0):
    residual_imf = data_smooth(data,imf,imf_index)
    imf_sample = residual_imf[-25:-15]

    result = []
    #for sample_step in range(1,2): 
    for order in range(3,6): 
        s = 0
        while s < 200:
            if longshort == "short":
                smooth_data = spline_smooth(data, 1, s, order)
            else:
                smooth_data = spline_smooth(ma_func(data,5), 1, s, order)
            smooth_data_sample = smooth_data[-25:-15]
            result.append(((s,order), calc_distance(smooth_data_sample,imf_sample)))
            #result.append((sample_step,s, calc_para(residual_imf,smooth_data)))
            #result.append((sample_step,s, calc_SMC(residual_imf,smooth_data)))
            s += 0.5
    result.sort(key = lambda x: x[1])
   # result.reverse()
    print result
    smooth_data = spline_smooth(data, 1, result[0][0][0], result[0][0][1])
   # print "haha"
   # print len(smooth_data)
   # print len(residual_imf)
    plt.plot(smooth_data[-70:],'r')
    plt.plot(residual_imf[-70:])
    plt.show()
    #result = result[:10]
    #result.sort(key = lambda x: x[1])
    
    return result[need_index]


def most_like_imf(data, imf, imf_index1, imf_index2): 


    residual_short_para, short_distance = most_like_spline(data,imf,imf_index1,"short")
    residual_long_para, long_distance = most_like_spline(data,imf,imf_index2,"long")
  #  i = 1
  #  while residual_long_para[0] - residual_short_para[0] < 10:
  #      residual_long_para, _ = most_like_spline(data,imf,imf_index2,"long",i)
  #      i += 1
    

    print "short sample step {}".format(residual_short_para)
    print "long sample step {}".format(residual_long_para)

    confidence = 0
    if (short_distance/np.mean(data[-20:]))<0.01 and (long_distance/np.mean(data[-20:]))<0.01:
        confidence = 1
    spline_long = spline_smooth(data, 1, residual_long_para[0], residual_long_para[1])
    spline_short = spline_smooth(data, 1, residual_short_para[0], residual_short_para[1])
 
    spline_diff = [spline_short[i]-spline_long[i] for i in range(len(spline_long))]
    
    data = np.array(spline_diff)
    max_index = list(argrelextrema(data,np.greater)[0])
    min_index = list(argrelextrema(data,np.less)[0])

    return max_index, min_index, residual_long_para, residual_short_para, spline_diff, confidence
    
    
    
            


if __name__ == "__main__":


    start = int(sys.argv[1])
    end = int(sys.argv[2])
    imf_number = int(sys.argv[3])

    data = load_data("../back_test/test_data/zgtb")
    data = [data[i]-ma_func(data,3)[i] for i in range(len(data))]
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
