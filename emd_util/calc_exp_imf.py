from emd import one_dimension_emd
import numpy as np
from scipy.signal import argrelextrema
from scipy.signal import spline_filter
import matplotlib.pyplot as plt
import sys
from smooth import exp_smooth
from smooth import exp_smooth_double
from smooth import exp_smooth_three
from smooth import cubicSmooth5



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


#def most_like_exp(data, imf, imf_index, need_index=0):
#    residual_imf = data_smooth(data,imf,imf_index)[-30:-3]
#
#    result = []
#    for alpha in list(np.linspace(0.01,0.7,99)): 
#        smooth_data = exp_smooth(data, alpha)[-30:-3]
#        result.append(((alpha,1), sum([abs(smooth_data[i]-residual_imf[i]) for i in range(len(smooth_data))])/(len(smooth_data))))
#    result.sort(key = lambda x: x[1])
#    print result
#    #plt.plot(smooth_data[-100:],'r')
#    #plt.plot(residual_imf[-100:])
#    #plt.show()
#    #result = result[:10]
#    #result.sort(key = lambda x: x[1])
#    
#    return result[need_index]

def most_like_exp(data, imf, imf_index, need_index=0):
    residual_imf = data_smooth(data,imf,imf_index)
    imf_sample = residual_imf[-20:]

    result = []
    alpha = 0.2
    while alpha<1:
        beta = 0.2
        while beta<1:
            gamma = 0.2
            while gamma<1:
                for season_period in range(3,20):
                    smooth_data = exp_smooth_three(data, alpha, beta, gamma, season_period)
                    smooth_sample = smooth_data[-20:]
                    result.append(((alpha,beta,gamma,season_period), sum([abs(smooth_sample[i]-imf_sample[i]) for i in range(len(smooth_sample))])/(len(smooth_sample))))
                gamma += 0.1
            beta += 0.1
        alpha += 0.1 
    result.sort(key = lambda x: x[1])
    print result
    print "len result {}".format(len(result))
    print result[0]
    smooth_data = exp_smooth_three(data, result[0][0][0], result[0][0][1], result[0][0][2],result[0][0][3])
    plt.plot(smooth_data[-50:],'r')
    plt.plot(residual_imf[-50:])
    plt.show()
    #result = result[:10]
    #result.sort(key = lambda x: x[1])

    return result[need_index]

#def most_like_exp(data, imf, imf_index, need_index=0):
#    residual_imf = data_smooth(data,imf,imf_index)
#    imf_sample = residual_imf[-30:-5]
#
#    result = []
#    alpha = 0.2
#    while alpha<1:
#        beta = 0.2
#        while beta<1:
#            smooth_data = exp_smooth_double(data, alpha, beta)
#            smooth_sample = smooth_data[-30:-5]
#            result.append(((alpha,beta), sum([abs(smooth_sample[i]-imf_sample[i]) for i in range(len(smooth_sample))])/(len(smooth_sample))))
#            beta += 0.1
#        alpha += 0.1
#    result.sort(key = lambda x: x[1])
#    print result
#    print "len result {}".format(len(result))
#    print result[0]
#    smooth_data = exp_smooth_double(data, result[0][0][0], result[0][0][1])
#    plt.plot(smooth_data[-100:],'r')
#    plt.plot(residual_imf[-100:])
#    plt.show()
#    #result = result[:10]
#    #result.sort(key = lambda x: x[1])
#
#    return result[need_index]


def most_like_imf(data, imf, imf_index1, imf_index2): 


    residual_short_para = most_like_exp(data,imf,imf_index1)
    residual_long_para= most_like_exp(data,imf,imf_index2)
    i = 1
    while residual_short_para[0] == residual_long_para[0]:
        residual_long_para = most_like_exp(data,imf,imf_index2,i)
        i += 1
    

    print "short alpha {}".format(residual_short_para)
    print "long alpha {}".format(residual_long_para)

    exp_long = exp_smooth_three(data, residual_long_para[0][0],residual_long_para[0][1],residual_long_para[0][2],residual_long_para[0][3])
    exp_short = exp_smooth_three(data, residual_short_para[0][0],residual_short_para[0][1],residual_short_para[0][2],residual_short_para[0][3])

   # for i in range(ma_short_shift):
   #     ma_short.pop(0)
   # for i in range(ma_long_shift):
   #     ma_long.pop(0)
    exp_diff = [exp_short[i]-exp_long[i] for i in range(len(exp_long))]
   # for i in range(ma_long_shift):
   #     ma_diff.insert(0,0)
   # print "ma diff value %s"%ma_diff
    
    data = np.array(exp_diff)
    max_index = list(argrelextrema(data,np.greater)[0])
    min_index = list(argrelextrema(data,np.less)[0])

    return max_index, min_index, residual_long_para, residual_short_para, exp_diff
    
    
    
            


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


    max_index, min_index, la, sa, expdiff = most_like_imf(data1, imf, imf_number-1, imf_number) 
    max_index = list(argrelextrema(np.array(expdiff),np.greater)[0])
    min_index = list(argrelextrema(np.array(expdiff),np.less)[0])
    max_index = [i+start for i in max_index]
    min_index = [i+start for i in min_index]

 #   imf = emd_decom(expdiff,3)
    print "exp"
    print max_index
    print min_index
    plt.plot(imf_base[imf_number][end-99:end+1])
    plt.plot(expdiff[-100:],'r')
 #   plt.figure(2)
 #   data1 = data_smooth(expdiff,imf,0)
 #   plt.plot(imf[2][-100:])
    plt.show()
