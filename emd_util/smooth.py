import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from scipy.signal import gaussian
from scipy.signal import convolve
from numpy import concatenate
from sample import system_sample
from spline_func import spline_sample

def cubicSmooth5 (data_input):
    N = len(data_input)
    data_output = [0 for i in range(N)]
    if N<5:
        data_output = data_input
    else:
        data_output[0] = (69.0 * data_input[0] + 4.0 * data_input[1] - 6.0 * data_input[2] + 4.0 * data_input[3] - data_input[4]) / 70.0;
        data_output[1] = (2.0 * data_input[0] + 27.0 * data_input[1] + 12.0 * data_input[2] - 8.0 * data_input[3] + 2.0 * data_input[4]) / 35.0;
        for i in range(2,N-2):
            data_output[i] = (-3.0 * (data_input[i - 2] + data_input[i + 2])+ 12.0 * (data_input[i - 1] + data_input[i + 1]) + 17.0 * data_input[i] ) / 35.0;
        data_output[N - 2] = (2.0 * data_input[N - 5] - 8.0 * data_input[N - 4] + 12.0 * data_input[N - 3] + 27.0 * data_input[N - 2] + 2.0 * data_input[N - 1]) / 35.0;
        data_output[N - 1] = (- data_input[N - 5] + 4.0 * data_input[N - 4] - 6.0 * data_input[N - 3] + 4.0 * data_input[N - 2] + 69.0 * data_input[N - 1]) / 70.0;
    return data_output

def exp_smooth(data, alpha):
    after_smooth = []
    for i in range(len(data)):
        if i == 0:
            after_smooth.append(data[0])
        else:
            after_smooth.append((1-alpha)*after_smooth[i-1]+alpha*data[i])

    return after_smooth

def exp_smooth_double(data, alpha, beta):
    s = []
    t = []
    for i in range(len(data)):
        if i == 0:
            s.append(data[0])
            t.append(data[1]-data[0])
        else:
            s.append((1-alpha)*(s[i-1]+t[i-1])+alpha*data[i])
            t.append(beta*(s[i]-s[i-1])+(1-beta)*t[i-1])

    
    return [s[i]+t[i] for i in range(len(s))]


def exp_smooth_three(data, alpha, beta, gamma, season_period):
    s = []
    t = []
    p = []
    k = season_period
    
    for i in range(season_period):
        p.append(0)
        s.append(data[i])
        t.append(0)

 
    for i in range(season_period,len(data)):
        s.append(alpha*(data[i]-p[i-k])+(1-alpha)*(s[i-1]+t[i-1]))
        t.append(beta*(s[i]-s[i-1])+(1-beta)*t[i-1])
        p.append(gamma*(data[i]-s[i])+(1-gamma)*p[i-k])

    return [s[i]+t[i]+p[i] for i in range(len(s))]
    #return s 
    


def gaussian_smooth(data, width, std):
    filt = gaussian(width, std)
    filt /= sum(filt)
    padded = concatenate((data[0]*np.ones(width//2),data,data[-1]*np.ones(width//2)))
    return convolve(padded, filt, mode='valid')



def spline_smooth(data, step, s, order):

    raw_data = data
    sample_data = system_sample(raw_data, step, "end")
    sample_index = [i[0] for i in sample_data]
    sample_value = [i[1] for i in sample_data]
    return spline_sample(sample_index, sample_value, raw_data, s, order)

 
def load_data(filename):
    with open(filename,'r') as f:
        lines = f.readlines()
        datalist = []
        for eachline in lines:
            eachline.strip("\n")
            datalist.append(float(eachline.split("\t")[4]))
    return datalist

def test_spline_smooth():
    data = [1,2,2.2,3,4,6,4,5,3,2.4,2,1]
    print data
    step = 2
    return spline_smooth(data,step)

def test_exp_smooth():
    data = load_data("../back_test/test_data/zgtb")
    data = data[-30:]
    smooth = exp_smooth_three(data,0.1,0.2,0.2,5)
    #smooth = exp_smooth(data,0.5)
    #smooth = spline_smooth(smooth,1,0.05,3)
    plt.plot(data)
    plt.plot(smooth,'r')
    plt.show()



if __name__ == "__main__":
#    datalist = [1,5,3,8,3,2,1,2,3,5,9,2,1,2,3,4,3,2,7,2,3,6,10,2,1]
#    f = open("zgtb",'r')
#    lines = f.readlines()
#    f.close()
#    datalist_raw = []
#    for eachline in lines:
#        eachline.strip("\n")
#        datalist_raw.append(float(eachline.split("\t")[4]))
#    datalist = datalist_raw[1350:]
    print test_exp_smooth()
#    datalist_out = cubicSmooth5(datalist)
#
#    data_array = np.array(datalist)
#    max_extreme_index_raw = argrelextrema(data_array,np.greater)[0]
#    min_extreme_index_raw = argrelextrema(data_array,np.less)[0]
#
#    data_array = np.array(datalist_out)
#    max_extreme_index = argrelextrema(data_array,np.greater)[0]
#    min_extreme_index = argrelextrema(data_array,np.less)[0]
#
#    print "raw max %s"%len(max_extreme_index_raw)
#    print "raw min %s"%len(min_extreme_index_raw)
#    print "out max %s"%len(max_extreme_index)
#    print "out min %s"%len(min_extreme_index)
#    
#    plt.subplot(211)
#    plt.plot(np.linspace(0,len(datalist)-1,len(datalist)),datalist,'r')
#    plt.plot(np.linspace(0,len(datalist)-1,len(datalist)),datalist,'g')
#    plt.subplot(212)
#    plt.plot(np.linspace(0,len(datalist_out)-1,len(datalist_out)),datalist_out,'r')
#    plt.show()
