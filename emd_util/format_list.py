import sys
from scipy.signal import argrelextrema
from scipy.signal import welch
from scipy.interpolate import UnivariateSpline
import math
from smooth import cubicSmooth5
import numpy as np
from scipy import interpolate
import copy
import matplotlib.pyplot as plt



def splinerestruct(raw_data,new_index,new_value,kk=1):

    x1 = new_index
    y1 = new_value


    sx1 = np.linspace(0,len(raw_data)-1,len(raw_data))
    sy1 = interpolate.UnivariateSpline(x1,y1,k=kk,s=0)(sx1)

    return sy1


def restruct_extrem(data):

    max_index = list(argrelextrema(np.array(data),np.greater)[0])
    min_index = list(argrelextrema(np.array(data),np.less)[0])

    new_value = []
    new_index = []
    for i in range(len(data)):
        if i in max_index:
            new_value.append(1)
            new_index.append(i)
        elif i in min_index:
            new_value.append(-1)
            new_index.append(i)

    new=splinerestruct(data,new_index,new_value)
    #return (new_index,new_value)
    return new


if __name__=="__main__":
    f=sys.argv[1]
    fp = open(f)
    lines=fp.readlines()
    fp.close()
    data=[]
    for eachline in lines:
        eachline.strip("\n")
        data.append(float(eachline.split("\t")[4]))

    #(new_index,new_value)=restruct_extrem(data)
    new=restruct_extrem(data)
    plt.subplot(211)
    plt.plot(data[300:700])
    plt.subplot(212)
    #plt.plot(new_index,new_value)
    plt.plot(new[300:700])
    plt.show()

