from monte_carlo import montecarlo_simulate
from monte_carlo import montecarlo_simulate2
from scipy.signal import argrelextrema
import statsmodels.tsa.stattools as ts
from load_data import load_data
import sys
import numpy as np
import matplotlib.pyplot as plt

def ma_func(data, period):
    ma_list=[]
    for i in range(period):
        ma_list.append(data[i])
    for i in range(period,len(data)):
        ma_list.append(np.mean(data[i-(period-1):i+1]))
    return ma_list


def calc_std(data1,data2):
    return    (np.std(data1)-np.std(data2))/(np.std(data1))

if __name__=="__main__":

    filename1=sys.argv[1]
    start=int(sys.argv[2])
    end=int(sys.argv[3])
    data1c =  load_data(filename1,4)[start:end]
    data1v =  load_data(filename1,5)[start:end]
    data1j =  load_data(filename1,6)[start:end]
    mp1 = [data1j[i]/data1v[i] for i in range(len(data1v))]


    cm1=[data1c[i]-mp1[i] for i in range(len(data1c))]
###############


    
    cpmpstd=[]
    p=500
    for i in range(p,len(data1c)):
        data1=data1c[i-p:i+1]
        data2=mp1[i-p:i+1]
        cpmpstd.append(calc_std(data1,data2))

    print np.mean(cpmpstd)
    print np.std(cpmpstd)

    min_index = argrelextrema(np.array(data1c),np.less)[0]
    min_index2 = argrelextrema(np.array(mp1),np.less)[0]
    max_index = argrelextrema(np.array(data1c),np.greater)[0]
    max_index2 = argrelextrema(np.array(mp1),np.greater)[0]
    
    c1=0
    c2=0
    s1=[]
    for i in min_index:
        if mp1[i]>data1c[i]:
            c1+=1
            s1.append(mp1[i]-data1c[i])
        if i in min_index2 or i+1 in min_index2 :#or i+1 in min_index2:
            c2+=1
    print "min {}".format(float(c1)/len(min_index))
    print "min {}".format(float(c2)/len(min_index))
    print "s1 {}".format(np.mean(s1))

    c1=0
    c2=0
    s2=[]
    for i in max_index:
        if mp1[i]<data1c[i]:
            c1+=1
            s2.append(mp1[i]-data1c[i])
        if i in max_index2 or i-1 in max_index2 or i+1 in max_index2:
            c2+=1
    print "max {}".format(float(c1)/len(max_index))
    print "max {}".format(float(c2)/len(max_index))
    print "s2 {}".format(np.mean(s2))
    print sum([abs((data1c[i]-mp1[i])/data1c[i]) for i in range(len(data1c))])

    print "minpro {}".format(len(min_index)/float(len(data1c)))
    print "maxpro {}".format(len(max_index)/float(len(data1c)))
    speed_c = np.diff(data1c)
    speed_mp = np.diff(mp1)
    print "speed c std {}".format(np.std(speed_c))
    print "speed mp std {}".format(np.std(speed_mp))
    print "speed"
    print sum([(speed_c[i]-speed_mp[i]) for i in range(len(speed_c))])

    t1=[]
    for i in range(1,len(data1c)):
        t1.append(np.log(data1c[i]/data1c[i-1]))
    print "t1 {}".format(np.std(t1))

    t2=[]
    for i in range(1,len(mp1)):
        t2.append(np.log(mp1[i]/mp1[i-1]))
    print "t2 {}".format(np.std(t2))
    plt.plot(data1c,'r')
    plt.plot(mp1)
    plt.show()

    
            
