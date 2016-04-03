
from load_data import load_data
from emd import one_dimension_emd
from scipy.signal import argrelextrema
import numpy as np
import sys
from hurst import hurst

def emd_decom(data,imf_number,precise=0.02):
    my_emd = one_dimension_emd(data,imf_number)
    (imf, residual) = my_emd.emd(precise,precise)
    return imf

def ma_func(data, period):
    ma_list=[]
    for i in range(period):
        ma_list.append(data[i])
    for i in range(period,len(data)):
        ma_list.append(np.mean(data[i-(period-1):i+1]))
    return ma_list

def mpdiff_func(datav,dataj,datac):
    mp=[]
    for i in range(len(datav)):
        if datav[i]==0:
            mp.append(datac[i])
        else:
            mp.append(dataj[i]/datav[i])
    return [datac[i]-mp[i] for i in range(len(datac))]

def ma_ex_data(data):
    data=ma_func(data,10)
    max_index = argrelextrema(np.array(data),np.greater)[0]
    min_index = argrelextrema(np.array(data),np.less)[0]
    print len(max_index)/float(len(data))
    print len(min_index)/float(len(data))
    print np.std(data)#/np.mean(data)




def repeat(filename,start):
    data=load_data(filename,4)
    imf_s=emd_decom(data,3,0.005)
    c1=0
    c=0
    for i in range(start,len(data)):
        c+=1
        emd_data=data[i-799:i+1]
        print i
        imf=emd_decom(emd_data,3,0.001)
        if calc_match(imf_s,imf,i):
            c1+=1

    print c1/float(c)
            



def compare_func2(filename,start,end,period):
    length = end-start
    epn = length/period
    for i in range(period):
        start1=start+i*epn
        end1=start1+epn
        print start1
        print end1
        compare_func(filename,start1,end1)
        
def calc_pro(l1,l2):
    c=0
    for i in l1:
        if i in l2 :#or i+1 in l2 or i-1 in l2:
            c+=1
    return float(c)/len(l1)

def calc_pro2(l1,l2,l3):
    c1=0
    c2=0
    for i in l1:
        if i in l2 :#or i+1 in l2 or i-1 in l2:
            c1+=1
    for i in l1:
        if i in l3 :#or i+1 in l3 or i-1 in l3:
            c2+=1
    return float(c1)/(c1+c2)


if __name__=="__main__":
    filename = sys.argv[1]
    #start = int(sys.argv[2])
    ma_ex_data(load_data(filename,4))
    #end = int(sys.argv[3])
    #data=load_data(filename,4)[start:end]
    #print hurst(data)
    #repeat(filename,start)
    #period = int(sys.argv[4])
    #compare_func2(filename,start,end,period)
