from scipy.signal import argrelextrema
from load_data import load_data
from emd import one_dimension_emd
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

def imf_percentage(imf, residual, source_data):
    sum_list = []
    source_data_without_residual = []
    for i in range(len(imf)):
        sum_list.append(sum([j*j for j in imf[i]]))
    sum_list.append(sum([j*j for j in residual]))
    source_data_without_residual = [source_data[i]-residual[i] for i in range(len(source_data))]
    source_square_sum = sum([j*j for j in source_data_without_residual])
    source_square_sum = sum([j*j for j in source_data])
    return [i/source_square_sum for i in sum_list]


def emd_decom(data,imf_number,precise=0.02):
    my_emd = one_dimension_emd(data,imf_number)
    (imf, residual) = my_emd.emd(precise,precise)
    return imf,residual

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

def calc_down(data):
    c=[]
    tmp=0
    for i in range(1,len(data)):
        if data[i]<data[i-1]:
            tmp+=data[i-1]-data[i]
        else:
            if tmp!=0:
                c.append(tmp)
                tmp=0
    return np.mean(c)

def calc_up(data):
    c=[]
    tmp=0
    for i in range(1,len(data)):
        if data[i]>data[i-1]:
            tmp+=data[i]-data[i-1]
        else:
            if tmp!=0:
                c.append(tmp)
                tmp=0
    return np.mean(c)

def ma_func(data, period):
    ma_list=[]
    for i in range(period):
        ma_list.append(data[i])
    for i in range(period,len(data)):
        ma_list.append(np.mean(data[i-(period-1):i+1]))
    return ma_list


def extrem_compare(imf,data):
    max_extreme_index = list(argrelextrema(np.array(imf),np.greater)[0])
    min_extreme_index = list(argrelextrema(np.array(imf),np.less)[0])
    tmp= max_extreme_index+min_extreme_index

    tmp.sort()
    if tmp[0] in max_extreme_index:
        tmp.pop(0) 
    if tmp[-1] in min_extreme_index:
        tmp.pop()
    r=[]
    for i in range(0,len(tmp),2):
        r.append(data[tmp[i+1]]-data[tmp[i]])

    print r
    print np.mean(r)

         
if __name__=="__main__":
    filename1=sys.argv[1]
    filename2=sys.argv[2]
    
    start1=int(sys.argv[3])
    end1=int(sys.argv[4])
    start2=int(sys.argv[5])
    end2=int(sys.argv[6])
    datac1=load_data(filename1,4)[start1:end1]
    datav1=load_data(filename1,5)[start1:end1]
    dataj1=load_data(filename1,6)[start1:end1]
    datac2=load_data(filename2,4)[start2:end2]
    datav2=load_data(filename2,5)[start2:end2]
    dataj2=load_data(filename2,6)[start2:end2]
    mp1=[dataj1[i]/datav1[i] for i in range(len(datac1))]
    mp_diff1=[datac1[i]-mp1[i] for i in range(len(datac1))]
    mp2=[dataj2[i]/datav2[i] for i in range(len(datac2))]
    mp_diff2=[datac2[i]-mp2[i] for i in range(len(datac2))]
    print pearsonr(mp_diff1,mp_diff2)
    print pearsonr(mp1,mp2)
   # start=int(sys.argv[2])
   # end=int(sys.argv[3])
   # datac=load_data(filename,4)[start:end]
   # datah=load_data(filename,2)[start:end]
   # datal=load_data(filename,3)[start:end]
   # datao=load_data(filename,1)[start:end]
   # datav=load_data(filename,5)[start:end]
   # dataj=load_data(filename,6)[start:end]

   # mp=[dataj[i]/datav[i] for i in range(len(datac))]
   # 
   # print calc_down(mp)
   # print calc_up(mp)
    
    
