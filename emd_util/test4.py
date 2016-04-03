
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

def calc_match(imf_s,imf,current_index):
    r=imf_s[2]
    s=imf[2]
    ss=[0]*(current_index-799)
    s=list(ss)+list(s)
    max_r_index = argrelextrema(np.array(r),np.greater)[0]
    min_r_index = argrelextrema(np.array(r),np.less)[0]
    max_s_index = argrelextrema(np.array(s),np.greater)[0]
    min_s_index = argrelextrema(np.array(s),np.less)[0]

    tmp = list(max_s_index)+list(min_s_index)
    tmp.sort()
    ei=2
    if tmp[-ei] in min_s_index and (tmp[-ei] in min_r_index or tmp[-ei]-1 in min_r_index or tmp[-ei]+1 in min_r_index):
        return True
    elif tmp[-ei] in max_s_index and (tmp[-ei] in max_r_index or tmp[-ei]-1 in max_r_index or tmp[-ei]+1 in max_r_index):
        return True
    else:
        return False
    
 




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
            

def compare_func(filename,start,end):
    datac =  load_data(filename,4)[start:end]
    datav =  load_data(filename,5)[start:end]
    dataj =  load_data(filename,6)[start:end]

    diffmp = mpdiff_func(datav,dataj,datac)

    max_diff_index = argrelextrema(np.array(diffmp),np.greater)[0]
    min_diff_index = argrelextrema(np.array(diffmp),np.less)[0]

    #print "diff max {}".format(max_diff_index)
    #print "diff min {}".format(min_diff_index)

    datacall =  load_data(filename,4)
    imf=emd_decom(datacall,3)
    imf_s=imf[0][start:end]
    max_index = argrelextrema(np.array(imf_s),np.greater)[0]
    min_index = argrelextrema(np.array(imf_s),np.less)[0]

    #print "max {}".format(max_index)
    #print "min {}".format(min_index)

    print "diffmin  {}".format(calc_pro2(min_diff_index,min_index,max_index))
    print "diffmax  {}".format(calc_pro2(min_diff_index,max_index,min_index))


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
    start = int(sys.argv[2])
    #end = int(sys.argv[3])
    #data=load_data(filename,4)[start:end]
    #print hurst(data)
    repeat(filename,start)
    #period = int(sys.argv[4])
    #compare_func2(filename,start,end,period)
