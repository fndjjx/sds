from scipy import signal 
from monte_carlo import montecarlo_simulate_array
from scipy.signal import argrelextrema
from monte_carlo import montecarlo_simulate2
import statsmodels.tsa.stattools as ts
from load_data import load_data
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

def ma_func(data, period):
    ma_list=[]
    for i in range(period):
        ma_list.append(data[i])
    for i in range(period,len(data)):
        ma_list.append(np.mean(data[i-(period-1):i+1]))
    return ma_list

def xielv(l):
    return (l[-1]-l[0])/float(len(l))

def pro_forward(datac,mp):

    c=0
    c1=0
    for i in range(1,len(l1)):
        c+=1
        if datac[i]<datac[i-1]:
            if datac[i-1]-mp[i-1]>datac[i]-mp[i]:
                c1+=1
        else:
            if datac[i-1]-mp[i-1]<datac[i]-mp[i]:
                c1+=1


    print c1/float(c)
    return c1/float(c)

def run(filename,start,end):
    datac =  load_data(filename,4)[start:end]
    datav =  load_data(filename,5)[start:end]
    dataj =  load_data(filename,6)[start:end]
    datao =  load_data(filename,1)[start:end]
    datah =  load_data(filename,2)[start:end]
    datal =  load_data(filename,3)[start:end]

    mp = [dataj[i]/datav[i] for i in range(len(datac))]
    mpdiff = [datac[i]-mp[i] for i in range(len(datac))]
    mp2 = [(datah[i]+datal[i])/2 for i in range(len(datac))]
    
    mpdiff2 = [datac[i]-mp2[i] for i in range(len(datac))]
    mas=ma_func(datac,10)
    mal=ma_func(datac,60)
    cdiff = list(np.diff(datac))
    cdiff.insert(0,0)
    c=0
    c1=0
    p=0
    x=[]
    s=0
    d=0
    ss=[]
    f=0
    t=10000
    tt=0

    min_index=list(argrelextrema(np.array(mp),np.less)[0])
    max_index=list(argrelextrema(np.array(mpdiff),np.greater)[0])
    tmp = min_index+max_index
    tmp.sort()
#    print np.mean(np.diff(tmp))

    cc=0
    for i in range(100,len(datac)-1):
#        c+=1
#        if mp[max_index[i]]<mp[max_index[i+1]]:
#            c1+=1
#
#    r=[] 
#    rr=[] 
#    c=0
#    c1=0
#    for j in range(1,len(datac)):
#        datac_r = datac[j-49:j+1]
#        mp_r = mp[j-49:j+1]
#        min_index=list(argrelextrema(np.array(datac_r),np.less)[0])
#        max_index=list(argrelextrema(np.array(datac_r),np.greater)[0])
#
#        for i in range(len(max_index)-1):
#            c+=1
#            if mp[max_index[i]]<mp[max_index[i+1]]:
#                c1+=1
#        #print min_index
#        #print min_index2
#        try:
#            r.append(c1/float(c))
#        except:
#            r.append(0)
#        c=0
#        c1=0
        


#    plt.subplot(411)
#    plt.plot(datac)
#    plt.plot(mp,'r')
#
#    plt.subplot(412)
#    data1=mpdiff
#    big0index=[i for i in range(len(data1)) if data1[i]>0 ]
#    small0index=[i for i in range(len(data1)) if data1[i]<0 ]
#    big0value=[i for i in data1 if i>0 ]
#    small0value=[i for i in data1 if i<0 ]
#    plt.plot([i for i in range(len(data1))],data1,'o',[i for i in range(len(data1))],data1,big0index,big0value,'ro',small0index,small0value,'go')
#
#    plt.subplot(413)
#    data2=mpdiff2
#    big0index2=[i for i in range(len(data2)) if data2[i]>0 ]
#    small0index2=[i for i in range(len(data2)) if data2[i]<0 ]
#    big0value2=[i for i in data2 if i>0 ]
#    small0value2=[i for i in data2 if i<0 ]
##
#    #plt.plot([i for i in range(len(data2))],data2,'o',[i for i in range(len(data2))],data2,big0index2,big0value2,'ro',small0index2,small0value2,'go')
#    plt.plot(datav)
#    plt.subplot(414)
#    plt.plot(dataj)
##
#    plt.show()
#  
    #print cc
        m,s = montecarlo_simulate_array(mpdiff[i-6:i],1000)
        if   m+2*s>mpdiff[i]:#>m-s :
            c+=1
            if mpdiff[i+1]>0:#mpdiff[i]:
                c1+=1
    print c
    return c1/float(c)

     

        

if __name__=="__main__":

    filename = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])

    cmd="ls ../back_test/test_data/ll"
    f=os.popen(cmd)
    file_name = f.readlines()
    f.close()

    file_name=[i.strip("\n") for i in file_name]

    r=[]
    d=[]
    f=[]
    rr=0
#    file_name=["axxt"]
#    file_name=["xyyh"]
#    file_name=["dhny"]
#    file_name=["zgtb"]
    for i in file_name:
        print i
        try:
            rr=run("../back_test/test_data/ll/{}".format(i),start,end)
        except:
            pass
        if rr!=0:
            r.append(rr)

    print np.mean(r)


