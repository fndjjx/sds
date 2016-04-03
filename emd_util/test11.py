from scipy import signal 
from monte_carlo import montecarlo_simulate
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

def run(filename,start,end):
    datac =  load_data(filename,4)[start:end]
    datav =  load_data(filename,5)[start:end]
    dataj =  load_data(filename,6)[start:end]
    datao =  load_data(filename,1)[start:end]
    datah =  load_data(filename,2)[start:end]
    datal =  load_data(filename,3)[start:end]

    mp = [dataj[i]/datav[i] for i in range(len(datac))]
    ma3=ma_func(datac,3)
    ma5=ma_func(datac,5)
    ma10=ma_func(datac,10)
    ma30=ma_func(datac,30)
    cdiff = list(np.diff(datac))
    cdiff.insert(0,0)
    c=0
    c1=0
    p=0
    x=[]
    s=0
    for i in range(100,len(datac)-2):
        #if  mp[i]<0.5*(datah[i]+datal[i]) and mp[i]<datac[i] and mp[i-1]>datac[i-1] and p==0:
        #if mp[i]<0.5*(datah[i]+datal[i]) and datac[i]<0.5*(datah[i]+datal[i]) and mp[i]<datac[i] and mp[i-1]>datac[i-1] and p==0:
        eln =  len(argrelextrema(np.array(ma3[i-9:i+1]),np.greater)[0])/10.0
        els =  len(argrelextrema(np.array(ma3[i-4:i+1]),np.greater)[0])/5.0
        mpd=signal.detrend(mp[i-9:i+1])
        mpd2=signal.detrend(mp[i-99:i+1])
        mpq=mp[i-9:i+1]-mpd
        mpq2=mp[i-99:i+1]-mpd2
        datacd=signal.detrend(datac[i-9:i+1])
        x.append(np.std(mpd))
        if  mp[i]>datac[i] and mp[i-1]<datac[i-1] and p==0:
            c+=1
            p=datac[i]
        if datac[i]>mp[i] and p!=0:
            if datac[i]>p:
                c1+=1
            s+=datac[i]-p
            p=0
        
    print c

    print "xielv {} {}".format(np.mean(x),np.std(x))
    if c!=0:
        print c1/float(c)
        #return c1/float(c)
        return s
    else:
        return 0
        

if __name__=="__main__":

    filename = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])

    cmd="ls test_data/ll"
    f=os.popen(cmd)
    file_name = f.readlines()
    f.close()

    file_name=[i.strip("\n") for i in file_name]

    r=[]
    for i in file_name:
        print i
        rr=run("test_data/ll/{}".format(i),start,end)
        if rr!=0:
            r.append(rr)

    print np.mean(r)


