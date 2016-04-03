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
    cdiff = list(np.diff(datac))
    cdiff.insert(0,0)
    c=0
    c1=0
    p=0
    x=[]
    s=0
    for i in range(30,len(datac)-1):
        #if  mp[i]<0.5*(datah[i]+datal[i]) and mp[i]<datac[i] and mp[i-1]>datac[i-1] and p==0:
        #if mp[i]<0.5*(datah[i]+datal[i]) and datac[i]<0.5*(datah[i]+datal[i]) and mp[i]<datac[i] and mp[i-1]>datac[i-1] and p==0:
        dd = datac[i-9:i+1]
        mpd = mp[i-9:i+1]
    
        min_index=argrelextrema(np.array(dd),np.less)[0] 
        for j in min_index:
            c+=1
            if dd[j]<mpd[j] :#or dd[j+1]<mpd[j+1] :#or dd[j-1]<mpd[j-1]:
                c1+=1
        if c!=0:
            x.append(c1/float(c))
        else:
            x.append(0)
        c=0
        c1=0

#    plt.plot(datac)
#    plt.plot(mp,'r')
#
#    plt.figure(2)
#    cd=signal.detrend(datac)
#    mpd=signal.detrend(mp)
#    plt.plot(cd)
#    plt.plot(mpd,'r')
#    
#    plt.show()

    datac=ma_func(datac,3)
    mp=ma_func(mp,3)
    c=c1=0
    min_index=argrelextrema(np.array(datac),np.less)[0]
    for i in min_index:
        c+=1
        if datac[i]<mp[i]:# or datac[i+1]<mp[i+1] or datac[i-1]<mp[i-1]:
            c1+=1
    print c1/float(c)

    c=c1=0
    max_index=argrelextrema(np.array(datac),np.greater)[0]
    for i in max_index:
        c+=1
        if datac[i]>mp[i]:# or datac[i+1]<mp[i+1] or datac[i-1]<mp[i-1]:
            c1+=1
    print c1/float(c)
#    if c!=0:
#        print np.mean(x)
#        return np.mean(x)
#    else:
#        return 0


        

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
    file_name=["axxt"]
    for i in file_name:
        print i
        rr=run("../back_test/test_data/ll/{}".format(i),start,end)
        if rr!=0:
            r.append(rr)

    print np.mean(r)


