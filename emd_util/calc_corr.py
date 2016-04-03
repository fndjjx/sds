import numpy as np
from smooth import spline_smooth
from load_data import load_data
import sys
import matplotlib.pyplot as plt


def autocorrelate(x):
    return  np.correlate(x, x, mode = 'same')

def autocorrelation(x,lags):
  n = len(x)
  x = np.array(x)
  result = [np.correlate(x[i:]-x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]\
    /(x[i:].std()*x[:n-i].std()*(n-i)) \
    for i in range(1,lags+1)]
  return result

def calc_diff(datac,dataj,datav):
    mp=[]
    for i in range(len(datav)):
        if datav[i]==0:
            mp.append(datac[i])
        else:
            mp.append(dataj[i]/datav[i])
    return [datac[i]-mp[i] for i in range(len(datac))]

if __name__=="__main__":
    filename=sys.argv[1]
    filename2=sys.argv[4]
    start=int(sys.argv[2])
    end=int(sys.argv[3])
    datacr = load_data(filename,4)[start:end]
    datajr = load_data(filename,6)[start:end]
    datajr2 = load_data(filename2,6)[start:end]
    datavr = load_data(filename,5)[start:end]
    datac=spline_smooth(datacr, 2, 0.1, 3)
    datav=spline_smooth(datavr, 2, 0.1, 3)
    dataj=spline_smooth(datajr, 2, 0.1, 3)
    data1 = calc_diff(datac,dataj,datav)
    data2 = calc_diff(datacr,datajr,datavr)
    a=autocorrelation(datac,end-start)
 
    t=list(np.diff(datac))
    t.insert(0,0)
    tt=[datac[i]-t[i] for i in range(len(t))]
    data=spline_smooth(datac, 2, 0.1, 3)
    plt.subplot(311)
    plt.plot(data1)
    plt.subplot(312)
    plt.plot(datajr)
    plt.subplot(313)
    plt.plot(datajr2)
    plt.show()

    
