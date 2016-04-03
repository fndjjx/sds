
from scipy.signal import argrelextrema
from load_data import load_data
from emd import one_dimension_emd
import numpy as np
import sys
import matplotlib.pyplot as plt



def dual_thrust(datac,datah,datal,k,o,c):
    hh = max(datah)
    lc = min(datac)
    hc = max(datac)
    ll = min(datal)
    ran = max(hh-lc,hc-ll)
    up_line = o+ran*k
    down_line = o-ran*k

    if c>up_line:
        return 1,up_line,down_line
    elif c<down_line:
        return -1,up_line,down_line
    else:
        return 0,up_line,down_line


if __name__=="__main__":
    filename=sys.argv[1]
    start=int(sys.argv[2])
    end=int(sys.argv[3])
    datac=load_data(filename,4)[start:end]
    datah=load_data(filename,2)[start:end]
    datal=load_data(filename,3)[start:end]
    datao=load_data(filename,1)[start:end]

    up_index=[]
    down_index=[]
    up_value=[]
    down_value=[]
    ull=[]
    dll=[]
    n=3
    for i in range(n,len(datac)):
        dc=datac[i-n+1:i+1]
        dh=datah[i-n+1:i+1]
        dl=datal[i-n+1:i+1]
        o=datao[i]
        c=datac[i]
        r,ul,dl=dual_thrust(dc,dh,dl,0.5,o,c)
        if r==1:
            up_index.append(i)
            up_value.append(datac[i])
        elif r==-1:
            down_index.append(i)
            down_value.append(datac[i])

        ull.append(ul)
        dll.append(dl)

    print up_value
    print down_value
    #plt.plot(datac[n:])
    plt.plot(datac[n:])
    plt.plot(ull,'r')
    plt.plot(dll,'g')
    #plt.plot(up_index,up_value,'ro',down_index,down_value,'go')
    plt.show()
        
    
    
