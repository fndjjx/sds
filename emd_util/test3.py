from hurst import hurst
from load_data import load_data
from emd import one_dimension_emd
import numpy as np
import sys
import matplotlib.pyplot as plt

def emd_decom(data,imf_number,precise=0.02):
    my_emd = one_dimension_emd(data,imf_number)
    (imf, residual) = my_emd.emd(precise,precise)
    return imf

def calc_below(data):
    tmp=0
    l=[]
    for i in data:
        if i>0:
            tmp+=1
        else:
            l.append(tmp)
            tmp=0

    l=filter(lambda x:x>0,l)
    if l==[]:
        return 0
    else:
        return np.mean(l)

def calc_diff(datac,dataj,datav):
    mp=[]
    for i in range(len(datav)):
        if datav[i]==0:
            mp.append(datac[i])
        else:
            mp.append(dataj[i]/datav[i])
    return [datac[i]-mp[i] for i in range(len(datac))]

def run(filename,period):
    datac = load_data(filename,4)
    dataj = load_data(filename,6)
    datav = load_data(filename,5)
    data = calc_diff(datac,dataj,datav)
    bz=[]
    for i in range(period,len(data),period):
        d=data[i-period:i]
        #bz.append(calc_below(d))
        bz.append(np.std(d))
    plt.plot(bz)
    plt.show()
    print bz
    print np.mean(bz)
    print np.std(datac)
    return sum([i-np.mean(bz) for i in bz])

def repeat(filename):
    result=[]
    for i in range(5,50):
        result.append((i,run(filename,i)))

    result.sort(key=lambda x:x[1])

    print result
    print result[0]
    
        
        
        
    
if __name__=="__main__":
    filename=sys.argv[1]
    #repeat(filename)
    run(filename,10)

