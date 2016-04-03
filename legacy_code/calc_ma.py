
import matplotlib.pyplot as plt
import numpy as np


def calc_ada_ma(data,n):


    ma = []
    length = len(data)
    c = 0
    for i in range(n-1):
        ma.append(data[0])
    ma.append(np.mean(data[:6]))
    for i in range(n,length):
        e1 = abs(data[i]-data[i-n+1])
        e2 = sum([abs(data[i-j-1]-data[i-j]) for j in range(n)])
        #print "haha"
#        print i
        #print data[i]
        #print data[i-1]
        #print data[i-2]
        #print data[i-3]
        #print data[i-4]
        #print e1
        #print e2
        e=e1/float(e2)
        #a = (2.0/(n+1))
        fast=(2.0/(3+1))
        slow=(2.0/(30+1))
        s=e*(fast-slow)+slow
        a=s*s
        ma.append((1-a)*ma[-1]+(a)*data[i])
        c+=1


    return ma

def calc_ema(data,n):


    ma = []
    length = len(data)
    c = 0
    for i in range(n-1):
        ma.append(data[i])
    ma.append(np.mean(data[:n+1]))
    for i in range(n,length):
        a=(2.0/(n+1))
        ma.append((1-a)*ma[-1]+(a)*data[i])
        c+=1


    return ma



if __name__=="__main__":

    f = open("qdpj",'r')
    lines = f.readlines()
    f.close()
    datalist = []
    vol = []
    for eachline in lines:
        eachline.strip("\n")
        datalist.append(float(eachline.split("\t")[4]))


    data = datalist
    ma5 = []
    for i in range(5-1):
        ma5.append(0)

    for i in range(5,len(data)+1):
        mean_5 = np.mean(data[i-(5-1):i+1])
        ma5.append(mean_5)
    ma=calc_ada_ma(data,5)
    print len(data)
    print len(ma)

    plt.figure(1)
    plt.plot([i for i in range(len(data[950:1150]))],data[950:1150],'b',[i for i in range(len(ma[950:1150]))],ma[950:1150],'r',[i for i in range(len(ma5[950:1150]))],ma5[950:1150],'g')
    plt.show()
    
