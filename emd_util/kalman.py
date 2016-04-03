
import sys
import numpy as np
import matplotlib.pyplot as plt
from leastsqt import leastsqt_para


def kalman_simple(o,c):

    # intial parameters
    n_iter = len(o)
    sz = (n_iter,) # size of array
    
 #   Q = np.var(np.diff(c)) # process variance
    
    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor
    
    
    # intial guesses
    xhat[0] = c[0]
    P[0] = 1.0

    z=o
    
    for k in range(1,n_iter):
        
        if k<2:
            Q = 0
        else:
            Q = np.var(np.diff(c[:k])) # process variance
        print "Q%s"%Q
        
        c_o=[c[i]-o[i] for i in range(k)]
        R = np.var(c_o) # estimate of measurement variance, change to see effect
        m = np.mean(c[:k])
        d = np.var(c[:k])
        A = np.exp(m-d**2)
        # time update
        if k>10:
            A=leastsqt_para(c[k-10:k])[0][0]
        else:
            A=0
        xhatminus[k] = A+xhat[k-1]  #X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
        Pminus[k] = P[k-1]+Q      #P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1
    
        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R ) #Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k]) #X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
        P[k] = (1-K[k])*Pminus[k] #P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1

    return xhat 


if __name__ == "__main__":

    datafile = sys.argv[1]
    begin = int(sys.argv[2])
    end = int(sys.argv[3])
    fp = open(datafile)
    lines = fp.readlines()
    fp.close()

    close_price = []
    open_price = []
    high_price = []
    low_price = []
    date = []
    macd = []
    vol = []
    je = []
    for eachline in lines:
        eachline.strip()
        close_price.append(float(eachline.split("\t")[4]))
        high_price.append(float(eachline.split("\t")[2]))
        low_price.append(float(eachline.split("\t")[3]))
        macd.append(float(eachline.split("\t")[5]))
        open_price.append(float(eachline.split("\t")[1]))
        date.append(eachline.split("\t")[0])
        vol.append(float(eachline.split("\t")[5]))
        je.append(float(eachline.split("\t")[5]))

    if end>len(close_price):
        end=len(close_price)+1
    
    ma2_list=[]
    for i in range(2):
        ma2_list.append(0)
    for i in range(2,len(close_price)):
        ma2_list.append(np.mean(close_price[i-1:i+1]))

    o_ma2=[]
    for i in range(2):
        o_ma2.append(0)
    for i in range(2,len(close_price)):
        o_ma2.append(np.mean(open_price[i-1:i+1]))

    o_ma5=[]
    for i in range(5):
        o_ma5.append(0)
    for i in range(5,len(close_price)):
        o_ma5.append(np.mean(open_price[i-4:i+1]))

    ma3_list=[]
    for i in range(3):
        ma3_list.append(0)
    for i in range(3,len(close_price)):
        ma3_list.append(np.mean(close_price[i-2:i+1]))

    ma5_list=[]
    for i in range(5):
        ma5_list.append(0)
    for i in range(5,len(close_price)):
        ma5_list.append(np.mean(close_price[i-4:i+1]))
    
    print len(o_ma2)
    print len(ma2_list)
    print len(o_ma5)

    real = [ma2_list[i]-ma5_list[i] for i in range(len(ma2_list))]
    ma2_pre = kalman_simple(o_ma2[100:],ma2_list[100:])
    ma2_pre = kalman_simple(ma2_list[99:-1],ma2_list[100:])
    ma5_pre = kalman_simple(ma5_list[99:-1],ma5_list[100:])
    #predict = kalman_simple(open_price,close_price)
#    predict = [ma2_pre[i]-ma5_pre[i] for i in range(len(ma2_pre))]
    

    #predict = [ma2_list[i]-ma5_list[i] for i in range(len(ma2_list))]
    plt.plot(ma5_pre[begin-100:end-100],'g')
    #plt.plot(close_price[begin:end],'r')
    plt.plot(ma5_list[begin:end],'r')
    plt.show()
