
import numpy as np
from scipy.optimize import leastsq
import sys
import matplotlib.pyplot as plt

def leastsqt_para(data):

    def fun(p, x):
        a, b = p
        return a*x+b

    def err(p, x, y):
        return abs(fun(p,x) -y)

    p0 = [1,1]

    x = range(len(data))
    y = data
    x1 = np.array(x)
    y1 = np.array(y)

    xishu = leastsq(err, p0, args=(x1,y1))

    return xishu

def leastsqt_para_2(data):

    def fun(p, x):
        a, b, c = p
        return a*x**2+b*x+c

    def err(p, x, y):
        return abs(fun(p,x) -y)

    p0 = [1,1,1]

    x = range(len(data))
    y = data
    x1 = np.array(x)
    y1 = np.array(y)

    xishu = leastsq(err, p0, args=(x1,y1))

    return xishu

def leastsqt_predict(data,n):

    def fun(p, x):
        a, b = p  
        return a*x+b
    
    def err(p, x, y):
        return abs(fun(p,x) -y)
    
    p0 = [1,1]   
    
    x = range(len(data))    
    y = data
    x1 = np.array(x)  
    y1 = np.array(y)
    
    xishu = leastsq(err, p0, args=(x1,y1))

    result= []
    for i in range(n):
        result.append(fun(xishu[0],len(data)+i))

    return result


def leastsqt_predict_2(data,n):

    def fun(p, x):
        a, b, c= p
        return a*x**2+b*x+c

    def err(p, x, y):
        return abs(fun(p,x) -y)

    p0 = [1,1,1]

    x = range(len(data))
    y = data
    x1 = np.array(x)
    y1 = np.array(y)

    xishu = leastsq(err, p0, args=(x1,y1))

    result= []
    for i in range(n):
        result.append(fun(xishu[0],len(data)+i))

    return result


def leastsqt_predict_3(data,n):

    def fun(p, x):
        a, b, c, d= p
        return a*x**3+b*x**2+c*x+d

    def err(p, x, y):
        return abs(fun(p,x) -y)

    p0 = [1,1,1,1]

    x = range(len(data))
    y = data
    x1 = np.array(x)
    y1 = np.array(y)

    xishu = leastsq(err, p0, args=(x1,y1))

    result= []
    for i in range(n):
        result.append(fun(xishu[0],len(data)+i))

    return result

def leastsqt_predict_4(data,n):

    def fun(p, x):
        a, b, c, d, e= p
        return a*x**4+b*x**3+c*x**2+d*x+e

    def err(p, x, y):
        return abs(fun(p,x) -y)

    p0 = [1,1,1,1,1]

    x = range(len(data))
    y = data
    x1 = np.array(x)
    y1 = np.array(y)

    xishu = leastsq(err, p0, args=(x1,y1))

    result= []
    for i in range(n):
        result.append(fun(xishu[0],len(data)+i))

    return result

def malist(data):

    close_price=data
    ma2_list=[]
    for i in range(2):
        ma2_list.append(0)
    for i in range(2,len(close_price)):
        ma2_list.append(np.mean(close_price[i-1:i+1]))

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

    ma10_list=[]
    for i in range(10):
        ma10_list.append(0)
    for i in range(10,len(close_price)):
        ma10_list.append(np.mean(close_price[i-9:i+1]))

    ma20_list=[]
    for i in range(20):
        ma20_list.append(0)
    for i in range(20,len(close_price)):
        ma20_list.append(np.mean(close_price[i-19:i+1]))

    ma_list=[ma2_list[i]-ma5_list[i] for i in range(len(close_price))]

    return ma_list



if __name__=="__main__":
    a= [1,2,3,4,5,6]
    print leastsqt_predict(a,2)
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

    data = close_price

    period=3
    ma = []
    for ii in range(period):
        ma.append(0)
    for ii in range(period,len(data)):
        ma.append(np.mean(data[ii-(period-1):ii+1]))

    ma2 = []
    for ii in range(period):
        ma2.append(0)
    for ii in range(period,len(data)):
        ma2.append(np.mean(ma[ii-(period-1):ii+1]))

    data=ma
    predict=[]
    for i in range(50):
        predict.append(data[i])

    for i in range(50,len(data)):
        t1 = data[i-10:i]
        t2 = data[i-10:i]
        t3 = data[i-10:i]
        t4 = data[i-10:i]

        #t1.append(open_price[i])
        #t2.append(open_price[i])
        #t3.append(open_price[i])

        p11=leastsqt_predict(t1,4)[0]
        p12=leastsqt_predict_2(t2,4)[0]
        p13=leastsqt_predict_3(t3,4)[0]
        p14=leastsqt_predict_4(t4,4)[0]


        p21=leastsqt_predict(t1,4)[1]
        p22=leastsqt_predict_2(t2,4)[1]
        p23=leastsqt_predict_3(t3,4)[1]
        p24=leastsqt_predict_4(t4,4)[1]

        p31=leastsqt_predict(t1,4)[2]
        p32=leastsqt_predict_2(t2,4)[2]
        p33=leastsqt_predict_3(t3,4)[2]
        p34=leastsqt_predict_4(t4,4)[2]


        pre1 = (p11+p12+p13)/3.0
        pre2 = (p21+p22+p23)/3.0
        pre3 = (p31+p32+p33)/3.0

        #pre1 = p11
        #pre2 = p21
        predict.append(pre1)
        predict.append(pre2)
        #predict.append(pre3)

    
    m1=malist(data)
    #m1=(data)
    m2=malist(predict)
    #m2=(predict)
    print len(m1)
    print len(m2)

    ma_list=[]
    for i in range(3):
        ma_list.append(0)
    for i in range(3,len(close_price)):
        ma_list.append(np.mean(m2[i-2:i+1]))

    #m2=ma_list

    plt.plot(m1[begin:end])
    plt.plot(m2[begin:end])
    plt.show()
























        
