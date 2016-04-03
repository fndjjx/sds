
import numpy as np
from scipy.stats import norm
import datetime
import timeit

def montecarlo_simulate(data,n):
    

    data1 = [data[i]/data[i-1] for i in range(1,len(data))]
    data_log = np.log(data1)
    mean=np.mean(data_log)
    std=np.std(data_log)
    #s=np.random.normal(mean, std, n)
    #s=np.random.lognormal(mean, std, n)
    #print (s)
    c=0
    t=[] 
    random_list = np.random.rand(n)
    for i in random_list:
        s=norm.ppf(i,mean,std)
        t.append(np.exp(s)*data[-1])

    mean=np.mean(t)
    std=np.std(t)
    return c/float(n),mean,std

def montecarlo_simulate2(data,n,step):


    data1 = [data[i]/data[i-1] for i in range(1,len(data))]
    data_log = np.log(data1)
    mean=np.mean(data_log)
    std=np.std(data_log)
    #s=np.random.normal(mean, std, n)
    #s=np.random.lognormal(mean, std, n)
    #print (s)
    c=0
    tt=[]
    for i in range(n):
        t=[]
        for j in range(step):
            s=norm.ppf(np.random.rand(1)[0],mean,std)
            t.append(np.exp(s))
        ss=reduce(lambda x,y:x*y,t) 
        tt.append(ss*data[-1])

    mean=np.mean(tt)
    std=np.std(tt)
    return mean,std


def montecarlo_simulate_array(data,n):

    data1 = [data[i]/data[i-1] for i in range(1,len(data))]
    data_log = np.log(data1)
    mean=np.mean(data_log)
    std=np.std(data_log)


    random_list = np.random.rand(n)
    s = norm.ppf(random_list, mean, std)
    ss = np.exp(s)
    sss = ss*data[-1]

    return np.mean(sss),np.std(sss)

def montecarlo_simulate_recur(data,n):


    data1 = [data[i]/data[i-1] for i in range(1,len(data))]
    data_log = np.log(data1)
    mean=np.mean(data_log)
    std=np.std(data_log)
    #s=np.random.normal(mean, std, n)
    #s=np.random.lognormal(mean, std, n)
    #print (s)
    t=[]
    random_list = np.random.rand(n)
    for i in random_list:
        s=norm.ppf(i,mean,std)
        t.append(np.exp(s)*data[-1])

    mean=np.mean(t)
    return mean

if __name__=="__main__":
    data=[4.93, 4.99, 5.02, 5.05, 5.09, 5.08, 5.0, 5.14, 5.43, 5.33]
    data=[242.16, 246.96, 248.83, 251.9, 252.8, 260.92, 248.21, 248.0, 247.5, 244.13]
    data=[243.05, 242.16, 246.96, 248.83, 251.9, 252.8, 260.92, 248.21, 248.0, 247.5]
    data = [18.83, 18.13, 19.43, 18.53, 18.54, 19.37, 19.37, 17.86, 16.09, 16.1, 16.14, 15.91, 15.69, 14.12, 13.83, 15.21, 15.54, 14.83, 14.54, 14.83]
    n=1000


    
    
    t1=timeit.Timer("montecarlo_simulate_array([18.83, 18.13, 19.43, 18.53, 18.54, 19.37, 19.37, 17.86, 16.09, 16.1, 16.14, 15.91, 15.69, 14.12, 13.83, 15.21, 15.54, 14.83, 14.54, 14.83],1000)","from __main__ import montecarlo_simulate_array")
    print t1.repeat(3,10)
    t2=timeit.Timer("montecarlo_simulate_recur([18.83, 18.13, 19.43, 18.53, 18.54, 19.37, 19.37, 17.86, 16.09, 16.1, 16.14, 15.91, 15.69, 14.12, 13.83, 15.21, 15.54, 14.83, 14.54, 14.83],1000)","from __main__ import montecarlo_simulate_recur")
    print t2.repeat(3,10)

 
