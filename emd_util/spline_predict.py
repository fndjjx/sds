import numpy as np
from scipy import interpolate
import copy

def splinepredict(raw_data,predict_num,through):

    len_raw_data = len(raw_data)
    
    x1 = np.linspace(0,len_raw_data-1,len_raw_data)
    y1 = raw_data


    sx1 = np.linspace(0,len_raw_data+predict_num-1,len_raw_data+predict_num)
    sy1 = interpolate.UnivariateSpline(x1,y1,s=through)(sx1)

    return sy1[len_raw_data:]

def splinerestruct(raw_data,kk=3):

    len_raw_data = len(raw_data)
    x1 = np.linspace(0,len_raw_data-1,len_raw_data)
    y1 = raw_data


    sx1 = np.linspace(0,len(raw_data)-1,len(raw_data))
    sy1 = interpolate.UnivariateSpline(x1,y1,k=kk,s=1000)(sx1)

    return sy1

def splinerestruct2(raw_data,data,index,kk=3,ss=3):

    x1 = index
    y1 = data

    print data
    print index
    print kk
    print ss
    sx1 = np.linspace(0,len(raw_data)-1,len(raw_data))
    sy1 = interpolate.UnivariateSpline(x1,y1,k=kk,s=ss)(sx1)

    return sy1

def linerestruct(x,y):
    A = np.vstack([x,np.ones_like(x)]).T
    a,b = np.linalg.lstsq(A,y)[0]
    line = [a*i+b for i in x]
    return line

if __name__ == "__main__":

    x = [0,4,5,7,12]
    test_data = [1,2,3,4,5]
    result_data = linerestruct(x,test_data)
    print result_data
   
    

