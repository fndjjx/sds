import matplotlib.pyplot as plt
from load_data import load_data
from emd import one_dimension_emd
from scipy.signal import argrelextrema
import numpy as np
import sys
from hurst import hurst
from sklearn import svm
from sklearn.ensemble import AdaBoostRegressor


def svm_predict(data, period):
    train_data = []
    lable = []
    for i in range(period,len(data)):
        tmp = data[i-period:i]
        train_data.append(tmp)
        lable.append(data[i])
    print train_data
    print lable
   # rng = np.random.RandomState(1)

    #clf = AdaBoostRegressor(svm.SVR(),n_estimators=10, random_state=rng)
    clf = svm.SVR()
    clf.fit(train_data, lable)

    predict_value=clf.predict(data[len(data)-period:])[0]

    return predict_value


def repeat(data,train_length,period):
    predict=[]
    for i in range(train_length,len(data)-1):
        real_data = data[i-train_length:i+1]
        predict.append(svm_predict(real_data,period))

    return predict


def mp_func(datav,dataj,datac):
    mp=[]
    for i in range(len(datav)):
        if datav[i]==0:
            mp.append(datac[i])
        else:
            mp.append(dataj[i]/datav[i])
    return mp

if __name__=="__main__":

    filename=sys.argv[1]
    start=int(sys.argv[2])
    end=int(sys.argv[3])
    train_length = 200 
    datac =  load_data(filename,4)
    datav =  load_data(filename,5)
    dataj =  load_data(filename,6)
    mp=mp_func(datav,dataj,datac)[start-train_length:end]
    predict = repeat(mp,train_length,10)
    print len(mp[start:])
    print len(predict)
    plt.plot(mp[train_length+1:],'r')
    plt.plot(predict)
    plt.show()
