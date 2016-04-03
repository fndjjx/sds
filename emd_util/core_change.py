import matplotlib.pyplot as plt
from load_data import load_data
from emd import one_dimension_emd
from scipy.signal import argrelextrema
import numpy as np
import sys
from hurst import hurst
from sklearn import svm
from sklearn.ensemble import AdaBoostRegressor

from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


def calc_down(data):
    c=[]
    tmp=0
    for i in range(1,len(data)):
        if data[i]<data[i-1]:
            tmp+=1#data[i-1]-data[i]
        else:
            if tmp!=0:
                c.append(tmp)
                tmp=0
    if c==[]:
        c=[0]
    return np.mean(c)

def calc_up(data):
    c=[]
    tmp=0
    for i in range(1,len(data)):
        if data[i]>data[i-1]:
            tmp+=1#data[i]-data[i-1]
        else:
            if tmp!=0:
                c.append(tmp)
                tmp=0
    if c==[]:
        c=[0]
    return np.mean(c)


def trend_choice(mp,datac,period):
    train_data=[]
    lable=[]
    f = 0
    for i in range(period,len(datac)):
        if f==0:
            #if datac[i]<mp[i]:
            if datac[i]>mp[i]:# and datac[i-1]<mp[i-1]:
                train_data.append([calc_down(mp[i-period+1:i+1]),calc_up(mp[i-period+1:i+1])]) 
                #train_data.append([calc_down(mp[i-period+1:i+1])]) 
                f=datac[i]
        else:
            #if datac[i]>mp[i]:
            if datac[i]<mp[i] :#and datac[i-1]>mp[i-1]:
                if datac[i]>f:
                    lable.append(1)
                else:
                    lable.append(0)
                f=0
     
    xl = len(train_data)
    yl = len(lable)
    if xl>yl:
        train_data.pop()
     
    return train_data,lable


def stable_choice(mp,datac,mpdiff,period):
    train_data=[]
    lable=[]
    f = 0
    for i in range(period,len(datac)):
        if f==0:
            if datac[i]<mp[i] and np.mean(mpdiff[i-2:i+1])<0:
                train_data.append([calc_down(mp[i-period+1:i+1]),calc_up(mp[i-period+1:i+1])])
                #train_data.append([calc_down(mp[i-period+1:i+1])])
                f=datac[i]
        else:
            if datac[i]>mp[i] and np.mean(mpdiff[i-2:i+1])>0:
                if datac[i]>f:
                    lable.append(1)
                else:
                    lable.append(0)
                f=0

    xl = len(train_data)
    yl = len(lable)
    if xl>yl:
        train_data.pop()

    return train_data,lable

def mp_func(datav,dataj,datac):
    mp=[]
    for i in range(len(datav)):
        if datav[i]==0:
            mp.append(datac[i])
        else:
            mp.append(dataj[i]/datav[i])
    return mp

def trend_predict(mp,datac,down_count,up_count,period,mpdiff):
    train_data,lable = trend_choice(mp,datac,period) 
    if lable_valid(lable)==1:
        print train_data
        print lable
        clf = LogisticRegression()
        #clf =  GaussianNB()
        #clf = svm.SVC()
        #clf  = RandomForestClassifier(n_estimators = 100)
        print [down_count,up_count]
        clf.fit(train_data,lable) 
        #return clf.predict(np.array([down_count]))[0]
        lr= clf.predict_proba(np.array([down_count,up_count]))[0]
        clf  = RandomForestClassifier(n_estimators = 100)
        clf.fit(train_data,lable) 
        rr= clf.predict_proba(np.array([down_count,up_count]))[0]
        if ((lr[0]+rr[0])/(lr[1]+rr[1]))<0.8:
            return 1
        else:
            return 0
    elif lable_valid(lable)==2:
        return 1
    else:
        return 0

def stable_predict(mp,datac,down_count,up_count,period,mpdiff):
    train_data,lable = stable_choice(mp,datac,mpdiff,period) 
    if lable_valid(lable)==1:
        clf = LogisticRegression()
        #clf = svm.SVC()
        #clf  = RandomForestClassifier(n_estimators = 100)
        #clf =  GaussianNB()
        print train_data
        print lable
        print [down_count,up_count]
        clf.fit(train_data,lable) 
        #return clf.predict(np.array([down_count]))[0]
        lr=clf.predict_proba(np.array([down_count,up_count]))[0]
        clf  = RandomForestClassifier(n_estimators = 100)
        clf.fit(train_data,lable)
        rr= clf.predict_proba(np.array([down_count,up_count]))[0]
        if ((lr[0]+rr[0])/(lr[1]+rr[1]))<0.8:
            return 1
        else:
            return 0
    elif lable_valid(lable)==2:
        return 1
    else:
        return 0
    
def lable_valid(data):
    c=len(data)
    c0=0
    c1=0
    for i in data:
        if i==0:
            c0+=1
        if i==1:
            c1+=1
    if c0==c:
        return 0
    elif c1==c:
        return 2
    else:
        return 1
if __name__=="__main__":

    filename=sys.argv[1]
    start=int(sys.argv[2])
    end=int(sys.argv[3])
    train_length = 200 
    datac =  load_data(filename,4)
    datav =  load_data(filename,5)
    dataj =  load_data(filename,6)
    mp=mp_func(datav,dataj,datac)[start:end]
    datac=datac[start:end]
    mpdiff=[datac[i]-mp[i] for i in range(len(mp))]
    train_data,lable = stable_choice(mp,datac,mpdiff,30)
    train_data,lable = trend_choice(mp,datac,30)
    print len(train_data)
    print len(lable)
    clf = LogisticRegression()
    
   # clf  = RandomForestClassifier(n_estimators = 500)
   # clf =  GaussianNB()
    clf.fit(train_data,lable)
    scores = cross_validation.cross_val_score(clf, np.array(train_data), np.array(lable), cv=10)
   # print clf.predict([0.3,1.1])
    print scores
    plt.scatter(np.array(train_data).T[0],np.array(train_data).T[1],c=lable)
    plt.show()
    
