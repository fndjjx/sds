
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from load_data import load_data
import matplotlib.pyplot as plt
import numpy as np
import sys


def ma_func(data, period):
    ma_list=[]
    for i in range(period):
        ma_list.append(data[i])
    for i in range(period,len(data)):
        ma_list.append(np.mean(data[i-(period-1):i+1]))
    return ma_list

def train_data_generate(filename):
    datav = load_data(filename,5)
    dataj = load_data(filename,6)
    datac = load_data(filename,4)
    mp = []
    for i in range(len(datav)):
        mp.append(dataj[i]/datav[i])
    datac = [i**1 for i in datac]
    mp = [i**1 for i in mp]


    ma10 = ma_func(datac,3)
    diff10 = list(np.diff(ma10))
    diff10.insert(0,0)

    sample = zip(datac,mp)
    sample = [list(i) for i in sample]
    lable = []
    b_flag=0
    tmp=0
#    for i in range(len(datac)):
#        if mp[i]>datac[i] and b_flag==0:
#            b_flag=datac[i]
#            lable.append(1)
#            tmp=i
#        elif mp[i]<datac[i] and b_flag!=0:
#            if b_flag>datac[i]:
#                lable[tmp]=2
#            lable.append(0)
#            b_flag=0
#        else:
#            lable.append(0)

    for i in range(1,len(datac)-3):
        if 0:#i == len(datac)-3:
            lable.append(0)
        else:
            if datac[i-1]<mp[i-1] and  (datac[i+1]<datac[i] or datac[i+2]<datac[i] or datac[i+3]>datac[i] or datac[i+4]>datac[i] or datac[i+5]>datac[i]):
                lable.append(1)
            else:#if diff10[i]<=0:
                lable.append(0)
        
    print "pro"
    print sum(lable)/float(len(lable))
#    print lable
#    newlist=[]
#    for i in range(len(lable)):
#        if lable[i]==0 and np.random.rand()>0.8:
#            newlist.append(i)
#        if lable[i]!=0:
#            newlist.append(i)


#    nl=[]
#    for i in range(len(lable)):
#        if  i in newlist:
#            nl.append(lable[i])
#
#    ns=[]
#    for i in range(len(sample)):
#        if i in newlist:
#            ns.append(sample[i])
#
#    sample=ns
#    lable=nl
            
            

    return np.array(sample),np.array(lable)


def judge(filename):

    sample,lable = train_data_generate(filename)
    clf  = RandomForestClassifier(n_estimators = 600)
    clf = LogisticRegression()
#    clf = svm.SVC(kernel='linear')
#    clf=GaussianNB()
    trainx = sample[:-30]
    trainy = lable[:-30]
    testx = sample[-30:]
    testy = lable[-30:]
    print trainx
    print trainy
    clf.fit(trainx,trainy)
    scores = cross_validation.cross_val_score(clf, sample, lable, cv=10)
    result = []
    
    for i in testx:
        result.append(clf.predict(np.array(i))[0])
    print testy
    print result
    print scores



if __name__ == "__main__":
    filename = sys.argv[1] 
    judge(filename)


