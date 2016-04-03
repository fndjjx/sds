
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def ma_func(data, period):
    ma_list=[]
    for i in range(period):
        ma_list.append(0)
    for i in range(period,len(data)):
        ma_list.append(np.mean(data[i-(period-1):i+1]))
    return ma_list

def combinelist_simple(listlist):
    listlist=[list(i) for i in listlist]
    print listlist


    length=len(listlist[0])

    colsum=[]
    for i in range((length)):
        tmp=[]
        for j in range(len(listlist)):
            tmp.append(listlist[j][i])
        colsum.append(sum(tmp))


    combinelist=list([(colsum[i])/len(listlist) for i in range(length)])
    print "comb1= %s"%combinelist
    combinelist=list(splinerestruct(combinelist))
    print "comb2= %s"%combinelist



    return combinelist

def combinelist(listlist,plist):
    listlist=[list(i) for i in listlist]
    print listlist
    print len(listlist)
    print len(plist)


    length=len(listlist[0])

    colsum=[]
    for i in range((length)):
        tmp=[]
        for j in range(len(listlist)):
            #tmp.append(listlist[j][i]*(1.0/plist[j]))
            tmp.append(listlist[j][i])
        colsum.append(sum(tmp))


    combinelist=list([(colsum[i])/len(listlist) for i in range(length)])
    print "comb1= %s"%combinelist
#    combinelist=list(splinerestruct(combinelist))
#    combinelist=ma_func(combinelist,3)
    print "comb2= %s"%combinelist
    return combinelist


def splinerestruct(raw_data,kk=3):

    len_raw_data = len(raw_data)
    x1 = np.linspace(0,len_raw_data-1,len_raw_data)
    y1 = raw_data


    sx1 = np.linspace(0,len(raw_data)-1,len(raw_data))
    sy1 = interpolate.UnivariateSpline(x1,y1,k=kk,s=100)(sx1)

    return sy1





def combinelist_max(listlist):
    print listlist

    listpair=[]
    for i in listlist:
        listpair.append((len(i),i))
    
    listpair.sort(key=lambda n:n[0])

    maxlength=listpair[-1][0]

    extenlist=[]
  
    for i in range(len(listlist)-1):
        x1 = np.linspace(0,listpair[i][0]-1,listpair[i][0])
        y1 = listpair[i][1]


        sx1 = np.linspace(0,maxlength-1,maxlength)
        sy1 = interpolate.UnivariateSpline(x1,y1,k=3,s=0)(sx1)

        extenlist.append(list(sy1))

    colsum=[]
    for i in range((maxlength)):
        tmp=[]
        for j in range(len(extenlist)):
            tmp.append(extenlist[j][i])
        colsum.append(sum(tmp))

    
    combinelist=[(listpair[-1][1][i]+colsum[i])/len(listlist) for i in range((listpair[-1][0]))] 
    return combinelist

def combinelist_min(listlist):

    listpair=[]
    for i in listlist:
        listpair.append((len(i),i))

    listpair.sort(key=lambda n:n[0])

    minlength=listpair[0][0]

    extenlist=[]

    for i in range(1,len(listlist)):
        x1 = np.linspace(0,listpair[i][0]-1,listpair[i][0])
        y1 = listpair[i][1]


        sx1 = np.linspace(0,minlength-1,minlength)
        sy1 = interpolate.UnivariateSpline(x1,y1,k=3,s=100)(sx1)

        extenlist.append(list(sy1))

    colsum=[]
    for i in range((minlength)):
        tmp=[]
        for j in range(len(extenlist)):
            tmp.append(extenlist[j][i])
        colsum.append(sum(tmp))


    combinelist=[(listpair[0][1][i]+colsum[i])/len(listlist) for i in range((listpair[0][0]))]
    return combinelist



if __name__=="__main__":

    a=[-0.12553837,-0.07723251,0.00489069,0.08668229,0.1322072,0.11518091,0.0494985,-0.03498483,-0.10683648,-0.13832941]
    b=[-0.11973996,-0.09484061,0.02607782,0.09089417,0.1422863,0.09408713,-0.00706206,-0.11571574]
    b=[-i for i in b]
    c=[-0.4*i for i in b]
    z=[a,b,c]
    plt.plot(a,'r')
    plt.plot(b,'y')
    plt.plot(combinelist(z),'b')
    plt.show()
    
