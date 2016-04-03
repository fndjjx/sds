
from scipy.signal import argrelextrema
import statsmodels.tsa.stattools as ts
from scipy import stats
from load_data import load_data
import matplotlib.pyplot as plt
import sys
import numpy as np
def draw_scatter(filename,start,end):
    datav = load_data(filename,5)[start:end]
    dataj = load_data(filename,6)[start:end]
    datac = load_data(filename,4)[start:end]
    mp = [dataj[i]/datav[i] for i in range(len(datav))]
    
    lable = []
    for i in range(len(datac)):
        if i == 0:
            lable.append(0)
        else:
            if datac[i]>datac[i-1]:
                lable.append(1)
            else:
                lable.append(0)
    datac = [i**1 for i in datac]
    mp = [i**1 for i in mp]
    plt.scatter(datac,mp,c=lable)
    plt.show()
    

def mp_func(datav,dataj,datac):
    mp=[]
    for i in range(len(datav)):
        if datav[i]==0:
            mp.append(datac[i])
        else:
            mp.append(dataj[i]/datav[i])
    return [datac[i]-mp[i] for i in range(len(datac))]

def ma_func(data, period):
    ma_list=[]
    for i in range(period):
        ma_list.append(data[i])
    for i in range(period,len(data)):
        ma_list.append(np.mean(data[i-(period-1):i+1]))
    return ma_list

def load_mp(filename,start,end):
    datac =  load_data(filename,4)[start:end]
    datav =  load_data(filename,5)[start:end]
    dataj =  load_data(filename,6)[start:end]

    return datac,dataj,datav

def stats_mpup(filename,start,end):
    datac,dataj,datav=load_mp(filename,start,end)
    mp=mp_func(datav,dataj,datac)
    c1=0
    c2=0
    l=[]
    for i in range(1,len(mp)):
        if mp[i]<0:
            c1+=1
            if datac[i]<datac[i-1]:
                c2+=1
    mpdowncdown=c2/float(c1)

    c1=0
    c2=0
    l=[]
    for i in range(1,len(mp)):
        if mp[i]>0:
            c1+=1
            if datac[i]>datac[i-1]:
                c2+=1
    mpupcup=c2/float(c1)
    return mpdowncdown,mpupcup

def compare_coef(filename,start,end):
    datac,dataj,datav=load_mp(filename,start,end)    
    mp=mp_func(datav,dataj,datac)
    ma1=ma_func(datac,2)
    ma2=ma_func(datac,5)
    madiff=[ma1[i]-ma2[i] for i in range(len(datac))]
    return np.corrcoef(mp,madiff)

def stats_diffmp(filename,start,end):
    datac,dataj,datav=load_mp(filename,start,end)
    mp=mp_func(datav,dataj,datac)
    tmp=0
    l=[]
    for i in range(len(mp)):
        if mp[i]<0:
            tmp+=1
        else:
            l.append(tmp)
            tmp=0
    l.append(tmp)
    l=filter(lambda x:x>0 ,l)
    rb=np.mean(l)
    print "below l {} {}".format(np.mean(l),np.std(l))


    tmp=0
    l=[]
    for i in range(len(mp)):
        if mp[i]>0:
            tmp+=1
        else:
            l.append(tmp)
            tmp=0
    l.append(tmp)
    l=filter(lambda x:x>0 ,l)
    ru=np.mean(l)
    print "up l {} {}".format(np.mean(l),np.std(l))
    return rb,ru
    

def repeat_func(filename,start,end,period,func):
    length = end-start
    epn = length/period
    rr=[]
    for i in range(period):
        start1=start+i*epn
        end1=start1+epn
        print start1
        print end1
        r = eval(func)(filename,start1,end1)
        print r
        rr.append(r)


def compare(filename1,start1,end1,filename2,start2,end2):
    data1c =  load_data(filename1,4)[start1:end1]
    data1v =  load_data(filename1,5)[start1:end1]
    data1j =  load_data(filename1,6)[start1:end1]

    data2c =  load_data(filename2,4)[start2:end2]
    data2v =  load_data(filename2,5)[start2:end2]
    data2j =  load_data(filename2,6)[start2:end2]

    mp1 = mp_func(data1v,data1j,data1c)
#    print mp1
    print "mp1 {} {}".format(np.mean(mp1),np.std(mp1))
    tmp=0
    l=[]
    for i in range(len(mp1)):
        if mp1[i]<0:
            tmp+=1
        else:
            l.append(tmp)
            tmp=0
    l=filter(lambda x:x>0 ,l)
    print l
    print "l {} {}".format(np.mean(l),np.std(l))


    tmp=0
    l=[]
    for i in range(len(mp1)):
        if mp1[i]<0:
            tmp+=1
        else:
            l.append(tmp)
            tmp=0
    l=filter(lambda x:x>0 ,l)
    print l
    print "l {} {}".format(np.mean(l),np.std(l))
   
    mp2 = mp(data2v,data2j,data2c)
#    print mp2
    print "mp2 {} {}".format(np.mean(mp2),np.std(mp2))
    ks_d,ks_p_value = stats.ks_2samp(mp1,mp2)

    print ks_p_value

def riseperiod(filename,start,end):
    data =  load_data(filename,4)[start:end]
    tmp = 0
    p=[]
    for i in range(1,len(data)):
        if data[i]<data[i-1]:
            tmp += 1
        else:
            p.append(tmp)
            tmp = 0
    print p
    p=filter(lambda x:x!=0, p)
    print p
    return np.mean(p),np.std(p)

def divide(data,n):
    l=[]
    ll=[]
    en1=len(data)/n
    en2=len(data)%n
    j=1
    for i in range(len(data)):
        if i<j*en1:
            ll.append(data[i])
        else:
            l.append(ll)
            ll=[]
            ll.append(data[i])
            j+=1
    return l
        
def calc_pro(l1,l2):
    c=0
    for i in l1:
        if i in l2:
            c+=1
    return float(c)/len(l1)
        
def pp(x):
    print "{} {}".format(np.mean(x),np.std(x))
if __name__=="__main__":
    filename1 = sys.argv[1]
    start1 = int(sys.argv[2])
    end1 = int(sys.argv[3])
    period = int(sys.argv[4])
#    compare_coef(filename1,start1,end1)
    repeat_func(filename1,start1,end1,period,"stats_mpup")
##    print riseperiod(filename,start,end)
##    start1 = int(sys.argv[2])
##    end1 = int(sys.argv[3])
#    filename2 = sys.argv[4]
#    start2 = int(sys.argv[5])
#    end2 = int(sys.argv[6])
#    compare(filename1,start1,end1,filename2,start2,end2)
#    draw_scatter(filename,start,end)
#    print "ksaxxt {} {}".format(np.mean(ks_axxt),np.std(ks_axxt))
#    print "kszgzg {} {}".format(np.mean(ks_zgzg),np.std(ks_zgzg))
#    a=divide(ks_axxt,50)
#    print len(a[0])
#    print len(a)
#    b=divide(ks_zgzg,50)
#    print "axxt"
#    map(pp,a)
#    print "zgzg"
#    map(pp,b)
    
#    datac =  load_data(filename1,4)[start1:end1]
#    datav =  load_data(filename1,5)[start1:end1]
#    dataj =  load_data(filename1,6)[start1:end1]
# #   
#    mp=[]
#    for i in range(len(datav)):
#        if datav[i]==0:
#            mp.append(datac[i])
#        else:
#            mp.append(dataj[i]/datav[i])
#    diffmp= mp_func(datav,dataj,datac)
#
#    max_diff_index = argrelextrema(np.array(diffmp),np.greater)[0]
#    min_diff_index = argrelextrema(np.array(diffmp),np.less)[0]
#
#    print "diff max {}".format(max_diff_index)
#    print "diff min {}".format(min_diff_index)
#    max_index = argrelextrema(np.array(datac),np.greater)[0]
#    min_index = argrelextrema(np.array(datac),np.less)[0]
#
#    print "max {}".format(max_index)
#    print "min {}".format(min_index)
#
#    print calc_pro(min_diff_index,max_index)
#    print calc_pro(min_diff_index,min_index)
#    plt.plot(datac,'r')
#    plt.plot(mp,'b')
#    plt.figure(2)
#    plt.plot(diffmp)
#    plt.show()
    
    
    
    

