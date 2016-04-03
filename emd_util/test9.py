from monte_carlo import montecarlo_simulate
from monte_carlo import montecarlo_simulate2
import statsmodels.tsa.stattools as ts
from load_data import load_data
import sys
import numpy as np
import matplotlib.pyplot as plt

def ma_func(data, period):
    ma_list=[]
    for i in range(period):
        ma_list.append(data[i])
    for i in range(period,len(data)):
        ma_list.append(np.mean(data[i-(period-1):i+1]))
    return ma_list

if __name__=="__main__":

    filename=sys.argv[1]
    start=int(sys.argv[2])
    end=int(sys.argv[3])
    datac =  load_data(filename,4)[start:end]
    datav =  load_data(filename,5)[start:end]
    dataj =  load_data(filename,6)[start:end]
    datao =  load_data(filename,1)[start:end]
    ma = ma_func(load_data(filename,4),60)
    allv = load_data(filename,5)
    allc = load_data(filename,4)
    allj = load_data(filename,6)
    ma = ma_func([allj[i]/allv[i] for i in range(len(allv))],3)
    ma = [allj[i]/allv[i] for i in range(len(allv))]
    ddd = ma

    r = [ddd[i]/ddd[i-1] for i in range(1,len(ddd))]
    r = [i-1 for i in r]

    cm=[allc[i]-ma[i] for i in range(len(allc))][start:end]
    mp = [dataj[i]/datav[i] for i in range(len(datav))]
    #mp=datac
    index_down=[]
    index_up=[]
    value_down=[]
    value_up=[]
    normal_down=[]
    value_normal_down=[]
    kk=[]
    kkk=[]
    for delay in range(1):
        c1=0
        c2=0
        c3=0
        c4=0
        delay=3
        for i in range(20,len(datac)):
            #rc=datac[i-20:i+1]
            rc1=mp[i-24:i-2]
            rc2=mp[i-21:i]
            rc3=datac[i-20:i+1]
            rc4=datac[i-21:i]
        #    p1,m1,s1=montecarlo_simulate(rc1,100)
          #  m2,s2=montecarlo_simulate2(rc1,50,delay)
        #    p3,m3,s3=montecarlo_simulate(rc3,100)
            #p4,m4,s4=montecarlo_simulate(rc4,100)
            stab=ts.adfuller(r[i-20:i+1],1)
            kk.append(stab[0])
            kkk.append(stab[4]["1%"])

         #   stddiff=np.std(cm[i-10:i+1])
         #   if m2-s2>mp[i] :#and mp[i]>datac[i]:#and len(s)>1 and s[-1]>s[-2]:
         #       index_down.append(i)
         #       value_down.append(mp[i])
         #       c1+=1
         #       if mp[i]>mp[i+3] :#and mp[i]>mp[i+2] and mp[i]>mp[i+1]:
         #           c2+=1
         #   if m2>mp[i]>m2-s2 :#and mp[i]>datac[i]:#and len(s)>1 and s[-1]>s[-2]:
         #       c3+=1
         #       index_up.append(i)
         #       value_up.append(mp[i])
         #       if mp[i]<mp[i+3] :#and mp[i]>mp[i+2] and mp[i]>mp[i+1]:
         #           c4+=1
         #   if m2<mp[i]<m2+s2 :#and mp[i]>datac[i]:#and len(s)>1 and s[-1]>s[-2]:
         #       c3+=1
         #       normal_down.append(i)
         #       value_normal_down.append(mp[i])

        print delay
        print c1
        print c3
       # print float(c2)/c1
       # print float(c4)/c3
    ddd = ma

    r = [ddd[i]/ddd[i-1] for i in range(1,len(ddd))]
    r = [i-1 for i in r]
    #r=ma_func(r,10)[start:end]
    #r=[sum(r[i-4:i+1]) for i in range(4,len(r))]
    r=r[start:end]

    print np.std(datac)
    print np.std(ma_func(datac,5))
    
    plt.subplot(311) 
    plt.plot([i for i in range(len(datac))],datac,index_up,value_up,'g*',normal_down,value_normal_down,'k*',index_down,value_down,'ro')
    plt.subplot(312) 
#    plt.plot(datac)
#    plt.plot(mp,'r')
    #plt.plot(r)
    kk=[0]*20+kk
    kkk=[0]*20+kkk
    plt.plot(kk)
    plt.plot(kkk,'r')
    plt.subplot(313) 
    datac=np.diff(datac)
    mp=np.diff(mp)
    f = list(datac-mp)
    f.insert(0,0)
    
    plt.plot(f)
#    plt.plot(datac)
#    plt.plot(mp,'r')
    print np.std(datac)
    print np.std(ma_func(datac,5))
    plt.show()
    print np.std(datac)
    print np.std(ma_func(datac,5))
            
