import matplotlib.pyplot as plt
from load_data import load_data
from load_data import load_date_data
import sys


def calc_power(datac,datav,datao,datad):
    power = []
    flag = []
    for i in range(len(datac)):
        if i == 0:
            power.append(0)
            flag.append(0)
        else:
            t = 0.5*datav[i]*((datac[i]-datao[i]))**2
            if datac[i]>datao[i]:
                power.append(power[-1]+t)
                flag.append(1)
            else:
                power.append(power[-1]-t)
                flag.append(-1)

    return zip(power,flag,datao,datac,datav,datad)


def calc_pro(data):
    c1=0
    c2=0
    c3=0
    c4=0
    c=float(len(data))
    cp=0
    cn=0
    for i in data:
        if i[0]<0 and i[1]<0:
            c1+=1
            cn+=1
        if i[0]<0 and i[1]>0:
            c2+=1
            cp+=1
        if i[0]>0 and i[1]<0:
            c3+=1
            cn+=1
        if i[0]>0 and i[1]>0:
            c4+=1
            cp+=1
    cp=float(cp)
    cn=float(cn)
    print "{} {} {} {}".format(c1/cn,c2/cp,c3/cn,c4/cp)

def calc_pro2(datac,dataj,datav,n):
    mp = [dataj[i]/datav[i] for i in range(len(datav))]
    c=0
    c1=0
    for i in range(1,len(datac)-n):
        if datac[i]<mp[i] and datac[i-1]>mp[i-1]:
            c+=1
            if datac[i+n]<datac[i]:
                c1+=1
    print "recent down pro{}".format(float(c1)/c)
    return float(c1)/c
    
if __name__=="__main__":

    filename=sys.argv[1]
    start=int(sys.argv[2])
    end=int(sys.argv[3])
    n=int(sys.argv[4])
    datac =  load_data(filename,4)[start:end]
    datav =  load_data(filename,5)[start:end]
    dataj =  load_data(filename,6)[start:end]
    datao =  load_data(filename,1)[start:end]
    datad =  load_date_data(filename,0)[start:end]
    #power = calc_power(datac,datav,datao,datad)
    #print power
    #print datac
    #calc_pro(power)
    calc_pro2(datac,dataj,datav,n)
