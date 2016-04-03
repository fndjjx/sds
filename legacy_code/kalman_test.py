
import sys
import os
import matplotlib.pyplot as plt
sys.path.append("../emd_util")
from generate_emd_data import generateEMDdata
from analyze import *
from spline_predict import linerestruct
from emd import *
from getMinData import *
from kalman import kalman_simple
from leastsqt import leastsqt_predict

def getanytime(file_data, date, time):
    data = getMinDataWholeday(file_data,date)

    p1 = getMinDatabyTime(data, time)
    return float(p1)


def getMinDataAllYears(datafile, years):
    datafile = datafile.split("/")[-1]
    current_path = os.getcwd()
    if datafile.startswith("sh"):
        minbasedir = current_path + "/test_data/mindata" + "/sh"
    else:
        minbasedir = current_path + "/test_data/mindata" + "/sz"


    mindatalist = []
    for i in years:
        mindir = minbasedir + "/%s"%i
        mindatalist.append(getMinDatabyFile(mindir + "/min1_%s.csv"%datafile))
    mindata = []
    for i in mindatalist:
        mindata += i

    return mindata

def ma_generate2(data,n):
    ma=[]
    for i in range(n):
        ma.append(i+np.random.standard_normal(1)[0])
    for i in range(n,len(data)):
        ma.append(np.mean(data[i-(n-1):i+1]))

    return ma

def main(datafile, begin, end):

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
        #macd.append(float(eachline.split("\t")[19]))
        macd.append(float(eachline.split("\t")[5]))
        open_price.append(float(eachline.split("\t")[1]))
        date.append(eachline.split("\t")[0])
        vol.append(float(eachline.split("\t")[5]))
        je.append(float(eachline.split("\t")[5]))


    mindata = getMinDataAllYears(datafile,(2010,2011,2012,2013))
    halfclose_list=[]
    near_list=[]
    near_list2=[]
    mc=[]
    ao=[]
    for i in range(begin,end):
        cdate = date[i].replace("/","-")
        cdate = cdate.replace(" ","")
        #p=float(getanytime(mindata,cdate,"11:30:00"))
        #halfclose_list.append(p)
        #p1=float(getanytime(mindata,cdate,"09:31:00"))
        #p2=float(getanytime(mindata,cdate,"10:00:00"))
        #p3=float(getanytime(mindata,cdate,"10:30:00"))
        #p4=float(getanytime(mindata,cdate,"11:00:00"))
        p5=float(getanytime(mindata,cdate,"11:30:00"))
        mc.append(p5)
        p5=float(getanytime(mindata,cdate,"13:01:00"))
        ao.append(p5)
        ####p6=float(getanytime(mindata,cdate,"09:45:00"))
        #p=(p1+p2+p3+p4+p5)/5.0
        p=float(getanytime(mindata,cdate,"14:00:00"))
        near_list.append(p)
        #near_list2.append(p5)
#
    #near_list = open_price[begin:end]
    
    open_ma = ma_generate2(open_price,2)
    print "openma%s"%open_ma
    close_ma = ma_generate2(close_price,2)
    print "openma%s"%close_ma
    print open_ma
    #close_ma = ma_generate2(ma_generate2(ma_generate2(close_price,3),3),3)
    #close_ma=close_ma[3:]
    print "close%s"%close_ma
    #k_v1 = list(kalman_simple(open_price[begin:],near_list))

    
    k_v2 = list(kalman_simple(close_price[begin-1:end-1],close_price[begin:end]))
    #k_v1 = list(kalman_simple(k_v[:-1],close_price[begin+1:end]))
    k_v1 = list(kalman_simple(open_ma,close_ma))
    print "k_v1%s"%k_v1
    k_v = list(kalman_simple(near_list,close_price[begin:end]))
    k_v1 = list(kalman_simple(near_list,close_price[begin:end]))
    print k_v

    cdiff = np.diff(close_price[begin:end])
    odiff = np.diff(open_price[begin:end])
    o = (open_price[begin:end])
    c = (close_price[begin:end])
    count1=0
    count2=0
    count3=0
    for i in range(len(ao)):
        if o[i]<mc[i] and ao[i]<c[i]:
            count1+=1 
        if o[i]<mc[i]:
            count2+=1
        if c[i]>ao[i]:
            count3+=1

    print "bayes %s"%(count1/float(count2))
    print "bayes %s"%(count3/float(len(ao)))
        
        

    open_ma = open_ma[begin:]
    #close_ma=close_ma[begin:]
    #diff = [abs(k_v[i]-halfclose_list[i]) for i in range(len(k_v))]

    #k_v1.insert(0,k_v[0])
    #diff = [abs(k_v1[i]-close_price[begin:end][i]) for i in range(len(k_v1))]
    #print "chaju %s"%sum(diff)


    
    count=0
    count2=0
    close=close_price[begin:end]
    kz_list=[]
    zhi_list=[]
    rz_list=[]
    #for i in range(len(k_v)):
        #if k_v[i]<close_price[begin:end][i-1] and close_price[begin:end][i]<close_price[begin:end][i-1]:
        #if k_v[i]>open_price[begin:end][i] and close_price[begin:end][i]>open_price[begin:end][i]:
    #    zhi5=close[i-1]+close[i-2]+close[i-3]+close[i-4]
    #    zhi5=zhi5+k_v[i]
    #    zhi2=close[i-1]+k_v[i]
    #    kz=zhi2/2.0-zhi5/5.0

    #    pp2=leastsqt_predict(close_price[begin+i-10:i+begin],1)[0]
    #    zhi5=close[i-1]+close[i-2]+close[i-3]+close[i-4]
    #    zhi5=zhi5+pp2
    #    zhi2=close[i-1]+pp2
    #    zhi=zhi2/2.0-zhi5/5.0

    #    zhi5=close[i-1]+close[i-2]+close[i-3]+close[i-4]
    #    zhi5=zhi5+close[i]
    #    zhi2=close[i-1]+close[i]
    #    rz=zhi2/2.0-zhi5/5.0
    #    kz_list.append(kz)
    #    rz_list.append(rz)
    #    zhi_list.append(zhi)


        #if zhi>close_ma[begin:end][i] and close_ma[begin:end][i]>close_ma[begin:end][i-1]:
    #    if len(zhi_list)>1 and zhi_list[-2]-rz_list[-2]>0 and zhi-rz<0 and  zhi-kz<0 :
    #    
    #        count+=1
    #    if len(zhi_list)>1 and zhi_list[-2]-rz_list[-2]>0 and zhi-rz<0  :
    #        count2+=1

    #print "count %s"%(float(count)/(len(k_v)-1))
    #print "count %s"%(float(count)/(count2))
        
    print len(close_price)
    print len(k_v)+3
    plt.figure(1)
    #plt.plot(close_price[begin+1:end],'g')
    plt.plot(close_price[begin:end],'g')
    plt.plot(k_v,'r')
#    plt.plot(k_v2,'y')
    plt.plot(k_v1,'b')
#    plt.figure(2)
#    plt.plot(diff,'r')
    plt.show()


def ma_generate(data):
    ma2_list=[]
    for i in range(2):
        ma2_list.append(0)
    for i in range(2,len(data)):
        ma2_list.append(np.mean(data[i-1:i+1]))


    ma5_list=[]
    for i in range(5):
        ma5_list.append(0)
    for i in range(5,len(data)):
        ma5_list.append(np.mean(data[i-4:i+1]))

    ma_list=[ma2_list[i]-ma5_list[i] for i in range(len(data))]
    return ma_list


if __name__ == "__main__":

    datafile = sys.argv[1]
    begin = int(sys.argv[2])
    end = int(sys.argv[3])
    main(datafile,begin,end)
