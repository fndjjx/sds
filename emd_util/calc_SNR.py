
from emd import *
import numpy as np
import sys
import matplotlib.pyplot as plt



def calc_SNR(data,imf,index):
    raw_data = data
    period=[]
    flag=0
    for i in range(len(imf)):
#        print i
        imf_single = imf[i]
        data = np.array(imf_single)
        max_index = list(argrelextrema(data,np.greater)[0])
        min_index = list(argrelextrema(data,np.less)[0])
        if len(max_index)!=0 and len(min_index)!=0:
            period.append(len(data)/float(len(max_index)))
 #           print period
            if len(period)>1 and ((period[-1]/float(period[-2]))<1.9 or (period[-1]/float(period[-2]))>2.1):
                flag=i
                break
            if i==len(imf)-1:
                flag=i+1
        else:
            flag=i+1
            break
  #      print "flag%s"%flag

   # print "signal imf %s"%(flag+1)
    para1= [np.var(imf[i][-index:]) for i in range(flag)]
    #print para1
    SNR1 = np.var(raw_data[-index:])-sum(para1)
    #print SNR1
    SNR2 = sum(para1)
    #print SNR2
    #print (SNR1/SNR2)
    SNR = 10*np.log10(SNR1/SNR2)
    #print SNR
    return (SNR,flag+1)

def calc_SNR2(data,imf,residual):

    para1= [data[i]-residual[i] for i in range(len(data))]
    SNR1 = np.std(para1)
    SNR2 = np.std(residual)
    print (SNR1/SNR2)
    SNR = 10*np.log10(SNR1/SNR2)
    print SNR
    return SNR

    






if __name__=="__main__":
    datafile = sys.argv[1]
    f=open(datafile)
    lines=f.readlines()
    f.close()
    close_data=[]
    for line in lines:
        close_data.append(float(line.split("\t")[4]))

    data=close_data
    ma5 = []
    for i in range(5):
        ma5.append(0)

    for i in range(5,len(data)):
        mean_5 = np.mean(data[i-4:i+1])
        ma5.append(mean_5)

    data = []
    for i in range(5):
        data.append(0)

    for i in range(5,len(ma5)):
        mean_5 = np.mean(ma5[i-4:i+1])
        data.append(mean_5)

    ma5 = []
    for i in range(5):
        ma5.append(0)

    for i in range(5,len(data)):
        mean_5 = np.mean(data[i-4:i+1])
        ma5.append(mean_5)


    SNR=[]
    for i in range(1000,len(close_data)):
        data = close_data[i-50:i]
        print "len %s"%len(data)
        my_emd = one_dimension_emd(data)
        (imf, residual) = my_emd.emd(0.01,0.01)
        print len(imf)
        SNR.append(calc_SNR(data,imf))
    #my_emd = one_dimension_emd(close_data)
    #(imf, residual) = my_emd.emd(0.01,0.01)
    #SNR.append(calc_SNR(close_data,imf))

    #print np.mean(SNR)
    #print np.std(SNR)
    #data=SNR

    plt.figure(1)
    plt.plot(close_data[1000:])
    plt.figure(2)
    plt.plot(data)
    plt.show()

    
    
