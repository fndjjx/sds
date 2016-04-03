import datetime
from multiprocessing import Process, Queue,Lock,Manager
from multiprocessing import Pool
from emd import *
from scipy.signal import argrelextrema
from scipy.signal import welch
from scipy.interpolate import UnivariateSpline
import numpy as np
import math
from smooth import cubicSmooth5

import matplotlib.pyplot as plt

def emd_process_once(data):
    my_emd = one_dimension_emd(list(data),3)
    (imf, residual) = my_emd.emd(0.005,0.005)
    return imf

def emd_process_once2(data,lock,result):
    r = emd_process_once(data)
    with lock:
        result.append(r)

class eemd():

    def __init__(self,data,n):
        self.data = data
        self.n = n
        self.f = emd_process_once
        self.f2 = emd_process_once2

    def eemd_process_single(self,data,n,percise):
        imf0 = []
        imf1 = []
        imf2 = []
        imf3 = []
        for i in range(n):
            noise = np.std(data)*0.1*np.random.standard_normal(len(data))
            data_addnoise = np.array(data)+np.array(noise)
            my_emd = one_dimension_emd(list(data_addnoise),3)
            (imf, residual) = my_emd.emd(percise,percise)
            imf0.append(imf[0])
            imf1.append(imf[1])
            imf2.append(imf[2])
            if len(imf)>3:
                imf3.append(imf[3])
        imf0_fin = self.mean_list(list(imf0))    
        imf1_fin = self.mean_list(list(imf1))    
        imf2_fin = self.mean_list(list(imf2))    
        if imf3!=[]:
            imf3_fin = self.mean_list(list(imf3))    
        else:
            imf3_fin = []
        return (imf0_fin,imf1_fin,imf2_fin,imf3_fin)

    def eemd_process_multi_pool(self,data,n,processor_num):
        imf0 = []
        imf1 = []
        imf2 = []
        imf3 = []

        pool = Pool(processes=processor_num)


        para = [list(data) for i in range(n)]
        result=pool.map(self.f, para)
        for i in range(len(result)):
            imf0.append(result[i][0])
            imf1.append(result[i][1])
            imf2.append(result[i][2])
        imf0_fin = self.mean_list(list(imf0))
        imf1_fin = self.mean_list(list(imf1))
        imf2_fin = self.mean_list(list(imf2))
        if imf3!=[]:
            imf3_fin = self.mean_list(list(imf3))
        else:
            imf3_fin = []
        return (imf0_fin,imf1_fin,imf2_fin,imf3_fin)
        

    def eemd_process_multi(self,data,n):

        imf0 = []
        imf1 = []
        imf2 = []
        imf3 = []
    
        result=[]
        lock = Lock()
        mgr=Manager()
        md=mgr.list(result)
    
        record=[]
        for i in range(n):
            process = Process(target=self.f2,args=(data,lock,md))
            process.start()
            record.append(process)
    
        for process in record:
            result=process.join()

        for i in range(len(md)):
            imf0.append(md[i][0])
            imf1.append(md[i][1])
            imf2.append(md[i][2])
        imf0_fin = self.mean_list(list(imf0))
        imf1_fin = self.mean_list(list(imf1))
        imf2_fin = self.mean_list(list(imf2))
        if imf3!=[]:
            imf3_fin = self.mean_list(list(imf3))
        else:
            imf3_fin = []
 #       my_emd = one_dimension_emd(list(imf1_fin))
 #       (imf, residual) = my_emd.emd(percise,percise)
 #       print "len mean imf%s"%len(imf) 
 #       if len(imf)>1:
 #           imf1_fin = imf[1]
 #       else:
 #           imf1_fin = imf[0]
 #       my_emd = one_dimension_emd(list(imf2_fin))
 #       (imf, residual) = my_emd.emd(0.05,0.05)
 #       print "len mean imf%s"%len(imf)
 #       if len(imf)>1:
 #           imf2_fin = imf[-2]
 #       else:
 #           imf2_fin = imf[-1]
        return (imf0_fin,imf1_fin,imf2_fin,imf3_fin)


    def eemd_process(self,data,n,processor_num,choice):
        if choice=="pool":
            return self.eemd_process_multi_pool(data,n,processor_num)
        elif choice=="multi":
            #noise = 2*(np.std(data)*0.2*np.random.standard_normal(len(data)))-1
            noise = ((np.std(data)*0.2* np.random.rand(len(data))))
            #empty = [0 for i in range(20)]
            #noise = list(noise) + empty 
            data_addnoise = np.array(data)+np.array(noise)
            return self.eemd_process_multi(data_addnoise,n)
        else:
            return self.eemd_process_single(data,n)
    

    def mean_list(self,list_list):
        new_list = []

        data = np.array(list(list_list))
        print data.shape[1]
        for i in range(data.shape[1]):
            col = data[:,i]
            mean_col = np.mean(col)
            new_list.append(mean_col)
        return new_list



if __name__=="__main__":
    #data = [[1,2,3],[3,2,1],[10,10,100]]

    starttime = datetime.datetime.now()
    f = open("../../emd_data/emd_600",'r')
    #f = open("data_zgpa",'r')
    #f = open("data_gongshang",'r')
    lines = f.readlines()
    f.close()
    datalist = []
    vol = []
    for eachline in lines:
        eachline.strip("\n")
        datalist.append(float(eachline.split("\t")[0]))
   
    my_eemd = eemd(datalist,10)
        
    (imf0,imf1,imf2,imf3) = my_eemd.eemd_process(datalist,100,4,"multi")            
     
    endtime = datetime.datetime.now()
    print (endtime - starttime).seconds
    my_emd = one_dimension_emd(list(imf1))
    (imf, residual) = my_emd.emd(0.03,0.03)
    print len(imf)
    plt.figure(1)
    plt.plot(datalist,'r')
    plt.figure(2)
    plt.plot(imf[0],'r')
    plt.figure(3)
    plt.plot(imf[1],'r')
    plt.figure(4)
    plt.plot(residual,'r')
#
    plt.show()
        
    
        



