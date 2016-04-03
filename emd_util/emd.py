from scipy.signal import argrelextrema
from scipy.signal import welch
from scipy.interpolate import UnivariateSpline
import numpy as np
import math
from smooth import cubicSmooth5
from generate_emd_data import generateEMDdata
from calc_match import *

import matplotlib.pyplot as plt

class one_dimension_emd(object):
    
    def __init__(self, source_data_list,stop_num=0):
        self.source_data_list = source_data_list
        self.pi = math.pi
        self.imf_stop_num = stop_num

    def get_max_extreme_point(self, data_list):
#        print "enter get max point"
        data_array = np.array(data_list)
        max_extreme_index = argrelextrema(data_array,np.greater)[0]
        max_extreme = [data_list[i] for i in max_extreme_index]
#        print "exit get max point"
#        print "max extreme point number %s"%len(max_extreme)
        return (max_extreme_index,max_extreme)
    
    def get_min_extreme_point(self, data_list):
#        print "enter get min point"
        data_array = np.array(data_list)
        min_extreme_index = argrelextrema(data_array,np.less)[0]
        min_extreme = [data_list[i] for i in min_extreme_index]
#        print "exit get min point"
#        print "min extreme point number %s"%len(min_extreme)
        return (min_extreme_index,min_extreme)
    
    def extension_data_match(self, data, max_extreme, min_extreme, u1=3, b1=0.5, fs=3):

#        print "enter extension match"
        
        max_extreme_point_index = list(max_extreme[0])
        max_extreme_point_value = list(max_extreme[1])

        min_extreme_point_index = list(min_extreme[0])
        min_extreme_point_value = list(min_extreme[1])

        T1 = u1*abs(max_extreme_point_index[0]-min_extreme_point_index[0])
        A1 = b1*abs(max_extreme_point_value[0]-min_extreme_point_value[0])
        M1 = max_extreme_point_value[0] - A1
        K1 = (data[0]-M1)/A1
        if K1 >= 1:
            O1 = np.arcsin(1)
            M1 = data[0] - A1
        elif K1 <= -1:
            O1 = np.arcsin(-1)
            M1 = data[0] + A1
        else:
            O1 = np.arcsin(K1)

        x1 = np.linspace(0,u1*T1-1,u1*T1*fs)
        y1 = A1*np.sin((2*self.pi*x1/T1)+O1)+M1
        n1 = u1*T1*fs

        extreme_index = max_extreme_point_index + min_extreme_point_index
        extreme_index.sort()

        sample = data[extreme_index[-2]:]
        len_sample = len(sample)
        result = []
        if max_extreme_point_index[-1]>min_extreme_point_index[-1]:
            for i in range(len(min_extreme_point_index)-1):
            #for i in range(len(max_extreme_point_index)-2):
                begin = min_extreme_point_index[i]
            #    begin = max_extreme_point_index[i]
                end = begin + len_sample
                #result.append((calc_para(sample,data[begin:end]),begin,end))
                result.append((calc_SMC(sample,data[begin:end]),begin,end))
        elif max_extreme_point_index[-1]<min_extreme_point_index[-1]:
            for i in range(len(max_extreme_point_index)-1):
            #for i in range(len(min_extreme_point_index)-2):
                begin = max_extreme_point_index[i]
             #   begin = min_extreme_point_index[i]
                end = begin + len_sample
                #result.append((calc_para(sample,data[begin:end]),begin,end))
                result.append((calc_SMC(sample,data[begin:end]),begin,end))

        result.sort(key=lambda x:x[0])
        end = result[-1][2]
        if len(data)-end>10:
            extension_part = data[end+1:end+11]
        else:
            extension_part = data[end+1:]
            

#        print "exit extension match"
        return (list(y1)+list(data)+list(extension_part),n1,len(extension_part))
        


    def extension_data(self, data_list, max_extreme, min_extreme, u1=3, u2=10, b1=0.5, b2=0.5, fs=3):
         
        max_extreme_point_index = list(max_extreme[0])
        max_extreme_point_value = list(max_extreme[1])

        min_extreme_point_index = list(min_extreme[0])
        min_extreme_point_value = list(min_extreme[1])

        T1 = u1*abs(max_extreme_point_index[0]-min_extreme_point_index[0])
        A1 = b1*abs(max_extreme_point_value[0]-min_extreme_point_value[0])
        M1 = max_extreme_point_value[0] - A1
        K1 = (data_list[0]-M1)/A1
        if K1 >= 1:
            O1 = np.arcsin(1)
            M1 = data_list[0] - A1
        elif K1 <= -1:
            O1 = np.arcsin(-1)
            M1 = data_list[0] + A1
        else:
            O1 = np.arcsin(K1)

        x1 = np.linspace(0,u1*T1-1,u1*T1*fs)
        y1 = A1*np.sin((2*self.pi*x1/T1)+O1)+M1
        n1 = u1*T1*fs

        T2 = u2*abs(max_extreme_point_index[-1]-min_extreme_point_index[-1])
        A2 = b2*abs(max_extreme_point_value[-1]-min_extreme_point_value[-1])
        M2 = max_extreme_point_value[-1] - A2
        K2 = (data_list[-1]-M2)/A2
        if K2 >= 1:
            O2 = np.arcsin(1)
            M2 = data_list[-1] - A2
        elif K2 <= -1:
            O2 = np.arcsin(-1)
            M2 = data_list[-1] + A2
        else:
            O2 = np.arcsin(K2)

        x2 = np.linspace(0,u2*T2-1,u2*T2*fs)
        y2 = A2*np.sin((2*self.pi*x2/T2)+O2)+M2
        n2 = u2*T2*fs
        
#        print "y1 %s"%y1
#        print "data_list %s"%data_list
#        print "y2 %s"%y2 
        extension_data = list(y1) + list(data_list) + list(y2)

  #      print "exit extension data"
        return (extension_data, n1, n2)


    def get_spline_interpolation(self, extreme_point, raw_data):
       
   #     print "enter interpolation" 
        extreme_point_index = np.array(extreme_point[0])
        #print extreme_point_index
        extreme_point_value = np.array(extreme_point[1])
        raw_data_index = np.linspace(0, len(raw_data)-1, len(raw_data))
        envelop = UnivariateSpline(extreme_point_index, extreme_point_value,s=0)(raw_data_index)
        
    #    print "exit interpolation"
        return envelop

    def imf_judge(self, imf_candidate, process_data_max, process_data_min, mean_data,th1,alpha):
        def extreme_point_criteria(imf_candidate):
#            print "enter extreme point criteria"
#            print self.get_max_extreme_point(imf_candidate)[0]
            if (len(self.get_max_extreme_point(imf_candidate)[0]) < 3) or (len(self.get_min_extreme_point(imf_candidate)[0]) < 3):
                return True
            else:
                return False

        def rilling_criteria(process_data_max, process_data_min, mean_data,th1=0.03,alpha=0.03):
#            print "enter rilling criteria"
            #th1 = 0.03
            th2 = 10*th1
            #alpha = 0.03
            sati_num_of_th1 = 0
            sati_num_of_th2 = 0
            data_length = len(mean_data)

            e = [(process_data_max[i]-process_data_min[i])/2 for i in range(len(process_data_max))]
            delta = [mean_data[i]/e[i] for i in range(data_length)]
 #           print "delta %s"%delta         
            for i in range(data_length):
                if delta[i] < th1:
                    sati_num_of_th1 = sati_num_of_th1 + 1
                if delta[i] < th2:
                    sati_num_of_th2 = sati_num_of_th2 + 1
            if sati_num_of_th1/data_length >= 1-alpha and sati_num_of_th2 == data_length:
                return True
            else:
                return False

        if extreme_point_criteria(imf_candidate) == True or rilling_criteria(process_data_max, process_data_min, mean_data,th1,alpha) == True:
    #    if  rilling_criteria(process_data_max, process_data_min, mean_data) == True:
            return True
        else:
            return False
            

    def emd_finish_criteria(self, imf, residual, original_data, period_num = 10):
        def extreme_point_criteria(residual):
  #          print "enter outside extreme point criteria"
            if (len(self.get_max_extreme_point(residual)[0]) < 4) or (len(self.get_min_extreme_point(residual)[0]) < 4):
                return True
            else:
                return False
        def power_stop_criteria(imf, original_data, period_num):
  #          print "enter power stop criteria"
            threshold = 0.9
            original_data_length = len(original_data)
            last_period_length = original_data_length%period_num
            period_length = (original_data_length-last_period_length)/period_num
            flag = 0
            imf_sum = 0
            imf_sum_list = []

            for i in range(len(imf[0])):
                for j in range(len(imf)-1):
                    imf_sum += imf[j][i]*imf[j][i] 
                imf_sum_list.append(imf_sum)
                imf_sum = 0
            original_data_squre = [ original_data[i]*original_data[i] for i in range(original_data_length)]
                
            for i in range(period_num):
                if i != period_num-1:
                    if (sum(imf_sum_list[i*period_length:(i+1)*period_length])/sum(original_data_squre[i*period_length:(i+1)*period_length])) > threshold:
                        flag += 1
                else:
                    if (sum(imf_sum_list[i*period_length:-1])/sum(original_data_squre[i*period_length:-1])) > threshold:
                        flag += 1
            
   #         print "flag %s"%flag
            if flag/period_num > 0.95:
                return True
            else: 
                return False    
        if extreme_point_criteria(residual) == True or power_stop_criteria(imf, original_data, period_num) == True or (self.imf_stop_num!=0 and len(imf) ==self.imf_stop_num):
            return True
        else:
            return False
                   

    def emd(self,th1,alpha):
        
        process_data_list = self.source_data_list
        imf_gen_flag = False
        emd_finish_flag = False    
        imf = []
        residual = []
        

        while emd_finish_flag != True:
            imf_process_data_list = process_data_list
           
            loop_count = 0
            while imf_gen_flag != True:
                
                max_extreme = self.get_max_extreme_point(imf_process_data_list)
                min_extreme = self.get_min_extreme_point(imf_process_data_list)
                #normal extension
                (extension_data_list, n1, n2) = self.extension_data(imf_process_data_list, max_extreme, min_extreme)
                #(extension_data_list, n1, n2) = self.extension_data_match(imf_process_data_list, max_extreme, min_extreme)
                
                max_extreme_after_extension = self.get_max_extreme_point(extension_data_list)
                min_extreme_after_extension = self.get_min_extreme_point(extension_data_list)

                max_envelop = self.get_spline_interpolation(max_extreme_after_extension, extension_data_list)[n1:-n2]
                min_envelop = self.get_spline_interpolation(min_extreme_after_extension, extension_data_list)[n1:-n2]
                
                mean_data_list = (max_envelop+min_envelop)/2
                
                imf_candidate_list = imf_process_data_list - mean_data_list
                if self.imf_judge(imf_candidate_list, max_envelop, min_envelop, mean_data_list,th1,alpha) :#or loop_count>1000:
                    imf.append(imf_candidate_list)
                    imf_gen_flag = True    
                else:
                    imf_process_data_list = imf_candidate_list
                loop_count += 1
         #       print "loop count%s"%loop_count

            print "loop count {}".format(loop_count)
            process_data_list = process_data_list - imf[-1]
            residual = process_data_list 
            imf_gen_flag = False
        #    print "imf"
        #    print imf
            #for i in range(len(imf)):
            #    plt.plot(np.linspace(0,len(imf[i])-1,len(imf[i])),imf[i],'r')
            #    plt.plot(np.linspace(0,len(residual)-1,len(residual)),residual,'y')
            #plt.show()

            if self.emd_finish_criteria(imf, residual, self.source_data_list):
                 emd_finish_flag = True
#        print "emd num%s"%len(imf)
        return (imf,residual)

    def imf_percentage(self, imf, residual, source_data):
        sum_list = []
        source_data_without_residual = []
        for i in range(len(imf)):
            sum_list.append(sum([j*j for j in imf[i]]))
        sum_list.append(sum([j*j for j in residual]))
        source_data_without_residual = [source_data[i]-residual[i] for i in range(len(source_data))]
        source_square_sum = sum([j*j for j in source_data_without_residual])
        return [i/source_square_sum for i in sum_list]


def ma_func(data, period):
    ma_list=[]
    for i in range(period):
        ma_list.append(0)
    for i in range(period,len(data)):
        ma_list.append(np.mean(data[i-(period-1):i+1]))
    return ma_list

def calc_ma(l):
    ma_l = ma_func(l,3)
    return [l[i]-ma_l[i] for i in range(len(l))]

def normalize(l):
    max_n = max(l)
    min_n = min(l)
    return map(lambda x: (x-min_n)/float(max_n-min_n), l)


if __name__ == "__main__":

#    (imf,residual) = my_emd.emd()
#    datalist = [1,5,3,8,3,2,1,2,3,5,9,2,1,2,3,4,3,2,7,2,3,6,10,2,1]
    #t = np.linspace(1,2,100)
    #datalist = 3*np.sin(2*10*t)*np.sin(2*20*t)
    end = int(sys.argv[1])
    pp = int(sys.argv[2])
    f = open("../back_test/test_data/shjh",'r')
    #f = open("../../emd_data/emd_666",'r')
    #f = open("../back_test/test_data/payh",'r')
    ##f = open("data_gongshang",'r')
    lines = f.readlines()
    f.close()
    datalist = []
    #vol = []
    for eachline in lines:
        eachline.strip("\n")
        datalist.append(float(eachline.split("\t")[4]))
    ##    vol.append(float(eachline.split("\t")[5]))

    ma3 = ma_func(datalist,3)
    datalist = [datalist[i]-ma3[i] for i in range(len(ma3))]

    ma2 = ma_func(datalist,2)
    ma3 = ma_func(datalist,3)
    ma4 = ma_func(datalist,4)
    ma5 = ma_func(datalist,5)
    ma7 = ma_func(datalist,7)
    ma9 = ma_func(datalist,9)
    ma10 = ma_func(datalist,10)
    ma8 = ma_func(datalist,8)


    ma_test = ma_func(datalist,pp)
#    datalist = datalist[end-499:end+1]
    #ma2 = ma2[end-99:end+1]
    ma_test = ma_test[end-99:end+1]
    ma9 = ma9[end-99:end+1]
    ma3 = ma3[end-99:end+1]
    #ma5 = ma5[end-99:end+1]


    #
    #ma5 = []
    #for i in range(2):
    #    ma5.append(0)
#
    #for i in range(3,len(datalist)+1):
    #    mean_5 = np.mean(datalist[i-2:i+1])
    #    ma5.append(mean_5)
    #ma5 = datalist


    #plt.plot(np.linspace(0,len(datalist)-1,len(datalist)),datalist, '.-')
    #plt.show()
    my_emd = one_dimension_emd(datalist,3)
#    max_extreme = my_emd.get_max_extreme_point(datalist) 
#    min_extreme = my_emd.get_min_extreme_point(datalist) 
#    (extension_data_list,n1,n2) = my_emd.extension_data(datalist, max_extreme, min_extreme)
#    max_extreme_after_extension = my_emd.get_max_extreme_point(extension_data_list)
#    print max_extreme_after_extension 
#    print "hah"
#    min_extreme_after_extension = my_emd.get_min_extreme_point(extension_data_list)
#    max_envelop_final = my_emd.get_spline_interpolation(max_extreme_after_extension, extension_data_list)[n1:-n2]
#    min_envelop_final = my_emd.get_spline_interpolation(min_extreme_after_extension, extension_data_list)[n1:-n2]
#                  
##    print max_envelop
#    plt.plot(np.linspace(0,len(datalist)-1,len(datalist)), datalist, '.-')    
#    plt.plot(np.linspace(0,len(datalist)-1,len(datalist)),max_envelop_final, 'g')
#    plt.plot(np.linspace(0,len(datalist)-1,len(datalist)),min_envelop_final, 'g')
##    print extension_data
##    plt.plot(np.linspace(0,len(extension_data)-1,len(extension_data)),extension_data, 'g')
#    plt.show()
    (imf, residual) = my_emd.emd(0.02,0.02)
    #for i in range(len(imf)):
    #    fp = open("imf%s"%i,'w')
    #    for j in range(len(imf[i])):
    #        fp.writelines(str(imf[i][j]))
    #        fp.writelines("\n")
    #    fp.close()
    print "imf num %s"%len(imf)


    residual_imf0 = [datalist[i]-imf[0][i] for i in range(len(imf[0]))]
    residual_imf1 = [residual_imf0[i]-imf[1][i] for i in range(len(imf[0]))]
    residual_imf2 = [residual_imf1[i]-imf[2][i] for i in range(len(imf[0]))]
    residual_imf2 = residual_imf2[end-99:end+1]
#    residual_imf2 = residual_imf2[-100:]
    residual_imf1 = residual_imf1[end-99:end+1]

    print sum([abs(ma_test[i]-residual_imf2[i-1]) for i in range(1,len(ma_test))])/(len(ma_test)-1)
    print sum([abs(ma_test[i]-residual_imf2[i-2]) for i in range(2,len(ma_test))])/(len(ma_test)-2)
    print sum([abs(ma_test[i]-residual_imf2[i-3]) for i in range(3,len(ma_test))])/(len(ma_test)-3)
    print sum([abs(ma_test[i]-residual_imf2[i-4]) for i in range(4,len(ma_test))])/(len(ma_test)-4)
    print sum([abs(ma_test[i]-residual_imf2[i-5]) for i in range(5,len(ma_test))])/(len(ma_test)-5)
    print sum([abs(ma_test[i]-residual_imf2[i-6]) for i in range(6,len(ma_test))])/(len(ma_test)-6)
    print sum([abs(ma_test[i]-residual_imf2[i-7]) for i in range(7,len(ma_test))])/(len(ma_test)-7)
    print sum([abs(ma_test[i]-residual_imf2[i-8]) for i in range(8,len(ma_test))])/(len(ma_test)-8)
    #print "imf percentage"
    #fp = open("residual",'w')
    #for i in range(len(residual)):
    #    fp.writelines(str(residual[i]))
    #    fp.writelines("\n")
    #fp.close()

    #corre = np.correlate(datalist,imf[1],"same")
    ##selfcorre = np.correlate(datalist,datalist,"full")
    ##t= np.linspace(1,100,1000)
    #tmp=datalist
    ##datalist = np.sin(t) 
    ##sin = datalist
    ##print datalist
    #datalist = [datalist[i+1]-datalist[i] for i in range(len(datalist)-1)]
    #data_0 = [datalist[i]-np.mean(datalist) for i in range(len(datalist))]
    #selfcorre = np.correlate(data_0,data_0,"full")
    #result = selfcorre
    #result = result[len(result)//2:]
    #result /= result[0]
    #selfcorre = result
    #print selfcorre
    #data = np.array(selfcorre)
    #corr_max_index = list(argrelextrema(data,np.greater)[0])
    #corr_min_index = list(argrelextrema(data,np.less)[0])

    #print "corr max%s"%corr_max_index
    #print "corr min%s"%corr_min_index


    #corr_diff = [corr_max_index[i+1]-corr_max_index[i] for i in range(len(corr_max_index)-1)]
    #corr_diff.remove(max(corr_diff))
    #corr_diff.remove(min(corr_diff))
    #print corr_diff
    #print np.mean(corr_diff)

    #datalist=tmp




    #data = np.array(imf[2][end-99:end+1])
    data = np.array(imf[2][end-99:end+1])
    imf_max_index = list(argrelextrema(data,np.greater)[0])
    imf_min_index = list(argrelextrema(data,np.less)[0])
    print "shouldbe"
    print imf_max_index
    print imf_min_index


    datalist = datalist[end-99:end+1]
    my_emd = one_dimension_emd(datalist,3)
    (imf, residual) = my_emd.emd(0.02,0.02)
    data=np.array(imf[2])
    imf_max_index = list(argrelextrema(data,np.greater)[0])
    imf_min_index = list(argrelextrema(data,np.less)[0])
    print "maybe"
    print imf_max_index
    print imf_min_index
    (data,extenflag1,ex_num,cha,cha2) = matchlist2(imf[2],5,datalist)
    data=np.array(data)
    imf_max_index = list(argrelextrema(data,np.greater)[0])
    imf_min_index = list(argrelextrema(data,np.less)[0])
    print "extenbe"
    print imf_max_index
    print imf_min_index


    ma4.pop(0)

    ma8.pop(0)
    ma8.pop(0)
    ma8.pop(0)



    ma_data = [ma4[i]-ma8[i] for i in range(len(ma8))]
    ma_data.insert(0,0)
    ma_data.insert(0,0)
    ma_data.insert(0,0)
    ma_data = np.array(ma_func(ma_data,3))
    ma_data = np.array(ma_func(ma_data,3))
    ma_data = np.array(ma_data[end-99:end+1])
    ma_max_index = list(argrelextrema(ma_data,np.greater)[0])
    ma_min_index = list(argrelextrema(ma_data,np.less)[0])
    ma_max_index = [i-3 for i in ma_max_index]
    ma_min_index = [i-3 for i in ma_min_index]
    print "ma"
    print ma_max_index
    print ma_min_index
    ma5.insert(0,0)
    ma5.insert(0,0)
    ma_max_index = list(argrelextrema(np.array(ma5[end-99:end+1]),np.greater)[0])
    ma_min_index = list(argrelextrema(np.array(ma5[end-99:end+1]),np.less)[0])
    print ma_max_index
    print ma_min_index

    #imf_max = imf_max_index
    #imf_min = imf_min_index
    #max_dis = [imf_max[i]-imf_max[i-1] for i in range(1,len(imf_max))]
    #min_dis = [imf_min[i]-imf_min[i-1] for i in range(1,len(imf_min))]
    #if imf_max[0]>imf_min[0] and imf_min[-1]<imf_max[-1]:
    #    min_max = [imf_max[i]-imf_min[i] for i in range(len(imf_max))]
    #    max_min = [imf_min[i+1]-imf_max[i] for i in range(len(imf_max)-1)]
    #elif imf_max[0]>imf_min[0] and imf_min[-1]>imf_max[-1]:
    #    min_max = [imf_max[i]-imf_min[i] for i in range(len(imf_max))]
    #    max_min = [imf_min[i+1]-imf_max[i] for i in range(len(imf_max))]
    #elif imf_max[0]<imf_min[0] and imf_min[-1]>imf_max[-1]:
    #    min_max = [imf_max[i+1]-imf_min[i] for i in range(len(imf_max)-1)]
    #    max_min = [imf_min[i]-imf_max[i] for i in range(len(imf_max))]
    #elif imf_max[0]<imf_min[0] and imf_min[-1]<imf_max[-1]:
    #    min_max = [imf_max[i+1]-imf_min[i] for i in range(len(imf_min))]
    #    max_min = [imf_min[i]-imf_max[i] for i in range(len(imf_min))]

    #mean_max_min = np.mean(max_min[:])
    #std_max_min = np.std(max_min[:])
    #mean_min_max = np.mean(min_max[:])
    #std_min_max = np.std(min_max[:])
    #print "mean"
    #print mean_max_min
    #print mean_min_max

    #for i in range(len(imf)):
    #    print "imf%s data coef %s"%(i,np.corrcoef(datalist,imf[i])[0][1])
    #print my_emd.imf_percentage(imf, residual, datalist)
    #plt.plot(np.linspace(0,len(imf[0])-1,len(imf[0])),imf[0], 'g')
    #plt.plot(np.linspace(0,len(imf[0])-1,len(imf[0])),imf[1], 'b')
    #plt.plot(np.linspace(0,len(datalist)-1,len(datalist)),datalist, '.-')
    plt.figure(1)
    num_fig = len(imf)+1
    for i in range(num_fig-1):
        x = "%d1%d"%(num_fig,i+1)
        print x
        plt.subplot(x)
        plt.plot(np.linspace(0,len(imf[i])-1,len(imf[i])),imf[i],'r')


    x = "%d1%d"%(num_fig,num_fig)
    plt.subplot(x)
    #plt.plot(np.linspace(0,len(residual)-1,len(residual)),residual,'y')
    plt.plot(np.linspace(0,len(residual)-1,len(residual)),datalist,'y')
    plt.figure(2)
    plt.plot(imf[2][end-99:end+1],'r')
    plt.plot(ma_data[end-99:end+1])
    
#
#
#    x = "%d1%d"%(num_fig,num_fig)
#    plt.subplot(x)
#    plt.plot(np.linspace(0,len(datalist)-1,len(datalist)),datalist, '.-')



    #plt.figure(2)
    #for i in range(num_fig):
    #    f, Pxx_den = welch(imf[i],5)
    #    x = "%d1%d"%(num_fig,i+1)
    #    plt.subplot(x)
    #    plt.plot(f, Pxx_den)

    #f, Pxx_den = welch(residual,5)
    #x = "%d1%d"%(num_fig,num_fig-1)
    #plt.subplot(x)
    #plt.plot(f, Pxx_den)

    #f, Pxx_den = welch(datalist,5)
    #x = "%d1%d"%(num_fig,num_fig)
    #plt.subplot(x)
    #plt.plot(f, Pxx_den)
    #plt.figure(3)
    #plt.plot(np.linspace(0,len(residual)-1,len(residual)),residual,'y')
    #plt.figure(4)
    #plt.plot(np.linspace(0,len(datalist)-1,len(datalist)),datalist,'y')
    #plt.figure(5)
    #plt.plot(np.linspace(0,len(datalist)-1,len(datalist)),datalist,'b',np.linspace(0,len(corre)-1,len(datalist)),corre,'y')
    #plt.figure(6)
    #plt.plot(np.linspace(0,len(selfcorre)-1,len(selfcorre)),selfcorre,'y')
    #plt.figure(7)
    #plt.subplot(211)
    #plt.plot(np.linspace(0,len(datalist)-1,len(datalist)),datalist,'y')
    #plt.subplot(212)
    #plt.plot(np.linspace(0,len(imf[1])-1,len(imf[i])),imf[1],'r')
    #plt.figure(8)
    #plt.plot(np.linspace(0,len(residual)-1,len(residual)),residual,'y')
    #plt.figure(9)
    #plt.plot(imf[2])
    plt.show()
