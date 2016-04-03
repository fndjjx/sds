from scipy.signal import argrelextrema
from scipy.signal import welch
from scipy.interpolate import UnivariateSpline
import numpy as np
import math
from smooth import cubicSmooth5
from generate_emd_data import generateEMDdata
import sys
import matplotlib.pyplot as plt

class emd_mean_pro_class(object):
    
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
    
    def extension_data(self, data_list, max_extreme, min_extreme, u1=3, u2=3, b1=0.5, b2=0.5, fs=1):
         
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
            if (len(self.get_max_extreme_point(residual)[0]) < 3) or (len(self.get_min_extreme_point(residual)[0]) < 3):
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


    def get_mean_data_list(self, max_extreme_after_extension, min_extreme_after_extension, extension_data_list):
    
        max_index = list(max_extreme_after_extension[0])
        max_value = list(max_extreme_after_extension[1])
        min_index = list(min_extreme_after_extension[0])
        min_value = list(min_extreme_after_extension[1])
  
        max_extreme_pair = []
        min_extreme_pair = []
        for i in range(len(max_index)):
            max_extreme_pair.append((max_index[i],max_value[i]))
        for i in range(len(min_index)):
            min_extreme_pair.append((min_index[i],min_value[i]))

        tmp = list(max_extreme_pair)+list(min_extreme_pair) 
        tmp.sort()
        
        mean_index = []
        mean_value = []

        for i in range(1,len(tmp)):
            mean_index.append(tmp[i][0])
            mean_value.append((tmp[i][1]+tmp[i-1][1])/2.0)

        mean = (mean_index, mean_value)

        mean_spline = self.get_spline_interpolation(mean, extension_data_list)
        return (mean_spline,mean)
                   

    def emd_mean_pro(self,th1,alpha):
        
        process_data_list = self.source_data_list
        imf_gen_flag = False
        emd_finish_flag = False    
        imf = []
        residual = []
        

        while emd_finish_flag != True:
            imf_process_data_list = process_data_list
            count = 0 
            while imf_gen_flag != True:
                
                max_extreme = self.get_max_extreme_point(imf_process_data_list)
                min_extreme = self.get_min_extreme_point(imf_process_data_list)
                (extension_data_list, n1, n2) = self.extension_data(imf_process_data_list, max_extreme, min_extreme)
                
                max_extreme_after_extension = self.get_max_extreme_point(extension_data_list)
                min_extreme_after_extension = self.get_min_extreme_point(extension_data_list)

                
#                max_envelop = self.get_spline_interpolation(max_extreme_after_extension, extension_data_list)[n1:-n2]
#                min_envelop = self.get_spline_interpolation(min_extreme_after_extension, extension_data_list)[n1:-n2]

                mean_data_list = self.get_mean_data_list(max_extreme_after_extension, min_extreme_after_extension, extension_data_list)[0][n1:-n2]
                
                imf_candidate_list = imf_process_data_list - mean_data_list
                if count>10:#self.imf_judge(imf_candidate_list, max_envelop, min_envelop, mean_data_list,th1,alpha):
                    imf.append(imf_candidate_list)
                    imf_gen_flag = True    
                else:
                    imf_process_data_list = imf_candidate_list
                    count += 1

            process_data_list = process_data_list - imf[-1]
            residual = process_data_list 
            imf_gen_flag = False
#            print "imf"
#            print imf
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


def main():

    datafile = sys.argv[1]
    begin = int(sys.argv[2])
    end = int(sys.argv[3])
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

    datalist = close_price[begin:end]
    my_emd = one_dimension_emd(datalist,3)
    (imf, residual) = my_emd.emd(0.02,0.02)
    plt.figure(1)
    num_fig = len(imf)+1
    for i in range(num_fig):
        if i == 0:
            x = "%d1%d"%(num_fig,1)
            plt.subplot(x)
            plt.plot(datalist)
        else:
            x = "%d1%d"%(num_fig,i+1)
            plt.subplot(x)
            plt.plot(np.linspace(0,len(imf[i-1])-1,len(imf[i-1])),imf[i-1],'r')


    plt.show()



def test():
    #main()
    datafile = sys.argv[1]
    begin = int(sys.argv[2])
    end = int(sys.argv[3])
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

    datalist = close_price[begin:end]
    my_emd = one_dimension_emd(datalist,3)
    max_extreme = my_emd.get_max_extreme_point(datalist)
    min_extreme = my_emd.get_min_extreme_point(datalist)
    (extension_data_list, n1, n2) = my_emd.extension_data(datalist, max_extreme, min_extreme)
    max_extreme_after_extension = my_emd.get_max_extreme_point(extension_data_list)
    min_extreme_after_extension = my_emd.get_min_extreme_point(extension_data_list)


    mean_data_list = my_emd.get_mean_data_list(max_extreme_after_extension, min_extreme_after_extension, extension_data_list)[0][n1:-n2]
    mean = my_emd.get_mean_data_list(max_extreme_after_extension, min_extreme_after_extension, extension_data_list)[1]

    plt.plot(datalist)
    plt.plot(extension_data_list[:],'r')
    plt.plot(mean_data_list,'g')
    plt.plot(mean[0],mean[1],'go')
    plt.show()

if __name__ == "__main__":
    t = np.linspace(1,2,100)
    datalist = 3*np.sin(2*10*t)*np.sin(2*20*t)
    data1 = np.sin(2*10*t)
    data2 = np.sin(2*20*t)
    my_emd = emd_mean_pro_class(datalist,3)
    (imf,residual) = my_emd.emd_mean_pro(0.01,0.01)
    plt.figure(1)
    num_fig = len(imf)+1
    for i in range(num_fig):
        if i == 0:
            x = "%d1%d"%(num_fig,1)
            plt.subplot(x)
            plt.plot(datalist)
        else:
            x = "%d1%d"%(num_fig,i+1)
            plt.subplot(x)
            plt.plot(np.linspace(0,len(imf[i-1])-1,len(imf[i-1])),imf[i-1],'r')


    plt.figure(2)
    plt.plot(residual)
    plt.figure(3)
    plt.subplot(211)
    plt.plot(data1)
    plt.subplot(212)
    plt.plot(data2)
    plt.show()
