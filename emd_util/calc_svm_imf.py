from emd import one_dimension_emd
import numpy as np
from scipy.signal import argrelextrema
from scipy.signal import spline_filter
import matplotlib.pyplot as plt
import sys
from smooth import exp_smooth
from smooth import cubicSmooth5
from smooth import spline_smooth
from eemd import *
from smooth import gaussian_smooth
from calc_match import calc_para
from calc_match import calc_SMC
from sklearn import svm
from sklearn.ensemble import AdaBoostRegressor



def load_data(filename,col):
    with open(filename,'r') as f:
        lines = f.readlines()
        datalist = []
        for eachline in lines:
            eachline.strip("\n")
            datalist.append(float(eachline.split("\t")[col]))
    return datalist


def ma_func(data, period):
    ma_list=[]
    for i in range(period):
        ma_list.append(data[i])
    for i in range(period,len(data)):
        ma_list.append(np.mean(data[i-(period-1):i+1]))
    return ma_list

def normalize(l):
    max_n = max(l)
    min_n = min(l)
    return map(lambda x: (x-min_n)/float(max_n-min_n), l)

def emd_decom(data,imf_number,precise=0.02):
    my_emd = one_dimension_emd(data,imf_number)
    (imf, residual) = my_emd.emd(precise,precise)
    return imf

def eemd_decom(data,imf_number,precise=0.02):
    my_eemd = eemd(data,30)
    (imf0,imf1,imf2,imf3)= my_eemd.eemd_process(data,30,4,'multi')
    imf = []
    imf.append(imf0)
    imf.append(imf1)
    imf.append(imf2)
    return imf


def data_smooth(data,imf,imf_index):
    for i in range(imf_index+1):
        residual = [data[j]-imf[i][j] for j in range(len(data))]
        data = residual

    return residual


def calc_distance(l1,l2):
    l1 = normalize(l1)
    l2 = normalize(l2)
    return sum([abs(l1[i]-l2[i]) for i in range(len(l1))])/(len(l1))


def svm_smooth(data, residual_imf, period):
    train_data = []
    lable = []
    for i in range(period,len(residual_imf)-20):
        tmp = data[i-period:i+1]
        train_data.append(tmp)
        lable.append(residual_imf[i])

    rng = np.random.RandomState(1)
    clf = AdaBoostRegressor(svm.SVR(),n_estimators=1, random_state=rng)
    clf.fit(train_data, lable) 
    smooth_data = []
    for i in range(len(data)):
        if i<=period:
            smooth_data.append(data[i])
        else:
            smooth_data.append(clf.predict([data[i-period:i+1]])[0])

    return smooth_data
    

#def most_like_smooth(data, imf, imf_index, longshort, need_index=0):
#    residual_imf = data_smooth(data,imf,imf_index)
#    imf_sample = residual_imf[-50:]
#
#    result = []
#    for period in range(10,100):
#        smooth_data = svm_smooth(data, residual_imf, period)
#        for width in range(5,40,2):
#            std = 1
#            while std < 5:
#                smooth_data = gaussian_smooth(smooth_data, width, std)
#                smooth_data_sample = smooth_data[-50:]
#                result.append(((period,width,std), sum([abs(smooth_data_sample[i]-imf_sample[i]) for i in range(len(smooth_data_sample))])/(len(smooth_data_sample))))
#                std += 1
#    result.sort(key = lambda x: x[1])
#   # result.reverse()
#    print result
#    smooth_data = svm_smooth(data, residual_imf, result[0][0][0])
#    smooth_data = gaussian_smooth(smooth_data, result[0][0][1],result[0][0][2])
#   # print "haha"
#   # print len(smooth_data)
#   # print len(residual_imf)
#    #plt.plot(smooth_data[-70:],'r')
#    #plt.plot(residual_imf[-70:])
#    #plt.show()
#    #result = result[:10]
#    #result.sort(key = lambda x: x[1])
#    
#    return smooth_data

def most_like_smooth(data, imf, imf_index, longshort, need_index=0):
    residual_imf = data_smooth(data,imf,imf_index)
    imf_sample = residual_imf[-150:-20]

    result = []
    for period in range(50,100):
        smooth_data = svm_smooth(data, residual_imf, period)
        smooth_data_sample = smooth_data[-150:-20]
        result.append(((period), sum([abs(smooth_data_sample[i]-imf_sample[i]) for i in range(len(smooth_data_sample))])/(len(smooth_data_sample))))
   # result.reverse()
    result.sort(key = lambda x: x[1])
    print result
    smooth_data = svm_smooth(data, residual_imf, result[0][0])
   # smooth_data = ma_func(smooth_data, 2)
   # print "haha"
   # print len(smooth_data)
   # print len(residual_imf)
#    plt.plot(smooth_data[-30:],'r')
#    plt.plot(residual_imf[-30:])
#    plt.show()
    #result = result[:10]
    #result.sort(key = lambda x: x[1])

    return smooth_data, residual_imf



def most_like_imf(data, imf, imf_index1, imf_index2): 


    smooth_data_short,ris = most_like_smooth(data,imf,imf_index1,"short")
    smooth_data_long ,ril= most_like_smooth(data,imf,imf_index2,"long")
  #  i = 1
  #  while residual_long_para[0] - residual_short_para[0] < 10:
  #      residual_long_para, _ = most_like_spline(data,imf,imf_index2,"long",i)
  #      i += 1
    


 
    diff = [smooth_data_short[i]-smooth_data_long[i] for i in range(len(smooth_data_long))]
    diff2 = [ris[i]-ril[i] for i in range(len(ril))]
    
    data = np.array(diff)
    max_index = list(argrelextrema(data,np.greater)[0])
    min_index = list(argrelextrema(data,np.less)[0])

    return max_index, min_index, smooth_data_long, smooth_data_short, diff, diff2
    
    
#############
def get_max_extreme_point(data_list):
    data_array = np.array(data_list)
    max_extreme_index = argrelextrema(data_array,np.greater)[0]
    max_extreme = [data_list[i] for i in max_extreme_index]
    return (max_extreme_index,max_extreme)

def get_min_extreme_point(data_list):
    data_array = np.array(data_list)
    min_extreme_index = argrelextrema(data_array,np.less)[0]
    min_extreme = [data_list[i] for i in min_extreme_index]
    return (min_extreme_index,min_extreme)
def extension_data(data_list, max_extreme, min_extreme, u1=3, u2=10, b1=0.5, b2=0.5, fs=3):

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
    y1 = A1*np.sin((2*np.pi*x1/T1)+O1)+M1
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
    y2 = A2*np.sin((2*np.pi*x2/T2)+O2)+M2
    n2 = u2*T2*fs

    extension_data = list(y1) + list(data_list) + list(y2)

    return (extension_data, n1, n2)

def get_spline_interpolation(extreme_point, raw_data):

    extreme_point_index = np.array(extreme_point[0])
    #print extreme_point_index
    extreme_point_value = np.array(extreme_point[1])
    raw_data_index = np.linspace(0, len(raw_data)-1, len(raw_data))
    envelop = UnivariateSpline(extreme_point_index, extreme_point_value,s=0)(raw_data_index)

#    print "exit interpolation"
    return envelop

def rilling_criteria(process_data_max, process_data_min, mean_data,th1=0.03,alpha=0.03):
    #th1 = 0.03
    th2 = 10*th1
    #alpha = 0.03
    sati_num_of_th1 = 0
    sati_num_of_th2 = 0
    data_length = len(mean_data)

    e = [(process_data_max[i]-process_data_min[i])/2 for i in range(len(process_data_max))]
    delta = [mean_data[i]/e[i] for i in range(data_length)]
    print "delta %s"%delta         
    for i in range(data_length):
        if delta[i] < th1:
            sati_num_of_th1 = sati_num_of_th1 + 1
        if delta[i] < th2:
            sati_num_of_th2 = sati_num_of_th2 + 1
    if sati_num_of_th1/data_length >= 1-alpha and sati_num_of_th2 == data_length:
        return True
    else:
        return False
    
            
def calc_recent_rilling(imf):
    max_extreme = get_max_extreme_point(imf)
    min_extreme = get_min_extreme_point(imf)
    (extension_data_list, n1, n2) = extension_data(imf, max_extreme, min_extreme)
 
    max_extreme_after_extension = get_max_extreme_point(extension_data_list)
    min_extreme_after_extension = get_min_extreme_point(extension_data_list)
 
    max_envelop = get_spline_interpolation(max_extreme_after_extension, extension_data_list)[n1:-n2]
    min_envelop = get_spline_interpolation(min_extreme_after_extension, extension_data_list)[n1:-n2]
 
    mean_data_list = (max_envelop+min_envelop)/2
 
    return mean_data_list



def compare_two_lists(l1,l2):
    result = {}
    for i in l1:
        result[i] = [1,0]

    for i in l2:
        if i in result :
            result[i] = [1, 1]
        elif i+1 in result:
            result[i+1] = [1, 1]
        elif i-1 in result:
            result[i-1] = [1, 1]
        else:
            result[i] = [0, 1]

    inl1notinl2 = filter(lambda x:x[1][1] == 0,result.items())
    inl2notinl1 = filter(lambda x:x[1][0] == 0,result.items())
    return result, inl1notinl2, inl2notinl1

if __name__ == "__main__":


    start = int(sys.argv[1])
    end = int(sys.argv[2])
    imf_number = int(sys.argv[3])

    dataj = load_data("../back_test/test_data/dhny",6)
    datav = load_data("../back_test/test_data/dhny",5)
    datac = load_data("../back_test/test_data/dhny",4)
    data = [dataj[i]/datav[i] for i in range(len(datav))]
    #data = [ma_func(data,3)[i]-ma_func(data,10)[i] for i in range(len(data))]
    #data = ma_func(data,3)
    print data[end-15:end+1]
    imf_base = emd_decom(data,4,0.5)
    print imf_base[imf_number][-15:]
    print "rl {}".format(calc_recent_rilling(imf_base[imf_number])[end-20:end+1])
    
    imf_component_base = np.array(imf_base[imf_number][:end+1])
    imf_max_index_base = list(argrelextrema(imf_component_base,np.greater)[0])
    imf_min_index_base = list(argrelextrema(imf_component_base,np.less)[0])
    print "standard"
    print imf_max_index_base
    print imf_min_index_base
    datasmooth = data_smooth(data,imf_base,imf_number)
    max_index = list(argrelextrema(np.array(datasmooth),np.greater)[0])
    min_index = list(argrelextrema(np.array(datasmooth),np.less)[0])
    print max_index
    print min_index

    data1 = data[start:end+1]
    imf = emd_decom(data1,3,0.1)
    print imf[imf_number][-15:]
    imf_component = np.array(imf[imf_number])
    imf_max_index = list(argrelextrema(imf_component,np.greater)[0])
    imf_min_index = list(argrelextrema(imf_component,np.less)[0])
    imf_max_index = [i+start for i in imf_max_index]
    imf_min_index = [i+start for i in imf_min_index]
    print "partial"
    print imf_max_index
    print imf_min_index
    
    datac1=datac[start:end+1]
    imf_c = emd_decom(datac1,3,0.001)
    resi_c = data_smooth(datac1,imf_c,imf_number)

    r,el1,el2 = compare_two_lists(imf_max_index_base,imf_max_index)
    el1 = filter(lambda x:x[0]>start,el1)
    el2 = filter(lambda x:x[0]>start,el2)
    print "max in base not in partial"
    for i in el1:
        print i[0]
    print "max in partial not in base"
    for i in el2:
        print i[0]
    r,el1,el2 = compare_two_lists(imf_min_index_base,imf_min_index)
    el1 = filter(lambda x:x[0]>start,el1)
    el2 = filter(lambda x:x[0]>start,el2)
    print "min in base not in partial"
    for i in el1:
        print i[0]
    print "min in partial not in base"
    for i in el2:
        print i[0]

    datasmooth = data_smooth(data1,imf,imf_number)
    max_index = list(argrelextrema(np.array(datasmooth),np.greater)[0])
    min_index = list(argrelextrema(np.array(datasmooth),np.less)[0])
    max_index = [i+start for i in max_index]
    min_index = [i+start for i in min_index]
    print "residual"
    print max_index
    print min_index
    #max_index, min_index, la, sa, splinediff, diff2 = most_like_imf(data1, imf, imf_number-1, imf_number) 
    #max_index = list(argrelextrema(np.array(splinediff),np.greater)[0])
    #min_index = list(argrelextrema(np.array(splinediff),np.less)[0])
    #max_index = [i+start for i in max_index]
    #min_index = [i+start for i in min_index]

    #print max_index
    #print min_index
######################################################
#    i = 0.003
#    imf_set = []
#    while i<0.1:
#        imf = emd_decom(data1,3,i)
#        imf_set.append(imf[2])
#        i += 0.001
#
#    
#    imf_fin = []
#    for i in  range(len(imf_set[0])):
#        tmp = 0
#        for j in range(len(imf_set)):
#            tmp += imf_set[j][i]        
#        imf_fin.append(tmp)
#    max_index = list(argrelextrema(np.array(imf_fin),np.greater)[0])
#    min_index = list(argrelextrema(np.array(imf_fin),np.less)[0])
#    max_index = [i+start for i in max_index]
#    min_index = [i+start for i in min_index]
#    print "combine"
#    print max_index
#    print min_index
#########################################################
    diff = [datac[i]-data[i] for i in range(len(datac))]
    diffma5=ma_func(diff,2)
    resi_base = data_smooth(data,imf_base,imf_number)
    resi = data_smooth(data1,imf,imf_number)
    #plt.plot(imf_base[imf_number][end-49:end+1])
    plt.subplot(211)
    plt.plot(data[end-20:end+1],'r')
    plt.plot(datac[end-20:end+1],'g')
    plt.subplot(212)
    plt.plot(diff[end-20:end+1],'go')
    plt.plot(diffma5[end-20:end+1],'ro')
    #plt.plot(resi_base[end-20:end+1])
    #plt.plot(splinediff[-50:],'r')
    #plt.plot(diff2[-50:],'g')
    #plt.plot(imf[imf_number][-50:],'y')
    #plt.plot(resi[-21:],'y')
    #plt.plot(resi_c[-21:],'g')
 #   plt.figure(2)
 #   data1 = data_smooth(expdiff,imf,0)
 #   plt.plot(imf[2][-100:])
    plt.show()
