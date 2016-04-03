import numpy as np
from emd import *
from scipy.signal import argrelextrema
import sys
import matplotlib.pyplot as plt
from combine_list import *

STOP_NUM=10

def format_data(data):
    if isinstance(data,list):
        return [2*((float(data[i])-min(data))/(max(data)-float(min(data))))-1 for i in range(len(data))]
def matchlist(data,throw_n,match_n):

    #data_max_index = list(argrelextrema(np.array(data),np.greater)[0])
    #data_min_index = list(argrelextrema(np.array(data),np.less)[0])
 
    sample_list = data[-(throw_n+match_n):-(throw_n)]
    waiting_match_list = data[20:-(throw_n+match_n)]
    throw_list=data[-(throw_n):]
    print "len w %s"%len(waiting_match_list)

    SMC = []
    for i in range(len(waiting_match_list)-len(sample_list)-throw_n):
        tmp_list = waiting_match_list[i:i+len(sample_list)]
        SMC.append((calc_SMC(tmp_list,sample_list),i+len(sample_list)))

    SMC.sort(key=lambda n:n[0])

    print "SMC %s"%SMC[-1][0]
    print "SMC %s"%SMC[-1][1]
    match_list = waiting_match_list[SMC[-1][1]:SMC[-1][1]+(throw_n)]
    print len(match_list)
    fin_list=[(throw_list[i]+match_list[i])/2 for i in range(len(match_list))]
    

    return list(data[:-(throw_n)])+list(match_list)
    #return list(data[:-(throw_n)])+list(fin_list)




def calc_SMC(list1,list2):
    pear = (float(np.cov(list1,list2)[0][1])/(np.std(list1)*np.std(list2)))
#    print pear
    MAD1 = [abs(list1[i]-list2[i]) for i in range(len(list1))]
#    print MAD1
    MAD = sum(MAD1)/float(len(MAD1))
#    print MAD
#    print 0.5*pear+0.5/float(MAD)
    return 0.5*pear+0.5/float(MAD) 



def matchlist2(data,pp,raw_data):

    combineflag=1

    data_max_index = list(argrelextrema(np.array(data),np.greater)[0])
    data_min_index = list(argrelextrema(np.array(data),np.less)[0])

#    data_max_index=filter(lambda n:n<len(data)-2,data_max_index)
#    data_min_index=filter(lambda n:n<len(data)-2,data_min_index)


    data_extreme = data_max_index+data_min_index
    data_extreme.sort()

    


    sample_list = data[data_extreme[-4]:(data_extreme[-2]+1)]
    throw_list=data[data_extreme[-2]:]

    para=[]
    if data_max_index[-1]<data_min_index[-1]:
        print "last min"
        if data_max_index[0]>data_min_index[0]:
            print "first min"
            for i in range(2,len(data_extreme)-7,2):
                tmp_list=data[data_extreme[i-1]:data_extreme[i+1]]
                para.append((calc_para(tmp_list,sample_list),data_extreme[i-1],data_extreme[i+1]))
        if data_max_index[0]<data_min_index[0]:
            print "first max"
            for i in range(3,len(data_extreme)-7,2):
                tmp_list=data[data_extreme[i-1]:data_extreme[i+1]]
                para.append((calc_para(tmp_list,sample_list),data_extreme[i-1],data_extreme[i+1]))

    if data_max_index[-1]>data_min_index[-1]:
        print "last max"
        if data_max_index[0]>data_min_index[0]:
            print "first min"
            for i in range(1,len(data_extreme)-7,2):
                tmp_list=data[data_extreme[i-1]:data_extreme[i+1]]
                para.append((calc_para(tmp_list,sample_list),data_extreme[i-1],data_extreme[i+1]))
        if data_max_index[0]<data_min_index[0]:
            print "first max"
            for i in range(2,len(data_extreme)-7,2):
                tmp_list=data[data_extreme[i-1]:data_extreme[i+1]]
                para.append((calc_para(tmp_list,sample_list),data_extreme[i-1],data_extreme[i+1]))


    para.sort(key=lambda n:n[0])
    min_para=para[0][0]
    flag=para[0][2]

    print "flag %s"%para[0][0]
    print "flag %s"%flag
    match_list = data[flag:flag+len(throw_list)+pp]
    m_list=[]
    p_list=[]
    #for i in range(20):
    #    if para[i][0]>2.5:
    #        break
    #stop_num=i-1
    #if stop_num<1:
    #    stop_num=1

    print "para len%s"%len(para)
    if len(para)<STOP_NUM:
        stop_num=len(para)
    else:
        stop_num=STOP_NUM
    if combineflag==1:
        for i in range(stop_num):
            m_list.append(data[para[i][2]:para[i][2]+len(throw_list)+pp])
            p_list.append(para[i][0])
        match_list=combinelist(m_list,p_list)

    match_max_index = list(argrelextrema(np.array(match_list),np.greater)[0])
    match_min_index = list(argrelextrema(np.array(match_list),np.less)[0])
    match_extrem_num=len(match_max_index)+len(match_min_index)
   # print "len matchlist %s"%len(match_list)
   # print "a=%s"%sample_list
    flag1=para[0][1]
   # print "b=%s"%data[flag1:flag]
  #  print "len  %s"%len(list(data[:-len(throw_list)]))

    cha=(list(data[:-len(throw_list)])[-1]-list(match_list)[0])
 #   print cha

    #fin_list=[(throw_list[i]*0.3+match_list[i]*0.7) for i in range(len(match_list))]
    #match_list=fin_list
    ###
    thrown_raw_data = format_data(raw_data[data_extreme[-2]:])
    match_thrown_data = format_data(list(match_list[:-pp]))
#    print "a1=%s"%thrown_raw_data
#    print "a2=%s"%match_thrown_data
    #real_cal_cha=calc_SMC(thrown_raw_data,match_thrown_data)
    real_cal_cha=0
    

####

    return (list(data[:-len(throw_list)])+list(match_list),min_para,match_extrem_num,cha,real_cal_cha)


def calc_para(list1,list2):
    

    list1_max_index = list(argrelextrema(np.array(list1),np.greater)[0])
    list1_min_index = list(argrelextrema(np.array(list1),np.less)[0])

    list2_max_index = list(argrelextrema(np.array(list2),np.greater)[0])
    list2_min_index = list(argrelextrema(np.array(list2),np.less)[0])

    error=10000

#######
#    if list1_max_index!=[] and list2_max_index!=[]:
#        error = abs(abs(list1[0]-list2[0])/list2[0])+abs(abs(list1[-1]-list2[-1])/list2[-1])+abs(abs(list1[list1_max_index[0]]-list2[list2_max_index[0]])/list2[list2_max_index[0]])+abs(1-list1_max_index[0]/list2_max_index[0])+abs(1-(len(list1)-list1_max_index[0])/(len(list2)-list2_max_index[0]))
#    if list1_min_index!=[] and list2_min_index!=[]:
#        error = abs(abs(list1[0]-list2[0])/list2[0])+abs(abs(list1[-1]-list2[-1])/list2[-1])+abs(abs(list1[list1_min_index[0]]-list2[list2_min_index[0]])/list2[list2_min_index[0]])+abs(1-list1_min_index[0]/list2_min_index[0])+abs(1-(len(list1)-list1_min_index[0])/(len(list2)-list2_min_index[0]))
        #error = abs(list1[0]-list2[0])+abs(list1[-1]-list2[-1])+abs(list1[list1_min_index[0]]-list2[list2_min_index[0]])
#######
    if list1_max_index!=[] and list2_max_index!=[]:
#        list1 = [i/abs(list1[list1_max_index[0]]-list1[0]) for i in list1]
#        list2 = [i/abs(list2[list2_max_index[0]]-list2[0]) for i in list2]
    #    error = abs((list2[0]/list1[0])-1)+abs((list2_max_index[0]/list1_max_index[0])-1)+abs((len(list2)-list2_max_index[0])/(len(list1)-list1_max_index[0])-1)
        #error = abs((list2[0]-list1[0])/list2[0])+abs((list2_max_index[0]-list1_max_index[0])/list2_max_index[0])+abs(((len(list2)-list2_max_index[0])-(len(list1)-list1_max_index[0]))/(len(list2)-list2_max_index[0]))+abs((list1[list1_max_index[0]]-list2[list2_max_index[0]])/list1[list1_max_index[0]])+abs((list2[-1]-list1[-1])/list2[-1])
        error = abs((list2[0]-list1[0])/min(list2[0],list1[0]))+abs((list2_max_index[0]-list1_max_index[0])/min(list2_max_index[0],list1_max_index[0]))+abs(((len(list2)-list2_max_index[0])-(len(list1)-list1_max_index[0]))/min((len(list2)-list2_max_index[0]),(len(list1)-list1_max_index[0])))+abs((list1[list1_max_index[0]]-list2[list2_max_index[0]])/min(list1[list1_max_index[0]],list2[list2_max_index[0]]))+abs((list2[-1]-list1[-1])/min(list1[-1],list2[-1]))+abs((len(list1)-len(list2))/min(len(list1),len(list2)))
    if list1_min_index!=[] and list2_min_index!=[]:
#        list1 = [i/abs(list1[list1_min_index[0]]-list1[0]) for i in list1]
#        list2 = [i/abs(list2[list2_min_index[0]]-list2[0]) for i in list2]
    #    error = abs((list2[0]/list1[0])-1)+abs((list2_min_index[0]/list1_min_index[0])-1)+abs((len(list2)-list2_min_index[0])/(len(list1)-list1_min_index[0])-1)
        #error = abs((list2[0]-list1[0])/list2[0])+abs((list2_min_index[0]-list1_min_index[0])/list2_min_index[0])+abs(((len(list2)-list2_min_index[0])-(len(list1)-list1_min_index[0]))/(len(list2)-list2_min_index[0]))+abs((list1[list1_min_index[0]]-list2[list2_min_index[0]])/list1[list1_min_index[0]])+abs((list2[-1]-list1[-1])/list2[-1])
        error = abs((list2[0]-list1[0])/min(list2[0],list1[0]))+abs((list2_min_index[0]-list1_min_index[0])/min(list2_min_index[0],list1_min_index[0]))+abs(((len(list2)-list2_min_index[0])-(len(list1)-list1_min_index[0]))/min((len(list2)-list2_min_index[0]),(len(list1)-list1_min_index[0])))+abs((list1[list1_min_index[0]]-list2[list2_min_index[0]])/min(list1[list1_min_index[0]],list2[list2_min_index[0]]))+abs((list2[-1]-list1[-1])/min(list1[-1],list2[-1]))+abs((len(list1)-len(list2))/min(len(list1),len(list2)))
    return error



def matchlist3(data,pp,raw_data):


    combineflag=1
    data_max_index = list(argrelextrema(np.array(data),np.greater)[0])
    data_min_index = list(argrelextrema(np.array(data),np.less)[0])

#    data_max_index=filter(lambda n:n<len(data)-2,data_max_index)
#    data_min_index=filter(lambda n:n<len(data)-2,data_min_index)


    data_extreme = data_max_index+data_min_index
    data_extreme.sort()

    sample_list = data[data_extreme[-5]:(data_extreme[-3]+1)]
    throw_list=data[data_extreme[-3]:]

    para=[]
    if data_max_index[-1]<data_min_index[-1]:
        print "last min"
        if data_max_index[0]>data_min_index[0]:
            print "first min"
            for i in range(1,len(data_extreme)-7,2):
                tmp_list=data[data_extreme[i-1]:data_extreme[i+1]]
                para.append((calc_para(tmp_list,sample_list),data_extreme[i-1],data_extreme[i+1]))
        if data_max_index[0]<data_min_index[0]:
            print "first max"
            for i in range(2,len(data_extreme)-7,2):
                tmp_list=data[data_extreme[i-1]:data_extreme[i+1]]
                para.append((calc_para(tmp_list,sample_list),data_extreme[i-1],data_extreme[i+1]))

    if data_max_index[-1]>data_min_index[-1]:
        print "last max"
        if data_max_index[0]>data_min_index[0]:
            print "first min"
            for i in range(2,len(data_extreme)-7,2):
                tmp_list=data[data_extreme[i-1]:data_extreme[i+1]]
                para.append((calc_para(tmp_list,sample_list),data_extreme[i-1],data_extreme[i+1]))
        if data_max_index[0]<data_min_index[0]:
            print "first max"
            for i in range(1,len(data_extreme)-7,2):
                tmp_list=data[data_extreme[i-1]:data_extreme[i+1]]
                para.append((calc_para(tmp_list,sample_list),data_extreme[i-1],data_extreme[i+1]))


    para.sort(key=lambda n:n[0])
    min_para=para[0][0]
    flag=para[0][2]
    print para

    print "flag %s"%para[0][0]
    print "flag %s"%flag
    match_list = data[flag:flag+len(throw_list)+pp]

    m_list=[]
    p_list=[]
    #for i in range(20):
    #    if para[i][0]>2.5:
    #        break
    #stop_num=i-1
    #if stop_num<1:
    #    stop_num=1

    print "para len%s"%len(para)
    if len(para)<STOP_NUM:
        stop_num=len(para)
    else:
        stop_num=STOP_NUM
    if combineflag==1:
        for i in range(stop_num):
            m_list.append(data[para[i][2]:para[i][2]+len(throw_list)+pp])
            p_list.append(para[i][0])
        match_list=combinelist(m_list,p_list)

    match_max_index = list(argrelextrema(np.array(match_list),np.greater)[0])
    match_min_index = list(argrelextrema(np.array(match_list),np.less)[0])
    match_extrem_num=len(match_max_index)+len(match_min_index)
    print "len matchlist %s"%len(match_list)
    print "a=%s"%sample_list
    flag1=para[0][1]
    print "b=%s"%data[flag1:flag]
    print "len  %s"%len(list(data[:-len(throw_list)]))
    cha=list(data[:-len(throw_list)])[-1]-list(match_list)[0]

    #fin_list=[(throw_list[i]*0.3+match_list[i]*0.7) for i in range(len(match_list))]
    #match_list=fin_list
    thrown_raw_data = format_data(raw_data[data_extreme[-3]:])
    match_thrown_data = format_data(list(match_list[:-pp]))
    print "a1=%s"%thrown_raw_data
    print "a2=%s"%match_thrown_data
    #real_cal_cha=calc_SMC(thrown_raw_data,match_thrown_data)
    real_cal_cha=0

    return (list(data[:-len(throw_list)])+list(match_list),min_para,match_extrem_num,cha,real_cal_cha)


def matchlist1(data,pp,raw_data):

    combineflag=1

    data_max_index = list(argrelextrema(np.array(data),np.greater)[0])
    data_min_index = list(argrelextrema(np.array(data),np.less)[0])

#    data_max_index=filter(lambda n:n<len(data)-2,data_max_index)
#    data_min_index=filter(lambda n:n<len(data)-2,data_min_index)


    data_extreme = data_max_index+data_min_index
    data_extreme.sort()

    sample_list = data[data_extreme[-3]:(data_extreme[-1]+1)]
    throw_list=data[data_extreme[-1]:]

    para=[]
    if data_max_index[-1]<data_min_index[-1]:
        print "last min"
        if data_max_index[0]>data_min_index[0]:
            print "first min"
            for i in range(1,len(data_extreme)-7,2):
                tmp_list=data[data_extreme[i-1]:data_extreme[i+1]]
                para.append((calc_para(tmp_list,sample_list),data_extreme[i-1],data_extreme[i+1]))
        if data_max_index[0]<data_min_index[0]:
            print "first max"
            for i in range(2,len(data_extreme)-7,2):
                tmp_list=data[data_extreme[i-1]:data_extreme[i+1]]
                para.append((calc_para(tmp_list,sample_list),data_extreme[i-1],data_extreme[i+1]))

    if data_max_index[-1]>data_min_index[-1]:
        print "last max"
        if data_max_index[0]>data_min_index[0]:
            print "first min"
            for i in range(2,len(data_extreme)-7,2):
                tmp_list=data[data_extreme[i-1]:data_extreme[i+1]]
                para.append((calc_para(tmp_list,sample_list),data_extreme[i-1],data_extreme[i+1]))
        if data_max_index[0]<data_min_index[0]:
            print "first max"
            for i in range(1,len(data_extreme)-7,2):
                tmp_list=data[data_extreme[i-1]:data_extreme[i+1]]
                para.append((calc_para(tmp_list,sample_list),data_extreme[i-1],data_extreme[i+1]))
    para.sort(key=lambda n:n[0])
    min_para=para[0][0]
    flag=para[0][2]

    print "flag %s"%para[0][0]
    print "flag %s"%flag
    match_list = data[flag:flag+len(throw_list)+pp]
    m_list=[]
    p_list=[]
    #for i in range(20):
    #    if para[i][0]>2.5:
    #        break
    #stop_num=i-1 
    #if stop_num<1:
    #    stop_num=1


    print "para len%s"%len(para)
    if len(para)<STOP_NUM:
        stop_num=len(para)
    else:
        stop_num=STOP_NUM
    if combineflag==1:
        for i in range(stop_num):
            m_list.append(data[para[i][2]:para[i][2]+len(throw_list)+pp])
            p_list.append(para[i][0])
        match_list=combinelist(m_list,p_list)
    match_max_index = list(argrelextrema(np.array(match_list),np.greater)[0])
    match_min_index = list(argrelextrema(np.array(match_list),np.less)[0])
    match_extrem_num=len(match_max_index)+len(match_min_index)
#    print "len matchlist %s"%len(match_list)
#    print "a=%s"%sample_list
    flag1=para[0][1]
#    print "b=%s"%data[flag1:flag]
#    print "len  %s"%len(list(data[:-len(throw_list)]))

    cha=list(data[:-len(throw_list)])[-1]-list(match_list)[0]

    #fin_list=[(throw_list[i]*0.3+match_list[i]*0.7) for i in range(len(match_list))]
    #match_list=fin_list
    #thrown_raw_data = format_data(raw_data[data_extreme[-1]:])
    #match_thrown_data = format_data(list(match_list[:-pp]))
    #print "a1=%s"%thrown_raw_data
    #print "a2=%s"%match_thrown_data
    #print len(thrown_raw_data)
    #print len(match_thrown_data)
    #real_cal_cha=calc_SMC(thrown_raw_data,match_thrown_data)
    real_cal_cha=0

    return (list(data[:-len(throw_list)])+list(match_list),min_para,match_extrem_num,cha,real_cal_cha)




def matchlist4(data):


    data_max_index = list(argrelextrema(np.array(data),np.greater)[0])
    data_min_index = list(argrelextrema(np.array(data),np.less)[0])

    data_max_index=filter(lambda n:n<len(data)-2,data_max_index)
    data_min_index=filter(lambda n:n<len(data)-2,data_min_index)


    data_extreme = data_max_index+data_min_index
    data_extreme.sort()




    sample_list = data[data_extreme[-6]:(data_extreme[-4])]
    throw_list=data[data_extreme[-4]:]

    para=[]
    if data_max_index[-1]<data_min_index[-1]:
        print "last min"
        if data_max_index[0]>data_min_index[0]:
            print "first min"
            for i in range(2,len(data_extreme)-8,2):
                tmp_list=data[data_extreme[i-1]:data_extreme[i+1]]
                para.append((calc_para(tmp_list,sample_list),data_extreme[i-1],data_extreme[i+1]))
        if data_max_index[0]<data_min_index[0]:
            print "first max"
            for i in range(3,len(data_extreme)-8,2):
                tmp_list=data[data_extreme[i-1]:data_extreme[i+1]]
                para.append((calc_para(tmp_list,sample_list),data_extreme[i-1],data_extreme[i+1]))

    if data_max_index[-1]>data_min_index[-1]:
        print "last max"
        if data_max_index[0]>data_min_index[0]:
            print "first min"
            for i in range(1,len(data_extreme)-8,2):
                tmp_list=data[data_extreme[i-1]:data_extreme[i+1]]
                para.append((calc_para(tmp_list,sample_list),data_extreme[i-1],data_extreme[i+1]))
        if data_max_index[0]<data_min_index[0]:
            print "first max"
            for i in range(2,len(data_extreme)-8,2):
                tmp_list=data[data_extreme[i-1]:data_extreme[i+1]]
                para.append((calc_para(tmp_list,sample_list),data_extreme[i-1],data_extreme[i+1]))


    para.sort(key=lambda n:n[0])
    min_para=para[0][0]
    flag=para[0][2]
    print para

    print "flag %s"%para[0][0]
    print "flag %s"%flag
    match_list = data[flag:flag+len(throw_list)+2]
    print "len matchlist %s"%len(match_list)
    print "a=%s"%sample_list
    flag1=para[0][1]
    print "b=%s"%data[flag1:flag]
    print "len  %s"%len(list(data[:-len(throw_list)]))

    #fin_list=[(throw_list[i]*0.3+match_list[i]*0.7) for i in range(len(match_list))]
    #match_list=fin_list

    return (list(data[:-len(throw_list)])+list(match_list),min_para)


if __name__=="__main__":
    a=[1,2,3,4,5,5,67,7]
    b=[2,2,3,4,5,5,67,7]
    num_imf=2
    print calc_SMC(a,b)
    datafile=sys.argv[1]
    n=int(sys.argv[2])
    fp = open(datafile)
    lines = fp.readlines()
    fp.close()
    close_price = []
    for eachline in lines:
        eachline.strip()
        close_price.append(float(eachline.split("\t")[4]))
    
    new=[]
    for i in range(3):
        new.append(0)
    for i in range(3,len(close_price)):
        new.append(np.mean(close_price[i-2:i+1]))
    #close_price=new
    data=close_price[n-1000:n]
    print data[-10:]
    print len(data)
    my_emd = one_dimension_emd(data,9)
    (imf, residual) = my_emd.emd(0.03,0.03)
    imfj=imf
    print data[-1]
    #new = matchlist(imf[2],15,100)

    print list(argrelextrema(np.array(imf[2]),np.greater)[0])
    print list(argrelextrema(np.array(imf[2]),np.less)[0])
    print "comm"
    print imf[2][-10:]
    (new,flag,f,cha) = matchlist2(list(imf[num_imf]),4)
    (new2,flag,f,cha) = matchlist3(list(imf[num_imf]),4)
    (new4,flag,f,cha) = matchlist1(list(imf[num_imf]),4)
    print "new"
    print list(argrelextrema(np.array(new2),np.greater)[0])
    print list(argrelextrema(np.array(new2),np.less)[0])

    data=close_price
    
    my_emd = one_dimension_emd(data,9)
    (imf, residual) = my_emd.emd(0.01,0.01)

    plt.figure(1)
    plt.plot(imf[num_imf][n-95:n+5],'r')
    plt.plot(imfj[num_imf][-100:],'y')
    plt.plot(new[-100:],'g')
    plt.plot(new2[-100:],'b')
    plt.plot(new4[-100:],'o')
    plt.figure(2)
    plt.plot(close_price[n-100:n])
    plt.show()
 
    
