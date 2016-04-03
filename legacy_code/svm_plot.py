
import os
import copy
import math
import sys
import matplotlib.pyplot as plt
import datetime
from scipy.signal import argrelextrema
import numpy as np



sys.path.append("../emd_util")
from generate_emd_data import generateEMDdata
from analyze import *
from spline_predict import linerestruct
from emd import *
from leastsqt import leastsqt_predict
from svm_uti import *

datafile=sys.argv[1]
fp = open(datafile)
lines = fp.readlines()
fp.close()
close_price = []
for eachline in lines:
    eachline.strip()
    close_price.append(float(eachline.split("\t")[4]))

data = close_price
print "len %s"%len(data)

ma5 = []
for i in range(3):
    ma5.append(0)

for i in range(3,len(data)):
    mean_5 = np.mean(data[i-2:i+1])
    ma5.append(mean_5)

data = ma5
print "len %s"%len(data)
ma5 = []
for i in range(3):
    ma5.append(0)

for i in range(3,len(data)):
    mean_5 = np.mean(data[i-2:i+1])
    ma5.append(mean_5)

data = ma5
print "len %s"%len(data)

ma5 = []
for i in range(3):
    ma5.append(0)

for i in range(3,len(data)):
    mean_5 = np.mean(data[i-2:i+1])
    ma5.append(mean_5)
##
#data = ma5
#
#ma5 = []
#for i in range(3):
#    ma5.append(0)
#
#for i in range(3,len(data)):
#    mean_5 = np.mean(data[i-2:i+1])
#    ma5.append(mean_5)
##
#data = ma5
#ma5 = []
#for i in range(3):
#    ma5.append(0)
#
#for i in range(3,len(data)):
#    mean_5 = np.mean(data[i-2:i+1])
#    ma5.append(mean_5)
##
#data = ma5
#ma5 = []
#for i in range(3):
#    ma5.append(0)
#
#for i in range(3,len(data)):
#    mean_5 = np.mean(data[i-2:i+1])
#    ma5.append(mean_5)
##
data = ma5
data = close_price
my_emd = one_dimension_emd(data,9)
(imf, residual) = my_emd.emd(0.01,0.01)
real_data=list(imf[2])
prediction=[]
prediction1=[]
prediction2=[]
raw_data=data
#######3########
for i in range(500,len(data)-3,2):
    p=[]
    pp=[]
    print "i %s"%i
    emd_data = raw_data[i-500:i+1]
    my_emd = one_dimension_emd(emd_data,9)
    (imf, residual) = my_emd.emd(0.01,0.01)
    print len(imf)
    #traindata=data[i-15:i-5]
    data=list(imf[2])
    traindata=data[-100:-90]
    #svm_train(traindata,3)
    testdata1 = traindata[-5:-2]
    print "testdata1%s"%testdata1
    #svm_train(traindata)
    #p.append(float(svm_predict(testdata1)))
    #prediction1.append(float(svm_predict(testdata1)))
    prediction1.append(imf[2][-16])
    #pp=(leastsqt_predict(data[i-1:i+1],1))
    #prediction2.append(pp[0])
    #print p
    #print pp
    #prediction.append(((prediction1[-1]+prediction2[-1])/2.0))

    testdata2 = traindata[-4:-2]
    testdata2.append(prediction1[-1])
    print "testdata2%s"%testdata2
    #traindata2=data[i-14:i-4]
    #traindata2.append(prediction1[-1])
    #svm_train(traindata2)
#    prediction1.append(float(svm_predict(testdata2)))

    prediction1.append(imf[2][-15])
#    prediction2.append(pp[0])
#    prediction.append(((prediction1[-1]+prediction2[-1])/2.0))
#
    testdata3 = data[i-0:i+1]
    #testdata3 = []
    testdata3.append(prediction1[-2])
    testdata3.append(prediction1[-1])
    print "testdata3%s"%testdata3
    #traindata3=data[i-13:i-3]
#    traindata3.append(prediction1[-2])
#    traindata3.append(prediction1[-1])
    #svm_train(traindata3)
#
#
    #prediction1.append(svm_predict(testdata3))
######


    testdata4 = []
    #testdata4.append(prediction1[-3])
    testdata4.append(prediction1[-2])
    testdata4.append(prediction1[-1])
#   svm_train(traindata3)
#
#
#    prediction1.append(svm_predict(testdata4))
#
#    testdata5 = []
#    testdata5.append(prediction1[-3])
#    testdata5.append(prediction1[-2])
#    testdata5.append(prediction1[-1])
#   svm_train(traindata3)
#
#
  #  prediction1.append(svm_predict(testdata5))



######

    #if i<len(data)-1:
    #    print data[i+1]
##############
print "len %s"%len(prediction1)
print "len data %s"%len(data)

data2 = data[:500]+prediction
data3 = real_data[:500]+prediction1
data4 = data[:500]+prediction2


plt.plot(real_data[450:],'bo',real_data[450:],'b')
#plt.plot(data2[400:],'r*',data2[400:],'r')
plt.plot(data3[465:],'g*',data3[465:],'g')
#plt.plot(data4[400:],'y*',data4[400:],'y')
plt.show()









