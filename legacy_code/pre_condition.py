#! /usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal
from scipy.signal import argrelextrema
from scipy.signal import *
import os
#matplotlib.use('Agg')
def openFile(filepath, period):
    os.system('mkdir train_data')
    fp = open(filepath,'r')
    lines = fp.readlines()
    fp.close()
    data_list = []
    print lines
    for eachline in lines:
        eachline.strip('\n')
	print "each line of data%s"%eachline
        data_list.append(float(eachline))
    x = np.arange(0,len(lines))
    y = data_list
    y_array = np.array(y)
    x_ymax = argrelextrema(y_array,np.greater)
    x_ymin = argrelextrema(y_array,np.less)
#ymax = []
#ymin = []
#
#for i in range(len(y)):
#    for j in range(len(x_ymax[0])):
#        if i == x_ymax[0][j]:
#            ymax.append(y[i])
#for i in range(len(y)):
#    for j in range(len(x_ymin[0])):
#        if i == x_ymin[0][j]:
#            ymin.append(y[i])
    y_new = []
    for i in range(len(y)):
        if i in x_ymax[0]:
            y_new.append(y[i])
            y_new.append("1")
        elif i in x_ymin[0]:
            y_new.append(y[i])
            y_new.append("-1")
        else:
            y_new.append(y[i])
            y_new.append("0")
    for i in range(len(y_new)):
        y_new[i] = "%s\n"%y_new[i]

    for i in range(0,2*len(y)-period,2): 
        count = 0
	if i+period*2 > len(y_new):
	    #count = (len(y_new) - i)/2
	    break
	else:
	    count = period
	fp = open("train_data/data_afterpre_%s"%i,'w')
        fp.writelines("     1 1")
	fp.write("\n")
	fp.writelines(y_new[i:i+period*2])

	fp.seek(0,0)
	fp.write("%s "%count)
        fp.close()

    
#plt.plot(x,y,'b',x_ymax[0],ymax,'ro',x_ymin[0],ymin,'yo')
#plt.show()

if __name__=="__main__":
   openFile("data_zhaoshang",5)
