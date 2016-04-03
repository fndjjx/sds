from scipy import interpolate
from smooth import cubicSmooth5
import numpy as np
from scipy import signal
from scipy import fftpack
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema


def fftfilter(rawdata,last_extrem,last_freq,max_freq = 0,min_freq = 0):

#    if last_extrem == "min":
#        tmp = []
#        for i in range(len(rawdata)):
#            if i<max_index[0] and  rawdata[i]<=rawdata[0]:
#                tmp.append(rawdata[0])
#            else:
#                tmp.append(rawdata[i])
#    elif last_extrem == "max":
#        tmp = []
#        for i in range(len(rawdata)):
#            if i<min_index[0] and  rawdata[i]>=rawdata[0]:
#                tmp.append(rawdata[0])
#            else:
#                tmp.append(rawdata[i])

#    data = tmp


#    data = [rawdata[i]-rawdata[i+1] for i in range(len(rawdata)-1)]
#    max_index = list(argrelextrema(np.array(data),np.greater)[0])
#    min_index = list(argrelextrema(np.array(data),np.less)[0])
#    if last_extrem == "max" and min_index[0]>max_index[0]: 
#        data = [rawdata[i+1]-rawdata[i] for i in range(len(rawdata)-1)]
#    if last_extrem == "min" and min_index[0]<max_index[0]: 
#        data = [rawdata[i+1]-rawdata[i] for i in range(len(rawdata)-1)]
#
#    print "after delta %s"%data


#    rawdata = data
#    data = [rawdata[i]-rawdata[i+1] for i in range(len(rawdata)-1)]
#    max_index = list(argrelextrema(np.array(data),np.greater)[0])
#    min_index = list(argrelextrema(np.array(data),np.less)[0])
#    if last_extrem == "max" and min_index[0]>max_index[0]:
#        data = [rawdata[i+1]-rawdata[i] for i in range(len(rawdata)-1)]
#    if last_extrem == "min" and min_index[0]<max_index[0]:
#        data = [rawdata[i+1]-rawdata[i] for i in range(len(rawdata)-1)]
#             
#    print "after delta %s"%data



    data = rawdata
    sig_fft = fftpack.fft(data)
    sample_freq = fftpack.fftfreq(len(data))
    print "sample freq %s"%list(sample_freq)
    pidxs = np.where(sample_freq > 0)
    freqs = sample_freq[pidxs]
    power = np.abs(sig_fft)[pidxs]
    freq = freqs[power.argmax()]
    print "main power freq %s"%freq
#    if freq>max_freq or freq<min_freq:
#        return (None,None,None)
#
    freq_cha = [abs(last_freq-list(sample_freq)[i]) for i in range(len(list(sample_freq)))]
    small_index = freq_cha.index(min(freq_cha))
#    for i in range(len(sample_freq)):
#        if last_freq*0.9<list(sample_freq)[i] :
#            print list(sample_freq)[i]
#            max_freq = list(sample_freq)[i]*1.1
#            min_freq = list(sample_freq)[i]*0.9
#            break
    
    max_freq = list(sample_freq)[small_index]*1.1
    min_freq = list(sample_freq)[small_index]*0.9
    print "max freq %s"%max_freq
    print "min freq %s"%min_freq
    if max_freq != 0 :
        sig_fft[(sample_freq) > max_freq] = 0
    if min_freq != 0 :
        sig_fft[(sample_freq) < min_freq] = 0

    #sig_fft[np.abs(sample_freq) > freq] = 0
    main_sig = fftpack.ifft(sig_fft)
    print "filtered sig %s"%(main_sig)

    filtered_sig = []
    for i in main_sig:
        filtered_sig.append(float(np.real(i)))

    print "filtered sig %s"%(filtered_sig)
    filtered_max_index = list(argrelextrema(np.array((filtered_sig)),np.greater)[0])
    filtered_min_index = list(argrelextrema(np.array((filtered_sig)),np.less)[0])

    
 
    #return (list(np.abs(main_sig)),filtered_max_index,filtered_min_index)
    return ((filtered_sig),filtered_max_index,filtered_min_index)
        



def fftfilter2(rawdata,last_extrem,max_freq = 0,min_freq = 0):
    
    data = [rawdata[i]-rawdata[i+1] for i in range(len(rawdata)-1)]
    (sig,max_index,min_index) = fftfilter(data,last_extrem,max_freq = 0,min_freq = 0)
    print max_index
    print min_index
    print last_extrem
    if max_index!=[] and min_index!=[] :
        print last_extrem
        if (last_extrem == "min" and max_index[0]<min_index[0]) or (last_extrem == "max" and max_index[0]>min_index[0]):
            return (sig,max_index,min_index)
        else:
            data = [rawdata[i+1]-rawdata[i] for i in range(len(rawdata)-1)]
            (sig,max_index,min_index) = fftfilter(data,last_extrem,max_freq = 0,min_freq = 0)
            return (sig,max_index,min_index)
    else:
        return (None,None,None) 
        








