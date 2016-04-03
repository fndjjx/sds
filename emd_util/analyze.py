import numpy as np
from spline_predict import linerestruct

def precondition(raw_list):
    print len(raw_list)
    new = []
    new.append(raw_list[0])
#    print new
    for i in range(len(raw_list)-1):
        if raw_list[i+1]==raw_list[i]:
            pass
        else:
            new.append(raw_list[i+1])
#        print new
#    print new
    return new
def success_ratio(result_list):
  
    result_list.pop(0)
    result_list.pop(0)
    good=0
    bad=0
    total=0
    diff_result = [result_list[i+1]-result_list[i] for i in range(len(result_list)-1)]
    total = len(diff_result)
    for i in diff_result:
        if i>0:
            good+=1
        else:
            bad+=1

    return (float(good)/total)
def income_mean_std(data):
    data.pop(0)
    data.pop(0)
    print data
    data = np.diff(data)
    positive = []
    negative = []
    for i in data:
        if i>0:
            positive.append(i)
        else:
            negative.append(i)
        
    return ((np.mean(positive),np.std(positive)),((np.mean(negative)),(np.std(negative))))



def profit_smooth(data):

    x=[]
    y=[]
    for i in range(1,len(data)):
        if abs(data[i]-data[i-1])>10:
            x.append(i)
            y.append(data[i])

    yy = linerestruct(x,y)

    cha= [] 
    for i in range(len(y)):
        cha.append((y[i]-yy[i])*(y[i]-yy[i]))
    print "rss  %s"%sum(cha)
    print "increase %s"%((yy[-1]-yy[0])/(x[-1]-x[0]))
    




if __name__=="__main__":
    asset_list=[1,1,3,4,5,2]
    print precondition(asset_list)
    print success_ratio(asset_list)
    a=[2,2,3,4,5]
    print profit_smooth(a)
