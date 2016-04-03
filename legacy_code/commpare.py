
'''
total asset is 38420.2491208

buy day179
'''

import sys
import os
import numpy as np



def f1(l):
    l.insert(0,10000)
    l_diff=np.diff(l)
    l.pop()
    l2=l
    per=[]
    for i in range(len(l2)):
        per.append(l_diff[i]/l2[i])

    sharp=np.mean(per)/np.std(per)

    total=[i*10000 for i in per]
    return (sum(total),per)


directory = sys.argv[1]
f = os.popen("ls %s"%directory)
file_name = f.readlines()
f.close()

file_name=[i.strip("\n") for i in file_name]

no_name=""
no_name2=""
no_name3=""
if sys.argv[2]:
    no_name=sys.argv[2]
if sys.argv[3]:
    no_name2=sys.argv[3]
if sys.argv[4]:
    no_name3=sys.argv[4]

print file_name
print no_name
if no_name in file_name:
    file_name.remove(no_name)
if no_name2 in file_name:
    file_name.remove(no_name2)
if no_name3 in file_name:
    file_name.remove(no_name3)


result=[]
rss_list = []
increase_list = []
drawback = {}
ass_10000_total=[]
t_s=[]
total_hb=[]

print file_name
for eachfile in file_name:
    
#    eachfile = eachfile[:-1]

    fullname = directory+"/"+eachfile
    print fullname
    cmd = "tail -n 20 %s | grep total"%fullname
    #print cmd
    f = os.popen(cmd)
    total = f.readline()
    f.close()
    total=total.split(" ")[-1]
    total=total[:-1]


    cmd = "tail -n 20 %s | grep day"%fullname
    #print cmd
    f = os.popen(cmd)
    day = f.readline()
    f.close()
    day = day[7:]
    day=day[:-1]

    result.append((total,day))


    cmd = "tail -n 20 %s | grep rss"%fullname
    #print cmd
    f = os.popen(cmd)
    rss = f.readline()
    f.close()
    rss=rss.split(" ")[-1]
    rss=rss[:-1]
    if rss:
        rss_list.append(float(rss))


    cmd = "tail -n 20 %s | grep increase"%fullname
    #print cmd
    f = os.popen(cmd)
    increase = f.readline()
    f.close()
    increase=increase.split(" ")[-1]
    increase=increase[:-1]

    if increase:
        increase_list.append(float(increase))
###################################################################

    cmd = "tail -n 20 %s | grep ^rihb"%fullname
    #cmd = "tail -n 20 %s | grep ^hb"%fullname
    #print cmd
    f = os.popen(cmd)
    hb = f.readline()
    f.close()
    hb = hb[6:-2]
    #hb = hb[4:-2]
    #print "hb%s"%hb
    hb = hb.split(",")
    #print "hb%s"%hb
    hb=[float(i) for i in hb]
    total_hb+=hb
    #print total_hb




###############
    cmd = "tail -n 2 %s | head -n 1"%fullname
    #print cmd
    f = os.popen(cmd)
    asset = f.readline()
    f.close()
    asset = asset[1:-2]
    asset = asset.split(",")
    if asset and len(asset)>20:
        diff_asset = [float(asset[i+1])-float(asset[i]) for i in range(len(asset)-1)]
        drawback[eachfile]=min(diff_asset)
    asset=[float(i) for i in asset]
    ass_10000_total.append( f1(asset)[0])
    for i in f1(asset)[1]:
        t_s.append(i)
    
count=0
for i in t_s:
    if i>0:
        count+=1
print "win%s"%(float(count)/len(t_s))
    

print "sharp %s"%((np.mean(total_hb)/np.std(total_hb))*np.sqrt(240))
print "mean pro%s"%np.mean(total_hb)

print "sum%s"%sum(ass_10000_total)

every_day_p = []
total_profit=0
total_day=0
for t,d in result:
    if t and d:
        t=float(t)-10000
        total_profit+=float(t)
        total_day+=float(d)
        d=float(d)
        every_day_p.append(t/d)

print "sum%s"%(sum(ass_10000_total)/total_day)
print "sharp%s"%(np.mean(t_s)/np.std(t_s))
print every_day_p
print "total profit%s"%total_profit
print "total day%s"%total_day
print "avg profit%s"%(total_profit/total_day)
print "avg profit%s"%((total_profit/total_day)*250)

print "drawback %s"%(drawback)
    
print "mean increase %s"%(np.mean(increase_list))
print "mean rss %s"%(np.mean(rss_list))
print "every day %s %s"%(np.mean(every_day_p),np.std(every_day_p))



