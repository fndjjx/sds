#! /usr/bin/python

def calc_power(price,ma5):
    if isinstance(price,list) and isinstance(ma5,list):
    	if len(price)==len(ma5):
    	    result = []
            for i in range(0,len(price)):
                result.append(price[i]-ma5[i])
        return result 
    else:
        print"calc_power:please input equal list"

def calc_flag(power):
    if isinstance(power,list):
        flag = []
        for i in range(0,len(power)-1):
            if power[i]>0 and power[i+1]<0:
                flag.append("1")
            elif power[i]<0 and power[i+1]>0:
            	flag.append("-1")
            else:
                flag.append("0")
        flag.append("0")
        return flag
    else:
        print "calc_flag:Please input list"

def generate_all(data,flag):
    if isinstance(data,list) and isinstance(flag,list):
        if len(data) == len(flag):
            data_new = []
            for i in range(len(data)):
                data_new.append(data[i])
                data_new.append(flag[i])
            return data_new
        else:
            print "Please input equal list"
    else:
        print "Please input list"

def generate_divide(data_raw,data_all,period,input_number):
    for i in range(len(data_raw)-period+1):
        fp = open("data_%s.txt"%i,'w')
        fp.write("%s %s 1\n"%(period,input_number))
        for j in range(i*2,(period+i)*2): 
            fp.writelines(str(data_all[j]))
            fp.writelines("\n")
        fp.close()



if __name__=="__main__":
    x = [1,2,3,4,5]
    y = [2,4,6,3,4]
    print type(x)
    power = calc_power(x,y)
    print power
    flag = calc_flag(power)
    print flag
    data = [[1,2],[2,4],[3,6],[4,3],[5,4]]
    print data
    data_all = generate_all(data,flag)
    print data_all
    generate_divide(x,data_all,2,2)

