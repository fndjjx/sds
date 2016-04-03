#! /usr/bin/python

import os
import libfann
import threading
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import numpy as np
import neurolab as nl


final_asset_list = []
price_max = []
price_min = []
price_max_index = []
price_min_index = []
result_index = []
flag_index = []


def get_real_price(pricefile):
    price_list = []
    fp = open(pricefile,'r')
    price_list_raw = fp.readlines()
    fp.close()
    for eachline in price_list_raw:
        eachline.rstrip('\n')
	price_list.append(float(eachline))
    print price_list

    return price_list

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
    print power
    global flag_index
    if isinstance(power,list):
        flag = []
        for i in range(len(power)-1):
	    if i == 0:
	        flag.append(float("0"))
            elif power[i]>0 and power[i+1]<0:
	        flag.pop()
                flag.append(float("1"))
                flag.append(float("0"))
            elif power[i]<0 and power[i+1]>0:
	        flag.pop()
            	flag.append(float("-1"))
            	flag.append(float("0"))
            else:
	        flag.pop()
                flag.append(float("0"))
                flag.append(float("0"))
        flag.append(float("0"))
	print "length flag %s"%len(flag)
	print "flag %s"%flag
	flag_index = flag
        return flag
        
    else:
        print "calc_flag:Please input list"
	
def calc_extrem_point(pricefile):
    global price_max
    global price_min
    global price_max_index
    global price_min_index
    price_list = get_real_price(pricefile)
    print "price_list%s"%price_list
    price_list_array = np.array(price_list)
    print "price_list_array%s"%price_list_array
    price_max_index = argrelextrema(price_list_array,np.greater)
    price_min_index = argrelextrema(price_list_array,np.less)
    print "price_max_index%s"%price_max_index
    print "price_min_index%s"%price_min_index
    price_max = [price_list[price_max_index[0][i]] for i in range(len(price_max_index[0])) ]
    price_min = [price_list[price_min_index[0][i]] for i in range(len(price_min_index[0])) ]

def calc_extrem_flag(price_list,price_max_index,price_min_index):
    flag = []
    global flag_index
    for i in range(len(price_list)):
        if i == 0:
	    flag.append(float("0"))
        elif i in price_max_index[0]:
	    flag.pop()
	    flag.append(float("1"))
	    flag.append(float("0"))
	elif i in price_min_index[0]:
	    flag.pop()
	    flag.append(float("-1"))
	    flag.append(float("0"))
	else:
	    flag.pop()
	    flag.append(float("0"))
	    flag.append(float("0"))
    flag_index = flag
    print "price_list %s"%len(price_list)
    print "flag %s"%len(flag)
    return flag
    


def generate_all(data,flag):
    print "length data %s"%len(data)
    print "length flag %s"%len(flag)
    if isinstance(data,list) and isinstance(flag,list):
        if len(data) == len(flag):
            data_new = []
            for i in range(len(data)):
                data_new.append(data[i])
                data_new.append(flag[i])
	    print "data_new%s"%data_new
            return data_new
        else:
            print "Please input equal list"
    else:
        print "Please input list"

def generate_divide(data_raw,data_all,period,input_number,train_data_dir):
    for i in range(len(data_raw)-period+1):
        fp = open("%s/data_%s"%(train_data_dir,i),'w')
        fp.write("%s %s 1\n"%(period,input_number))
        for j in range(i*2,(period+i)*2): 
            fp.writelines(str(data_all[j]))
            fp.writelines("\n")
        fp.close()

def train_nn(data_file,period):

    connection_rate = 1
    learning_rate = 0.7
    num_input = 2
    num_hidden = 4
    num_output = 1

    desired_error = 0.0001
    max_iterations = 10000
    iterations_between_reports = 10000

    fp = open(data_file,'r')
    lines = fp.readlines()
    fp.close()

    price_list = []
    ma5_list = []
    ma10_list = []
    vol5_list = []
    vol10_list = []
    macd_list = []
    vol_list = []
    
    target_list = []

    for i in range(len(lines)):
        if i != 0:
	    eachline = lines[i]
            if i%2 != 0:
                 price_list.append(eachline.split(" ")[0])
                 ma5_list.append(eachline.split(" ")[1])
                 ma10_list.append(eachline.split(" ")[2])
                 vol5_list.append(eachline.split(" ")[3])
                 vol10_list.append(eachline.split(" ")[4])
                 macd_list.append(eachline.split(" ")[5])
                 vol_list.append(eachline.split(" ")[6])
            if i%2 == 0:
                 target_list.append(eachline.split("\n")[0])
    input = np.array([price_list,ma5_list,ma10_list,vol5_list,vol10_list,macd_list,vol_list])
    input = np.transpose(input)
    target = np.array([target_list])
    target = np.transpose(target)
    #print "target %s"%target
    net = nl.net.newelm([[0, 1], [0, 1], [0,1], [0,1], [0,1], [0,1], [0,1]], [10,1])
    err = net.train(input, target, show=250,epochs=500)

    return net



def sell(share,price):
    return float(share)*float(price)

def buy(money,price):
    return float(money)/float(price)


def file_count(train_data_dir):
    f = os.popen("ls %s|wc -l"%train_data_dir)
    data = f.readline()
    f.close()
    return data

def format_data(data):
    if isinstance(data,list):
        return [ (float(data[i])-min(data))/(max(data)-float(min(data))) for i in range(len(data))] 

def test_each(net,price,ma5,ma10,vol5,vol10,macd,vol):
   
    print "test data%s"%[price,ma5,ma10,vol5,vol10,macd,vol]
    print "result %s"%net.sim([[price,ma5,ma10,vol5,vol10,macd,vol]])

    return net.sim([[price,ma5,ma10,vol5,vol10,macd,vol]])[0][0]

def format_train_data(datafile,price_file,ma_file):
    fp = open(datafile,'r')
    lines = fp.readlines()
    fp.close()
    lists = [[] for i in range(len(lines[0].split("\t")))]
    for eachline in lines:
        data_list = eachline.split("\t")
        for i in range(len(data_list)):
            lists[i].append(float(data_list[i].strip("\n")))   
    print "lists %s"%lists
    fp = open(price_file,'w')
    for price in lists[0]:
        fp.write(str(price))
	fp.write("\n")
    fp.close()
    fp = open(ma_file,'w')
    for ma in lists[1]:
        fp.write(str(ma))
	fp.write("\n")
    fp.close()
    for i in range(len(lines[0].split("\t"))):
        lists[i] = format_data(lists[i])
    return lists

def write_two_dimension_list(two_dimension_list,target_file):
    lists = [[] for i in range(len(two_dimension_list[0]))]
    for i in range(len(two_dimension_list[0])):
        for j in range(len(two_dimension_list)):
            lists[i].append(str(two_dimension_list[j][i]))
	    lists[i].append(" ")
    print "after format lists %s"%lists
    fp = open(target_file,'w')
    for i in range(len(lists)):
        fp.writelines(lists[i])
	fp.write("\n")
    fp.close
	    
def generate_train_data(rawdatafile,period,train_data_dir,input_number,pricefile,mafile):
    
    fp = open(rawdatafile,'r')
    lines = fp.readlines()
    fp.close()

    new_lines = []

    price = []
    ma5 = []
    for eachline in lines:
#        print "eachline %s"%eachline
#	print eachline.split("\t")
	eachline.strip("\n")
#        print "eachline %s"%eachline
	new_lines.append(eachline)
        price.append(float(eachline.split(" ")[0]))
        ma5.append(float(eachline.split(" ")[1]))
#    print "new_lines%s"%new_lines
#    print "price %s"%price
#    print "ma5 %s"%ma5
    os.system("rm -r %s"%train_data_dir)
    os.system("mkdir %s"%train_data_dir)
    power = calc_power(price,ma5)
    flag = calc_flag(power)
#    calc_extrem_point(mafile)
#    flag = calc_extrem_flag(price,price_max_index,price_min_index)
    data_after_flag = generate_all(new_lines,flag)     
    generate_divide(lines,data_after_flag,period,input_number,train_data_dir)
    os.system("sed -i /^$/d %s/*"%train_data_dir)
    
   
def train(train_data_dir,price_file,period,ma_file):
    money = 10000
    share = 0
    buy_price = 0
    final_asset = 0



    global final_asset_list
    global result_index
 #   for i in range(period):
 #       result_index.append(float("0"))
    fp = open(price_file,'r')
    price_list = fp.readlines()
    fp.close()

    fp = open(ma_file,'r')
    ma_list = fp.readlines()
    fp.close()
    
    sell_x_index = []
    sell_y_index = []
    sell_lost_x_index = []
    sell_lost_y_index = []
    buy_x_index = []
    buy_y_index = []
    
    (flag_buy_x_index,flag_buy_y_index,flag_sell_x_index,flag_sell_y_index) = flag_value(flag_index,price_list)

    for i in range(int(file_count(train_data_dir))-2):
        print "loop %s"%i
        file = "%s/data_%s"%(train_data_dir,i)
        file_next = "%s/data_%s"%(train_data_dir,i+1)
        f=os.popen("tail %s -n 2|head -n 1"%file_next)
	line=f.readline()
	print "price ma5 %s"%line
        price=float(line.split(" ")[0])
#	print "price%s"%price
	ma5=float(line.split(" ")[1])
	ma10=float(line.split(" ")[2])
	vol5=float(line.split(" ")[3])
	vol10=float(line.split(" ")[4])
	macd=float(line.split(" ")[5])
	vol=float(line.split(" ")[6])
#	print "ma5%s"%ma5
        f.close()

	f = os.popen("tail %s -n 1"%file_next)
	f_t = f.readline()
	f.close()
	result_list = []
	for j in range(10):
            net = train_nn(file,period)
            result_list.append(test_each(net,price,ma5,ma10,vol5,vol10,macd,vol))
	result_list.remove(max(result_list))
	result_list.remove(min(result_list))
        print "result_list%s"%result_list	
	result = sum(result_list)/len(result_list)
#	if result>0.6:
#	    result_index.append(float("1"))
#	elif result<-0.8:
#	    result_index.append(float("-1"))
#	else:
#	    result_index.append(float("0"))

        result_index.append(result)
        
	print "avg_reslut%s"%result
	print "flag real %s"%f_t
	
#        file_next_next = "%s/data_%s"%(train_data_dir,i+2)
#	f=os.popen("tail %s -n 2|head -n 1"%file_next_next)
#        price = f.readline().split(" ")[0]
#	f.close()



        
        delta_current_ma = float(ma_list[period+i])-float(ma_list[period+i-1])
	delta_last_ma = float(ma_list[period+i-1])-float(ma_list[period+i-2])
	delta_ma = delta_current_ma-delta_last_ma



        trade_price = price_list[period+i+1] 
        current_price = price_list[period+i]
	#trade_price = current_price
	print "trade_price%s"%trade_price
        if float(result)>0.5 and share>0 and (float(current_price)-float(buy_price))/float(buy_price)>0.02 and delta_ma<0:
	    print "sell"
            money = sell(share,trade_price)
            share = 0
            buy_price = 0
	    sell_x_index.append(period+i+1)
	    sell_y_index.append(trade_price)
        if float(buy_price)!=0 and (float(buy_price)-float(current_price))/float(buy_price)>0.02 and share>0:
	    print "sell"
            money = sell(share,trade_price)
            share = 0
            buy_price = 0
	    sell_lost_x_index.append(period+i+1)
            sell_lost_y_index.append(trade_price)
#
        if float(result)<-0.5  and money>0 and delta_ma>0:
	    print "buy"
            share = buy(money,trade_price)
            money = 0
            buy_price = trade_price
            buy_x_index.append(period+i+1)
	    buy_y_index.append(trade_price)

        print "result %s"%result
        print "money %s"%money
        print "share %s"%share
        print "price %s"%trade_price
        print "total asset is %s"%(float(money)+float(share)*float(trade_price))
	final_asset = float(money)+float(share)*float(trade_price)
    final_asset_list.append(float(final_asset))
    print "flag index length: %s result_index: %s"%(len(flag_index[10:-1]),len(result_index))
    print "result index %s"%result_index
    diffnum = calc_diff_power(flag_index[10:-1],result_index)
    print "diff power %s"%diffnum
    max_diffnum = calc_max_diff_power(flag_index[10:-1])
    print "max diff power %s"%(max_diffnum)
    print "parameter of power diff is %s"%(diffnum/max_diffnum)
#    plt.plot([i for i in range(len(price_list))],price_list,'b',buy_x_index,buy_y_index,'ro',sell_x_index,sell_y_index,'go',price_min_index[0],price_min,'r*',price_max_index[0],price_max,'g*')
#    plt.figure(1)
#    plt.plot([i for i in range(len(price_list))],price_list,'b',buy_x_index,buy_y_index,'r*',sell_x_index,sell_y_index,'g*',sell_lost_x_index,sell_lost_y_index,'g*')
#    plt.plot([i for i in range(len(price_list))],price_list,'b',flag_buy_x_index,flag_buy_y_index,'ro',flag_sell_x_index,flag_sell_y_index,'go')
#    plt.plot([i for i in range(len(price_list))],price_list,'b',buy_x_index,buy_y_index,'ro',sell_x_index,sell_y_index,'g*')
#    plt.plot([i for i in range(len(price_list))],price_list,'b',buy_x_index,buy_y_index,'ro')
#    plt.figure(2)
#    plt.plot([i for i in range(len(flag_index))],flag_index,'g',[i for i in range(len(flag_index))],flag_index,'go',[i for i in range(len(result_index))],result_index,'ro')
#    plt.show()


def flag_value(flag,price_list):
    print "flag_value flag%s"%flag
    print "price_list %s"%price_list
    flag_x_buy_index = []
    flag_y_buy_index = []
    flag_x_sell_index = []
    flag_y_sell_index = []

    for i in range(len(flag)):
        if flag[i] == 1:
	    flag_x_sell_index.append(float(i+1))
	if flag[i] == -1:
	    flag_x_buy_index.append(float(i+1))
    for i in range(len(price_list)):
        if i in flag_x_sell_index:
	    flag_y_sell_index.append(float(price_list[i]))
	if i in flag_x_buy_index:
	    flag_y_buy_index.append(float(price_list[i]))
    print "length of buy x %s"%len(flag_x_buy_index)
    print "length of buy y %s"%len(flag_y_buy_index)
    print "length of sell x %s"%len(flag_x_sell_index)
    print "length of sell y %s"%len(flag_y_sell_index)
    return (flag_x_buy_index,flag_y_buy_index,flag_x_sell_index,flag_y_sell_index)
    

def calc_diff_power(flag_list,result_list):
    power =  map(lambda x,y:(x-y)**2,flag_list,result_list)
    return sum(power)

def calc_max_diff_power(flag_list):
    reverse_flag_list = []
    for i in range(len(flag_list)):
        if flag_list[i] == 1:
	    reverse_flag_list.append(float("-1"))
	if flag_list[i] == -1:
	    reverse_flag_list.append(float("1"))
	if flag_list[i] == 0:
	    reverse_flag_list.append(float("1"))
    return calc_diff_power(flag_list,reverse_flag_list)
if __name__=="__main__":
#    x = [1,2,3,4,5]
#    y = [2,4,6,3,4]
#    print type(x)
#    power = calc_power(x,y)
#    print power
#    flag = calc_flag(power)
#    print flag
#    data = ["1 2","2 4","3 6","4 3","5 4"]
#    print data
#    data_all = generate_all(data,flag)
#    print data_all
#    generate_divide(x,data_all,2,2)

#    money = 10000
#    share = 0
#    buy_price = 0
#
#
#    openFile('data_hxdq',10)
#    for i in range(0,int(file_count())*2-1,2):
#        print "loop %s"%i
#        file = "train_data/data_afterpre_%s"%i
#        file_next = "train_data/data_afterpre_%s"%(i+2)
#        f=os.popen("tail %s -n 2|head -n 1"%file_next)
#        price=f.readline()
#        f.close()
#        train_nn(file,'prediction.net')
#        result = test_each('prediction.net',float(price))
#        if float(result)>0.8 and share>0:
#            money = sell(share,price)
#            share = 0
#            buy_price = 0
#        if float(buy_price)!=0 and abs(float(buy_price)-float(price))/float(buy_price)>0.02 and share>0:
#            money = sell(share,price)
#            share = 0
#            buy_price = 0
#        if float(result)<-0.8 and money>0:
#            share = buy(money,price)
#            money = 0
#            buy_price = price
#        print "result %s"%result
#        print "money %s"%money
#        print "share %s"%share
#        print "price %s"%price
#        print "total asset is %s"%(float(money)+float(share)*float(price))
    
    write_two_dimension_list(format_train_data("data_hxdq","data_price_hxdq","data_ma5_hxdq"),"data_after_format")
    generate_train_data("data_after_format",10,"train_data",7,"data_price_hxdq","data_ma5_hxdq")
    loops = 100
    threads = []
    for i in range(loops):
        t = threading.Thread(target=train,args=("train_data","data_price_hxdq",10,"data_ma5_hxdq"))
        threads.append(t)
	print threads
    for i in range(loops):
        threads[i].start()
    for i in range(loops):
        threads[i].join()
    print "total_asset%s"%final_asset_list 
    print "max_asset%s"%max(final_asset_list)
    print "min_asset%s"%min(final_asset_list)
    print "avg_asset%s"%(sum(final_asset_list)/len(final_asset_list))

        


