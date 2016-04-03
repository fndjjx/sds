#!/usr/bin/python

import libfann
from pre_condition import *

def train_nn(data_file,net_file):

    connection_rate = 1
    learning_rate = 0.7
    num_input = 2
    num_hidden = 4
    num_output = 1
    
    desired_error = 0.0001
    max_iterations = 100000
    iterations_between_reports = 10000
    ann = libfann.neural_net()
    ann.create_standard_array((1,10,1))
    ann.set_learning_rate(0.3)
    ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)
    
    ann.train_on_file(data_file, max_iterations, iterations_between_reports, desired_error)
    
    ann.save(net_file)


def sell(share,price):
    return float(share)*float(price)

def buy(money,price):
    return float(money)/float(price)


def file_count():
    f = os.popen("ls train_data|wc -l")
    data = f.readline()
    f.close()
    return data


def test_each(net_file,test_data):

    ann = libfann.neural_net()
    ann.create_from_file(net_file)

    return ann.run([test_data])[0]




if __name__ == '__main__':
    #train_nn('train_data/data_afterpre_0','prediction.net')
    #print test_each('prediction.net',4.6)
    money = 10000
    share = 0
    buy_price = 0


    openFile('data_zhaoshang',10)
    for i in range(0,int(file_count())*2-1,2):
        print "loop %s"%i
        file = "train_data/data_afterpre_%s"%i
	file_next = "train_data/data_afterpre_%s"%(i+2)
	f=os.popen("tail %s -n 2|head -n 1"%file_next)
	price=f.readline()
	f.close()
        train_nn(file,'prediction.net')
	result = test_each('prediction.net',float(price))
        if float(result)>0.8 and share>0:
	    money = sell(share,price) 
	    share = 0
	    buy_price = 0
	if float(buy_price)!=0 and abs(float(buy_price)-float(price))/float(buy_price)>0.02 and share>0:
	    money = sell(share,price)
	    share = 0
	    buy_price = 0
	if float(result)<-0.8 and money>0:
	    share = buy(money,price)
	    money = 0
	    buy_price = price
	print "result %s"%result
	print "money %s"%money
	print "share %s"%share
	print "price %s"%price
	print "total asset is %s"%(float(money)+float(share)*float(price))













