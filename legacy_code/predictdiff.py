import libfann
import numpy as np
import math

import matplotlib.pyplot as plt
import os

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
    ann.create_standard_array((20,20,1))
    ann.set_learning_rate(0.3)
    ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)

    ann.train_on_file(data_file, max_iterations, iterations_between_reports, desired_error)

    ann.save(net_file)



def test_each(net_file,test_data):

    ann = libfann.neural_net()
    ann.create_from_file(net_file)

    return ann.run(test_data)[0]

def generate_test_data(input_num,eachfile_group_num):

    fp = open("imf5",'r')
    data = fp.readlines()
    fp.close()

    data_new = []
    for i in data:
        data_new.append(i.strip("\n"))

    raw_data_len = len(data_new)

    #print data_new
    expect = 0 
    fp = open("train_data_all", 'w')
    for i in range(len(data_new)-input_num+1):
        for j in range(i,input_num+i):
            fp.writelines(str(data_new[j]))
            fp.writelines(" ")
            expect = j+1
        fp.writelines("\n")
        if expect <= len(data_new)-1:
            fp.writelines(str(data_new[expect]))
            fp.writelines("\n")
    fp.writelines("0")
    fp.close()

    fp = open("train_data_all",'r')
    data = fp.readlines()
    fp.close() 
    total_group_num = raw_data_len-input_num+1
    total_file_num = total_group_num - eachfile_group_num +1
    print "total group num %s"%total_group_num
    print "total file num%s"%total_file_num
    
    for i in range(total_file_num):
        fp = open("../../train_data/data_%s"%i,'w')
        fp.write("%s %s 1\n"%(eachfile_group_num,input_num))
        for j in range(i*2,(eachfile_group_num+i)*2):
            fp.writelines(str(data[j]))
        fp.close()


def file_count():
    f = os.popen("ls ../../train_data|wc -l")
    data = f.readline()
    f.close()
    return data





generate_test_data(20,10)

result_list = []

for i in range(int(file_count())-1):
    print "loop %s"%i
    file_current = "../../train_data/data_%s"%i
    file_next = "../../train_data/data_%s"%(i+1)
    f=os.popen("tail %s -n 2|head -n 1"%file_next)
    line=f.readline()
    test_data=line.split(" ")
    test_data.pop()
    test_data1=[]
    print test_data
    for i in range(len(test_data)):
        test_data1.append(float(test_data[i]))
    f.close()
    train_nn(file_current,'prediction.net')
    result = test_each('prediction.net',test_data1)
    result_list.append(result)

fp = open("imf5",'r')
data = fp.readlines()
fp.close()

data_new = []
for i in data:
    data_new.append(i.strip("\n"))

print "length predict %s"%len(result_list)
print "length real %s"%len(data_new[20:])

source_data=data_new
real_list = source_data[29:]








plt.plot(np.linspace(0,len(result_list)-1,len(result_list)),result_list,'g')
plt.plot(np.linspace(0,len(real_list)-1,len(real_list)),real_list,'r')
plt.show()

