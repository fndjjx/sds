import libfann
import matplotlib.pyplot as plt
import os



class neuroprediction():

    def __init__(self,train_file,net_file,num_input,num_hidden1,num_hidden2,num_output,desired_error = 0.0001,max_iterations = 500000,learning_rate = 0.5):

        self.learning_rate = learning_rate
        self.num_input = num_input
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.num_output = num_output
        self.desired_error = desired_error
        self.max_iterations = max_iterations
        self.train_file = train_file
        self.net_file = net_file

    def train_nn(self):

        connection_rate = 1
        iterations_between_reports = 100000

        ann = libfann.neural_net()
        if self.num_hidden2 == 0:
            ann.create_standard_array((self.num_input,self.num_hidden1,self.num_output))
        elif self.num_hidden2 > 0:
            ann.create_standard_array((self.num_input,self.num_hidden1,self.num_hidden2,self.num_output))
        ann.set_learning_rate(self.learning_rate)

#        ann.set_activation_function_hidden(libfann.SIGMOID)
#        ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC)
        ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC_STEPWISE)
#        ann.set_activation_function_output(libfann.LINEAR)
#        ann.set_activation_function_output(libfann.ELLIOT_SYMMETRIC)
        ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)

        ann.train_on_file(self.train_file, self.max_iterations, iterations_between_reports, self.desired_error)

        ann.save(self.net_file)



    def test_each(self,test_data):

        ann = libfann.neural_net()
        ann.create_from_file(self.net_file)

        return ann.run(test_data)

def test_file(net_file,test_data):

    ann = libfann.neural_net()
    ann.create_from_file(net_file)

    return ann.run(test_data)



if __name__ == "__main__":

    f = os.popen("tail sin_data/data_431 -n 2|head -n 1")
    line = f.readline()
    f.close()
    test_data = line.split(" ")
    test_data.pop()
    print "test_data%s"%test_data
    test_data_float = [float(each) for each in test_data]
    
    result = []
    fin_result = []
    sum = 0
    for i in range(20):
        neuro_prediction = neuroprediction("sin_data/data_430","net_file",50,15,0,10)
        neuro_prediction.train_nn()
        result.append( neuro_prediction.test_each(test_data_float))
        print result[-1]
    for i in range(10):
        for j in range(20):
            sum += result[j][i]
        fin_result.append(sum)
        sum = 0
    fin_result1 = [fin_result[i]/20 for i in range(10)] 
    print fin_result1 


    data = []
    fp = open("imf_raw2")
    lines = fp.readlines()
    fp.close()
 
    for eachline in lines:
        eachline.strip("\n")
        data.append(float(eachline))

    data1 = []
    fp = open("imf2")
    lines = fp.readlines()
    fp.close()

    for eachline in lines:
        eachline.strip("\n")
        data1.append(float(eachline))

 
    data_new = data1[:-10]+fin_result1
    print len(data_new)    
    plt.plot([i for i in range(len(data[300:600]))],data[300:600],'b')
    plt.plot([i for i in range(len(data1[300:600]))],data1[300:600],'g')
    plt.plot([i for i in range(len(data_new[300:600]))],data_new[300:600],'*')
    plt.show()
    

