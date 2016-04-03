import os
import sys
import libfann
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from emd import *
import datetime




class nero_func():

    def __init__(self, train_data_dir, net_file):
        
        self.train_data_dir = train_data_dir
        self.net_file = net_file
    
        
    def train_nn(self, data_file, net_file):
    
        connection_rate = 1
        learning_rate = 0.7
        num_input = 2
        num_hidden = 4
        num_output = 1
    
        desired_error = 0.0001
        max_iterations = 100000
        iterations_between_reports = 100000
        ann = libfann.neural_net()
        ann.create_standard_array((25,6,1))
        ann.set_learning_rate(0.3)
        ann.set_activation_function_hidden(libfann.SIGMOID)
        ann.set_activation_function_output(libfann.LINEAR)
    
        ann.train_on_file(data_file, max_iterations, iterations_between_reports, desired_error)
    
        ann.save(net_file)
    
    
    
    def test_each(self, net_file, test_data):
    
        ann = libfann.neural_net()
        ann.create_from_file(net_file)
    
        return ann.run(test_data)[0]



    def draw_fig(self,datafile,start,save=0):
        
        data = self.close_price

        fp = open("imf3",'r')
        lines = fp.readlines()
        fp.close()
        
        imf = []
        for eachline in lines:
            imf.append(float(eachline.split("\n")[0]))

        fp = open("residual",'r')
        lines = fp.readlines()
        fp.close()

        residual = []
        for eachline in lines:
            residual.append(float(eachline.split("\n")[0]))

        

        imf_buy_value = [imf[i] for i in self.buy_x_index]
        imf_sell_value = [imf[i] for i in self.sell_x_index]
        plt.figure(1)
        plt.subplot(211)#.axis([900,1100,8,15])      
        plt.plot([i for i in range(len(data))],data,'b',self.buy_x_index,self.buy_y_index,'r*',self.sell_x_index,self.sell_y_index,'g*')             
        plt.subplot(212).axis([1170,1400,-2,2])
        plt.plot([i for i in range(len(imf))],imf,'b',self.buy_x_index,imf_buy_value,'r*',self.sell_x_index,imf_sell_value,'g*')   
        figname = "fig1_"+datafile         
        if save==1:
            savefig(figname)
        plt.figure(2)
        plt.subplot(211)
        plt.plot([i for i in range(len(self.total_asset_list))],self.total_asset_list,'b')
        plt.subplot(212)
        plt.plot([i for i in range(len(imf[start:]))],imf[start:],'b',[i for i in range(len(self.predict_list))],self.predict_list,'r')
        figname = "fig2_"+datafile
        if save==1:         
            savefig(figname)
        else:
            plt.show() 


if __name__ == "__main__":

