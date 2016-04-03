import os
import sys
import libfann
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from emd import *
import datetime
from scipy.signal import argrelextrema
import numpy as np
import time




class process_util():

    def __init__(self,open_price,close_price,macd,date,vol):
        
        self.money = 10000
        self.share = 0
        self.sell_x_index = []
        self.sell_y_index = []
        self.buy_x_index = []
        self.buy_y_index = []
        self.open_price = open_price
        self.close_price = self.format_data(close_price)
        
        self.total_asset_list = []
        self.total_asset = 0
        self.buy_price = 0
        self.predict_list = []

        self.period_decision_buy_point = []
        self.period_decision_sell_point = []

        self.max_min_period_delta = []
        self.min_max_period_delta = []
        self.last_extrem = ""
        self.current_extrem = ""

        self.last_period_decision = 0
        self.last_last_period_decision = 0

        self.last_rise_flag = 0
        self.current_rise_flag = 0
        self.last_buy_flag = 0

        self.last_now_max = 0
        self.last_now_min = 0
        self.before_last_max = 0
        self.before_last_min = 0
        self.max_keep_flag = 0    
        self.macd = macd
        self.last_bottom = 0
        self.date = date
        self.buy_macd = 0
        self.down_macd_counter = 0 
        self.imf_pre = []
        self.vol = vol
        self.predict_price = []


    def sell(self, share, price):
        return float(share)*float(price)

    def buy(self, money, price):
        return float(money)/float(price) 
    
    def generate_all(self, data, flag, input_number):
        print "length data %s"%len(data)
        print "length flag %s"%len(flag)
        fp = open("train_data_all", 'w')
        for i in range(len(data)-input_number+1):
           for j in range(i,input_number+i):
               fp.writelines(str(data[j]))
               fp.writelines(" ")
           fp.writelines("\n")
           fp.writelines(str(flag[i]))
           fp.writelines("\n")
        fp.close()


    def generate_all_price(self,data,input_num):
        expect = 0
        fp = open("train_data_all",'w')
        for i in range(len(data)-input_num+1): 
            for j in range(i,input_num+i):
                fp.writelines(str(data[j]))
                fp.writelines(" ")
                expect = j+1
            fp.writelines("\n")
            if expect<len(data):
                fp.writelines(str(data[expect]))
                fp.writelines("\n")
        fp.writelines("0")
        fp.close()
    
    def generate_divide(self, data_raw, eachfile_group_num, input_num):
        fp = open("train_data_all",'r')
        data = fp.readlines()
        fp.close()
        total_group_num = len(data_raw)-input_num+1
        total_file_num = total_group_num - eachfile_group_num-1 
        for i in range(total_file_num):
            fp = open("../../train_data/data_%s"%i,'w')
            fp.write("%s %s 1\n"%(eachfile_group_num,input_num))
            for j in range(i*2,(eachfile_group_num+i)*2):
                fp.writelines(str(data[j]))
            fp.close()


    def file_count(self, train_data_dir):
        f = os.popen("ls %s|wc -l"%train_data_dir) 
        file_num = f.readline()
        f.close()
        return file_num
        


    def sell_fee(self, money):
        
        return money*0.0002+money*0.001

    def buy_fee(self, share, price):
      
        return share*price*0.0002



    def generate_emd_data(self,data,start):
        
        #for i in range(start,len(data)-2):
        for i in range(start,len(data)+1):
            fp = open("../../emd_data/emd_%s"%i,'w')
            for j in range(i):
                fp.writelines(str(data[j]))
                fp.writelines("\n")
   #         fp.writelines(str(self.open_price[i]))
   #         fp.writelines("\n")
   #         fp.writelines(str(self.close_price[i-1]))
   #         fp.writelines("\n")
            fp.close()

    def generate_emd_data_fix(self,data,start,period):

        for i in range(start,len(data)+1):
            fp = open("../../emd_data_fix/emd_%s"%i,'w')
            for j in range(i-period,i):
                fp.writelines(str(data[j]))
                fp.writelines("\n")
            fp.close()

    def train_nn(self, data_file,net_file):

        connection_rate = 1
        learning_rate = 0.7
        num_input = 2
        num_hidden = 4
        num_output = 1

        desired_error = 0.0001
        max_iterations = 100000
        iterations_between_reports = 100000
        ann = libfann.neural_net()
        ann.create_standard_array((25,15,1))
        ann.set_learning_rate(0.2)
        #ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC_STEPWISE)
        ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC)
        #ann.set_activation_function_hidden(libfann.SIGMOID)
        ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC)

        ann.train_on_file(data_file, max_iterations, iterations_between_reports, desired_error)

        ann.save(net_file)



    def test_each(self, net_file,test_data):

        ann = libfann.neural_net()
        ann.create_from_file(net_file)

        return ann.run(test_data)[0]


    def predict_price_func(self, train_data, net_file, input_num):


        file = train_data
        f = os.popen("tail %s -n 2|head -n 1"%file)
        line = f.readline()
        f.close()
        test_data = line.split(" ")
        test_data.pop()
        print "test_data%s"%test_data
        test_data_float = [float(each) for each in test_data]

        fp = open("train_data_all",'r')
        lines = fp.readlines()
        fp.close()
        
        lines.pop()
        lines.pop()
        
        f = os.popen("wc -l train_data_all")
        line = f.readline()
        f.close()
        
        pair_num = (int(line.split(" ")[0])-1)/2
        
        for i in range(pair_num-50):
            lines.pop(1)
            lines.pop(1)

        newline = "50 25 1 \n"
        os.popen("rm train_data_all_new") 
        fp = open("train_data_all_new",'w')
        fp.write(newline)
        for i in range(len(lines)):
            fp.writelines(str(lines[i]))
        time.sleep(3)
        fp.close()


         
        result_list = []
        for j in range(5):
            self.train_nn("train_data_all_new",net_file)
            result_list.append(self.test_each(net_file,test_data_float))
        result = sum(result_list)/len(result_list)
        print "result%s"%result
        return result





    def span_estimate(self, data):
        return (round(np.mean(data)-np.std(data),0),round(np.mean(data)+np.std(data),0))

    def format_data(self, data):
        return [ ((float(data[i])-min(data))/(max(data)-float(min(data)))) for i in range(len(data))]

    def run2(self,emd_data, datafile,current_index,date):
        starttime = datetime.datetime.now()
        self.run_predict_price_macd(current_index,date,datafile)
        endtime = datetime.datetime.now()
        print "run time"
        print (endtime - starttime).seconds



    def run(self,emd_data_raw, datafile,current_index,date):
        starttime = datetime.datetime.now()
        self.clear_emd_data()
        emd_data = self.cubicSmooth5(emd_data_raw) 
        emd_data = self.format_data(emd_data_raw) 
        my_emd = one_dimension_emd(emd_data_raw)
        (imf, residual) = my_emd.emd()
        for i in range(len(imf)):
            fp = open("imf%s"%i,'w')
            for j in range(len(imf[i])):
                fp.writelines(str(imf[i][j]))
                fp.writelines("\n")
            fp.close()
        print "imf num %s"%len(imf)
        fp = open("residual",'w')
        for i in range(len(residual)):
            fp.writelines(str(residual[i]))
            fp.writelines("\n")
        fp.close()

        fp = open("imf0")
        lines = fp.readlines()
        fp.close()

        #print lines

        imf_raw = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            imf_raw.append(float(eachline.split("\n")[0]))
   #     for i in range(2):
   #         imf.pop()
        imf_without = [emd_data[i]-imf_raw[i] for i in range(len(emd_data))]

        fp = open("imf1")
        lines = fp.readlines()
        fp.close()

        #print lines

        imf_raw1 = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            imf_raw1.append(float(eachline.split("\n")[0]))
        imf_without2 = [imf_without[i]-imf_raw1[i] for i in range(len(emd_data))]


        fp = open("imf2")
        lines = fp.readlines()
        fp.close()
        imf_raw2 = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            imf_raw2.append(float(eachline.split("\n")[0]))
        imf_without3 = [imf_without2[i]-imf_raw2[i] for i in range(len(emd_data))]
        fp = open("imf3")
        lines = fp.readlines()
        fp.close()
        imf_raw3 = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            imf_raw3.append(float(eachline.split("\n")[0]))
        imf_without4 = [imf_without3[i]-imf_raw3[i] for i in range(len(emd_data))]


        fp = open("residual")
        lines = fp.readlines()
        fp.close()
        residual_raw = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            residual_raw.append(float(eachline.split("\n")[0]))
        imf_without6 = [imf_without2[i]-residual_raw[i] for i in range(len(emd_data))]



        imf.append(residual)
        predict_imf = []
        for i in range(len(imf)):
        #    self.clear_nn_train_data()
            self.clear_train_data_all()
            self.generate_all_price(imf[i],25)
        #    self.generate_divide(imf[i],20,25)
            result = self.predict_price_func("train_data_all","nn_file",25)
            predict_imf.append(result)

        self.predict_price.append(sum(predict_imf))
        print "predict %s"%self.predict_price[-1]
        print "real %s"%(self.close_price[current_index+1])
        endtime = datetime.datetime.now()
        print "run time"
        print (endtime - starttime).seconds

    def clear_train_data_all(self):
        os.popen("rm train_data_all*")

                 
        
    def clear_emd_data(self):
        os.popen("rm imf*")    

    def clear_nn_train_data(self):
        os.popen("rm -r ../../train_data")
        os.popen("mkdir ../../train_data")

    def top_period(self, data):
        top_period_list = []
        top_period_value = []
        data_set = set(data)
        data_pair = [(i,data.count(i)) for i in data_set]
        data_pair.sort(key = lambda k:k[1],reverse=True)
        for i in range(2):
            top_period_list.append(data_pair[i][0])
            top_period_value.append(data_pair[i][1])
        result_list = range(min(top_period_list),max(top_period_list)+1)
        portion = float(sum(top_period_value))/len(data)
        print "top period list %s"%result_list
        print "top period value %s"%portion
        return (result_list,portion)


    def ma(self, data, period, start, current_index):
        sum_period = 0
        for i in range(period):
            sum_period += data[current_index + start - i]
        return float(sum_period)/period    
    
    def cubicSmooth5(self,data_input):
        N = len(data_input)
        data_output = [0 for i in range(N)]
        if N<5:
            data_output = data_input
        else:
            data_output[0] = (69.0 * data_input[0] + 4.0 * data_input[1] - 6.0 * data_input[2] + 4.0 * data_input[3] - data_input[4]) / 70.0;
            data_output[1] = (2.0 * data_input[0] + 27.0 * data_input[1] + 12.0 * data_input[2] - 8.0 * data_input[3] + 2.0 * data_input[4]) / 35.0;
            for i in range(2,N-2):
                data_output[i] = (-3.0 * (data_input[i - 2] + data_input[i + 2])+ 12.0 * (data_input[i - 1] + data_input[i + 1]) + 17.0 * data_input[i] ) / 35.0;
            data_output[N - 2] = (2.0 * data_input[N - 5] - 8.0 * data_input[N - 4] + 12.0 * data_input[N - 3] + 27.0 * data_input[N - 2] + 2.0 * data_input[N - 1]) / 35.0;
            data_output[N - 1] = (- data_input[N - 5] + 4.0 * data_input[N - 4] - 6.0 * data_input[N - 3] + 4.0 * data_input[N - 2] + 69.0 * data_input[N - 1]) / 70.0;
        return data_output


    def success_portion(self):
        data = self.total_asset_list
        success_flag = 0
        fail_flag = 0
        for i in range(len(data)-1):
            if data[i+1]>data[i]:
                success_flag += 1
            elif data[i+1]<data[i]:
                fail_flag += 1
        return float(success_flag)/(success_flag+fail_flag)

    def peilv(self):
        data = self.total_asset_list
        delta = []
        delta_new = []
        positive = 0
        negative = 0
        p_flag = 0
        n_flag = 0
        for i in range(len(data)-1):
            delta.append(data[i+1]-data[i])
        for j in range(len(delta)):
            if delta[j] != 0:
                delta_new.append(delta[j])
        if len(delta_new)>0:
            delta_new.remove(delta_new[0])
            print "asset delta %s"%delta_new
            for p in range(len(delta_new)):
                if delta_new[p] > 0:
                    positive += delta_new[p]
                    p_flag += 1
                if delta_new[p] < 0:
                    negative += delta_new[p]
                    n_flag += 1
            if p_flag != 0 and n_flag != 0:
                p_mean = float(positive)/p_flag
                n_mean = float(negative)/n_flag
                print "portion corr %s"%(float(p_flag)/(p_flag+n_flag))
                print "p_mean %s"%p_mean
                print "n_mean %s"%n_mean
         
        
    def hold_time(self):
        buy_time = self.buy_x_index
        sell_time = self.sell_x_index
        if len(buy_time)>len(sell_time):
            hold_time_list = [sell_time[i]-buy_time[i] for i in range(len(buy_time)-1)]
        elif len(buy_time) == len(sell_time):
            hold_time_list = [sell_time[i]-buy_time[i] for i in range(len(buy_time))]
        print "hold time %s"%sum(hold_time_list)


    def rsi(self,data,period,start):
        down_num = 0
        down_sum = 0
        up_num = 0
        up_sum = 0
        down_mean = 0
        up_mean = 0
        for i in range(start-period,start):
            if data[i]>data[i+1]:
                down_num += 1
                down_sum += data[i]-data[i+1]
            if data[i]<data[i+1]:
                up_num += 1
                up_sum += data[i+1]-data[i] 
        if down_num != 0 and up_num != 0 :
            down_mean = float(down_sum)/down_num
            up_mean = float(up_sum)/up_num
            rs = up_mean/down_mean
        elif down_num == 0:
            rs = 100
        elif up_num == 0:
            rs = 0
            
        return (float(rs)/(rs+1))*100

    def ma_line(self,n,index):
        ma = []
        for i in range(index+1):
            sum = 0
            if i == 0 or i == 1:
                ma.append(self.close_price[i])
            else:
                for j in range(i-n+1,i+1):
                    sum += self.close_price[j]
                ma.append(sum/n)
        return ma
        
        

    def draw_fig(self,datafile,start,save=0):
        
        data = self.close_price

        fp = open("imf0")
        lines = fp.readlines()
        fp.close()

        #print lines

        imf_raw = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            imf_raw.append(float(eachline.split("\n")[0]))
   #     for i in range(2):
   #         imf.pop()
        imf_without = [emd_data[i]-imf_raw[i] for i in range(len(emd_data))]

        fp = open("imf1")
        lines = fp.readlines()
        fp.close()
        
        #print lines
        
        imf_raw1 = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            imf_raw1.append(float(eachline.split("\n")[0]))
        imf_without2 = [imf_without[i]-imf_raw1[i] for i in range(len(emd_data))]

        fp = open("imf2")
        lines = fp.readlines()
        fp.close()
        imf_raw2 = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            imf_raw2.append(float(eachline.split("\n")[0]))
        imf_without3 = [imf_without2[i]-imf_raw2[i] for i in range(len(emd_data))]


        fp = open("imf3")
        lines = fp.readlines()
        fp.close()
        imf_raw3 = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            imf_raw3.append(float(eachline.split("\n")[0]))
        imf_without4 = [imf_without3[i]-imf_raw3[i] for i in range(len(emd_data))]

        fp = open("residual")
        lines = fp.readlines()
        fp.close()
        residual_raw = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            residual_raw.append(float(eachline.split("\n")[0]))
        imf_without6 = [imf_without2[i]-residual_raw[i] for i in range(len(emd_data))]

        imf = self.cubicSmooth5(imf_raw2)

        fp = open("residual",'r')
        lines = fp.readlines()
        fp.close()

        residual = []
        for eachline in lines:
            residual.append(float(eachline.split("\n")[0]))

        plt.figure(1)
        plt.subplot(211).axis([start,len(data),min(data[start:]),max(data[start:])])      
        plt.plot([i for i in range(len(data))],data,'b')             
        plt.plot([i for i in range(len(self.predict_price))],self.predict_price,'r')   
        plt.subplot(212)
        plt.plot([i for i in range(len(self.predict_price))],self.predict_price,'b')   
        figname = "fig1_"+datafile         
        plt.show() 


if __name__ == "__main__":

    datafile = sys.argv[1]
    begin = int(sys.argv[2])
    fp = open(datafile)
    lines = fp.readlines()
    fp.close()

    close_price = []
    open_price = []
    date = [] 
    macd = []
    vol = []
    for eachline in lines:
        eachline.strip()
        close_price.append(float(eachline.split("\t")[4]))
        macd.append(float(eachline.split("\t")[6]))
        open_price.append(float(eachline.split("\t")[1]))
        date.append(eachline.split("\t")[0])
        vol.append(float(eachline.split("\t")[5]))
    
   
    process = process_util(open_price,close_price,macd,date,vol)
    os.popen("rm -r ../../emd_data_fix")
    os.popen("mkdir ../../emd_data_fix")
    #process.generate_emd_data(close_price,begin)
    print "begin %s"%begin
    process.generate_emd_data_fix(process.format_data(close_price),begin,300)
    for i in range(begin,begin+int(process.file_count("../../emd_data_fix"))-1):
        emd_data = []
        print "emd file %s"%i
        fp = open("../../emd_data_fix/emd_%s"%i,'r')
        lines = fp.readlines()
        fp.close()
#        
        for eachline in lines:
            eachline.strip("\n")
            emd_data.append(float(eachline))
#    #    print "emd_data %s"%emd_data
        process.run(emd_data,datafile,i-1,date[-1])
#        process.run2(emd_data,datafile,i-1,date[-1])
    process.peilv()
    process.hold_time()
    process.draw_fig(datafile,begin)

    

            
