import os
import sys
import libfann
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from emd import *
import datetime
from scipy.signal import argrelextrema




class process_util():

    def __init__(self,open_price,close_price):
        
        self.money = 10000
        self.share = 0
        self.sell_x_index = []
        self.sell_y_index = []
        self.buy_x_index = []
        self.buy_y_index = []
        self.open_price = open_price
        self.close_price = close_price
        
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
        ann.create_standard_array((3,25,6,1))
        ann.set_learning_rate(0.3)
        ann.set_activation_function_hidden(libfann.SIGMOID)
        ann.set_activation_function_output(libfann.LINEAR)
    
        ann.train_on_file(data_file, max_iterations, iterations_between_reports, desired_error)
    
        ann.save(net_file)
    
    
    
    def test_each(self, net_file,test_data):
    
        ann = libfann.neural_net()
        ann.create_from_file(net_file)
    
        return ann.run(test_data)[0]


    def sell_fee(self, money):
        
        return money*0.0002+money*0.001

    def buy_fee(self, share, price):
      
        return share*price*0.0002

    def format_data(data):
        if isinstance(data,list):
            return [ (float(data[i])-min(data))/(max(data)-float(min(data))) for i in range(len(data))]
            
   
          
    def run_predict_price(self, data, train_data_dir, net_file, test_group_num, input_num, imf, open_price,predict_num,datafile):

        buy_price = 0
        decision = 0
        total_asset = 0
        loops = 0
        file_num = self.file_count(train_data_dir)

        for i in range(int(file_num)-predict_num,int(file_num)-2):
            print "loop %s"%i
            file_current = "%s/data_%s"%(train_data_dir,i)
            file_next = "%s/data_%s"%(train_data_dir,i+1)
            f = os.popen("tail %s -n 2|head -n 1"%file_next)
            line = f.readline()
            f.close()
            test_data = line.split(" ")
            test_data.pop()
            print "test_data%s"%test_data
            test_data_float = [float(each) for each in test_data]

            result_list = []
            for j in range(5):
                self.train_nn(file_current,net_file)
                result_list.append(self.test_each(net_file,test_data_float))
            result = sum(result_list)/len(result_list)
            print "result%s"%result


            current_index = i+input_num+test_group_num-2+1
            current_price = data[current_index]
            trade_price = open_price[current_index+1]

            print "current close price %s"%current_price
            print "trade price%s"%trade_price
            imf_ma5_current = (imf[current_index-4]+imf[current_index-3]+imf[current_index-2]+imf[current_index-1]+imf[current_index])/5.0
            imf_ma5_next = (imf[current_index-3]+imf[current_index-2]+imf[current_index-1]+imf[current_index]+result)/5.0
            power_current = imf[current_index] - imf_ma5_current
            power_next = result - imf_ma5_next



            if power_current>0 and power_next<0:
                decision = 1
            if power_current<0 and power_next>0:
                decision = -1
            
            print "ma5 current%s"%imf_ma5_current
            print "ma5 next%s"%imf_ma5_next
        
            print "decision%s"%decision 

            if decision == 1 and self.share>0 and (float(current_price)-float(buy_price))/float(buy_price)>0.02:
                print "sell"
                self.money = self.sell(self.share,trade_price)
                self.share = 0
                buy_price = 0
                total_asset = self.money - self.sell_fee(self.money)
                self.sell_x_index.append(current_index)
                self.sell_y_index.append(current_price)
            if decision !=1 and decision != -1 and float(buy_price)!=0 and (float(buy_price)-float(current_price))/float(buy_price)>0.02 and self.share>0:
                print "sell"
                self.money = self.sell(self.share,trade_price)
                self.share = 0
                buy_price = 0
                total_asset = self.money - self.sell_fee(self.money)
                self.sell_x_index.append(current_index)
                self.sell_y_index.append(current_price)
            if decision == -1  and self.money>0:
                print "buy"
                self.share = self.buy(self.money,trade_price)
                self.money = 0
                buy_price = trade_price
                total_asset = self.share*trade_price - self.buy_fee(self.share,trade_price)
                self.buy_x_index.append(current_index)
                self.buy_y_index.append(current_price)
            decision = 0
            self.total_asset_list.append(total_asset)
            print "total asset is %s"%(total_asset)
            print "money is %s"%(self.money)
            print "share is %s"%(self.share)
            loops = i
        lines = []
        if os.path.exists("result"):
            fp = open("result",'r')
            lines = fp.readlines()
            fp.close()
            fp = open("result",'w')
            lines.append("\n")
            lines.append(datafile)
            lines.append(str(self.total_asset_list[-1]))
            lines.append("\n")
            lines.append(str(loops))
            lines.append("\n")
            fp.writelines(lines)
        else:
            fp = open("result",'w')
            lines.append(datafile)
            lines.append("\n")
            lines.append(str(self.total_asset_list[-1]))
            lines.append("\n")
            lines.append(str(loops))
            lines.append("\n")
            fp.writelines(lines)
        fp.close()



    def run_predict_price_real(self, train_data_dir, net_file, test_group_num, input_num, imf):
        

        decision = 0
        file_num = int(self.file_count(train_data_dir))
        print "file num %s"%file_num

        file_current = "%s/data_%s"%(train_data_dir,file_num-2)
        file_next = "%s/data_%s"%(train_data_dir,file_num-1)
        f = os.popen("tail %s -n 2|head -n 1"%file_next)
        line = f.readline()
        f.close()
        test_data = line.split(" ")
        test_data.pop()
        print "test_data%s"%test_data
        test_data_float = [float(each) for each in test_data]

        result_list = []
        for j in range(5):
            self.train_nn(file_current,net_file)
            result_list.append(self.test_each(net_file,test_data_float))
        result_list.remove(max(result_list))
        result_list.remove(min(result_list))
        result = sum(result_list)/len(result_list)
        print "result%s"%result
        self.predict_list.append(result)


        current_index = file_num-1+input_num+test_group_num-2
        print "current imf value%s"%imf[current_index]

        imf_ma5_current = (imf[current_index-4]+imf[current_index-3]+imf[current_index-2]+imf[current_index-1]+imf[current_index])/5.0
        imf_ma5_next = (imf[current_index-3]+imf[current_index-2]+imf[current_index-1]+imf[current_index]+result)/5.0
        power_current = imf[current_index] - imf_ma5_current
        power_next = result - imf_ma5_next

  #      imf_ma3_current = (imf[current_index-2]+imf[current_index-1]+imf[current_index])/3.0
  #      imf_ma3_next = (imf[current_index-1]+imf[current_index]+result)/3.0
  #      power_current = imf[current_index] - imf_ma3_current
  #      power_next = result - imf_ma3_next

                

        if power_current>0 and power_next<0:
            decision = 1
            print "sell"
        if power_current<0 and power_next>0:
            decision = -1
            print "buy"
        print "decision %s"%decision 

  #      if result>imf[current_index] and imf[current_index]<imf[current_index-1]:
  #          decision = -1
  #      if result<imf[current_index] and imf[current_index]>imf[current_index-1]:
  #          decision = 1

        current_price = self.close_price[current_index]
        trade_price = self.open_price[current_index+1]
        print "current price%s"%current_price
        print "trade price%s"%trade_price
        print "buy price%s"%self.buy_price

        if decision == 1 and self.share>0 and (float(current_price)-float(self.buy_price))/float(self.buy_price)>0.02:
            print "sell"
            self.money = self.sell(self.share,trade_price)
            self.share = 0
            self.buy_price = 0
            self.total_asset = self.money - self.sell_fee(self.money)
            self.sell_x_index.append(current_index)
            self.sell_y_index.append(current_price)
        if decision == 0 and float(self.buy_price)!=0 and (float(self.buy_price)-float(current_price))/float(self.buy_price)>0.02 and self.share>0:
            print "sell"
            self.money = self.sell(self.share,trade_price)
            self.share = 0
            self.buy_price = 0
            self.total_asset = self.money - self.sell_fee(self.money)
            self.sell_x_index.append(current_index)
            self.sell_y_index.append(current_price)
        if decision == -1  and self.money>0:
            print "buy"
            self.share = self.buy(self.money,trade_price)
            self.money = 0
            self.buy_price = trade_price
            self.total_asset = self.share*trade_price - self.buy_fee(self.share,trade_price)
            self.buy_x_index.append(current_index)
            self.buy_y_index.append(current_price)
        self.total_asset_list.append(self.total_asset)
        print "total asset is %s"%(self.total_asset)
        print "money is %s"%(self.money)
        print "share is %s"%(self.share)


    def generate_emd_data(self,data,start):
        
        for i in range(start,len(data)-2):
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
                print j
                fp.writelines(str(data[j]))
                fp.writelines("\n")
            fp.close()

    def run_predict_price_period(self,imf,current_price_index,residual):

        decision = 0
        power_decision = 0
        imf_power_decision = 0
        period_decision = 0
        extrem_decision = 0
        current_rise_flag = 0

        data = np.array(imf)
        imf_max_index = argrelextrema(data,np.greater)[0]
        imf_min_index = argrelextrema(data,np.less)[0]

        max_period_delta = [imf_max_index[i+1]-imf_max_index[i] for i in range(len(imf_max_index)-1)]
        min_period_delta = [imf_min_index[i+1]-imf_min_index[i] for i in range(len(imf_min_index)-1)]

        if imf_max_index[0]>imf_min_index[0]:
            self.max_min_period_delta = [imf_max_index[i]-imf_min_index[i] for i in range(min(len(imf_max_index),len(imf_min_index)))]
            self.min_max_period_delta = [imf_min_index[i+1]-imf_max_index[i] for i in range(min(len(imf_max_index),len(imf_min_index))-1)]
        else:
            self.max_min_period_delta = [imf_max_index[i+1]-imf_min_index[i] for i in range(min(len(imf_max_index),len(imf_min_index))-1)]
            self.min_max_period_delta = [imf_min_index[i]-imf_max_index[i] for i in range(min(len(imf_max_index),len(imf_min_index)))]

        max_period_delta.remove(max(max_period_delta))
        max_period_delta.remove(min(max_period_delta))
        min_period_delta.remove(max(min_period_delta))
        min_period_delta.remove(min(min_period_delta))
        #print "mean max%s"%max_period_delta
        #print "mean min%s"%min_period_delta
        #print "mean max min%s"%max_min_period_delta
        mean_max_delta = float(sum(max_period_delta))/len(max_period_delta)
        mean_min_delta = float(sum(min_period_delta))/len(min_period_delta)



        max_min_period_delta = self.max_min_period_delta[-5:]
        max_min_period_delta.remove(max(max_min_period_delta))
        max_min_period_delta.remove(min(max_min_period_delta))
        
        min_max_period_delta = self.min_max_period_delta[-5:]
        min_max_period_delta.remove(max(min_max_period_delta))
        min_max_period_delta.remove(min(min_max_period_delta))
        mean_max_min = float(sum(max_min_period_delta))/len(max_min_period_delta)
        mean_min_max = float(sum(min_max_period_delta))/len(min_max_period_delta)


        #print "mean max delta%s"%mean_max_delta
        #print "mean min delta%s"%mean_min_delta
        #print "mean max min%s"%mean_max_min

        now_position_last_min_delta = len(imf)-1-imf_min_index[-1]
        #print "now min delta%s"%now_position_last_min_delta
        now_position_last_max_delta = len(imf)-1-imf_max_index[-1]
        #print "now max delta%s"%now_position_last_max_delta
         
        (max_min_low,max_min_high) = self.span_estimate(max_min_period_delta)
        (min_max_low,min_max_high) = self.span_estimate(min_max_period_delta)
        


#        if abs(now_position_last_min_delta-mean_min_delta)/mean_min_delta < 0.3 and abs(now_position_last_max_delta-mean_max_min)/mean_max_min < 0.3:
        if imf_min_index[-1]<imf_max_index[-1]:
            print "now max delta%s"%now_position_last_max_delta
            print "min max%s"%min_max_period_delta
            print "mean min max%s"%mean_min_max
            self.current_extrem = "max"
            print "low high %s %s"%(min_max_low,min_max_high)
        #    if  abs(now_position_last_max_delta-mean_min_max)/mean_min_max < 0.2:
        #    if now_position_last_max_delta+1 in self.top_period(min_max_period_delta)[0]:
            if now_position_last_max_delta+1 >= min_max_low and now_position_last_max_delta+1 <= min_max_high:
                period_decision = -1
                print "period decision is -1"
                self.period_decision_buy_point.append(current_price_index)
#        if abs(now_position_last_max_delta-mean_max_delta)/mean_max_delta < 0.3 and abs(now_position_last_min_delta-mean_max_min)/mean_max_min < 0.3:
        elif imf_min_index[-1]>imf_max_index[-1]:
            print "now min delta%s"%now_position_last_min_delta
            print "max min%s"%max_min_period_delta
            print "mean max min%s"%mean_max_min
            self.current_extrem = "min"
            print "low high %s %s"%(max_min_low,max_min_high)
        #    if abs(now_position_last_min_delta-mean_max_min)/mean_max_min < 0.2:
        #    if now_position_last_min_delta+1 in self.top_period(max_min_period_delta)[0]:
            if now_position_last_min_delta+1 >= max_min_low and now_position_last_min_delta+1 <= max_min_high:  
                period_decision = 1
                print "period decision is 1"
                self.period_decision_sell_point.append(current_price_index)

        print "last extre%s"%self.last_extrem
        print "current extre %s"%self.current_extrem
        if self.last_extrem == "max" and self.current_extrem == "min":
            extrem_decision = -1
        elif self.last_extrem == "min" and self.current_extrem == "max":
            extrem_decision = 1
        elif self.last_extrem == "min" and self.current_extrem == "min":
            extrem_decision = 0.5
        elif self.last_extrem == "max" and self.current_extrem == "max":
            extrem_decision = -0.5
        print "extre decision %s"%extrem_decision        
        self.last_extrem = self.current_extrem
        current_price_index = current_price_index#-1
        current_price = self.close_price[current_price_index]
        trade_price = self.open_price[current_price_index+1]
        print "current price%s"%current_price
        print "trade price%s"%trade_price
        print "buy price%s"%self.buy_price

        current_index = len(imf)-1
        imf_ma5_before = (imf[current_index-5]+imf[current_index-4]+imf[current_index-3]+imf[current_index-2]+imf[current_index-1])/5.0
        imf_ma5_current = (imf[current_index-4]+imf[current_index-3]+imf[current_index-2]+imf[current_index-1]+imf[current_index])/5.0
        imf_power_before = imf[current_index-1] - imf_ma5_before
        imf_power_current = imf[current_index] - imf_ma5_current
        if imf_power_before>0 and imf_power_current<0:
            imf_power_decision = 1
        if imf_power_before<0 and imf_power_current>0:
            imf_power_decision = -1


#        imf_ma3_before = (imf[current_index-3]+imf[current_index-2]+imf[current_index-1])/3.0
#        print "ma before%s"%imf_ma3_before
#        imf_ma3_current = (imf[current_index-2]+imf[current_index-1]+imf[current_index])/3.0
#        print "ma current%s"%imf_ma3_current
#        power_before = imf[current_index-1] - imf_ma3_before
#        print "power before%s"%power_before
#        power_current = imf[current_index] - imf_ma3_current
#        print "power current%s"%power_current

        price_ma5_before2 = (self.close_price[current_price_index-6]+self.close_price[current_price_index-5]+self.close_price[current_price_index-4]+self.close_price[current_price_index-3]+self.close_price[current_price_index-2])/5.0
        price_ma5_before = (self.close_price[current_price_index-5]+self.close_price[current_price_index-4]+self.close_price[current_price_index-3]+self.close_price[current_price_index-2]+self.close_price[current_price_index-1])/5.0
        price_ma5_current = (self.close_price[current_price_index-4]+self.close_price[current_price_index-3]+self.close_price[current_price_index-2]+self.close_price[current_price_index-1]+self.close_price[current_price_index])/5.0
        power_before2 = self.close_price[current_price_index-2] - price_ma5_before2
        power_before = self.close_price[current_price_index-1] - price_ma5_before
        power_current = self.close_price[current_price_index] - price_ma5_current
#
##        price_ma3_before = (self.close_price[current_price_index-3]+self.close_price[current_price_index-2]+self.close_price[current_price_index-1])/5.0
##        price_ma3_current = (self.close_price[current_price_index-2]+self.close_price[current_price_index-1]+self.close_price[current_price_index])/5.0
##        power_before = self.close_price[current_price_index-1] - price_ma3_before
##        power_current = self.close_price[current_price_index] - price_ma3_current
        if power_before>0 and power_current<0:
            power_decision = 1
        if power_before<0 and power_current>0:
            power_decision = -1 


        print "ma 60 -1 %s"%self.ma(self.close_price, 60, -1, current_price_index)
        print "ma 60 %s"%self.ma(self.close_price, 60, 0, current_price_index)
        print "ma 30 -1 %s"%self.ma(self.close_price, 30, -1, current_price_index)
        print "ma 30 %s"%self.ma(self.close_price, 30, 0, current_price_index)

        print "ma 20 -1 %s"%self.ma(self.close_price, 20, -1, current_price_index)
        print "ma 20 %s"%self.ma(self.close_price, 20, 0, current_price_index)
        print "ma 10 -1 %s"%self.ma(self.close_price, 10, -1, current_price_index)
        print "ma 10 %s"%self.ma(self.close_price, 10, 0, current_price_index)
        print "ma 5 -1 %s"%self.ma(self.close_price, 5, -1, current_price_index)
        print "ma 5 %s"%self.ma(self.close_price, 5, 0, current_price_index)
        #if self.close_price[current_price_index-1] > self.ma(self.close_price, 5 , -1, current_price_index) and self.close_price[current_price_index] > self.ma(self.close_price, 5 , 0, current_price_index):
        if self.close_price[current_price_index] > self.ma(self.close_price, 5 , 0, current_price_index):
            ma_decision = 1
            print "ma decision is 1"
        else:
            ma_decision = 0
            print "ma decision is 0"

        if self.ma(self.close_price, 60, 0, current_price_index) > self.ma(self.close_price, 60, -1, current_price_index):
            ma_decision4 = 1
        else:
            ma_decision4 = 0
#        if self.ma(self.close_price, 5, 0, current_price_index) > self.ma(self.close_price, 20, 0, current_price_index):
#            ma_decision3 = 1
#        else:
#            ma_decision3 = 0
        if self.ma(self.close_price, 10, -1, current_price_index) < self.ma(self.close_price, 10, 0, current_price_index):
            ma_decision5 = 1
        else:
            ma_decision5 = 0

        if self.ma(imf, 5, 0, current_price_index) > self.ma(imf, 10, 0, current_price_index):
            ma_decision2 = 1
        else:
            ma_decision2 = 0
        if self.ma(imf, 5, 0, current_price_index) > self.ma(imf, 20, 0, current_price_index):
            ma_decision3 = 1
        else:
            ma_decision3 = 0

        
        if self.ma(self.close_price, 5, 0, current_price_index)-self.ma(self.close_price, 5, -1, current_price_index) > self.ma(self.close_price, 5, -1, current_price_index)-self.ma(self.close_price, 5, -2, current_price_index):
            ma_decision6 = 1
        else:
            ma_decision6 = 0

        if self.ma(self.close_price, 10, 0, current_price_index)-self.ma(self.close_price, 10, -1, current_price_index) > self.ma(self.close_price, 10, -1, current_price_index)-self.ma(self.close_price, 10, -2, current_price_index):
            ma_decision9 = 1
        else:
            ma_decision9 = 0

        if self.ma(self.close_price, 20, 0, current_price_index) > self.ma(self.close_price, 20, -1, current_price_index):
            ma_decision7 = 1
        else:
            ma_decision7 = 0

        if (self.close_price[current_index]-self.close_price[current_index-1] )/ self.close_price[current_index-1] > 0.05:
            price_decision = 0
        else:
            price_decision = 1
        if (self.close_price[current_index-1]-self.close_price[current_index-2] )< (self.close_price[current_index]-self.close_price[current_index-1] ):
            price_decision2 = 1
        else:
            price_decision2 = 0

        if self.close_price[current_index]<self.close_price[current_index-1]:
            price_decision3 = 1
        else:
            price_decision3 = 0
        
        if self.ma(self.close_price, 20, -1, current_price_index) < self.ma(self.close_price, 20, 0, current_price_index):
            ma_decision8 = 1
        else:
            ma_decision8 = 0
#        if residual[current_index-1]<residual[current_index]:
#            residual_decision = 1
#        else:
#            residual_decision = 0
           
#        if power_before2<power_before and power_before<power_current:
#            power_decision = -1
#        if power_before2>power_before and power_before>power_current:
#            power_decision = 1

#        if power_before > power_current:
#            power_decision = 1
#        elif power_before < power_current:
#            power_decision = -1

#        print "imf current%s"%imf[current_index]
#        print "imf before current%s"%imf[current_index-1]
#        if imf[current_index]>imf[current_index-1]:
#            power_decision = -1
#        if imf[current_index]<imf[current_index-1]:
#            power_decision = 1

#        power_decision = period_decision

        if  period_decision == -1:
            decision = -1
        if  period_decision == 1:
            decision = 1


        if self.close_price[current_price_index] > self.close_price[current_price_index-1]:
            current_rise_flag = 1
        else:
            current_rise_flag = 0
        print "last last period decision%s"%self.last_last_period_decision 
        print "last period decision%s"%self.last_period_decision 
#        if self.last_last_period_decision == 1 and self.last_period_decision == 1  and extrem_decision > 0 and self.share>0 and float(current_price)>float(self.buy_price) and ma_decision6 == 0:
#            print "sell"
#            self.money = self.sell(self.share,trade_price)
#            self.share = 0
#            self.buy_price = 0
#            self.last_buy_flag = 0
#            self.total_asset = self.money - self.sell_fee(self.money)
#            self.money = self.total_asset
#            self.sell_x_index.append(current_price_index)
#            self.sell_y_index.append(current_price)
        if self.last_period_decision == 1 and decision == 0 and extrem_decision > 0  and self.share>0 and float(current_price)>float(self.buy_price) and ma_decision6 == 0:
            print "sell"
            self.money = self.sell(self.share,trade_price)
            self.share = 0
            self.buy_price = 0
            self.total_asset = self.money - self.sell_fee(self.money)
            self.last_buy_flag = 0
            self.money = self.total_asset
            self.sell_x_index.append(current_price_index)
            self.sell_y_index.append(current_price)
        elif  float(self.buy_price)!=0 and (float(self.buy_price)-float(current_price))/float(self.buy_price)>0.03 and self.share>0:
            print "sell"
            self.money = self.sell(self.share,trade_price)
            self.share = 0
            self.buy_price = 0
            self.total_asset = self.money - self.sell_fee(self.money)
            self.money = self.total_asset
            self.sell_x_index.append(current_price_index)
            self.last_buy_flag = 0
            self.sell_y_index.append(current_price)
#        if self.last_rise_flag == 1  and current_rise_flag ==0 and self.share>0 :
#            print "sell"
#            self.money = self.sell(self.share,trade_price)
#            self.share = 0
#            self.buy_price = 0
#            self.total_asset = self.money - self.sell_fee(self.money)
#            self.money = self.total_asset
#            self.sell_x_index.append(current_price_index)
#            self.sell_y_index.append(current_price)
#         
#        elif  decision == 0 and float(self.buy_price)!=0 and self.share> 0 and (float(current_price)-float(self.buy_price))/float(self.buy_price)>0.01:
#            print "sell"
#            self.money = self.sell(self.share,trade_price)
#            self.share = 0
#            self.buy_price = 0
#            self.total_asset = self.money - self.sell_fee(self.money)
#            self.money = self.total_asset
#            self.sell_x_index.append(current_price_index)
#            self.sell_y_index.append(current_price)
#        elif  decision == -1 and float(self.buy_price)!=0 and self.share> 0 and (float(current_price)-float(self.buy_price))/float(self.buy_price)>0.01:
#            print "sell"
#            self.money = self.sell(self.share,trade_price)
#            self.share = 0
#            self.buy_price = 0
#            self.total_asset = self.money - self.sell_fee(self.money)
#            self.money = self.total_asset
#            self.sell_x_index.append(current_price_index)
#            self.sell_y_index.append(current_price)
#        elif self.last_last_period_decision == -1 and self.last_period_decision == -1 and extrem_decision < 0 and decision == 0  and self.money>0 and  ma_decision6 == 1 and ma_decision7 == 1 :#and ma_decision5 == 1:
#            print "buy"
#            self.share = self.buy(self.money,trade_price)
#            self.money = 0
#            self.buy_price = trade_price
#            self.total_asset = self.share*trade_price 
#            self.buy_x_index.append(current_price_index)
#            self.buy_y_index.append(current_price)
#            self.last_buy_flag = 1
        elif self.last_period_decision == -1 and decision >= 0 and extrem_decision == -1  and self.money>0 and  ma_decision6 == 1 and ma_decision9 == 1 and price_decision == 1 and ma_decision8 == 1 and ma_decision4 == 1:
            print "buy"
            self.share = self.buy(self.money,trade_price)
            self.money = 0
            self.buy_price = trade_price
            self.total_asset = self.share*trade_price
            self.buy_x_index.append(current_price_index)
            self.buy_y_index.append(current_price)
            self.last_buy_flag = 1
        elif self.last_last_period_decision == -1 and self.last_period_decision == 0 and decision >= 0 and extrem_decision == -1  and self.money>0 and  ma_decision6 == 1 and ma_decision9 == 1 and price_decision == 1 and ma_decision8 == 1 :
            print "buy"
            self.share = self.buy(self.money,trade_price)
            self.money = 0
            self.buy_price = trade_price
            self.total_asset = self.share*trade_price
            self.buy_x_index.append(current_price_index)
            self.buy_y_index.append(current_price)
            self.last_buy_flag = 1
        elif self.last_last_period_decision == 0 and self.last_period_decision == 0 and decision == 0 and extrem_decision == -1  and self.money>0 and  ma_decision6 == 1  and price_decision == 1 and ma_decision8 == 1 and price_decision2 == 1:
            print "buy"
            self.share = self.buy(self.money,trade_price)
            self.money = 0
            self.buy_price = trade_price
            self.total_asset = self.share*trade_price
            self.buy_x_index.append(current_price_index)
            self.buy_y_index.append(current_price)
            self.last_buy_flag = 1

        elif self.last_period_decision == -1 and decision >= 0 and extrem_decision == -1  and self.money>0 and  ma_decision6 == 1 and ma_decision9 == 1 and price_decision3 == 1 and ma_decision8 == 1 and ma_decision4 == 1:
            print "buy"
            self.share = self.buy(self.money,trade_price)
            self.money = 0
            self.buy_price = trade_price
            self.total_asset = self.share*trade_price
            self.buy_x_index.append(current_price_index)
            self.buy_y_index.append(current_price)
            self.last_buy_flag = 1
        elif self.last_last_period_decision == -1 and self.last_period_decision == 0 and decision >= 0 and extrem_decision == -1  and self.money>0 and  ma_decision6 == 1 and ma_decision9 == 1 and price_decision3 == 1 and ma_decision8 == 1 :
            print "buy"
            self.share = self.buy(self.money,trade_price)
            self.money = 0
            self.buy_price = trade_price
            self.total_asset = self.share*trade_price
            self.buy_x_index.append(current_price_index)
            self.buy_y_index.append(current_price)
            self.last_buy_flag = 1
        elif self.last_last_period_decision == 0 and self.last_period_decision == 0 and decision == 0 and extrem_decision == -1  and self.money>0 and  ma_decision6 == 1  and price_decision3 == 1 :
            print "buy"
            self.share = self.buy(self.money,trade_price)
            self.money = 0
            self.buy_price = trade_price
            self.total_asset = self.share*trade_price
            self.buy_x_index.append(current_price_index)
            self.buy_y_index.append(current_price)
            self.last_buy_flag = 1
#        elif self.last_now_min > now_position_last_min_delta and  now_position_last_min_delta == 1 and self.money>0 and decision != 1 and  ma_decision6 == 1 and ma_decision7 == 1 and ma_decision5 == 1 and ma_decision8 == 1 and price_decision == 1:
#            print "buy"
#            self.share = self.buy(self.money,trade_price)
#            self.money = 0
#            self.buy_price = trade_price
#            self.total_asset = self.share*trade_price
#            self.buy_x_index.append(current_price_index)
#            self.buy_y_index.append(current_price)
#            self.last_buy_flag = 1
        
        else:
            self.last_buy_flag = 0

        self.last_last_period_decision = self.last_period_decision
        self.last_period_decision = period_decision

        if imf_min_index[-1]<imf_max_index[-1]:
            self.last_now_max = now_position_last_max_delta
            self.last_now_min = 0
        elif imf_min_index[-1]>imf_max_index[-1]:
            self.last_now_min = now_position_last_min_delta
            self.last_now_max = 0
        self.last_rise_flag = current_rise_flag
        self.total_asset_list.append(self.total_asset)
        print "total asset is %s"%(self.total_asset)
        print "money is %s"%(self.money)
        print "share is %s"%(self.share)

    def span_estimate(self, data):
        return (round(np.mean(data)-np.std(data),0),round(np.mean(data)+np.std(data),0))
    def run_predict_price_ma5(self, imf):


        decision = 0


        current_index = len(imf)-1
        print "current index%s"%current_index
        imf_ma5_before = (imf[current_index-5]+imf[current_index-4]+imf[current_index-3]+imf[current_index-2]+imf[current_index-1])/5.0
        imf_ma5_current = (imf[current_index-4]+imf[current_index-3]+imf[current_index-2]+imf[current_index-1]+imf[current_index])/5.0
        power_before = imf[current_index-1] - imf_ma5_before
        power_current = imf[current_index] - imf_ma5_current

  #      imf_ma3_current = (imf[current_index-2]+imf[current_index-1]+imf[current_index])/3.0
  #      imf_ma3_next = (imf[current_index-1]+imf[current_index]+result)/3.0
  #      power_current = imf[current_index] - imf_ma3_current
  #      power_next = result - imf_ma3_next



        if power_before>0 and power_current<0:
            decision = 1
            print "sell"
        if power_before<0 and power_current>0:
            decision = -1
            print "buy"
        print "decision %s"%decision

        current_index = len(imf)-1
        current_price = self.close_price[current_index]
        trade_price = self.open_price[current_index+1]
        print "current price%s"%current_price
        print "trade price%s"%trade_price
        print "buy price%s"%self.buy_price
        
        if decision == 1 and self.share>0 and (float(current_price)-float(self.buy_price))/float(self.buy_price)>0.04:
            print "sell" 
            self.money = self.sell(self.share,trade_price)
            self.share = 0
            self.buy_price = 0
            self.total_asset = self.money - self.sell_fee(self.money)
            self.money = self.total_asset
            self.sell_x_index.append(current_index)
            self.sell_y_index.append(current_price)
        if decision == 0 and float(self.buy_price)!=0 and (float(self.buy_price)-float(current_price))/float(self.buy_price)>0.02 and self.share>0:
            print "sell" 
            self.money = self.sell(self.share,trade_price)
            self.share = 0
            self.buy_price = 0
            self.total_asset = self.money - self.sell_fee(self.money)
            self.money = self.total_asset
            self.sell_x_index.append(current_index)
            self.sell_y_index.append(current_price)
        if decision == -1  and self.money>0:
            print "buy"
            self.share = self.buy(self.money,trade_price)
            self.money = 0
            self.buy_price = trade_price
            self.total_asset = self.share*trade_price - self.buy_fee(self.share,trade_price)
            self.buy_x_index.append(current_index)
            self.buy_y_index.append(current_price)
        self.total_asset_list.append(self.total_asset)
        print "total asset is %s"%(self.total_asset)
        print "money is %s"%(self.money)
        print "share is %s"%(self.share)

    def format_data(self, data):
        if isinstance(data,list):
            return [ 10*((float(data[i])-min(data))/(max(data)-float(min(data)))) for i in range(len(data))]

    def run(self,emd_data, datafile,current_index):
        starttime = datetime.datetime.now()
        self.clear_emd_data()
        my_emd = one_dimension_emd(emd_data)
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

        fp = open("imf2")
        lines = fp.readlines()
        fp.close()
    
        #print lines
    
        imf_raw = []
        for eachline in lines:
        #    print eachline.split(" ")[0]
            imf_raw.append(float(eachline.split("\n")[0]))
   #     for i in range(2):
   #         imf.pop()
        imf = self.format_data(imf_raw)
        #self.clear_nn_train_data()
        #self.generate_all_price(imf,25)
        #self.generate_divide(imf,20,25) 
        #self.run_predict_price_real("../../train_data","nn_file",20,25,imf)
        #self.run_predict_price_period(imf)
        print "\n\n\n"
        self.run_predict_price_period(imf,current_index,residual)
        endtime = datetime.datetime.now()
        print "run time"
        print (endtime - starttime).seconds

                 
        
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
        print "asset delta %s"%delta
        for j in range(len(delta)):
            print delta[j]
            if delta[j] != 0:
                delta_new.append(delta[j])
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
        

    def draw_fig(self,datafile,start,save=0):
        
        data = self.close_price

        fp = open("imf2",'r')
        lines = fp.readlines()
        fp.close()
        
        imf_raw = []
        for eachline in lines:
            imf_raw.append(float(eachline.split("\n")[0]))

        imf = self.format_data(imf_raw)
        fp = open("residual",'r')
        lines = fp.readlines()
        fp.close()

        residual = []
        for eachline in lines:
            residual.append(float(eachline.split("\n")[0]))

        

        print "buy x index%s"%self.buy_x_index 
        print "sell x index%s"%self.sell_x_index 
        imf_buy_value = [imf[i] for i in self.buy_x_index]
        imf_sell_value = [imf[i] for i in self.sell_x_index]
        #imf_buy_value = [imf[i] for i in self.period_decision_buy_point]
        #imf_sell_value = [imf[i] for i in self.period_decision_sell_point]
        #self.buy_x_index = self.period_decision_buy_point
        #self.sell_x_index = self.period_decision_sell_point
        plt.figure(1)
        plt.subplot(211).axis([start,len(data),min(data),max(data)])      
        plt.plot([i for i in range(len(data))],data,'b',self.buy_x_index,self.buy_y_index,'r*',self.sell_x_index,self.sell_y_index,'g*')             
        plt.subplot(212).axis([start,len(data),min(imf),max(imf)])
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
        plt.figure(3)
        plt.subplot(211)
        plt.plot([i for i in range(len(self.max_min_period_delta))],self.max_min_period_delta,'b')
        plt.subplot(212)
        plt.plot([i for i in range(len(self.min_max_period_delta))],self.min_max_period_delta,'b')
       
        if save==1:         
            savefig(figname)
        else:
            plt.show() 


if __name__ == "__main__":

    datafile = sys.argv[1]
    begin = int(sys.argv[2])
    fp = open(datafile)
    lines = fp.readlines()
    fp.close()
#
#    #print lines
#
    close_price = []
    open_price = []
    date = [] 
    for eachline in lines:
#    #    print eachline.split(" ")[0]
        eachline.strip()
        close_price.append(float(eachline.split("\t")[2]))
        open_price.append(float(eachline.split("\t")[1]))
        date.append(eachline.split("\t")[0])
    
    process = process_util(open_price,close_price)
    os.popen("rm -r ../../emd_data")
    os.popen("mkdir ../../emd_data")
    process.generate_emd_data(close_price,begin)
    #process.generate_emd_data_fix(close_price,begin,500)
    for i in range(begin,begin+int(process.file_count("../../emd_data"))-1):
        emd_data = []
        print "emd file %s"%i
        fp = open("../../emd_data/emd_%s"%i,'r')
        lines = fp.readlines()
        fp.close()
#        
        for eachline in lines:
            eachline.strip("\n")
            emd_data.append(float(eachline))
#    #    print "emd_data %s"%emd_data
        process.run(emd_data,datafile,i-1)
    process.peilv()
    process.hold_time()
    process.draw_fig(datafile,begin)

#if __name__ == "__main__":
#
#    #fp = open("data_dbn")
#
#
#    datafile = sys.argv[1]
#    fp = open(datafile)
#    lines = fp.readlines()
#    fp.close()
#
#    print lines
#
#    price = []
#    open_price = []
#    date = [] 
#    for eachline in lines:
#        print eachline.split(" ")[0]
#        price.append(float(eachline.split("\t")[2]))
#        open_price.append(float(eachline.split("\t")[1]))
#        date.append(eachline.split("\t")[0])
#
#    my_emd = one_dimension_emd(price)
#    (imf, residual) = my_emd.emd()
#    for i in range(len(imf)):
#        fp = open("imf%s"%i,'w')
#        for j in range(len(imf[i])):
#            fp.writelines(str(imf[i][j]))
#            fp.writelines("\n")
#        fp.close()
#    print "imf num %s"%len(imf)
#    fp = open("residual",'w')
#    for i in range(len(residual)):
#        fp.writelines(str(residual[i]))
#        fp.writelines("\n")
#    fp.close()
#
#
#
#    fp = open("imf2")
#    lines = fp.readlines()
#    fp.close()
#
#    print lines
#
#    imf = []
#    for eachline in lines:
#        print eachline.split(" ")[0]
#        imf.append(float(eachline.split("\n")[0]))
#
#    fp = open("residual")
#    lines = fp.readlines()
#    fp.close()
#
#    print lines
#
#    residual = []
#    for eachline in lines:
#        print eachline.split(" ")[0]
#        residual.append(float(eachline.split("\n")[0]))
#
#    #process = process_util()
#    #process.generate_all_price(imf,10)
#    #process.generate_divide(imf,10,10) 
#    #process.run_predict_price_real(price,"train_data","nn_file",10,10,imf,datafile,date[-1])
    

            
