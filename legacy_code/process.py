import os
import sys
import libfann
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from emd import *




class process_util():

    def __init__(self):
        
        self.money = 10000
        self.share = 0
        self.sell_x_index = []
        self.sell_y_index = []
        self.buy_x_index = []
        self.buy_y_index = []
        
        self.total_asset_list = []
    
    def data_mean(self, mean_period, data):
        mean = []
        for i in range(mean_period-1,len(data)):
            period_sum = 0.0
            for j in range(mean_period):
                period_sum += data[i-j]
            mean.append(period_sum/mean_period)
        return mean

    def calc_power(self, data, ma):
        power = []
        if isinstance(data,list) and isinstance(ma,list):
            if len(data)==len(ma):
                for i in range(0,len(data)):
                    power.append(data[i]-ma[i])
            return power 
        else:
            print"calc_power:please input equal list"
    
    def calc_flag(self, power):
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
        total_file_num = total_group_num - eachfile_group_num + 1
        for i in range(total_file_num):
            fp = open("train_data/data_%s"%i,'w')
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
        ann.create_standard_array((10,10,1))
        ann.set_learning_rate(0.3)
        ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)
    
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


    def open_close_diff(self,open_price_file,close_price_file):

        open_price = []
        close_price = []
        delta = []
        positive = []
        negative = []
        
        fp = open(open_price_file,'r')
        lines = fp.readlines()
        fp.close()

        for eachline in lines:
            open_price.append(float(eachline.split("\n")[0]))
            

        fp = open(close_price_file,'r')
        lines = fp.readlines()
        fp.close()

        for eachline in lines:
            close_price.append(float(eachline.split("\n")[0]))


        close_price.pop()
        for i in range(len(close_price)):
            delta.append(open_price[i+1]-close_price[i])

        max_delta = max(delta)
        min_delta = min(delta)

        for i in delta:
            if i > 0:
                positive.append(i)
            elif i < 0:
                negative.append(i)
        #positive.remove(max(positive))
        #negative.remove(min(negative))
        mean_positive = sum(positive)/len(positive)
        mean_negative = sum(negative)/len(negative)
  
        return (max_delta, min_delta, mean_positive, mean_negative)

            
   
          
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



    def run_predict_price_real(self, data, train_data_dir, net_file, test_group_num, input_num, imf, datafile, date):

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
        for j in range(20):
            self.train_nn(file_current,net_file)
            result_list.append(self.test_each(net_file,test_data_float))
        result_list.remove(max(result_list))
        result_list.remove(min(result_list))
        print "result list %s"%result_list
        result = sum(result_list)/len(result_list)
        print "result%s"%result


        current_index = file_num-1+input_num+test_group_num-2
        print "current imf value%s"%imf[current_index]
        print "current price%s"%data[current_index]

        imf_ma5_current = (imf[current_index-4]+imf[current_index-3]+imf[current_index-2]+imf[current_index-1]+imf[current_index])/5.0
        imf_ma5_next = (imf[current_index-3]+imf[current_index-2]+imf[current_index-1]+imf[current_index]+result)/5.0
        power_current = imf[current_index] - imf_ma5_current
        power_next = result - imf_ma5_next

        if power_current>0 and power_next<0:
            decision = 1
            print "sell"
        if power_current<0 and power_next>0:
            decision = -1
            print "buy"
        print "decision %s"%decision 

        lines = []
        if os.path.exists("decision"):
            fp = open("decision",'r')
            lines = fp.readlines()
            fp.close()
            fp = open("decision",'w')
            lines.append("\n")
            lines.append(date)
            lines.append(datafile)
            lines.append(str(decision))
            fp.writelines(lines)
        else:
            fp = open("decision",'w')
            lines.append(date)
            lines.append(datafile)
            lines.append(str(decision))
            fp.writelines(lines)
        fp.close()


    def run(self, data, train_data_dir, net_file, test_group_num, input_num):

        buy_price = 0
        
        for i in range(int(self.file_count(train_data_dir))-2):
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
            for j in range(1):
                self.train_nn(file_current,net_file)
                result_list.append(self.test_each(net_file,test_data_float))
            result = sum(result_list)/len(result_list)
            print "result%s"%result

            current_index = i+input_num+test_group_num-2
            trade_price = data[current_index+1]
            current_price = data[current_index]
            
            if float(result)>0.5 and self.share>0 and (float(current_price)-float(buy_price))/float(buy_price)>0.02:
                print "sell"
                self.money = self.sell(self.share,trade_price)
                self.share = 0
                buy_price = 0
                self.sell_x_index.append(current_index)
                self.sell_y_index.append(current_price)
            if float(buy_price)!=0 and (float(buy_price)-float(current_price))/float(buy_price)>0.02 and self.share>0:
                print "sell"
                self.money = self.sell(self.share,trade_price)
                self.share = 0
                buy_price = 0
                self.sell_x_index.append(current_index)
                self.sell_y_index.append(current_price)
            if float(result)<-0.9  and self.money>0:
                print "buy"
                self.share = self.buy(self.money,trade_price)
                self.money = 0
                buy_price = trade_price
                self.buy_x_index.append(current_index)
                self.buy_y_index.append(current_price)
            total_asset = float(self.money)+float(self.share)*float(trade_price)
            self.total_asset_list.append(total_asset)
            print "total asset is %s"%(total_asset) 
            print "total asset list  is %s"%(self.total_asset_list) 
            print "money is %s"%(self.money) 
            print "share is %s"%(self.share) 


    def draw_fig(self,data,imf,residual,datafile,save=0):
        


        imf_buy_value = [imf[i] for i in self.buy_x_index]
        imf_sell_value = [imf[i] for i in self.sell_x_index]
        plt.figure(1)
        plt.subplot(211).axis([900,1100,8,15])      
        plt.plot([i for i in range(len(data))],data,'b',self.buy_x_index,self.buy_y_index,'r*',self.sell_x_index,self.sell_y_index,'g*')             
        plt.subplot(212).axis([900,1100,-2,2])
        plt.plot([i for i in range(len(imf))],imf,'b',self.buy_x_index,imf_buy_value,'r*',self.sell_x_index,imf_sell_value,'g*')   
        figname = "fig1_"+datafile         
        if save==1:
            savefig(figname)
        plt.figure(2)
        plt.subplot(211)
        plt.plot([i for i in range(len(self.total_asset_list))],self.total_asset_list,'b')
        plt.subplot(212)
        plt.plot([i for i in range(len(residual))],residual,'b',self.buy_x_index,self.buy_y_index,'r*',self.sell_x_index,self.sell_y_index,'g*')
        figname = "fig2_"+datafile
        if save==1:         
            savefig(figname)
        else:
            plt.show() 



if __name__ == "__main__":

    #fp = open("data_dbn")


    datafile = sys.argv[1]
    fp = open(datafile)
    lines = fp.readlines()
    fp.close()

    print lines

    price = []
    open_price = []
    date = [] 
    for eachline in lines:
        print eachline.split(" ")[0]
        price.append(float(eachline.split("\t")[2]))
        open_price.append(float(eachline.split("\t")[1]))
        date.append(eachline.split("\t")[0])

    my_emd = one_dimension_emd(price)
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

    print lines

    imf = []
    for eachline in lines:
        print eachline.split(" ")[0]
        imf.append(float(eachline.split("\n")[0]))

    fp = open("residual")
    lines = fp.readlines()
    fp.close()

    print lines

    residual = []
    for eachline in lines:
        print eachline.split(" ")[0]
        residual.append(float(eachline.split("\n")[0]))

    process = process_util()
    ma =  process.data_mean(10,imf)
    print ma
    power = process.calc_power(imf[9:],ma)
    print power
    flag = process.calc_flag(power)
    print flag
    process.generate_all_price(imf,10)
    process.generate_divide(imf,10,10) 
    file_num = int(process.file_count("train_data"))
    if sys.argv[2] == "trend":
        #process.run_predict_price(price,"train_data","nn_file",10, 10,imf,open_price,file_num,datafile)
        process.run_predict_price(price,"train_data","nn_file",10, 10,imf,open_price,170,datafile)
        process.draw_fig(price,imf,residual,datafile)
    elif sys.argv[2] == "predict":
        process.run_predict_price_real(price,"train_data","nn_file",10,10,imf,datafile,date[-1])
    

            
