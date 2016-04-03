import os
import sys
import libfann
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from emd import *
import datetime




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


    def generate_all_price(self,data,input_num,imf_num):
        expect = 0
        fp = open("imf%s_train_data_all"%imf_num,'w')
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
    
    def generate_divide(self, data_raw, eachfile_group_num, input_num,imf_num):
        fp = open("imf%s_train_data_all"%imf_num,'r')
        data = fp.readlines()
        fp.close()
        total_group_num = len(data_raw)-input_num+1
        total_file_num = total_group_num - eachfile_group_num-1 
        for i in range(total_file_num):
            fp = open("../../train_data_imf%s/data_%s"%(imf_num,i),'w')
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
        ann.create_standard_array((25,15,1))
        ann.set_learning_rate(0.3)
        ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC)
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


    def run_predict_price_each(self, train_data_dir, net_file, test_group_num, input_num, imf):


        decision = 0
        file_num = int(self.file_count(train_data_dir))
        print train_data_dir
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
        return result

    def generate_emd_data(self,data,start):
        
        for i in range(start,len(data)-2):
            fp = open("../../emd_data/emd_%s"%i,'w')
            for j in range(i):
                fp.writelines(str(data[j]))
                fp.writelines("\n")
            fp.writelines(str(self.open_price[i]))
            fp.writelines("\n")
            fp.writelines(str(self.close_price[i-1]))
            fp.writelines("\n")
            fp.close()

    def generate_emd_data_fix(self,data,start,period):

        for i in range(start,len(data)+1):
            fp = open("emd_data_fix/emd_%s"%i,'w')
            for j in range(i-period,i):
                print j
                fp.writelines(str(data[j]))
                fp.writelines("\n")
            fp.close()

    def judge(self, result, train_data_dir, test_group_num, input_num):


        decision = 0

        file_num = int(self.file_count(train_data_dir))

        current_index = file_num-1+input_num+test_group_num-2
      

        data_ma5_current = (self.close_price[current_index-4]+self.close_price[current_index-3]+self.close_price[current_index-2]+self.close_price[current_index-1]+self.close_price[current_index])/5.0
        data_ma5_next = (self.close_price[current_index-3]+self.close_price[current_index-2]+self.close_price[current_index-1]+self.close_price[current_index]+result)/5.0
        power_current = self.close_price[current_index] - data_ma5_current
        power_next = result - data_ma5_next
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

    def run(self,emd_data, datafile,close_price):
        predict_imf = []
        predict_price = 0
        starttime = datetime.datetime.now()
        self.clear_emd_data()
        my_emd = one_dimension_emd(emd_data)
        (imf, residual) = my_emd.emd()
        #for i in range(len(imf)):
        #    fp = open("imf%s"%i,'w')
        #    for j in range(len(imf[i])):
        #        fp.writelines(str(imf[i][j]))
        #        fp.writelines("\n")
        #    fp.close()
        #print "imf num %s"%len(imf)
        #fp = open("residual",'w')
        #for i in range(len(residual)):
        #    fp.writelines(str(residual[i]))
        #    fp.writelines("\n")
        #fp.close()

        #fp = open("imf3")
        #lines = fp.readlines()
        #fp.close()
    
        ##print lines
    
        #imf = []
        #for eachline in lines:
        ##    print eachline.split(" ")[0]
        #    imf.append(float(eachline.split("\n")[0]))
        #for i in range(2):
        #    imf.pop()

        for i in range(len(imf)):
            self.clear_nn_train_data()
            self.generate_all_price(imf[i],25,i)
            self.generate_divide(imf[i],20,25,i) 
            predict_imf.append(self.run_predict_price_each("../../train_data_imf%s"%i,"nn_file",20,25,imf[i]))
        self.clear_nn_train_data()
        self.generate_all_price(residual,25,100)
        self.generate_divide(residual,20,25,100)
        predict_imf.append(self.run_predict_price_each("../../train_data_imf100","nn_file",20,25,residual))
        predict_price = sum(predict_imf)
        print "predict price %s"%predict_price
        self.judge(predict_price,"train_data_100",20,25)
        endtime = datetime.datetime.now()
        print "run time"
        print (endtime - starttime).seconds

                 
        
    def clear_emd_data(self):
        os.popen("rm imf*")    

    def clear_nn_train_data(self):
        os.popen("rm -r ../../train_data_imf0")
        os.popen("mkdir ../../train_data_imf0")
        os.popen("rm -r ../../train_data_imf1")
        os.popen("mkdir ../../train_data_imf1")
        os.popen("rm -r ../../train_data_imf2")
        os.popen("mkdir ../../train_data_imf2")
        os.popen("rm -r ../../train_data_imf3")
        os.popen("mkdir ../../train_data_imf3")
        os.popen("rm -r ../../train_data_imf4")
        os.popen("mkdir ../../train_data_imf4")
        os.popen("rm -r ../../train_data_imf5")
        os.popen("mkdir ../../train_data_imf5")
        os.popen("rm -r ../../train_data_imf6")
        os.popen("mkdir ../../train_data_imf6")
        os.popen("rm -r ../../train_data_imf7")
        os.popen("mkdir ../../train_data_imf7")
        os.popen("rm -r ../../train_data_imf8")
        os.popen("mkdir ../../train_data_imf8")
        os.popen("rm -r ../../train_data_imf9")
        os.popen("mkdir ../../train_data_imf9")
        os.popen("rm -r ../../train_data_imf100")
        os.popen("mkdir ../../train_data_imf100")







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
        plt.subplot(212).axis([1200,1500,-3,3])
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

    datafile = sys.argv[1]
    begin = int(sys.argv[2])
    fp = open(datafile)
    lines = fp.readlines()
    fp.close()

    #print lines

    close_price = []
    open_price = []
    date = [] 
    for eachline in lines:
    #    print eachline.split(" ")[0]
        eachline.strip()
        close_price.append(float(eachline.split("\t")[2]))
        open_price.append(float(eachline.split("\t")[1]))
        date.append(eachline.split("\t")[0])
    
    process = process_util(open_price,close_price)
    os.popen("rm -r ../../emd_data")
    os.popen("mkdir ../../emd_data")
    process.generate_emd_data(close_price,begin)
##    process.generate_emd_data_fix(close_price,815,200)
    for i in range(begin,begin+int(process.file_count("../../emd_data"))-1):
        emd_data = []
        print "emd file %s"%i
        fp = open("../../emd_data/emd_%s"%i,'r')
        lines = fp.readlines()
        fp.close()
        
        for eachline in lines:
            eachline.strip("\n")
            emd_data.append(float(eachline))
    #    print "emd_data %s"%emd_data
        process.run(emd_data,datafile,close_price)
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
    

            
