
import numpy as np
from calc_ma import *
from macd import ema


class generateEMDdata():
    
    def __init__(self,raw_data_file,start_position,each_file_period,emd_data_dir):
        self.raw_data_file = raw_data_file 
        self.start_position = start_position
        self.each_file_period = each_file_period

        self.emd_data_dir = emd_data_dir
        self.raw_data = self.load_raw_data()
        self.vol_data = self.load_vol_data()
        self.high_data = self.load_high_data()
        self.low_data = self.load_low_data()
#        self.je = self.load_je_data()
#        self.macd_data = self.load_macd_data()
        

    def load_raw_data(self):

        data = []
        fp = open(self.raw_data_file)
        lines = fp.readlines()
        fp.close()

        for eachline in lines:
            eachline.strip("\n")
            data.append(float(eachline.split("\t")[4]))

        return data

    def load_je_data(self):

        data = []
        fp = open(self.raw_data_file)
        lines = fp.readlines()
        fp.close()

        for eachline in lines:
            eachline.strip("\n")
            data.append(float(eachline.split("\t")[6]))

        return data

    def load_macd_data(self):

        data = []
        fp = open(self.raw_data_file)
        lines = fp.readlines()
        fp.close()

        for eachline in lines:
            eachline.strip("\n")
            data.append(float(eachline.split("\t")[6]))

        return data
    def load_vol_data(self):

        data = []
        fp = open(self.raw_data_file)
        lines = fp.readlines()
        fp.close()

        for eachline in lines:
            eachline.strip("\n")
            data.append(float(eachline.split("\t")[5]))

        return data

    def load_high_data(self):

        data = []
        fp = open(self.raw_data_file)
        lines = fp.readlines()
        fp.close()

        for eachline in lines:
            eachline.strip("\n")
            data.append(float(eachline.split("\t")[2]))

        return data

    def load_low_data(self):

        data = []
        fp = open(self.raw_data_file)
        lines = fp.readlines()
        fp.close()

        for eachline in lines:
            eachline.strip("\n")
            data.append(float(eachline.split("\t")[3]))

        return data


    def generate_emd_data(self):

        for i in range(self.start_position,len(self.raw_data)+1):
            fp = open("%s/emd_%s"%(self.emd_data_dir,i),'w')
            for j in range(i):
                fp.writelines(str(self.raw_data[j]))
                fp.writelines("\n")
            fp.close()

    def generate_emd_data_fix(self):
        for i in range(self.start_position,len(self.raw_data)):
            fp = open("%s/emd_%s"%(self.emd_data_dir,i),'w')
            for j in range(i-self.each_file_period,i):
                fp.writelines(str(self.raw_data[j]))
                fp.writelines("\n")
            fp.close()

    def generate_ma_emd_data_fix(self,n):

        data = self.raw_data 
        #data = [self.vol_data[i] for i in range(len(self.raw_data))]
        ma5 = []
        for i in range(n):
            ma5.append(0)

        for i in range(n,len(data)):
            mean_5 = np.mean(data[i-(n-1):i+1])
            ma5.append(mean_5)
        print len(data)
        print len(ma5)

#        data = []
#        for i in range(n):
#            data.append(0)
##
#        for i in range(n,len(ma5)):
#            mean_5 = np.mean(ma5[i-(n-1):i+1])
#            data.append(mean_5)


   #     ma5=data

   #     data = []
   #     for i in range(n):
   #         data.append(0)

   #     for i in range(n,len(ma5)):
   #         mean_5 = np.mean(ma5[i-(n-1):i+1])
   #         data.append(mean_5)


  #      ma5=data

  #      data = []
  #      for i in range(n):
  #          data.append(0)

  #      for i in range(n,len(ma5)):
  #          mean_5 = np.mean(ma5[i-(n-1):i+1])
  #          data.append(mean_5)


  #      ma5=data
  #      data = []
  #      for i in range(n):
  #          data.append(0)

  #      for i in range(n,len(ma5)):
  #          mean_5 = np.mean(ma5[i-(n-1):i+1])
  #          data.append(mean_5)


  #      ma5=data
  #      data = []
  #      for i in range(n):
  #          data.append(0)

  #      for i in range(n,len(ma5)):
  #          mean_5 = np.mean(ma5[i-(n-1):i+1])
  #          data.append(mean_5)


   #     ma5=data



        for i in range(self.start_position,len(self.raw_data)+1):
            fp = open("%s/emd_%s"%(self.emd_data_dir,i),'w')
            for j in range(i-self.each_file_period,i):
                fp.writelines(str(ma5[j]))
                fp.writelines("\n")
            fp.close()


    def generate_jema_emd_data_fix(self,n):

        data = self.je
        #data = [self.vol_data[i] for i in range(len(self.raw_data))]
        ma5 = []
        for i in range(n):
            ma5.append(0)

        for i in range(n,len(data)):
            mean_5 = np.mean(data[i-(n-1):i+1])
            ma5.append(mean_5)
        print len(data)
        print len(ma5)

        data = []
        for i in range(n):
            data.append(0)
#
        for i in range(n,len(ma5)):
            mean_5 = np.mean(ma5[i-(n-1):i+1])
            data.append(mean_5)


        ma5=data

        #data = []
        #for i in range(n):
        #    data.append(0)

        #for i in range(n,len(ma5)):
        #    mean_5 = np.mean(ma5[i-(n-1):i+1])
        #    data.append(mean_5)


        #ma5=data



        for i in range(self.start_position,len(self.raw_data)+1):
            fp = open("%s/emd_%s"%(self.emd_data_dir,i),'w')
            for j in range(i-self.each_file_period,i):
                fp.writelines(str(ma5[j]))
                fp.writelines("\n")
            fp.close()


    def generate_machao_emd_data_fix(self,n):

        data = self.raw_data
        #data = [self.vol_data[i] for i in range(len(self.raw_data))]
        ma5 = []
        for i in range(n-1):
            ma5.append(0)

        for i in range(n,len(data)+1):
            mean_5 = np.mean(data[i-(n-1):i+1])
            ma5.append(mean_5)
        print len(data)
        print len(ma5)

#        data = []
#        for i in range(n-1):
#            data.append(0)
##
#        for i in range(n,len(ma5)+1):
#            mean_5 = np.mean(ma5[i-(n-1):i+1])
#            data.append(mean_5)


#        ma5=data
#
#        data = []
#        for i in range(n-1):
#            data.append(0)
#
#        for i in range(n,len(ma5)+1):
#            mean_5 = np.mean(ma5[i-(n-1):i+1])
#            data.append(mean_5)


#        ma5=data



        for i in range(self.start_position,len(self.raw_data)+1):
            fp = open("%s/emd_%s"%(self.emd_data_dir,i),'w')
            for j in range(i-self.each_file_period,i):
                fp.writelines(str(ma5[j]))
                fp.writelines("\n")
            fp.close()

    def generate_ma_emd_data_fix2(self,n):

        p = self.raw_data





        data = []
        for i in range(len(p)):
            #data.append((p[i-3]+3*p[i-2]+6*p[i-1]+7*p[i]+6*pp[0]+3*pp[1]+pp[2])/27.0)
            data.append((p[i-3]+3*p[i-2]+6*p[i-1]+17*p[i])/27.0)

        print len(p)
        print len(data)
        ma5=data



        for i in range(self.start_position,len(self.raw_data)+1):
            fp = open("%s/emd_%s"%(self.emd_data_dir,i),'w')
            for j in range(i-self.each_file_period,i):
                fp.writelines(str(ma5[j]))
                fp.writelines("\n")
            fp.close()

    def generate_mavol_emd_data_fix(self,n):

        ma5_p = []
        for i in range(n-1):
            ma5_p.append(0)

        for i in range(n,len(self.raw_data)+1):
            mean_5 = np.mean(self.raw_data[i-(n-1):i+1])
            ma5_p.append(mean_5)
        print len(self.raw_data)
        print len(ma5_p)

        ma5_v = []
        for i in range(n-1):
            ma5_v.append(0)

        for i in range(n,len(self.raw_data)+1):
            mean_5 = np.mean(self.vol_data[i-(n-1):i+1])
            ma5_v.append(mean_5)
        print len(self.vol_data)
        print len(ma5_v)

        ma5 = [ma5_p[i]*ma5_v[i] for i in range(len(ma5_p))]



        for i in range(self.start_position,len(self.raw_data)+1):
            fp = open("%s/emd_%s"%(self.emd_data_dir,i),'w')
            for j in range(i-self.each_file_period,i):
                fp.writelines(str(ma5[j]))
                fp.writelines("\n")
            fp.close()

    def generate_mavol5_emd_data_fix(self,n):

        ma5_p = []
        for i in range(n-1):
            ma5_p.append(0)

        for i in range(n,len(self.raw_data)+1):
            mean_5 = np.mean(self.raw_data[i-(n-1):i+1])
            ma5_p.append(mean_5)
        print len(self.raw_data)
        print len(ma5_p)

        ma5_v = []
        for i in range(n-1):
            ma5_v.append(0)

        for i in range(n,len(self.raw_data)+1):
            mean_5 = np.mean(self.vol_data[i-(n-1):i+1])
            ma5_v.append(mean_5)
        print len(self.vol_data)
        print len(ma5_v)

        ma5 = [ma5_p[i]*ma5_v[i] for i in range(len(ma5_p))]

        data = []
        for i in range(n-1):
            data.append(0)

        for i in range(n,len(ma5)+1):
            mean_5 = np.mean(ma5[i-(n-1):i+1])
            data.append(mean_5)


        ma5=data
        for i in range(self.start_position,len(self.raw_data)+1):
            fp = open("%s/emd_%s"%(self.emd_data_dir,i),'w')
            for j in range(i-self.each_file_period,i):
                fp.writelines(str(ma5[j]))
                fp.writelines("\n")
            fp.close()

    def generate_mavol3_emd_data_fix(self,n):

######
#        ma5_p = []
#        for i in range(n-1):
#            ma5_p.append(0)
#        mean_data = [(self.high_data[i]+self.low_data[i])/2 for i in range(len(self.high_data))]
#        for i in range(n,len(self.raw_data)+1):
#            mean_5 = np.mean(mean_data[i-(n-1):i+1])
#            ma5_p.append(mean_5)
#        print len(self.raw_data)
#        print len(ma5_p)
#
#        ma5_v = []
#        for i in range(n-1):
#            ma5_v.append(0)
#
#        for i in range(n,len(self.raw_data)+1):
#            mean_5 = np.mean(self.vol_data[i-(n-1):i+1])
#            ma5_v.append(mean_5)
#        print len(self.vol_data)
#        print len(ma5_v)
#
#        ma5 = [ma5_p[i]*ma5_v[i] for i in range(len(ma5_p))]

#####
        data = [((self.high_data[i]+self.low_data[i])/2)*self.vol_data[i] for i in range(len(self.vol_data))]

        ma5 = []
        for i in range(n-1):
            ma5.append(0)
#
        for i in range(n,len(data)+1):
            mean_5 = np.mean(data[i-(n-1):i+1])
            ma5.append(mean_5)
############

        for i in range(self.start_position,len(self.raw_data)+1):
            fp = open("%s/emd_%s"%(self.emd_data_dir,i),'w')
            for j in range(i-self.each_file_period,i):
                fp.writelines(str(ma5[j]))
                fp.writelines("\n")
            fp.close()

    def generate_mavol2_emd_data_fix(self,n):

        obv = [((2*self.raw_data[i]-self.high_data[i]-self.low_data[i])/(self.high_data[i]-self.low_data[i]))*self.vol_data[i] for i in range(len(self.raw_data))]
        ma5 = []
        for i in range(n-1):
            ma5.append(0)

        for i in range(n,len(obv)+1):
            mean_5 = np.mean(obv[i-(n-1):i+1])
            ma5.append(mean_5)


        for i in range(self.start_position,len(self.raw_data)+1):
            fp = open("%s/emd_%s"%(self.emd_data_dir,i),'w')
            for j in range(i-self.each_file_period,i):
                fp.writelines(str(ma5[j]))
                fp.writelines("\n")
            fp.close()


    def generate_ada_ma_emd_data_fix(self):


        ma = calc_ada_ma(self.raw_data,5)
        
        
        for i in range(self.start_position,len(self.raw_data)+1):
            fp = open("%s/emd_%s"%(self.emd_data_dir,i),'w')
            for j in range(i-self.each_file_period,i):
                fp.writelines(str(ma[j]))
                fp.writelines("\n")
            fp.close()


    def generate_ema_emd_data_fix(self,n):

        ma1 = []
        for i in range(n):
            ma1.append(0)
        ma2 = [ema(self.raw_data,i,n) for i in range(n,len(self.raw_data))]
        print len(ma2)

        ma = ma1+ma2

        print "len raw%s"%len(self.raw_data)
        print "len ema%s"%len(ma)
       


        for i in range(self.start_position,len(self.raw_data)+1):
            fp = open("%s/emd_%s"%(self.emd_data_dir,i),'w')
            for j in range(i-self.each_file_period,i):
                fp.writelines(str(ma[j]))
                fp.writelines("\n")
            fp.close()




if __name__ == "__main__":
    generate_emd_func = generateEMDdata("gmry",800,500)
    generate_emd_func.generate_ema_emd_data_fix(5)
#    generate_emd_func.generate_emd_data()
