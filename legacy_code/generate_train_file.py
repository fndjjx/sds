

class genTraindata():

    def __init__(self,raw_data_file,train_data_all_file,train_data_divide_dir,group_number,train_number,result_number):

        self.raw_data_file = raw_data_file
        self.group_number = group_number
        self.train_number = train_number
        self.result_number = result_number        
        self.train_data_all_file = train_data_all_file
        self.train_data_divide_dir = train_data_divide_dir
        self.raw_data = self.load_raw_data()

    def load_raw_data(self):
        
        data = []
        fp = open(self.raw_data_file)
        lines = fp.readlines()
        fp.close()

        for eachline in lines:
            eachline.strip("\n")
            data.append(float(eachline))

        data_format = self.format_data(data)

        return data_format

    def format_data(self, data):
        if isinstance(data,list):
            return [2*((float(data[i])-min(data))/(max(data)-float(min(data))))-1 for i in range(len(data))]
        
#  generate decrease number result
#    def generate_all(self):
#
#        len_raw_data = len(self.raw_data)
#        fp = open(self.train_data_all_file,'w')
#        for i in range(len_raw_data-self.train_number+1):
#            for j in range(i,self.train_number+i):
#                fp.writelines(str(self.raw_data[j]))
#                fp.writelines(" ")
#                result_begin = j+1
#            fp.writelines("\n")
#            if result_begin+self.result_number<=len_raw_data:
#                result_end = result_begin+self.result_number
#                for k in range(result_begin,result_end):
#                    fp.writelines(str(self.raw_data[k]))
#                    fp.writelines(" ")
#                fp.writelines("\n")
#            elif (result_begin+self.result_number>len_raw_data) and result_begin<len_raw_data:
#                result_end = len_raw_data
#                for k in range(result_begin,result_end):
#                    fp.writelines(str(self.raw_data[k]))
#                    fp.writelines(" ")
#                fp.writelines("\n")
#            else:
#                fp.writelines("0")
#                
#        fp.close()

    def generate_all(self):

        len_raw_data = len(self.raw_data)
        fp = open(self.train_data_all_file,'w')
        for i in range(len_raw_data-self.train_number-self.result_number+1):
            for j in range(i,self.train_number+i):
                fp.writelines(str(self.raw_data[j]))
                fp.writelines(" ")
                result_begin = j+1
            fp.writelines("\n")
            if result_begin+self.result_number<=len_raw_data:
                result_end = result_begin+self.result_number
                for k in range(result_begin,result_end):
                    fp.writelines(str(self.raw_data[k]))
                    fp.writelines(" ")
                fp.writelines("\n")
            else:
                fp.writelines("0")

        fp.close()


    def generate_divide(self):

        len_raw_data = len(self.raw_data)

        
        fp = open(self.train_data_all_file,'r')
        data = fp.readlines()
        fp.close()

        #total_group_num = len_raw_data-self.train_number+1  #decrease result number
        total_group_num = len_raw_data-self.train_number-self.result_number+1 #equel result number
        total_file_num = total_group_num-self.group_number+1
        for i in range(total_file_num):
            fp = open("%s/data_%s"%(self.train_data_divide_dir,i),'w')
            fp.write("%s %s %s\n"%(self.group_number,self.train_number,self.result_number))
            for j in range(i*2,(self.group_number+i)*2):
                fp.writelines(str(data[j]))
            fp.close()


if __name__ == "__main__":
    raw_file = "imf2"
    train_data_all = "sin_data_all"
    train_data_dir = "sin_data"
    group_number = 10
    train_number = 50
    result_number = 10
    genfunc = genTraindata(raw_file,train_data_all,train_data_dir,group_number,train_number,result_number) 
    genfunc.generate_all()
    genfunc.generate_divide()
