import numpy as np
from neuro_predict import neuroprediction


def load_file_data(file_data):
    return np.loadtxt(file_data)


def format_data(data):
    return [2*((float(data[i])-min(data))/(max(data)-float(min(data))))-1 for i in range(len(data))]





if __name__=="__main__":
    ma_train_file="ma_train"
    train_file="ma_train_data_set"
    data_mat = load_file_data(ma_train_file)
    row_number = data_mat.shape[0] 
    col_number = data_mat.shape[1] 
    col_after_format=[]
    for i in range(col_number-1):
        col_list = list(data_mat[:,i])
        #col_after_format.append(format_data(col_list))
        col_after_format.append((col_list))
    new_array = np.array(col_after_format)
    array_after_format = new_array.transpose()
    #print array_after_format
    label = list(data_mat[:,-1])
    fp=open(train_file,'w')

    fp.write(str(row_number))
    fp.write(" ")
    fp.write(str(col_number-1))
    fp.write(" ")
    fp.write("1")
    fp.write("\n")


    for i in range(row_number):
        for j in range(col_number-1):
            fp.write(str(array_after_format[i][j]))
            fp.write(" ")
        fp.write("\n")
        fp.write(str(label[i]))
        fp.write("\n")
    fp.close()


    my_nn = neuroprediction(train_file,"net_file",4,10,0,1)
    my_nn.train_nn()
    
    

        
        
    
