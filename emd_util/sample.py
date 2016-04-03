
import numpy as np

def system_sample(data, step, direction):

    sample_data = []
    if direction == "start":
        for i in range(0,len(data),step):
            sample_data.append((i,data[i]))
        return sample_data
    elif direction == "end":
        for i in range(len(data)-1,-1,-step):
            sample_data.append((i,data[i]))
        sample_data.sort(key=lambda x:x[0])
        return sample_data

    else:
        raise Exception("need a direction")



if __name__ == "__main__":

    
    print system_sample([1,2,3,4,5],2,"start")
    print system_sample([1,2,3,4,5],2,"end")
    
