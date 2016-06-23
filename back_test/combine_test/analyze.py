import numpy as np

for num in range(0,100,2):
    with open("log",'r') as f:
        lines = f.readlines()
        #line = lines[558944+num]
        line = lines[-1]
        line = line[1:-2]
        line = line.split(",")
        newline = []
        for i in line:
            newline.append(float(i))
    
    c=0
    c1=0
    for i in newline:
        c+=1
        if i>0:
            c1+=1
    
    
    
    
    print float(c1)/c


