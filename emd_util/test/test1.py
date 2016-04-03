
import numpy as np


a = [1,2,3,4,5,6]
ma = []
for i in range(3):
    ma.append(0)

for i in range(3,len(a)):
    mean = np.mean(a[i-2:i+1])
    ma.append(mean)
print ma
