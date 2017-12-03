import numpy as np
from random import choice
import math

dataset = np.array([[1, 0.17532942, 4],
                    [2, 0.30256783, 1],
                    [3, 0.27721028, 2],
                    [4, 0.24489247, 3],
                    [5, 0, 5],
                    [6, 0, 5]])
 

r = [1, 2, 3, 4, 5]
w = 0
b = [0, 0, 0, 0, math.inf]

for t in range(0, len(dataset)) :
    x = dataset[t]
    min_r = [math.inf, 0] # List with [rank, value]
    for i in range(0, len(r)):
        print("aa", np.dot(w, x[:2]))
        print("bb", b[i])
        value = np.dot(w, x[:2]).sum() - b[i]
        if (value < 0 and r[i] < min_r[0]) :
            min_r = [r[i], value]
    predict_rank = min_r[0]
    real_rank = x[2]
    
    if (predict_rank != real_rank) :
        min_value = min_r[1]

        y_r = np.zeros(len(r)-1)
        for i in range(0, len(r)-1) :
            if (real_rank <= r[i]) :
                y_r[i] = -1
            else :
                y_r[i] = 1
                
        T_r = np.zeros(len(r)-1)
        for i in range(0, len(r)-1) :
            if (min_value * y_r[i] <= 0) :
                T_r[i] = y_r[i]
            else :
                T_r[i] = 0
        w = w + (np.sum(T_r) * x[:2]).sum()
        
        for i in range(0, len(r)-1) :
            b[i] = b[i] - T_r[i]
 
 
values_chosen = [math.inf, 0]
x = np.array([4, 0.24489247, 3])
for i in range(0, len(r)) :
    value = np.dot(w, x[:2]).sum() - b[i]
    print("value", value)
    if (value < 0 and r[i] < values_chosen[0]) :
        values_chosen = [r[i], value]
        
print(values_chosen)