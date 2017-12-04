#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 17:31:23 2017

@author: RitaRamos
"""

import numpy as np
from random import choice
import math
import operator

#tmbm acerta este, ms é + dificil, isto deve ser pq se calhar n é linear; 
# o perceptron é melhor para dataset lineares...
dataset = np.array([[1, 0.17532942, 4],
                    [2, 0.30256783, 1],
                    [3, 0.27721028, 2],
                    [4, 0.24489247, 3],
                    [5, 0, 5],
                    [6, 0, 5]])

#ele acerta + neste!!! q é mais facil, é so por o 1º peso w + alto!

dataset = np.array([[4, 0.17532942, 4],
                    [1, 0.30256783, 1],
                    [2, 0.27721028, 2],
                    [3, 0.24489247, 3],
                    [5, 0, 5],
                    [5, 0, 5]])



r = [1, 2, 3, 4,5]
w = [0,0]  #igual ao teu numero de features!
b = [0, 0, 0, 0, math.inf]
n_loop=2000


count_corrects=0
for t in range(0, n_loop) :
    
    x = choice(dataset)
    #min_r = [math.inf, 0] # List with [rank, value]
    
    #min_r=[math.inf,5]
    print("x", print(x))
    predict_rank=5
    for i in range(0, len(r)):
        print("dot", np.dot(w, x[:2]))
        print("r[i]",r[i], ": b", b[i])
        
        value = np.dot(w, x[:2])
        if(value < b[i]):
            #dic[i]=r[i]
            predict_rank=r[i]
            break  #podes fazer break pq o rank + baixo é logo o 1º rank q encontras (ate diz no paper)

    print("predicted_rank", predict_rank)  
    real_rank = x[2]
    print("real rank", x[-1])
    
    if (predict_rank != real_rank) :
        print("errei no valor")

        y_r = np.zeros(len(r)-1)
        for i in range(0, len(r)-1) :
            if (real_rank <= r[i]) :
                y_r[i] = -1
            else :
                y_r[i] = 1
        print("\n y_r", y_r)
        T_r = np.zeros(len(r)-1)
        for i in range(0, len(r)-1) :
            
            print("np dop-b",(np.dot(w, x[:2]) -b[i] ))
            print("np dop-b*y",(np.dot(w, x[:2]) -b[i] )*  y_r[i])
            if (  (np.dot(w, x[:2]) -b[i] ) * y_r[i] <= 0) :  #tinhamos mal, era para aquele b em especifico, 
                                                               #tavamos so a chamar min <y
                print("vou ter valor -1  ou 1")
                T_r[i] = y_r[i]
            else :
                print("vou ter valor 0 ")

                T_r[i] = 0
                   
        print("\n T_r[i]", T_r)
        print(" \n np sum", np.sum(T_r) )
        print(" \n np sum", np.sum(T_r)*(x[:2]))

        
        w = w + (np.sum(T_r))*(x[:2])
        
        print("w", w)
        
        for i in range(0, len(r)-1) :
            b[i] = b[i] - T_r[i]
        print("\n b value", b)
    else:
        count_corrects+=1
        print("acertei no valor")
 
print("count corrects",count_corrects )




print("\n \n \n FINAL")
print(" w", w)
print(" b", b)

x=dataset[2]
print("x", x)
predict_rank=5
for i in range(0, len(r)):
    value = np.dot(w, x[:2])
    print("value", value)
    print("r[i]",r[i], ": b", b[i])
    if(value < b[i]):
        print("entrei aqui, é menor")
        predict_rank=r[i]
        break
        #dic[r[i]]=value
           
           

print("predicted rank", predict_rank)
print("predicted prob", np.dot(w, x[:2])- b[i])

#print("value rank", np.dot(w, x[:2]) - )




''' 
values_chosen = [math.inf, 0]
x = np.array([4, 0.24489247, 3])
for i in range(0, len(r)) :
    value = np.dot(w, x[:2]).sum() - b[i]
    print("value", value)
    if (value < 0 and r[i] < values_chosen[0]) :
        values_chosen = [r[i], value]
        
print(values_chosen)

'''