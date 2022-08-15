# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 16:42:40 2022

@author: Amrutha
"""
e = int(input("Prime number's from:"))
r = int(input("Prime number's upto:"))
import numpy as np
PrimeNumbers = []
for i in range(e,r):
    set=[]
    for j in range(1,r):
        x = i%j
        if x==0:
            set.append(i)
        # print(set,end='___')
    if len(set)==2:
        # print(set,end='  ')
        for k in set:
            PrimeNumbers.append(k)
a = np.array(PrimeNumbers)  
print(np.unique(a))                         
             