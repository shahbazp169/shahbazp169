# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 23:25:28 2022

@author: Amrutha
"""
import math
for i in range(0,20000):
    u = int(i)  
    m = [int(a) for a in str(i)]
    
    if len(m)==1:
        w = m[0]
        j = (math.pow(w, 3))
        if j==u:
            print(u,end=' ')
    if len(m)==2:
        w = m[0]
        x = m[1]
        j = (math.pow(w, 3))+(math.pow(x, 3))
        if j==u:
            print(u,end=' ')
    elif len(m)==3:
        w = m[0]
        x = m[1]
        y = m[2]
        j = (math.pow(w, 3))+(math.pow(x, 3))+(math.pow(y, 3))
        if j==u:
            print(u,end=' ')
    elif len(m)==4:
        w = m[0]
        x = m[1]
        y = m[2]
        z = m[3]
        j = (math.pow(w, 3))+(math.pow(x, 3))+(math.pow(y, 3))+(math.pow(z, 3))
        if j==u:
            print(u,end=' ')
        
