# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 13:00:25 2022


"""
import math
x = int(input('Triangle 1 and Rectangle 2:'))
if x==1:#area of any triangle whose sides are given
    a = int(input('Side 1:'))
    b = int(input('Side 2:'))
    c = int(input('Side 3:'))
    s=(a+b+c)/2
    print(math.sqrt(s*(s-a)*(s-b)*(s-c)))
elif x==2:#area of any rectangle whose side are given
    a = int(input('Side 1:'))
    b = int(input('Side 2:'))  
    
    print(a*b)
