# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 11:42:57 2022

@author: Amrutha
"""
x = 0
j = int(input('No. of contact numbers:'))
Contact = {}
while (x<j):
    x = x + 1
    z= input('Name{}:'.format(x))
    d= int(input('Phone number{}:'.format(x)))
    Contact[z] = d
print(Contact)