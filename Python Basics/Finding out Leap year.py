# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 19:11:58 2022

@author: Amruthamol N A
"""
x = int(input('Type the year:'))

if x%4==0:
    if x%100==0:
        if x%400==0:
            print('{} year is a leap year')
        else:
            print('{} year is not a leap year')
    else:
        print('{} year is a leap year')
else:
    print('{} year is not a leap year')