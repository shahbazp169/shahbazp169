# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 10:45:12 2022


"""
b=int(input('Fibonacci series upto:'))
def f(x):
    if x==0:
        return(0)
    elif x==1:
         return(1)
    elif x>=2:
         return(f(x-1)+f(x-2))
x=0     
while x<b:
    x=x+1
    print('{}'.format(f(x)), end =" ")
     
a=int(input('Fibonacci number:'))
print(f(a))



