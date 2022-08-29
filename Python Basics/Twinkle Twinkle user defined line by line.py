# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 12:35:32 2022


"""
c = int(input('Input:'))
a = c//4
r = c%4

    
def space(x):
    return('   '*x)
y = 0
while y<a:
    y = y+1
    print('Twinkle Twinkle little star \n{}How I wonder what you are \n{}Up above the world so high \n{}Like a dimond in the sky'.format(space(1),space(2),space(2)))
if r == 1:
    print('Twinkle Twinkle little star')
elif r == 2:
    print('Twinkle Twinkle little star \n{}How I wonder what you are'.format(space(1)))
elif r == 3:
    print('Twinkle Twinkle little star \n{}How I wonder what you are \n{}Up above the world so high'.format(space(1),space(2)))


