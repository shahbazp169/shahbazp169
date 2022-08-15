# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 18:42:02 2022

@author: shahb
"""

import sys
import cv2
import numpy as np

# Load image
# im = cv2.imread(sys.path[0]+'/im.png', cv2.IMREAD_GRAYSCALE)
im = cv2.imread('G:/Internship/Akaike assignment/data/train/full_grain_1.jpg', cv2.IMREAD_GRAYSCALE)
H, W = im.shape[:2]

# Remove noise
im = cv2.medianBlur(im, 9)
im = cv2.GaussianBlur(im, (11, 11), 21)

# Find outlines
im = cv2.adaptiveThreshold(
    im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, 2)
top1 = im.copy()

# Fill area with black to find seeds
cv2.floodFill(im, np.zeros((H+2, W+2), np.uint8), (0, 0), 0)
im = cv2.erode(im, np.ones((5, 5)))
top2 = im.copy()

# Find seeds
cnts, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Convert GRAY instances to BGR
top1 = cv2.cvtColor(top1, cv2.COLOR_GRAY2BGR)
top2 = cv2.cvtColor(top2, cv2.COLOR_GRAY2BGR)
im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

# Draw circle around detected seeds
c = 0
for cnt in cnts:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.circle(im, (x+w//2, y+h//2), max(w, h)//2, (c, 150, 255-c), 3)
    c += 5
im=~im

# Print number of seeds
print(len(cnts))

# Save output
image = np.hstack((top1,top2,im))
image = cv2.resize(image, (720,460))
cv2.imwrite('grain.png', image)
cv2.imshow('grain',image)
cv2.waitKey(1)
