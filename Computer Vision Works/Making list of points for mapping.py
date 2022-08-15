# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 23:18:47 2022

@author: Shahbaz P
"""
# Importing libraries
import cv2
import numpy as np
import pickle

cornerpoints = []  # all the polygons and their points
path = []  # current single polygon

# Read Image
img = cv2.imread('G:/Internship/Chaos Assignment/imgmain.png')
img = cv2.resize(img, (720, 460))

# Function to append the pixel points when clicked by mouse
def mousePoints(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        path.append([x, y])


while True:
    # To draw the path line
    for point in path:
        cv2.circle(img, point, 7, (0, 0, 255), cv2.FILLED)

    pts = np.array(path, np.int32).reshape((-1, 1, 2))
    img = cv2.polylines(img, [pts], True, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mousePoints)
    key = cv2.waitKey(1)
    
    # if loop for making the polygon complete
    if key == ord('q'):
        cornerpoints.append([path])
        print("Cornerpoints: ", len(cornerpoints))
        path = []
        
    # if loop for finishing the code
    if key == ord("p"):
        with open('cornerpoints', 'wb') as f:
            print(cornerpoints[0][0])
            pickle.dump(cornerpoints, f)
        break