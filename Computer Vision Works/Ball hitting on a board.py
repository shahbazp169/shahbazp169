# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:55:12 2022

@author: Shahbaz P
"""
# Importing libraries
import cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder
import pickle
# Importing video
cap = cv2.VideoCapture('G:/Internship/Chaos Assignment/assignmentVideo.mp4')
frameCounter = 0
# corner points of the images for mapping
cornerPoints = [[129, 47], [528, 74], [143, 371], [531, 317]]
# cornerPoints = [[348,113],[1400,173],[383,870],[1416,738]]
# Finding the ball using color finder since colors are different
colorFinder = ColorFinder(False)
hsvVals = {'hmin': 30, 'smin': 94, 'vmin': 154, 'hmax': 63, 'smax': 181, 'vmax': 255}
countHit = 0
imgListBallsDetected = []
hitDrawBallInfoList = []
totalScore = []
timestamps = []

# with open('polygons', 'rb') as f:
#     polygonsWithScore = pickle.load(f)
# print(polygonsWithScore)

# List containing the path of each box and assigned letters

polygonsWithScore = [[[[70, 81], [231, 81], [231, 210], [68, 210]], 'A'], [[[284, 81], [444, 80], [444, 210], [282, 213]], 'B'], [[[492, 79], [653, 78], [655, 210], [491, 211]], 'C'], [[[69, 256], [231, 255], [231, 386], [67, 389]], 'D'], [[[283, 256], [443, 256], [444, 387], [282, 386]], 'E'], [[[493, 255], [656, 254], [655, 387], [492, 386]], 'F']]


# Function defined for mapping
def getBoard(img):
    width, height = int(1500), int(1000)
    pts1 = np.float32(cornerPoints)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (width, height))
    for x in range(4):
        cv2.circle(img, (cornerPoints[x][0], cornerPoints[x][1]), 15, (0, 255, 0), cv2.FILLED)

    return imgOutput

# Function defined for reducing noices and making a perfect masking
def detectColorDarts(img):
    imgBlur = cv2.GaussianBlur(img, (7, 7), 2)
    imgColor, mask = colorFinder.update(imgBlur, hsvVals)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask, 9)
    mask = cv2.dilate(mask, kernel, iterations=4)
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("Image Color", imgColor)
    return mask


# Running the video frame by frame using while loop
while True:
    
    #only 1 time video will be run
    frameCounter += 1
    if frameCounter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frameCounter = 0
        break
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    img = cv2.resize(img, (720, 460))
    imgBoard = getBoard(img)
    imgBoard = cv2.resize(imgBoard, (720, 460))
    mask = detectColorDarts(imgBoard)
    
    # using cvzone for detecting the ball
    imgContours, conFound = cvzone.findContours(imgBoard, mask, 2500)

    if conFound:
        countHit += 1
        if countHit == 7:
            # imgListBallsDetected.append(mask)
            print("Hit Detected")
            countHit = 0
            for polyScore in polygonsWithScore:
                # print(polyScore[1])
                # center = max(imgBoard, key = cv2.contourArea)
                # center = cv2.drawContours(imgBoard, [max(imgContours, key = len)], 0, (0,255,0),2, cv2.LINE_AA, maxLevel=1)
                center = conFound[0]['center']
                # cnts = cv2.findContours(imgBoard.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                # cnts = imutils.grab_contours(cnts)
                # center = max(cnts, key=cv2.contourArea)
                # cv2.circle(imgBoard, center, 5, (255, 0, 255), cv2.FILLED)
                # print(center)
                # center = max(imgContours, key = cv2.contourArea)
                polyScore[0] = np.int32(polyScore[0])
                poly = np.array([polyScore[0]])
                inside = cv2.pointPolygonTest(poly, center, False)
                # print(inside)
                if inside == 1:
                    # print("Yes")
                    # hitDrawBallInfoList.append([conFound[0]['bbox'], conFound[0]['center'], poly])
                    # polyScore[1] = str(polyScore[1])
                    # print(polyScore[1])
                    
                    # timestamp
                    timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
                    # print(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
                    
                    # Ball hitting which box? 
                    totalScore.append(polyScore[1])
                    # print(polyScore[1])
                    print('Ball Hitted on box {} at {} second'.format(polyScore[1],cap.get(cv2.CAP_PROP_POS_MSEC)/1000))
                    # print(totalScore)
    # #                 totalScore += polyScore[1]
    # print(totalScore)
    # imgBlank = np.zeros((imgContours.shape[0], imgContours.shape[1], 3), np.uint8)

    # for bbox, center, poly in hitDrawBallInfoList:
    #     cv2.rectangle(imgContours, bbox, (255, 0, 255), 2)
    #     cv2.circle(imgContours, center, 5, (0, 255, 0), cv2.FILLED)
    #     cv2.drawContours(imgBlank, poly, -1, color=(0, 255, 0), thickness=cv2.FILLED)


    # imgBoard = cv2.addWeighted(imgBoard, 0.7, imgBlank, 0.5, 0)

    # imgBoard,_ = cvzone.putTextRect(imgBoard, f'Total Score: {totalScore}',
    #                               (10, 40), scale=2, offset=20)

    # imgStack = cvzone.stackImages([imgContours, imgBoard], 2, 1)

    # cv2.imwrite('imgBoard.png',imgBoard)
    # cv2.imshow("Image", img)
    # cv2.imshow("Image Board", imgBoard)

    # cv2.imshow("Image Mask", mask)
    # imgContours = cv2.resize(imgContours, (720, 460))
    cv2.imshow("Image Contours", imgContours)

    cv2.waitKey(1)
print('Labels on boxes:{}'.format(totalScore)) 
print('Corresponding time:{}'.format(timestamps)) 
 