import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
cap = cv.VideoCapture('movingball.mp4')
kernel = np.ones((5,5),np.uint8)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # lower boundary RED color range values; Hue (0 - 20)
    lower1 = np.array([0, 100, 10])
    upper1 = np.array([10, 255, 255])
 
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])

    #light pink
    #lower3 = np.array([150,10,20])
    #upper3 = np.array([170,70,255])
    
    mask1 = cv.inRange(hsv, lower1, upper1)
    mask2 = cv.inRange(hsv, lower2, upper2)
    #mask3 = cv.inRange(hsv, lower3, upper3)

    full_mask = mask1 + mask2;

    # Bitwise-AND mask and original image
    mask = cv.bitwise_and(frame,frame, mask= full_mask)
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    dilation = cv.dilate(opening,kernel,iterations = 1)
    opAndClos = cv.morphologyEx(dilation, cv.MORPH_CLOSE, kernel)

    edges = cv.Canny(opAndClos,100,200)
    edges = cv.dilate(edges,kernel,iterations = 1)
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    gray_img = cv.cvtColor(opAndClos, cv.COLOR_BGR2GRAY)

    # convert the grayscale image to binary image
    ret,thresh = cv.threshold(gray_img,127,255,0)
 
    # calculate moments of binary image
    M = cv.moments(thresh)
 
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
 
    # put text and highlight the center
    cv.circle(frame, (cX, cY), 5, (255, 100, 100), -1)
    cv.putText(frame, "centroid", (cX - 25, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)
    
    # display the image
    cv.imshow('edge',edges)
    cv.imshow('mask',full_mask)
    cv.imshow('res',opAndClos)
    cv.imshow("Image", frame)
    
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()