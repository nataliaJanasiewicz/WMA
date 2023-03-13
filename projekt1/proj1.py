import cv2 as cv
import numpy as np
cap = cv.VideoCapture('movingball.mp4')
kernel = np.ones((5,5),np.uint8)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # lower boundary RED color range values; Hue (0 - 20)
    lower1 = np.array([0, 100, 10])
    upper1 = np.array([10, 255, 255])
 
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])
    
    lower_mask = cv.inRange(hsv, lower1, upper1)
    upper_mask = cv.inRange(hsv, lower2, upper2)
 
    full_mask = lower_mask + upper_mask;

    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= full_mask)
    opening = cv.morphologyEx(res, cv.MORPH_OPEN, kernel)
    opAndClos = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    
    cv.imshow('frame',frame)
    cv.imshow('mask',full_mask)
    cv.imshow('res',opAndClos)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()