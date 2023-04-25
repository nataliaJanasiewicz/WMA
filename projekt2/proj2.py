import cv2 as cv
import numpy as np
import os
from os import listdir

def chceck(x,y,x0,y0,x1,y1):
    if( x0 < x < x1 and y0 < y < y1):
        #print('w srodku')
        return 1
    else: 
        #print('na zewnatrz')
        return 0

folder_dir = '/Users/nacia/Desktop/WMA/projekt2/trays'

for image in os.listdir(folder_dir):
    if(image.endswith('.jpg')):
        print(image)
        frameColour = (255,170,255)
        img = cv.imread('trays/'+ image, cv.IMREAD_COLOR)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        #img for circles
        img_cir = cv.GaussianBlur(img,(5,5),0)
        img_cir = cv.medianBlur(img_cir,5)
        #img_cir = cv.dilate(img_cir,np.ones((2,2),np.uint8))
        img_cir = cv.cvtColor(img_cir,cv.COLOR_BGR2GRAY)
        kernel = np.ones((5,5),np.float32)/25
        dst = cv.filter2D(img_cir,-1,kernel)

        #img for lines
        #lower1 = np.array([0, 150, 180])
        #upper1 = np.array([30, 255, 255])
        #low_up = cv.inRange(hsv, lower1, upper1)
        #imask = cv.bitwise_and(img,img, mask= low_up)
        edges = cv.Canny(gray,50,150,apertureSize = 3)

        #kernelcl = np.ones((17,17),np.uint16)
        #clos = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernelcl)
        #gray = cv.cvtColor(dst,cv.COLOR_BGR2GRAY)


        #using lines
        lines = cv.HoughLinesP(edges ,1,np.pi/180,100, minLineLength=50, maxLineGap=10)
        #print(lines)

        x0 = lines[0][0][0]
        x1 = lines[0][0][0]
        y0 = lines[0][0][1]
        y1 = lines[0][0][1]
        x = []
        y = []

        for line in lines:
            x.append(line[0][0])
            x.append(line[0][2])
            y.append(line[0][1])
            y.append(line[0][3])

        x0 = min(x)
        x1 = max(x)
        y0 = min(y)
        y1 = max(y)


        cv.line(img,(x0,y0),(x1,y0),frameColour,5)
        cv.line(img,(x0,y0),(x0,y1),frameColour,5)
        cv.line(img,(x1,y1),(x1,y0),frameColour,5)
        cv.line(img,(x1,y1),(x0,y1),frameColour,5)

        cv.circle(img, (x0, y0), 5, (255, 0, 0), -1)
        cv.circle(img, (x1, y1), 5, (0, 255, 0), -1)

        for line in lines:
            x2,y2,x3,y3 = line[0]
            cv.line(img,(x2,y2),(x3,y3),(0,255,0),2)

        circles = cv.HoughCircles(dst,cv.HOUGH_GRADIENT,1,20, param1=50,param2=30,minRadius=20,maxRadius=40)
        circles = np.uint16(np.around(circles))
        radiusSize = []
        for i in circles[0,:]:
            radiusSize.append(i[2])

        #criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        #flags = cv.KMEANS_RANDOM_CENTERS
        #compactness,labels,centers = cv.kmeans(radiusSize,2,None,criteria,10,flags)
        #A = radiusSize[labels==0]
        #B = radiusSize[labels==1]
        #print(A)
        #print(B)
        sumIn = 0
        countIn =0
        sumOut = 0
        countOut =0

        for i in circles[0,:]:
            # draw the outer circle
            zm = 0.05
            radC = (255,0,0)
            if(i[2] >= max(radiusSize)-3):
                radC = (0,255,255)
                zm = 5
            cv.circle(img,(i[0],i[1]),i[2],radC,2)
            # draw the center of the circle
            centerC = (0,255,0)
            if (chceck(i[0],i[1],x0,y0,x1,y1) == 0):
                centerC = (0,0,255)
                sumOut = sumOut + zm
                countOut = countOut + 1
            else:
                sumIn = sumIn + zm
                countIn = countIn +1
            cv.circle(img,(i[0],i[1]),2,centerC,3)

        print('w srodku jest: ', countIn, ' monet, suma: ', round(sumIn,2))
        print('na zewnatrz jest: ', countOut, ' monet, suma: ', round(sumOut,2))
        suma = sumIn + sumOut
        print('suma: ', round(suma,2))
        cv.imshow('img',img)
        #cv.imshow("Linie", img)
        #cv.imshow("mask", clos)
        #cv.imshow("edges", edges2)
        k = cv.waitKey(0)
#k = cv.waitKey(0)
