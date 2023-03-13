import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
#0- kamera wbudowana 1- USB chyba

if not cap.isOpened():
    print('nie moge otworzyc kamery')
    exit()
while True:
    #Capture fram-by-frame
    ret,frame = cap.read()
    #if frame is read correctly ret is True
    if not ret:
        print("cant recive frame")
        break
    #Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    color = cv.cvtColor(frame,1)

    #display the resulting frame
    cv.imshow('frame',gray)
    #imshow('frame',color)

    if cv.waitKey(1) == ord('q'):
        break
#hen everything done
cap.release()
cv.destroyAllWindows()