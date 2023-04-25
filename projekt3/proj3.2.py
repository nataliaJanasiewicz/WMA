import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

cap = cv.VideoCapture('sawmovie.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
MIN_MATCH_COUNT = 10

gray1 = cv.imread('saw1.jpg', cv.IMREAD_GRAYSCALE)# queryImage
img1 = cv.imread('saw1.jpg')

# Initiate video writer
#fourcc = cv.VideoWriter_fourcc(*'mp4v')
#out = cv.VideoWriter('output.avi', fourcc, 20.0, (frame_width,frame_height))


while(1):
    _, frame = cap.read()
    if frame is None:
        break
    gray2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # trainImage

    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(gray1,None)
    kp2, des2 = orb.detectAndCompute(gray2,None)
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    img3 = cv.drawMatches(img1,kp1,frame,kp2,matches[:20],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv.imshow('frame',img3)
    if cv.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()  # zamknięcie okna