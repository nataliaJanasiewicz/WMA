import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

cap = cv.VideoCapture('sawmovie.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
MIN_MATCH_COUNT = 70
sift = cv.SIFT_create()
sift2 = cv.SIFT_create()

gray1 = cv.imread('saw1.jpg', cv.IMREAD_GRAYSCALE)# queryImage
gray1 = cv.resize(gray1, None, fx=0.5, fy=0.5)

gray4 = cv.imread('saw1.jpg', cv.IMREAD_GRAYSCALE)# queryImage
gray4 = cv.resize(gray4, None, fx=0.5, fy=0.5)

img1 = cv.imread('saw1.jpg')
img1 = cv.resize(img1, None, fx=0.5, fy=0.5)

gray3 = cv.imread('saw3.jpg', cv.IMREAD_GRAYSCALE)# queryImage
gray3 = cv.resize(gray3, None, fx=0.5, fy=0.5)
img3 = cv.imread('saw3.jpg')
img3 = cv.resize(img3, None, fx=0.5, fy=0.5)

kp1, des1 = sift.detectAndCompute(gray1,None)
kp4, des4 = sift2.detectAndCompute(gray4,None)

while(1):
    _, frame = cap.read()
    if frame is None:
        break
    gray2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # trainImage

    kp2, des2 = sift.detectAndCompute(gray2,None)
    kp3, des3 = sift2.detectAndCompute(gray2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    index_params2 = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params2 = dict(checks = 50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    flann3 = cv.FlannBasedMatcher(index_params2, search_params2)
    matches = flann.knnMatch(des1,des2,k=2)
    matches3 = flann3.knnMatch(des4,des3,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    good3 = []

    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good3.append(m)

    matches = flann.knnMatch(des1,des2,k=2)
    flaga = 1
    if len(good)>=MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = gray1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        frame = cv.polylines(frame,[np.int32(dst)],True,255,3, cv.LINE_AA)
        print( "Matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )

    elif len(good3)>=MIN_MATCH_COUNT:
        #flaga =3
        #src_pts = np.float32([ kp1[m.queryIdx].pt for m in good3 ]).reshape(-1,1,2)
       # dst_pts = np.float32([ kp3[m.trainIdx].pt for m in good3 ]).reshape(-1,1,2)
       # M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
       # matchesMask = mask.ravel().tolist()
      #  h,w = gray1.shape
        #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        #pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1,1,2)
       # dst = cv.perspectiveTransform(pts,M)
       # frame = cv.polylines(frame,[np.int32(dst)],True,255,3, cv.LINE_AA)
        print( "Matches are found 3- {}/{}".format(len(good3), MIN_MATCH_COUNT) )
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
    
    draw_params = dict(matchColor = (255,50,200), 
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    
    img4 = cv.drawMatches(img1,kp1,frame,kp2,good,None,**draw_params)
    if flaga == 3:
        img4 = cv.drawMatches(img1,kp1,frame,kp3,good3,None,**draw_params)
    cv.imshow('frame',img4)
    if cv.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()  # zamknięcie okna