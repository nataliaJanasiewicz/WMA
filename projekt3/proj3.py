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

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1,None)
    kp2, des2 = sift.detectAndCompute(gray2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = gray1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        gray2 = cv.polylines(gray2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
    
    draw_params = dict(matchColor = (255,30,220), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img3 = cv.drawMatches(img1,kp1,frame,kp2,good,None,**draw_params)
    img3 = cv.resize(img3, None, fx=0.5, fy=0.5)
    #out.write(img3)
    #plt.imshow(img3, 'gray'),plt.show()
    #img3 = cv.resize(img3, None, fx=0.7, fy=0.7)
    cv.imshow('frame',img3)
    if cv.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()  # zamknięcie okna