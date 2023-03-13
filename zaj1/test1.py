import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("/Users/nacia/Desktop/IMG_6123.jpg")
assert img is not None, "nie ma pliku :("

b,g,r = cv.split(img)
img = cv.merge((r,g,b))

cv.rectangle(img,(220,110),(165,50),(0,255,0),3)
cv.rectangle(img,(220,110),(165,50),(0,255,0),3)
cv.circle(img,(105,250),25,(0,0,255),-1)
cv.circle(img,(297,255),20,(0,0,255),-1)

#img[30:100,40:150,0:2] = 0
#bierzemy cały zakres składowych do : i składowej 2 -> (r,g,b) -> i przypisujemy jej 0
#0:2 -> wyrzucamy pierwsze 2 czyli zostaje tylko składowa b
#[1,2] -> zostaje tylko 0 wiec r 
#[20:130,50:200] - obszar X i Y
#[20:-1] -> od 20 do konca

print(type(img))
print(img.shape)

plt.imshow(img)
plt.show()