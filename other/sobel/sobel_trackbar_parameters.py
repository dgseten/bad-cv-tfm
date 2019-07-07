import cv2
import numpy as np
import time

def nothing(x):
    pass

# Create a black image, a window
img = cv2.imread('C:\\TFM\\imagenes\\finalRioWS-35880-897.jpeg',0)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)

# create trackbars for color change
cv2.createTrackbar('ksize','image',3,31,nothing)



# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)

last_state = 0
ksize = 5

while 1:
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    cv2.imshow('image',sobelx)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    ksize = cv2.getTrackbarPos('ksize','image')
    if ksize % 2 ==0:
        ksize += 1
    print(ksize)
    s = cv2.getTrackbarPos(switch,'image')

    if s == 0:
        img[:] = 0
        last_state = 0
    elif last_state==0:
        img = cv2.imread('C:\\TFM\\imagenes\\finalRioWS-35880-897.jpeg', 0)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=ksize)
        cv2.imshow('image', sobelx)
        last_state =1

    time.sleep(0.1)

cv2.destroyAllWindows()
