import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('C:\\TFM\\imagenes\\finalRioWS-35880-897.jpeg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,25,0.5,120)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

cv2.imwrite("C:\\TFM\\ws1\\test_corners\\results\\goog_f_to_track_100_0.05_120.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])