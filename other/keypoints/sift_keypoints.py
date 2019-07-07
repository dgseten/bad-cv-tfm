import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('C:\\TFM\\imagenes\\finalRioWS-35880-897.jpeg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,100,200,apertureSize = 3)

cv2.imwrite("C:\\TFM\\ws1\\test_keypoints\\canny_sift_1.jpg", edges, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
sift = cv2.xfeatures2d.SIFT_create()
(kps, descs) = sift.detectAndCompute(edges, None)

img=cv2.drawKeypoints(img,kps,img)

cv2.imwrite("C:\\TFM\\ws1\\test_keypoints\\sift_1.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])