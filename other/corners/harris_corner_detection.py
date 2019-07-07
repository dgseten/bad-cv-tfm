import cv2
import numpy as np

img = cv2.imread('C:\\TFM\\imagenes\\finalRioWS-35880-897.jpeg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]


cv2.imwrite("C:\\TFM\\ws1\\test_corners\\results\\cornersHarris_100_200_1_180_250.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])