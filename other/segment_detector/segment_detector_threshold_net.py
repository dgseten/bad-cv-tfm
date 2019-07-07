import cv2
import numpy as np

img = cv2.imread('C:\\TFM\\imagenes\\finalRioWS-35880-897.jpeg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh3 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)


#Create default parametrization LSD
lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV,_scale=0.3,_density_th=0.3)

#Detect lines in the image
lines = lsd.detect(thresh3)[0] #Position 0 of the returned tuple are the detected lines

#Draw detected lines in the image
img = cv2.imread('C:\\TFM\\imagenes\\finalRioWS-35880-897.jpeg')
drawn_img = lsd.drawSegments(img,lines)

#Show image
cv2.imwrite("C:\\TFM\\ws1\\test_segments\\lsd_segments_threshold_net_lines_0.jpg", drawn_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])