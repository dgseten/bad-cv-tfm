import cv2
import numpy as np

img = cv2.imread('C:\\TFM\\imagenes\\finalRioWS-35880-897.jpeg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,100,200,apertureSize = 3)


minLineLength = 100
maxLineGap = 30
lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength,maxLineGap)
if not lines is None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite("C:\\TFM\\ws1\\test_hought\\results\\postes\\houghP_canny_5.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])