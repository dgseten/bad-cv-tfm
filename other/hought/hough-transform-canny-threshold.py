import cv2
import numpy as np

img = cv2.imread('C:\\TFM\\imagenes\\finalRioWS-35880-897.jpeg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh2 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

cv2.imwrite("C:\\TFM\\ws1\\test_hought\\results\\hough_thresd.jpg", thresh2, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
edges = cv2.Canny(thresh2,100,200,apertureSize = 3)
cv2.imwrite("C:\\TFM\\ws1\\test_hought\\results\\hough_thresd_canny.jpg", edges, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


lines = cv2.HoughLines(edges,2,np.pi/180,250)
for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite("C:\\TFM\\ws1\\test_hought\\results\\hough_thresd_canny__100_200_1_180_250.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])