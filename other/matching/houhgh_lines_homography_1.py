import cv2
import numpy as np


img = cv2.imread('C:\\TFM\\imagenes\\finalRioWS-35880-897.jpeg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh3 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
cv2.imwrite("C:\\TFM\\ws1\\test_final_alg\\threshold.jpg", thresh3, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
edges = cv2.Canny(thresh3,100,200,apertureSize = 3)
#cv2.imwrite("C:\\TFM\\ws1\\test_final_alg\\canny.jpg", edges, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


"""
line ecuation: y = m*x +n
polar:

    punto(3,6): indica que estamos a una distancia de 3 desde el origen de coordenadas con un angulo de 6

    p = x * cos(o) + y* sin(o)
    p =

"""

lines = cv2.HoughLines(thresh3,1,np.pi/180,400)
for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 10000*(-b))
        y1 = int(y0 + 10000*(a))
        x2 = int(x0 - 10000*(-b))
        y2 = int(y0 - 10000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite("C:\\TFM\\ws1\\test_final_alg\\hough_canny_100_200_1_180_250.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])