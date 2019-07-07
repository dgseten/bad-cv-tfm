import cv2
import numpy as np

img = cv2.imread('C:\\TFM\\imagenes\\finalRioWS-35880-897.jpeg',0)
edges = cv2.Laplacian(img, cv2.CV_8UC1,ksize=1)
cv2.imwrite("C:\\TFM\\ws1\\test_hought\\results\\hough_laplacian_edges.jpg", edges, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

lines = cv2.HoughLines(edges,2,np.pi/180,900)
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


cv2.imwrite("C:\\TFM\\ws1\\test_hought\\results\\hough_laplacian_k1_th_900.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])