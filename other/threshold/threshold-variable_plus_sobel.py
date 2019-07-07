import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('C:\\TFM\\imagenes\\finalRioWS-35880-897.jpeg',0)
ret,thresh1 = cv2.threshold(img,240,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
ret,thresh3 = cv2.threshold(img,160,255,cv2.THRESH_BINARY)
ret,thresh4 = cv2.threshold(img,140,255,cv2.THRESH_BINARY)
ret,thresh5 = cv2.threshold(img,120,255,cv2.THRESH_BINARY)



titles = ['Original Image','240','200','160','140','120']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(cv2.Sobel(images[i], cv2.CV_64F, 1, 1, ksize=5),'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()