import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("C:\\Users\\Diego\\Desktop\\img040.jpg")
rows,cols,ch = img.shape

pts1 = np.float32([[10,145],[2196,109],[61,2748],[2245,2709]])
pts2 = np.float32([[10,145],[2196,145],[10,2748],[2196,2748]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(2250,2800))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()

cv2.imwrite('C:\\Users\\Diego\\Desktop\\img040_out.jpg',dst)