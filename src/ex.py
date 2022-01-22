import cv2 as cv
import sys
from matplotlib import pyplot as plt
img = cv.imread("storch.jpg")


if img is None:
    sys.exit("Could not read the image.")

edges = cv.Canny(img,50,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


# cv.imshow("Display window", edges)
# k = cv.waitKey(0)
# if k == ord("s"):
#     print("s gedrückt...")
