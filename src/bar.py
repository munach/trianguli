import cv2
import numpy as np 
# import matplotlib.pyplot as plt 

def callback(x):
    pass

def analyse(name, img):
    if (len(img.shape) == 2):
        print(name, "Grey:   Size width", orig.shape[1], "height", orig.shape[0])
    else:
        print(name, "Colors: Size width", orig.shape[1], "height", orig.shape[0], "channels", orig.shape[2])


orig = cv2.imread('unke.jpg') #read image as grayscale
analyse("orig", orig)
# img = cv2.resize(orig, (500, 400))

grey = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
analyse("grey", grey)

img = grey

canny = cv2.Canny(img, 85, 255) 
canny1 = canny

cv2.namedWindow('image') # make a window with name 'image'
cv2.createTrackbar('L', 'image', 0, 255, callback) #lower threshold trackbar for window 'image
cv2.createTrackbar('U', 'image', 0, 255, callback) #upper threshold trackbar for window 'image
cv2.createTrackbar('A', 'image', 0, 2, callback) #upper threshold trackbar for window 'image

while(1):
    
    numpy_horizontal_concat = np.concatenate((img, canny, canny1), axis=1) # to display image side by side
    cv2.imshow('image', numpy_horizontal_concat)
    k = cv2.waitKey(10) & 0xFF
    if k == 27: #escape key
        break
    l = cv2.getTrackbarPos('L', 'image')
    u = cv2.getTrackbarPos('U', 'image')
    a = cv2.getTrackbarPos('A', 'image')
    
    # aperaturSize can be 3, 5, or 7
    canny = cv2.Canny(img, l, u, apertureSize=a*2+3)
    canny1 = cv2.Canny(img, l, u, apertureSize=a*2+3, L2gradient=True)
    

cv2.destroyAllWindows()