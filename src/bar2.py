# from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from lines import intersects

max_lowThreshold = 100

window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3
nonzero = None   
scale = 0.3


class Triangle: 
    def __init__(self, src):
        self.src = src
        self.src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        self.src_hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
        self.img = src.copy()
        self.nonzero = None             # koordinaten von canny
        self.detected_edges = None      # canny
        self.points = []
        self.lines = []

    def reset(self):
        self.img = src.copy()
        self.points = []
        self.lines = []


    def doCanny(self, val):
        low_threshold = val
        img_blur = cv.blur(self.src_gray, (3,3))
        self.detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
        # indices = np.where(detected_edges != [0])
        self.nonzero = cv.findNonZero(self.detected_edges)


    def find_nearest_white(self, target):
        distances = np.sqrt((self.nonzero[:,:,0] - target[0]) ** 2 + (self.nonzero[:,:,1] - target[1]) ** 2)
        nearest_index = np.argmin(distances)
        p = self.nonzero[nearest_index][0]
        pp = (p[0], p[1])
        return pp

    def find_nearest(self, point, max):
        """returns nearest point withn max (pixels)"""
        p = self.find_nearest_white(point)
        if (sqrt( (p[0]-point[0]) ** 2 + (p[1]-point[1]) ** 2) > max):
            return point
        else: 
            return p


    def addPoint(self, p):
        self.points.append(p)

    def addLine(self, p, q):
        self.lines.append((p, q))


    def linePossible(self, p, q):
        for line in self.lines: 
            if intersects((p, q), line): 
                print('intersects:', (p, q), "with", line)
                return False
        print('no intersection found')
        return True
        



    def threshold(self, low, high):
        th1 = cv.inRange(self.src_gray, low, high)
        # ret,th1 = cv.threshold(self.src_gray,low,high,cv.THRESH_BINARY_INV)
        return th1
    
    def getavg(self, a, b, c): 
        """ reuturs avg. color of triangle a b c"""    
        mask = np.zeros(self.img.shape[:2], dtype="uint8")
        triangle_cnt = np.array( [a, b, c] )
        cv.drawContours(mask, [triangle_cnt], 0, 255, -1)       # Farbe = 255
        masked = cv.bitwise_and(self.img, t.img, mask=mask)
        avg_color_per_row = np.average(masked, axis=0)
        return np.average(avg_color_per_row, axis=0)

    def getColorDelta(self, p, q):
        """ returns the delta of the colors on both sides of (p,q) """
        m = (int((p[0]+q[0]) / 2), int((p[1]+q[1]) / 2))    # middle point
        d = (int((p[0]-q[0])/4), int((p[1]-q[1])/4))    # vector with 1/4 length
        m1 = (m[0] + d[1], m[1]-d[0])       # add 90 Degree d
        m2 = (m[0] - d[1], m[1]+d[0])

        c1 = self.getavg(p, q, m1)
        c2 = self.getavg(p, q, m2)
        # cv.line(t.img, p, q, (255,0,0), 1)
        # cv.line(t.img, p, m1, (255,0,0), 1)
        # cv.line(t.img, p, m2, (255,0,0), 1)
        # cv.line(t.img, q, m1, (255,0,0), 1)
        # cv.line(t.img, q, m2, (255,0,0), 1)
        return np.sum(np.abs(c1 - c2))
        



parser = argparse.ArgumentParser(description='Code for Canny Edge Detector tutorial.')
parser.add_argument('--input', help='Path to input image.', default='storch.jpg')

args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image: ', args.input)
    exit(0)

t = Triangle(src)

def CannyThreshold(val):
    t.doCanny(val)
    dst = cv.resize(t.detected_edges, (int(src.shape[1] * scale), int(src.shape[0] * scale)))
    cv.imshow("canny", dst)



def drawline(p, q):
    m = (int((p[0]+q[0]) / 2), int((p[1]+q[1]) / 2))    # middle point
    cv.line(t.img, p, q, (255,0,0), 1)
    d = (int((p[0]-q[0])/4), int((p[1]-q[1])/4))    # vector with 1/4 length
    m1 = (m[0] + d[1], m[1]-d[0])       # add 90 Degree d
    m2 = (m[0] - d[1], m[1]+d[0])
    cv.line(t.img, m1, m2, (255,0,0), 1)

    mask = np.zeros(t.img.shape[:2], dtype="uint8")
    triangle_cnt = np.array( [p, q, m1] )
    cv.drawContours(mask, [triangle_cnt], 0, 255, -1)       # Farbe = 255
    masked = cv.bitwise_and(t.img, t.img, mask=mask)
    avg_color_per_row = np.average(masked, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    print(avg_color)

    cv.imshow(window_name, masked)

# myimg = cv2.imread('image.jpg')
# avg_color_per_row = numpy.average(myimg, axis=0)
# avg_color = numpy.average(avg_color_per_row, axis=0)
# print(avg_color)


def line():
    drawline(t.points[0], t.points[1])

# mouse callback function
def draw_circle(event,x,y,flags,param):
    # print("mouse:" + str(event))
    if event == cv.EVENT_LBUTTONUP:
        # print("Mouse " + str(x) + ", " + str(y))
        onedge = t.find_nearest_white((x, y))
        # print("Edge:" + str(onedge))
        cv.circle(t.img, onedge,2,(255,0,0),-1)
        cv.imshow(window_name, t.img)
        t.addPoint(onedge)
        # if (len(t.points) == 2):
        #     t.colors(t.points[0], t.points[1])
        #     cv.imshow(window_name, t.img)
            


def draw_histogramm():
    #for i, col in enumerate(['b', 'g', 'r']):
    #    hist = cv.calcHist([t.src], [i], None, [256], [0, 256])
    #    plt.plot(hist, color = col)
    #    plt.xlim([0, 256])
    histogram = cv.calcHist([t.src_gray], [0], None, [256], [0, 256])
    y = np.convolve(histogram.flatten(), np.ones(20)/20, mode='valid')
    hp1 = np.gradient(histogram.flatten())
    hp1 = np.gradient(y)
    hp2 = np.gradient(hp1)
    # plt.plot(histogram, color='k')
    # plt.plot(y, color='r')
    plt.plot(hp1, color = 'y')
    plt.plot(hp2, color = 'g')
    low = 0
    maximum = histogram[0][0]
    minimum = histogram[0][0]
    if (hp2[0] >= 0): 
        up = True
    else: 
        up = False 
    for (i, val) in enumerate(hp2):
        # print(">>>", up, i, histogram[i][0], val)
        if (val >= 0): # up
            if (up):  # immer noch nach oben
                maximum = max(histogram[i][0], maximum) 
            else:     # jetzt kehrt es
                print(low, i-1, ": ", minimum)
                low = i
                maximum = histogram[i][0]
                up = not up
        else: #down    
            if (up):
                print(low, i-1, ": ", maximum)
                minimum = histogram[i][0]
                low = i
                up = not up
            else: 
                minimum = min(histogram[i][0], minimum) 
    plt.show()




def color_mask_black(b = 30): 
    """ creates a black-selection mask with black having less than 'b' 30 value"""
    # lower1 = np.array([0, 0, 0])
    # upper1 = np.array([180, 255, 0])
    lower2 = np.array([0,0,0])
    upper2 = np.array([180,255,b])
    # lower_mask = cv.inRange(t.src_hsv, lower1, upper1)
    mask = cv.inRange(t.src_hsv, lower2, upper2)
 
    # mask = cv.bitwise_not(lower_mask + upper_mask)
    return mask

def color_mask_white(w = 10): 
    """ creates a white-selection mask with with having less than 'w' 10 saturation"""
    lower = np.array([0,0,0])
    upper = np.array([180,w,255])
    mask = cv.inRange(t.src_hsv, lower, upper)
    return mask

def color_mask_hue(lo, hi, b = 30, w = 10): 
    if (lo < hi): 
        lower = np.array([lo, w, b])
        upper = np.array([hi, 255, 255])
        mask = cv.inRange(t.src_hsv, lower, upper)
        return mask
    else: 
        lower1 = np.array([0, w, b])
        upper1 = np.array([hi, 255, 255])
        lower2 = np.array([lo,w,b])
        upper2 = np.array([180,255,255])
        lower_mask = cv.inRange(t.src_hsv, lower1, upper1)
        upper_mask = cv.inRange(t.src_hsv, lower2, upper2)
        mask = lower_mask + upper_mask
        return mask


def pick_color():
    # lower boundary RED color range values; Hue (0 - 10)
    # rot nach https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv/
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
    
    # upper boundary RED color range values; Hue (160 - 180)
    # rot nach https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv/
    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])
    
    lower_mask = cv.inRange(t.src_hsv, lower1, upper1)
    upper_mask = cv.inRange(t.src_hsv, lower2, upper2)
 
    full_mask = lower_mask + upper_mask
    # mask_inv = cv.bitwise_not(full_mask)
    # result = t.src.copy()   # bgr
    result = cv.bitwise_and(t.src, t.src, mask=full_mask)
    cv.imshow(window_name, full_mask)
    # cv2.imshow('result', result)


def draw_hsv_histogramm():
    mask = cv.bitwise_not(cv.bitwise_or(color_mask_white(), color_mask_black()))
    img = cv.bitwise_and(t.src, t.src, mask=mask)
    cv.imshow(window_name, img)
            
    for i, col in enumerate(['b', 'g', 'r']):
        hist = cv.calcHist([t.src_hsv], [i], mask, [256], [0, 256])
        plt.plot(hist, color = col)
        plt.xlim([0, 256])
    plt.show()


def harris(): 
    gray = np.float32(t.detected_edges)
    dst = cv.cornerHarris(gray,3,5,0.04)
    #result is dilated for marking the corners, not important
    print('dst:', dst)
    dst = cv.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    img = t.src.copy()
    img[dst>0.01*dst.max()]=[0,0,255]
    cv.imshow(window_name,img)

def find_line():
    start = t.points[len(t.points)-1]
    print('start point:', start)
    count = 7       # number of point in x and y axis
    delta = 30      # pixels between
    d2 = delta // 2
    max = 0.0
    best = start
    img = t.img.copy()
    cv.imshow(window_name,img)
    for i in range(start[0]-(count-1)*d2, start[0]+(count)*d2, delta): 
        for j in range(start[1]-(count-1)*d2, start[1]+(count)*d2, delta):
            target = t.find_nearest((i, j), delta)
            cv.circle(img, target,1,(255,0,0),-1)
            cv.imshow(window_name,img)
            shift = (start[0]-1 if target[0]<=start[0] else start[0]+1, 
                     start[1]-1 if target[1]<=start[1] else start[1]+1) 
            print("start:", start, "shift", shift, "target:", target)
            if ((target != start) and t.linePossible(shift, target)):
                # cv.circle(img, target,1,(255,0,0),-1)
                cd = t.getColorDelta(start, target)
                if (cd > max):
                    best = target
                    max = cd
    if (max == 0.0): 
        print('oups, no point found')
    else: 
        print('target point:', best, 'with color delta', max)
        cv.line(img, start, best, (255,0,0), 1)
        cv.line(t.img, start, best, (255,0,0), 1)
        t.addLine(start, best)
        t.addPoint(best)
        cv.imshow(window_name,img)


# src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.namedWindow(window_name)
cv.namedWindow("canny")
cv.createTrackbar(title_trackbar, window_name , 75, max_lowThreshold, CannyThreshold)
cv.setMouseCallback(window_name, draw_circle)
cv.imshow(window_name, src)
CannyThreshold(75)

help = """q - quit
r - reset 
b - black-white
c - color (HSV) histogramm
p - pick color
t - threshold
h - histogram
a - harris corner detection
l - line
t - toggle img, lines ...
w - select white """

k = 0

while(1):
    # cv.imshow(window_name,img)
    r = cv.waitKey(1)    # 20 ms to fetch key
    # if (r != -1): print("r " + str(r))
    if (r >=0):
        if (chr(r) == 'q'): 
            break
        elif (chr(r) == 'r'):
            t.reset()
            cv.imshow(window_name, src)
        elif (chr(r) == 'h'):
            draw_histogramm()
        elif (chr(r) == 'a'):
            harris()
        elif (chr(r) == 'c'):
            draw_hsv_histogramm()
        elif (chr(r) == 'l'):
            find_line()
        elif (chr(r) == 't'):
            if (k == 0):
                cv.imshow(window_name, t.lines)
                k = 1
            else: 
                cv.imshow(window_name, t.img)
                k = 0
        elif (chr(r) == 'p'):
            # mask = color_mask_hue(18, 36)  # alles grüne
            # mask = color_mask_hue(113, 133)  # alles weisse?
            # mask = color_mask_hue(134, 163)  # alles dunkelbraun
            mask = color_mask_hue(164, 4)  # alles rot
            res = cv.bitwise_and(t.src, t.src, mask=mask)
            cv.imshow(window_name, res)
        elif (chr(r) == 'b'):
            cv.imshow(window_name, t.src_gray)
        elif (chr(r) == 'w'):
            mask = cv.bitwise_or(color_mask_white(), color_mask_black())
            res = cv.bitwise_and(t.src, t.src, mask=mask)
            cv.imshow(window_name, mask)
        elif (chr(r) == 't'):
            # 0 - 78 - 130 - 161 - 255
            bounds = [40, 130, 161, 255]
            fig = plt.figure()
            for i, b in enumerate(bounds):
                if (i == 0):
                    img = t.threshold(0, b)
                    title = "0 - " + str(b)
                else:  
                    img = t.threshold(bounds[i-1]+1, b)
                    title = str(bounds[i-1]+1) + " - " + str(b)    
                p = fig.add_subplot(2,2,i+1)
                p.set_title(title)
                p.imshow(img, 'gray')
                # plt.subplot(2,2,i+1),plt.imshow(imgs[i],'gray')
                # plt.title(titles[i])
                # plt.xticks([]),plt.yticks([])
            fig.canvas.draw()
            imgp = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            imgp  = imgp.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # img is rgb, convert to opencv's default bgr
            imgp = cv.cvtColor(imgp,cv.COLOR_RGB2BGR)
            # plt.show()
            cv.imshow(window_name, imgp)
        elif (chr(r) == '?'):
            print(help)
        else: 
            print("key ('" + chr(r) + "') pressed. Press '?' for help")   

cv.destroyAllWindows()
