# from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 
from math import sqrt
from lines import intersects, intersectR, orientation

max_lowThreshold = 100

window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3
nonzero = None   
scale = 0.3

col = [                 # rot...
    [204, 204, 255],    # 0
    [153, 153, 255],
    [102, 102, 255],
    [51, 51, 255],    # 3
    [0, 0, 255],
    [0, 0, 204],
    [0, 0, 153],    # 6 
    [0, 0, 102],
    [0, 0, 51],
    [0, 0, 0],     # 9
    [0, 0, 0]
]

col1 = [            # blau
    [204, 0, 0],
    [204, 51, 0], 
    [204, 102, 0],
    [204, 153, 0], 
    [204, 204, 0],
    [204, 255, 0]
]


def kindOfEnlargement(o, n, a, b):
    """ moving o to point n , with respect to segment a-b, return the kind of enlargement"""
    if intersects((o, a),(n,b)) or intersects((o, b), (n, a)):
        # shift...
        return 0
    else: 
        # real enlargement
        return 1


class Triangle: 
    def __init__(self, src):
        self.src = src
        self.src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        self.src_hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
        self.img = src.copy()
        self.nonzero = None             # koordinaten von canny
        self.detected_edges = None      # canny
        self.z = []
        self.lines = []
        self.raster = None
        self.points = None  #  [[layer, x, y]...
        self.initEdges()
        self.initRaster(50) 
        self.triangle = None # [[x1, y1], [x2, y2], [x3, y3]]
        self.itriangle = None # index des Triangles [i0, i1, i2]
        # letzer triangle
        self.enlargement = []
        self.index = 0  # index for 'current' enlargement


    def drawRasterX(self, delta):
        for i in range(0, self.src.shape[1]+delta, delta):
            for j in range(0, self.src.shape[0]+delta, delta):
                # cv.circle(self.img, (i,j),2,(255,0,0),-1)
                (k, p) = self.findZ((i,j), delta)
                # print("p:", (i, j), "got: ", k, p)
                cv.circle(self.img, p,2,col[k],-1)

    def drawRaster(self):
        (x,y,c) = self.raster.shape
        for pp in self.raster.reshape(x*y, c):                 
            cv.circle(self.img, (pp[1], pp[2]),2,col[pp[0]],-1)


    def findZ(self, p, delta): 
        """ return k, point where the lower k means the better edge for p"""
        max = sqrt(2)*(delta/2)
        for k, edges in enumerate(self.z):
            distances = np.sqrt((edges[:,:,0] - p[0]) ** 2 + (edges[:,:,1] - p[1]) ** 2)
            nearest_index = np.argmin(distances)
            point = edges[nearest_index][0]
            # check if this point is within delta bound
            if (sqrt( (p[0]-point[0]) ** 2 + (p[1]-point[1]) ** 2) < max):
                return (k, point)
        return (10, p)

        
    def initEdges(self): 
        img_blur = cv.blur(self.src_gray, (3,3))
        for i in range(10, 0, -1):
            threshold = i*10
            detected_edges = cv.Canny(img_blur, threshold, threshold*ratio, kernel_size)
        # indices = np.where(detected_edges != [0])
            self.z.append(cv.findNonZero(detected_edges))
        print("Edges initi, ", len(self.z))
        print('shape', self.src.shape)

    def initRaster(self, delta):
        x = (self.src.shape[1]+delta) // delta
        y = (self.src.shape[0]+delta) // delta
        # print('size', x, y)
        self.raster = np.zeros((x, y, 3), dtype = np.int64)
        for i in range(0, x):
            for j in range(0, y):
                # cv.circle(self.img, (i,j),2,(255,0,0),-1)
                (k, p) = self.findZ((i*delta,j*delta), delta)
                # print("p:", (i, j), "got: ", k, p)
                self.raster[i, j] = [k, p[0], p[1]]
                # cv.circle(self.img, p,2,col[k],-1)
        # print('raster', self.raster)
        (x,y,c) = self.raster.shape
        self.points = self.raster.reshape(x*y, c)

    def reset(self):
        self.img = src.copy()
        self.lines = []

    def at(self, i): 
        """ returns coordinates of index i"""
        return self.points[i][1:3]

    def tria(self, l):
        """ returns coordinate triance of the list of 3 indices"""
        (a,b,c) = l
        return [self.at(a), self.at(b), self.at(c)]

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

    def set_triangle(self, itria):
        self.itriangle = itria
        self.triangle = self.tria(itria)
        self.enlargement = []           # reset enlargement
        self.index = 0
        cv.drawContours(self.img, [np.array(self.triangle)], 0, (255,0,0, 1))


    def find_triangle(self, point):
        distances = np.sqrt(((self.points[:,1] - point[0]) ** 2) + ((self.points[:,2] - point[1]) ** 2))
        l = distances.argsort()

        t = self.points[l[0:3]][:,1:3]       # nearest 3 corners ([0:3]), select col 1 and 2 ([:,1:3])
        cv.line(self.img, t[0], t[1], (255,0,0), 1)
        cv.line(self.img, t[0], t[2], (255,0,0), 1)
        cv.line(self.img, t[1], t[2], (255,0,0), 1)
        self.triangle = t
        self.itriangle = l[0:3]
        self.enlargement = []           # reset enlargement
        self.index = 0
        return t

    def findXnearest(self, i0, i1, i2):
        """ return a set of indices near i0, without i0, i1, and i2
        and not crossing line segment i1-i2 """
        x = 2
        origin = self.points[i0][1:3]
        distances = np.sqrt(((self.points[:,1] - origin[0]) ** 2) + ((self.points[:,2] - origin[1]) ** 2))
        l = distances.argsort()
        res = []
        for i in l:
            if (i != i0 and i!=i1 and i!=i2 and 
                not intersects((self.at(i1), self.at(i2)), (self.at(i), origin))):
                res.append(i)
            if (len(res) > x): return res


    def find_more(self, i0, i1, i2):
        """ return a set of indices near i0, without i0, i1, and i2"""
        x = 4
        (p0, p1, p2) = self.tria((i0, i1, i2))
        orientation_orig = orientation(p0, p1, p2)
        distances = np.sqrt(((self.points[:,1] - p0[0]) ** 2) + ((self.points[:,2] - p0[1]) ** 2))
        l = distances.argsort()
        res = []
        for i in l:
            pn = self.at(i)
            if ((orientation(p1, p2, pn) == orientation_orig) and     # oberhalb base 
                ((orientation(p1, p0, pn) == orientation_orig) or     # oberhalb linke achse
                 (orientation(p2, pn, p0) == orientation_orig))): 
                res.append(i)
                if (len(res) >= x): break
        return res



    def find_enlarge(self, i0, i1, i2):
        """ return a set of indices near i0, without i0, i1, and i2
        and not crossing line segment i1-i2 
        and enlarging the triangle i0,i1,i2 (such that i0 is inside the enlarged triangle"""
        x = 2
        origin = self.points[i0][1:3]
        p1 = self.at(i1)
        p2 = self.at(i2)
        distances = np.sqrt(((self.points[:,1] - origin[0]) ** 2) + ((self.points[:,2] - origin[1]) ** 2))
        l = distances.argsort()
        res = []
        orientation_orig = orientation(self.at(i0), p1, p2)
        for i in l:
            if (i != i0 and i!=i1 and i!=i2 and 
                orientation(self.at(i), p1, p2) == orientation_orig and
                not intersects((self.at(i1), self.at(i2)), (self.at(i), origin))):
                if (kindOfEnlargement(self.at(i0), self.at(i), self.at(i1), self.at(i2)) == 1):
                    # print('orientation new', orientation(self.at(i), self.at(i1), self.at(i2)))
                    res.append(i)
            if (len(res) >= x): return res
        print('ous, ', res)
        return res

    def analyze_only(self, itria):
        orig = self.tria(itria)
        mask = np.zeros(t.src.shape[:2], dtype="uint8")
        triangle_cnt = np.array( orig )
        cv.drawContours(mask, [triangle_cnt], 0, 255, -1)       # Farbe = 255

        n = np.count_nonzero(mask == 255)
        Tvar = 0.0
        for i, col in enumerate(['b', 'g', 'r']):
            hist = cv.calcHist([self.src], [i], mask, [256], [0, 256])
            hist = hist.flatten()
            np_img = np.ma.masked_where(mask == 0, self.src[:,:,i])
            var = np.sqrt(np.var(np_img))
            Tvar = Tvar + var

        return (n / Tvar)


    def analyze(self, fig, mask, title = None):
        info = []
        n = np.count_nonzero(mask == 255)
        info.append('$N$ = %d'%(n))     # N = count
        Tvar = 0.0
        for i, col in enumerate(['b', 'g', 'r']):
            hist = cv.calcHist([self.src], [i], mask, [256], [0, 256])
            hist = hist.flatten()
            info.append('$%s_{max}$ = %d'%(col, np.argmax(hist)))
            np_img = np.ma.masked_where(mask == 0, self.src[:,:,i])
            info.append('$%s_{avg}$ = %f'%(col, np.average(np_img)))
            var = np.sqrt(np.var(np_img))
            Tvar = Tvar + var
            info.append('$%s_{var}$ = %f'%(col, var))
            fig.plot(hist, color = col)
        if title is not None: 
            fig.set_title(title)

        info.append('$Total_{var}$ = %f'%(Tvar))

        # join all info elements for the text
        fig.text(0, 0, '\n'.join(info))
        return (n / Tvar)

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
        mask = np.zeros(self.src.shape[:2], dtype="uint8")
        triangle_cnt = np.array( [a, b, c] )
        cv.drawContours(mask, [triangle_cnt], 0, 255, -1)       # Farbe = 255
        masked = cv.bitwise_and(self.src, self.src, mask=mask)
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





def compare_shift(orig, d1, fig):
    mask = np.zeros(t.src.shape[:2], dtype="uint8")
    cv.drawContours(mask, [np.array( d1 )], 0, 255, -1)       # Farbe = 255
    t.analyze(fig, mask, 'better?')

def compare_enlarged(orig, d1, fig):
    mask = np.zeros(t.src.shape[:2], dtype="uint8")
    cv.drawContours(mask, [np.array( d1 )], 0, 255, -1)       # Farbe = 255
    cv.drawContours(mask, [np.array( orig )], 0, 0, -1)       # Farbe = 0 überschreibt altes dreieck

    # masked_d1 = cv.bitwise_and(t.src, t.src, mask=mask)
    # cv.imshow('x2',masked_d1)
    t.analyze(fig, mask, 'add?')




def enlarge(itriangle): 
    (a,b,c) = itriangle
    n = t.findXnearest(a, b, c)
    # cv.line(t.img, t.at(a), t.at(n[0]), (255,0,0), 1)
    # cv.line(t.img, t.at(a), t.at(n[1]), (255,0,0), 1)
    cv.imshow(window_name,t.img)
    orig = t.tria(itriangle)

    fig, axs = plt.subplots(2)
    fig.suptitle('bessere Dreiecke?')
    
    # hist for origianl
    mask = np.zeros(t.src.shape[:2], dtype="uint8")
    triangle_cnt = np.array( orig )
    cv.drawContours(mask, [triangle_cnt], 0, 255, -1)       # Farbe = 255
    t.analyze(axs[0], mask, 'Original')
    
    cv.drawContours(t.img, [np.array(t.tria((n[0], b, c)))], 0, col1[3], 1)       # neues dreieck zeigen
    cv.imshow(window_name, t.img)

    if kindOfEnlargement(t.at(a), t.at(n[0]), t.at(b), t.at(c)) == 0:
        compare_shift(orig, t.tria((n[0],b,c)), axs[1])
    else:     
        compare_enlarged(orig, t.tria((n[0],b,c)), axs[1])

    # plt.show():
    fig.canvas.draw()
    # convert canvas to image
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,)).copy()
    # img.resize((600, 600))
    # img is rgb, convert to opencv's default bgr
    img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
    cv.imshow('plot', img)


def enlarge1(itriangle): 
    (a,b,c) = itriangle

    for i in t.find_more(a, b, c): 
        cv.line(t.img, t.at(a), t.at(i), col1[3], 1)

    for i in t.find_more(b, a, c): 
        cv.line(t.img, t.at(b), t.at(i), col1[3], 1)

    for i in t.find_more(c, b, a): 
        cv.line(t.img, t.at(c), t.at(i), col1[3], 1)

    cv.imshow(window_name,t.img)


def analyse_triangle(itriangle, name = 'Original'): 
    (a,b,c) = itriangle
    orig = t.tria(itriangle)

    fig, axs = plt.subplots(1)
    
    # hist for origianl
    mask = np.zeros(t.src.shape[:2], dtype="uint8")
    triangle_cnt = np.array( orig )
    cv.drawContours(mask, [triangle_cnt], 0, 255, -1)       # Farbe = 255
    x = t.analyze(axs, mask)

    plt.suptitle(name + ' ' + str(x))

    fig.canvas.draw()
    # convert canvas to image
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,)).copy()
    # img is rgb, convert to opencv's default bgr
    img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
    cv.imshow('plot', img)
    plt.close()


def next(itriangle): 
    t.reset()

    cv.drawContours(t.img, [np.array(t.tria(itriangle))], 0, col1[0], 1)       # 
    if len(t.enlargement) == 0:
        (a,b,c) = itriangle
        # get new triangles
        for i in t.find_more(a, b, c): 
            t.enlargement.append((i, b, c))

        for i in t.find_more(b, a, c): 
            t.enlargement.append((i, a, c))

        for i in t.find_more(c, b, a): 
            t.enlargement.append((i, b, a))

    # now pick t.next'th enlargment, display it, and show its analyse
    itria = t.enlargement[t.index]
    tria = np.array(t.tria(itria))
    cv.drawContours(t.img, [tria], 0, col1[2], 1)       # 
    cv.imshow(window_name, t.img)
    analyse_triangle(itria, 'option %d'%t.index)

    t.index = t.index+1
    if t.index >= len(t.enlargement): t.index = 0

def best(itriangle): 
    t.reset()
    (a,b,c) = itriangle
        # get new triangles
    for i in t.find_more(a, b, c): 
        t.enlargement.append((i, b, c))

    for i in t.find_more(b, a, c): 
        t.enlargement.append((i, a, c))

    for i in t.find_more(c, b, a): 
        t.enlargement.append((i, b, a))

    xo = t.analyze_only(itriangle)
    best = None
    for nt in t.enlargement: 
        xn = t.analyze_only(nt)
        if (xn > xo):
            best = nt
            xo = xn

    if (best is not None): 
        t.set_triangle(best)
        cv.imshow(window_name, t.img)


def draw_triangle():
    d = t.triangle

    mask = np.zeros(t.src.shape[:2], dtype="uint8")
    triangle_cnt = np.array( d )
    cv.drawContours(mask, [triangle_cnt], 0, 255, -1)       # Farbe = 255
    masked = cv.bitwise_and(t.src, t.src, mask=mask)

    # inv_mask = cv.bitwise_not(mask)


    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    # fig = plt.figure(figsize=(4,3))

    avg_color_per_row = np.average(masked, axis=0)
    avg = np.average(avg_color_per_row, axis=0)
    print('avg:', avg)

    for i, col in enumerate(['b', 'g', 'r']):
        hist = cv.calcHist([masked], [i], mask, [256], [0, 256])
        hist = hist.flatten()
        axs[0].plot(hist, color = col)
        # axs[0].xlim([0, 256])

        con = np.convolve(hist, np.ones(5)/5, mode='valid')
        # axs[1].plot(con, color = col)
        max = np.array(signal.argrelmax(con)).flatten()
        axs[1].plot(max, con[max], color=col, linestyle="--")
        
        # axs[3].xlim([0, 256])
        # sg = signal.savgol_filter(hist, 101, 5)
        # axs[2].plot(sg, color = col)
        # max = signal.argrelmax(sg)[0]
        # axs[2].scatter(max, sg[max], color=col)


    
    # plt.show():
    fig.canvas.draw()
    # convert canvas to image
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,)).copy()
    # img.resize((600, 600))
    # img is rgb, convert to opencv's default bgr
    img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
    cv.imshow('plot', img)

k = 0


# mouse callback function
def draw_circle(event,x,y,flags,param):
    # print("mouse:" + str(event))
    if event == cv.EVENT_LBUTTONUP:
        # print("Mouse " + str(x) + ", " + str(y))
        # onedge = t.find_nearest_white((x, y))
        # print("Edge:" + str(onedge))
        cv.circle(t.img, (x, y),2,(255,0,0),-1)
        t.find_triangle((x,y))
        cv.imshow(window_name, t.img)
        if (k == 1):
            draw_triangle()
        # t.addPoint(onedge)
        # if (len(t.points) == 2):
        #     t.colors(t.points[0], t.points[1])
        #     cv.imshow(window_name, t.img)
            



# src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.namedWindow(window_name)
# cv.namedWindow("canny")
# cv.createTrackbar(title_trackbar, window_name , 75, max_lowThreshold, CannyThreshold)
cv.setMouseCallback(window_name, draw_circle)
cv.imshow(window_name, src)
# CannyThreshold(75)

help = """q - quit
r - reset 
g - gray
a - analyze Triangle
b - find best enlarged triange, and select this triangle
i - show Raster
e - enlarge triangle
d - toggle histogram on triangle
n - analyze next enlared triangle...
t - toggle img, lines ...
"""


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
        elif (chr(r) == 'a'):
            # harris()
            if (t.itriangle is not None):
                 analyse_triangle(t.itriangle)
                
        elif (chr(r) == 'e'):
            if (t.itriangle is not None):
                enlarge1(t.itriangle)
        elif (chr(r) == 'n'):
            if (t.itriangle is not None):
                next(t.itriangle)
        elif (chr(r) == 'b'):
            if (t.itriangle is not None):
                best(t.itriangle)
        elif (chr(r) == 'd'):
            if (k==0):
                if (t.triangle is not None):
                    draw_triangle()
                k = 1
            else: 
                cv.destroyWindow('plot')
                k = 0
        elif (chr(r) == 'i'):
            t.drawRaster()
            cv.imshow(window_name, t.img)
        elif (chr(r) == 'b'):
            cv.imshow(window_name, t.src_gray)
        elif (chr(r) == '?'):
            print(help)
        else: 
            print("key ('" + chr(r) + "') pressed. Press '?' for help")   

cv.destroyAllWindows()
