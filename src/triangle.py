# from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 
from math import sqrt, pi, exp
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


class Control:
    def __init__(self):
        self.show_Points = False
        self.show_CannyEdges = False
        self.show_lines = False
        self.analyze = False
        self.itriangle = None       # np.array with 3 indices
        self.contour = None         # np.array of contour (points)
        self.candidates = []        # itria of candidates triangles
        self.candidateIdx = 0
        self.ipoint = None          # index of focus point
        self.iline = None

    def toggle_points(self):
        self.show_Points = not self.show_Points

    def toggle_cannyEdges(self):
        self.show_CannyEdges = not self.show_CannyEdges    

    def toggle_lines(self):
        self.show_lines = not self.show_lines

    def toggle_analyze(self):
        self.analyze = not self.analyze

    def set_triangle(self, t):
        self.itriangle = t
        self.contour = None
        self.reset_candidates()

    def set_point(self, p):
        if (self.ipoint is not None):
            self.set_line(self.ipoint, p)
        self.ipoint = p
        

    def set_line(self, a, b):
        self.iline = [a, b]

    def set_contour(self, c):
        self.contour = c

    def reset_candidates(self): 
        self.candidates = []

    def add_candidate(self, itria): 
        self.candidates.append(itria)

    def has_candidates(self):
        return len(self.candidates) != 0

    def next_candidate(self):
        self.candidateIdx = self.candidateIdx+1
        if self.candidateIdx >= len(self.candidates): self.candidateIdx = 0

    def current_candidate(self):
        if self.has_candidates(): 
            return self.candidates[self.candidateIdx]
        else: return None


class Triangle: 
    def __init__(self, src, lenx, n):
        self.src = src
        self.lenx = lenx    # number of points in x direction
        self.avg_pixels = self.src.shape[0] * self.src.shape[1] // n
        self.src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        self.src_hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
        self.img = src.copy()
        self.points = None  #  [[x, y], ...]    len = n
        self.plevel = None  #  [l,...]          len = n
        self.edges = []     # list of edges (canny) found in initPoints()
        self.line = None   
        self.window_name = 'main'
        self.state = Control()
        self.initPoints()
        print('shape', self.src.shape)
        print('number of points', len(self.points)) 
        print('avg px', self.avg_pixels)

    def toggle_points(self):
        self.state.toggle_points()

    def draw(self):
        if (not self.state.show_CannyEdges):
            self.img = self.src.copy()
        else: 
            # self.img = self.edges[0].copy()
            self.img = cv.dilate(self.edges[0], cv.getStructuringElement(cv.MORPH_RECT, (2,2)))
            self.img = cv.cvtColor(self.img, cv.COLOR_GRAY2BGR)
        if self.state.show_Points: 
            self.drawPoints()         
        if self.state.itriangle is not None: 
            p = self.tria(self.state.itriangle)
            if (not self.state.show_CannyEdges):
                cv.drawContours(self.img, [p], 0, (255,0,0), 1)
            else: 
                cv.drawContours(self.img, [p], 0, (255,0,0), 4)
        if self.state.contour is not None: 
            c = self.state.contour
            cv.drawContours(self.img, c, -1, col1[2], 1)
        if (self.state.current_candidate() is not None):
            tria = self.tria(self.state.current_candidate())
            cv.drawContours(self.img, [tria], 0, col1[2], 1)       # 
            #  analyse_triangle(t, itria, 'option %d'%t.index)

        if (self.state.ipoint is not None):
            p = self.at(self.state.ipoint)
            cv.circle(self.img, p, 4, col1[0],1)

        if (self.state.iline is not None):
            a = self.state.iline[0]
            b = self.state.iline[1]
            cv.line(self.img, self.at(a), self.at(b), col[4], 2)
            self.analyze_line(a, b)

    def show(self):
        cv.imshow(self.window_name, self.img)

    def drawPoints(self):
        for i, point in enumerate(self.points):                 
            cv.circle(self.img, point, 2, col[self.plevel[i]],-1)
            # if (self.plevel[i] == 1):
            #     cv.circle(self.img, point, 4, col1[0],1)
            # if (self.plevel[i] == 0):
            #     cv.circle(self.img, point, 6, col1[0],1)


    def findZ(self, p, delta, zeroes): 
        """ return k, point where the lower k means the better edge for p"""
        max = sqrt(2)*(delta/2)
        for k, edges in enumerate(zeroes):
            distances = np.sqrt((edges[:,:,0] - p[0]) ** 2 + (edges[:,:,1] - p[1]) ** 2)
            nearest_index = np.argmin(distances)
            point = edges[nearest_index][0]
            # check if this point is within delta bound
            if (sqrt( (p[0]-point[0]) ** 2 + (p[1]-point[1]) ** 2) < max):
                return (k, point)
        return (10, [p[0], p[1]])
        
    def initPoints(self): 
        src_gray = cv.cvtColor(self.src, cv.COLOR_BGR2GRAY)
        img_blur = cv.blur(self.src_gray, (3,3))
        y,x = self.src.shape[0:2] 
        zeroes = [1]   # add dummy value for corners and line segements 
        self.edges = []
        for i in range(10, 0, -1):
            threshold = i*10
            detected_edges = cv.Canny(img_blur, threshold, threshold*ratio, kernel_size)
            self.edges.append(detected_edges)
            zeroes.append(cv.findNonZero(detected_edges))
        # print('z0:', zeroes[0].__repr__())

        ## add zeros[0] with line segments
        # img = self.edges[0].copy()
        # 20 = min lenght of line, 4 = ???
        # self.lines = cv.HoughLinesP(img, 1, np.pi / 180, 50, None, 20, 4)
        ## append all line points to zeroes[0]:        
        corners = [[[0,0]], [[0, y]], [[x, 0]], [[x,y]]]
        #if self.lines is not None:
        #    for houghLine in self.lines:
        #        l = houghLine[0]
        #        corners.append([[l[0], l[1]]])
        #        corners.append([[l[2], l[3]]])
        zeroes[0] = np.array(corners, dtype=np.int32)

        a = np.linspace(0, x, num=self.lenx, dtype=np.uint16)
        b = np.linspace(0, y, num=int(self.lenx*y / x), dtype=np.uint16)

        points = np.column_stack(( 
            np.full((len(b),len(a)), a).T.flatten(), 
            np.full((len(a),len(b)), b).flatten()))
        self.points = np.zeros((len(points), 2), dtype=np.int)
        self.plevel = np.zeros(len(points), dtype=np.int)
        for i,p in enumerate(points):
            pl, point = self.findZ(p, a[1]-a[0], zeroes)
            self.points[i] = point
            self.plevel[i] = pl 

    def analyze_line(self, a, b):
        ### line a - b has x pixels, 
        ### of these, y are on canny[0]
        mask = np.zeros(self.img.shape[0:2], dtype=np.uint8)
        cv.line(mask, self.at(a), self.at(b), 255, 2)
        x = len(cv.findNonZero(mask))

        dilated = cv.dilate(self.edges[0], cv.getStructuringElement(cv.MORPH_RECT, (2,2)))

        masked = cv.bitwise_and(dilated,dilated,mask = mask)
        y = len(cv.findNonZero(masked))
        print('#linie', x, '#edge', y)
        return (x,y)


    def reset(self):
        self.img = self.src.copy()
        self.lines = []

    def at(self, i): 
        """ returns coordinates of index i"""
        return self.points[i]

    def tria(self, l):
        """ returns coordinate triance of the list of 3 indices, 
            l must be an array"""
        return self.points[l]



    def find_triangle(self, point):
        """ find and set next 3 point to given point"""
        distances = np.sqrt(((self.points[:,0] - point[0]) ** 2) + ((self.points[:,1] - point[1]) ** 2))
        l = distances.argsort()
        self.state.set_triangle(l[0:3])

    def find_point(self, point):
        """ find and set next the point """
        distances = np.sqrt(((self.points[:,0] - point[0]) ** 2) + ((self.points[:,1] - point[1]) ** 2))
        l = distances.argsort()
        for i in l:
            if (self.plevel[i] == 1): 
                self.state.set_point(i)
                break

    def find_line(self):
        """ find a line from self.state.ipoint"""
        if (self.state.ipoint is None): return
        point = self.at(self.state.ipoint)
        distances = np.sqrt(((self.points[:,0] - point[0]) ** 2) + ((self.points[:,1] - point[1]) ** 2))
        l = distances.argsort()
        x = []
        for i in l:
            if (self.plevel[i] == 1): 
                x.append(i)
                if len(x) > 2: break
        ## interessante linien sind jetzt x[0] - x[1] und x[0] - x[2]
        self.state.set_line(x[0], x[1])



    def find_more(self, i0, i1, i2):
        """ return a set of indices near i0, without i0, i1, and i2"""
        x = 4
        (p0, p1, p2) = self.tria([i0, i1, i2])
        orientation_orig = orientation(p0, p1, p2)
        distances = np.sqrt(((self.points[:,0] - p0[0]) ** 2) + ((self.points[:,1] - p0[1]) ** 2))
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


    def size_bonus(self, n):
        """ returns adapted size, where the optimal size is best"""
        return 2*n - (n**2 / self.avg_pixels)
        

    def analyze_only(self, itria):
        orig = self.tria(itria)
        mask = np.zeros(self.src.shape[:2], dtype="uint8")
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

        return (self.size_bonus(n) / Tvar)


    def analyze(self, fig, mask, title = None):
        info = []
        n = np.count_nonzero(mask == 255)
        info.append('$N$ = %d'%(n))     # N = count
        bonus = self.size_bonus(n)
        info.append('bonus = %f' %(bonus))
        Tvar = 0.0
        for i, col in enumerate(['b', 'g', 'r']):
            hist = cv.calcHist([self.src], [i], mask, [256], [0, 256])
            hist = hist.flatten()
            info.append('$%s_{max}$ = %d'%(col, np.argmax(hist)))
            np_img = np.ma.masked_where(mask == 0, self.src[:,:,i])
            mu = np.average(np_img)
            info.append('$%s_{\\mu}$ = %f'%(col, mu))
            sigma = np.sqrt(np.var(np_img))
            Tvar = Tvar + sigma
            info.append('$%s_{\\sigma}$ = %f'%(col, sigma))
            fig.plot(hist, color = col, linewidth=1)
            bins = np.arange(0,255)
            fig.plot(bins, 1000/(sigma * np.sqrt(2 * np.pi)) *
                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
                linewidth=2, color=col, ls='--')
        if title is not None: 
            fig.set_title(title)

        info.append('$Total_{var}$ = %f'%(Tvar))

        # join all info elements for the text
        fig.text(0, 0, '\n'.join(info))
        return (bonus / Tvar)

    def find_candidates(self, itria):
        (a,b,c) = itria
        self.state.reset_candidates()
        for i in self.find_more(a, b, c): 
            self.state.add_candidate([i, b, c])

        for i in self.find_more(b, a, c): 
            self.state.add_candidate([i, a, c])

        for i in self.find_more(c, b, a): 
            self.state.add_candidate([i, b, a])



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

        





