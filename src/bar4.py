# from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 
from math import sqrt
from lines import intersects, intersectR, orientation
from triangle import Triangle


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


def getTriangleFromArgs(): 
    parser = argparse.ArgumentParser(description='Code for Canny Edge Detector tutorial.')
    parser.add_argument('--input', help='Path to input image.', default='storch.jpg')
    parser.add_argument('--len', help='Number of Points in x direction', type=int, default=20)
    parser.add_argument('-n', help='Avg Number of Triangles', type=int, default=100)
    args = parser.parse_args()
    src = cv.imread(cv.samples.findFile(args.input))
    if src is None:
        print('Could not open or find the image: ', args.input)
        exit(0)

    return Triangle(src, args.len, args.n)


def enlarge(t:Triangle):
    if (t.state.itriangle is None): return 
    (a,b,c) = t.state.itriangle

    res = []
    count = 0

    for i in t.find_more(a, b, c): 
        res.append(t.at(a))
        res.append(t.at(i))
        count = count+1

    for i in t.find_more(b, a, c): 
        res.append(t.at(b))
        res.append(t.at(i))
        count = count+1

    for i in t.find_more(c, b, a): 
        res.append(t.at(c))
        res.append(t.at(i))
        count = count+1
    t.state.set_contour(np.array(res).flatten().reshape(count,2,2))


def analyse_triangle(t:Triangle, itriangle, name = 'Original'):
    if (itriangle is None): return 

    (a,b,c) = itriangle
    orig = t.tria(itriangle)
    # quality = t.plevel[itriangle]
    # print('quality', quality)     # levels of edges...

    fig, axs = plt.subplots(1)
    
    # hist for origianl
    mask = np.zeros(t.src.shape[:2], dtype="uint8")
    cv.drawContours(mask, [orig], 0, 255, -1)       # Farbe = 255
    t.analyze(axs, mask)
    x = t.analyze_only(itriangle)

    plt.suptitle(name + ' ' + str(x))

    fig.canvas.draw()
    # convert canvas to image
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,)).copy()
    # img is rgb, convert to opencv's default bgr
    img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
    cv.imshow('plot', img)
    plt.close()

def next(t:Triangle): 
    if not t.state.has_candidates():
        if (t.state.itriangle is None): return
        t.find_candidates(t.state.itriangle)
    else:
        t.state.next_candidate()

    if (t.state.analyze): 
        analyse_triangle(t, t.state.current_candidate(), name = "Cand %d:"%t.state.candidateIdx)



def best(t:Triangle): 
    if not t.state.has_candidates():
        if (t.state.itriangle is None): return
        t.find_candidates(t.state.itriangle)

    # now we have a itriangle and candidates
    xo = t.analyze_only(t.state.itriangle)
    print('current triangle', xo)
    best = None
    for nt in t.state.candidates: 
        xn = t.analyze_only(nt)
        if (xn > xo):
            best = nt
            xo = xn

    if (best is not None): 
        print('better triangle found', xo)
        t.state.set_triangle(best)
    else: 
        print('no better triangle found')
        t.state.reset_candidates()

def lines(t:Triangle):
    img = t.edges[0].copy()

    linesP = cv.HoughLinesP(img, 1, np.pi / 150, 50, None, 20, 4)
    # Draw the lines
    dest_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(dest_img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv.LINE_AA)
    cv.imshow("LSD",dest_img )
    

def click(t:Triangle, event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONUP:
        cv.circle(t.img, (x,y) ,2,(255,0,0),-1)
        t.find_triangle((x,y))
        t.draw()
        t.show()
        if (t.state.analyze): analyse_triangle(t, t.state.itriangle)
    if event == cv.EVENT_RBUTTONUP:
        t.find_point((x,y))
        t.draw()
        t.show()
        if (t.state.analyze): analyse_triangle(t, t.state.itriangle)
    # draw_triangle()



help = """q - quit
r - reset 
g - gray
a - analyze Triangle
b - find best enlarged triange, and select this triangle
c - (toggle) show canny edges on triangle 
i - toggle Raster
e - enlarge triangle
n - next candidate triangle...
l - find line: 
"""

if __name__=='__main__': 
    t = getTriangleFromArgs()
    cv.namedWindow(t.window_name)
    cv.setMouseCallback(t.window_name, lambda event, x, y, flags, param: 
        click(t, event, x, y, flags, param))
    t.draw()
    t.show()

    while(1):
        r = cv.waitKey(1)    # 20 ms to fetch key
        if (r >=0):
            if (chr(r) == 'q'): break
            elif (chr(r) == 'r'): t.reset()
            elif (chr(r) == 'a'): 
                t.state.toggle_analyze()
                if (t.state.analyze): analyse_triangle(t, t.state.itriangle)
                else: cv.destroyWindow('plot')
            elif (chr(r) == 'e'): enlarge(t)
            elif (chr(r) == 'n'): next(t)
            elif (chr(r) == 'b'): best(t)
            elif (chr(r) == 'i'): t.toggle_points()
            elif (chr(r) == 'c'): t.state.toggle_cannyEdges()
            elif (chr(r) == '?'): print(help)
            elif (chr(r) == 'l'): t.find_line() 
            else: print("key ('" + chr(r) + "') pressed. Press '?' for help")
            t.draw()
            t.show()
    cv.destroyAllWindows()
