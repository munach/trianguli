import numpy as np


def on_segment(p, q, r):
    """check if r lies on (p,q)"""

    if r[0] <= max(p[0], q[0]) and r[0] >= min(p[0], q[0]) and r[1] <= max(p[1], q[1]) and r[1] >= min(p[1], q[1]):
        return True
    return False

def orientation(p, q, r):
    """return 0/1/-1 for colinear/clockwise/counterclockwise"""

    val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
    if val == 0 : return 0
    return 1 if val > 0 else -1

def intersects(seg1, seg2):
    """check if seg1 and seg2 intersect"""

    p1, q1 = seg1
    p2, q2 = seg2

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        # check general case

        return True

    if o1 == 0 and on_segment(p1, q1, p2) : return True
        # check special cases

    if o2 == 0 and on_segment(p1, q1, q2) : return True
    if o3 == 0 and on_segment(p2, q2, p1) : return True
    if o4 == 0 and on_segment(p2, q2, q1) : return True

    return False

def intersectR(seg1, seg2):
    "check if seg1 and seg2 really intersect (not the same Corner)" 
    p1, q1 = seg1
    p2, q2 = seg2
    if (p1[0] == p2[0] and p1[1] == p2[1]): return False
    if (p1[0] == q2[0] and p1[1] == q2[1]): return False
    if (q1[0] == p2[0] and q1[1] == p2[1]): return False
    if (q1[0] == q2[0] and q1[1] == q2[1]): return False
    return intersects(seg1, seg2)
    

def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)


if __name__ == "__main__":
    segment_one = ((-1, 0), (1, 0))
    segment_two = ((0, 1), (0, -1))
    print(True, intersects(segment_one, segment_two))

    segment_one = ((-1, 0), (1, 1))
    segment_two = ((0, 2), (1, -1))
    print(True, intersects(segment_one, segment_two))
    
    segment_one = ((-1, 0), (1, 1))
    segment_two = ((0, 2), (4, -1))
    print(False, intersects(segment_one, segment_two))
    
    segment_one = ((0, 0), (2, 1))
    segment_two = ((0, 0), (4, 2))
    print(True, intersects(segment_one, segment_two))
    
    segment_one = ([0, 0], [2, 1])
    segment_two = ((0, 0), (-1, -1))
    print(False, intersectR(segment_one, segment_two))
    