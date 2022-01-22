import numpy as np
import cv2 as cv
from numpy.lib.function_base import insert


b = np.arange(0,24,2)
i = np.array([3,5,7])

res = np.c_[i, b[i]]
print(res)
