import os
import numpy as np
import taichi as ti
import open3d as o3d
import math
import taichi_tools
import time
import sys
from multiprocessing import Process
import multiprocessing
from threading import Thread
from qef import *
import meshio

@ti.func
def isnan(x):
    return not (x < 0 or 0 < x or x == 0)

def export_partial_vdb_str(return_dict, data, begin, end, threadId):
    print("Joining thread #{}".format(threadId))
    return_dict[threadId] = ""
    result = ""
    for i in range(begin, end):
        if i % 500000 == 0:
            print("Thread {} finished build {}%".format(threadId, (i - begin) * 100.0 / (end - begin)))
        vec = data[i]
        result += "{} {} {} {}\n".format(int(vec[0]), int(vec[1]), int(vec[2]), float(vec[3]))
    return_dict[threadId] = result

@ti.data_oriented
class _SRBuiltInData:
    def __init__(self):
        # These following data came from Paul Bourke's site:
        # http://paulbourke.net/geometry/polygonise/
        tri_table_data = np.array(
            [
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
                [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
                [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
                [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
                [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1], [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
                [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
                [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
                [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
                [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
                [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
                [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
                [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
                [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
                [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
                [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
                [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
                [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
                [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
                [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
                [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
                [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
                [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
                [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
                [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
                [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
                [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
                [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
                [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
                [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
                [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
                [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
                [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
                [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
                [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
                [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
                [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
                [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
                [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
                [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
                [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
                [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
                [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
                [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
                [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
                [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
                [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
                [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
                [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
                [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
                [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
                [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
                [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
                [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
                [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
                [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
                [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
                [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
                [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
                [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
                [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
                [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
                [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
                [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
                [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
                [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
                [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
                [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
                [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
                [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
                [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
                [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
                [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
                [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
                [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
                [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
                [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
                [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
                [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
                [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
                [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
                [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
                [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
                [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
                [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
                [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
                [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
                [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
                [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
                [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
                [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
                [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
                [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
                [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
                [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
                [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
                [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
                [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
                [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
                [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
                [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
                [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
                [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
                [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
                [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
                [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
                [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
                [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
                [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
                [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
                [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
                [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
                [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
                [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
                [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
                [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
                [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
                [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
                [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
                [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
                [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
                [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
                [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
                [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
                [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
                [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
                [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
                [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
                [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
                [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
                [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
                [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
                [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
                [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
                [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
                [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
                [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
                [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
                [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
                [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
                [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
                [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
                [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
                [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
                [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
                [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
                [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
                [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
                [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
                [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
                [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
                [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
                [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
                [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
                [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
                [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
                [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
                [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
                [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
                [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
                [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
                [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
                [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
                [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
                [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
                [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
                [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
                [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
                [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
                [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
                [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
                [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
                [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
                [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
                [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
                [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
                [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
                [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
                [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
                [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
                [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
                [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
                [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
                [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
                [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
                [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
                [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
                [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
                [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            ]
        )
        edge_table_data = np.array(
            [
                0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
                0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
                0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
                0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
                0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
                0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
                0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
                0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
                0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
                0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
                0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
                0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
                0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
                0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
                0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
                0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
                0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
                0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
                0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
                0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
                0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
                0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
                0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
                0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
                0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
                0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
                0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
                0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
                0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
                0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
                0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
                0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0]
        )

        # Initialize the data as taichi fields
        self.edge_table = ti.field(dtype=ti.i32, shape=256)
        self.tri_table = ti.field(dtype=ti.i32, shape=(256, 16))

        self.edge_table.from_numpy(edge_table_data)
        self.tri_table.from_numpy(tri_table_data)


###############################
# Sparse Grid Data

@ti.data_oriented
class _SRVoxel:
    def __init__(self, bounding_box_extent, voxel_size=0.01, particle_radius=0.01, level_dimensions=None):
        assert particle_radius > 0, "Particle radius must be positive!"
        assert voxel_size > 0, "Voxel size must be positive!"
        assert level_dimensions is None or len(level_dimensions) >= 2, "Need at least two levels for marching cube grids!"

        self.voxel_size = voxel_size
        self.bounding_box_extent = bounding_box_extent
        if level_dimensions is None:
            # Use default levels to initiate marching cube grid levels
            self.level_dimensions = [
                ti.Vector([3, 3, 3]),
                ti.Vector([3, 3, 3]),
                ti.Vector([3, 3, 3]),
                ti.Vector([3, 3, 3]),
                ti.Vector([3, 3, 3]),
                ti.Vector([3, 3, 3]),
                ti.Vector([5, 5, 5])
            ]
        else:
            # Use custom levels to initiate marching cube grid levels
            self.level_dimensions = level_dimensions

        self.max_num_particle_per_voxel = int(np.ceil(pow(voxel_size / particle_radius, 3.0)))
        if self.max_num_particle_per_voxel <= 2:
            self.max_num_particle_per_voxel = 2

        # Initiailize:
        # Values at each grid vertex
        # Particles inside each grid
        # Number of particles inside each grid
        self.voxel_vertex_value = ti.field(ti.f32)
        self.voxel_vertex_velocity = ti.field(ti.f32)
        self.particle_hash = ti.field(ti.i32)
        self.particle_hash_length = ti.field(ti.i32)
        self.voxel_visit_hash = ti.field(ti.i32)
        self.voxel_value = ti.field(ti.f32)

        # Points used by dual contouring
        self.dc_point = ti.field(ti.f32)


        self.voxel_vertex_value_levels = [ti.root.pointer(ti.ijk, self.level_dimensions[0])]
        self.particle_hash_levels = [ti.root.pointer(ti.ijk, self.level_dimensions[0])]
        self.particle_hash_length_levels = [ti.root.pointer(ti.ijk, self.level_dimensions[0])]
        self.voxel_visit_hash_levels = [ti.root.pointer(ti.ijk, self.level_dimensions[0])]
        self.voxel_value_levels = [ti.root.pointer(ti.ijk, self.level_dimensions[0])]

        self.dc_point_levels = [ti.root.pointer(ti.ijk, self.level_dimensions[0])]

        # Sample a velocity field
        self.voxel_vertex_velocity_levels = [ti.root.pointer(ti.ijk, self.level_dimensions[0])]

        for i in range(1, len(self.level_dimensions) - 1):
            self.voxel_vertex_value_levels.append(
                self.voxel_vertex_value_levels[i - 1].pointer(ti.ijk, self.level_dimensions[i])
            )
            self.particle_hash_levels.append(
                self.particle_hash_levels[i - 1].pointer(ti.ijk, self.level_dimensions[i])
            )
            self.particle_hash_length_levels.append(
                self.particle_hash_length_levels[i - 1].pointer(ti.ijk, self.level_dimensions[i])
            )
            self.voxel_visit_hash_levels.append(
                self.voxel_visit_hash_levels[i - 1].pointer(ti.ijk, self.level_dimensions[i])
            )
            self.voxel_value_levels.append(
                self.voxel_value_levels[i - 1].pointer(ti.ijk, self.level_dimensions[i])
            )
            self.dc_point_levels.append(
                self.dc_point_levels[i - 1].pointer(ti.ijk, self.level_dimensions[i])
            )
            self.voxel_vertex_velocity_levels.append(
                self.voxel_vertex_velocity_levels[i - 1].pointer(ti.ijk, self.level_dimensions[i])
            )

        self.voxel_vertex_value_levels.append(
            self.voxel_vertex_value_levels[-1].dense(ti.ijk, self.level_dimensions[-1])
        )
        self.voxel_vertex_value_levels[-1].place(self.voxel_vertex_value)

        self.voxel_value_levels.append(
            self.voxel_value_levels[-1].dense(ti.ijk, self.level_dimensions[-1])
        )
        self.voxel_value_levels[-1].place(self.voxel_value)

        self.particle_hash_levels.append(
            self.particle_hash_levels[-1].dense(ti.ijk, self.level_dimensions[-1])
        )
        self.particle_hash_levels.append(
            self.particle_hash_levels[-1].dynamic(ti.l, self.max_num_particle_per_voxel)
        )
        self.particle_hash_levels[-1].place(self.particle_hash)

        self.particle_hash_length_levels.append(
            self.particle_hash_length_levels[-1].dense(ti.ijk, self.level_dimensions[-1])
        )
        self.particle_hash_length_levels[-1].place(self.particle_hash_length)

        self.voxel_visit_hash_levels.append(
            self.voxel_visit_hash_levels[-1].dense(ti.ijk, self.level_dimensions[-1])
        )
        self.voxel_visit_hash_levels[-1].place(self.voxel_visit_hash)

        # Data used to sample velocity field at the vertex
        self.voxel_vertex_velocity_levels.append(
            self.voxel_vertex_velocity_levels[-1].pointer(ti.ijk, self.level_dimensions[-1])
        )
        self.voxel_vertex_velocity_levels.append(
            self.voxel_vertex_velocity_levels[-1].dense(ti.l, 3)
        )
        self.voxel_vertex_velocity_levels[-1].place(self.voxel_vertex_velocity)

        self.dc_point_levels.append(
            self.dc_point_levels[-1].pointer(ti.ijk, self.level_dimensions[-1])
        )
        self.dc_point_levels.append(
            self.dc_point_levels[-1].dense(ti.l, 4)
        )
        self.dc_point_levels[-1].place(self.dc_point)



        # Initialize offset maps that will be used in utility functions

        self.voxel_vertex_id_map = ti.field(dtype=ti.i32, shape=(2, 2, 2))
        self.voxel_vertex_id_map[0, 0, 0] = 3
        self.voxel_vertex_id_map[1, 0, 0] = 2
        self.voxel_vertex_id_map[0, 1, 0] = 0
        self.voxel_vertex_id_map[0, 0, 1] = 7
        self.voxel_vertex_id_map[1, 1, 0] = 1
        self.voxel_vertex_id_map[0, 1, 1] = 4
        self.voxel_vertex_id_map[1, 0, 1] = 6
        self.voxel_vertex_id_map[1, 1, 1] = 5

        self.voxel_vertex_offset_map = ti.Vector.field(3, dtype=ti.i32, shape=8)
        self.voxel_vertex_offset_map[0] = ti.Vector([0, 1, 0])
        self.voxel_vertex_offset_map[1] = ti.Vector([1, 1, 0])
        self.voxel_vertex_offset_map[2] = ti.Vector([1, 0, 0])
        self.voxel_vertex_offset_map[3] = ti.Vector([0, 0, 0])
        self.voxel_vertex_offset_map[4] = ti.Vector([0, 1, 1])
        self.voxel_vertex_offset_map[5] = ti.Vector([1, 1, 1])
        self.voxel_vertex_offset_map[6] = ti.Vector([1, 0, 1])
        self.voxel_vertex_offset_map[7] = ti.Vector([0, 0, 1])

    @ti.func
    def particle_id(self, x : ti.i32, y : ti.i32, z : ti.i32, w : ti.i32):
        return self.particle_hash[x, y, z, w]

    @ti.func
    def particle_len(self, x : ti.i32, y : ti.i32, z : ti.i32):
        return self.particle_hash_length[x, y, z]

    @ti.func
    def get_mapped_position(self, x : ti.i32, y : ti.i32, z : ti.i32):
        return ti.Vector([x, y, z]) * self.voxel_size + self.bounding_box_extent[0]

    @ti.func
    def get_voxel_index(self, pos):
        x = int((pos[0] - self.bounding_box_extent[0][0]) / self.voxel_size)
        y = int((pos[1] - self.bounding_box_extent[0][1]) / self.voxel_size)
        z = int((pos[2] - self.bounding_box_extent[0][2]) / self.voxel_size)
        return x, y, z

    @ti.func
    def get_voxel_vertex_mapped_position(self, x : ti.i32, y : ti.i32, z : ti.i32, w : ti.i32):
        offset = self.voxel_vertex_offset_map[w]
        return self.get_mapped_position(x + offset[0], y + offset[1], z + offset[2])

    @ti.func
    def get_voxel_vertex_position(self, x : ti.i32, y : ti.i32, z : ti.i32, w : ti.i32):
        return ti.Vector([x, y, z]) + self.voxel_vertex_offset_map[w]

    @ti.func
    def get_voxel_vertex_value(self, x : ti.i32, y : ti.i32, z : ti.i32, w : ti.i32) -> ti.f32:
        x, y, z = self.get_voxel_vertex_position(x, y, z, w)
        return self.voxel_vertex_value[x, y, z]

    @ti.func
    def add_particle(self, pos, particle_id):
        x, y, z = self.get_voxel_index(pos)
        ti.append(self.particle_hash.parent(), (x, y, z), particle_id)
        self.particle_hash_length[x, y, z] = ti.length(self.particle_hash.parent(), (x, y, z))

    @ti.func
    def visit_voxel(self, x: ti.i32, y: ti.i32, z: ti.i32, w: ti.i32 = 1):
        return ti.atomic_max(self.voxel_visit_hash[x, y, z], w)



    def clear_visit(self):
        self.voxel_visit_hash_levels[0].deactivate_all()

    def deactivate_particle_hash(self):
        self.particle_hash_levels[0].deactivate_all()
        self.particle_hash_length_levels[0].deactivate_all()

    def deactivate(self):
        # self.voxel_vertex_value_levels[0].deactivate_all()
        # self.voxel_visit_hash_levels[0].deactivate_all()
        # self.voxel_value_levels[0].deactivate_all()
        # self.particle_hash_levels[0].deactivate_all()
        # self.particle_hash_length_levels[0].deactivate_all()
        # self.dc_point_levels[0].deactivate_all()
        # self.voxel_vertex_velocity_levels[0].deactivate_all()
        ti.deactivate_all_snodes()
        self.dc_cubes_cnt = 0

    @ti.func
    def get_voxel_center_pos(self, x: ti.i32, y: ti.i32, z: ti.i32):
        center_pos = ti.Vector([0.0, 0.0, 0.0])
        for i in range(8):
            center_pos += self.get_voxel_vertex_mapped_position(x, y, z, i)
        return center_pos / 8.0






##############################



# used grid


##############################

#
# Marching Cube Implementation for Fluid Surface Reconstruct
#
@ti.data_oriented
class SRTool:
    #@detail:
    # Grids are arranged so that:
    # Grid Index: [0, 0, 0]      <-------> The grid at lower-left corner
    # Grid Vert Index: [0, 0, 0] <-------> The vertex coordinate at the lower-left corner
    # Each grid has vertex id from 0 - 7 indicating the 8 vertices, and the ids are arrange as:
    #
    #
    #       4 ---------- 5
    #       / |        /|
    #      /  |       / |
    #     7----------6  |
    #     |   |      |  |
    #     |   0------|--1
    #     |  /       | /
    #     | /        |/
    #     3----------2

    def __init__(self, bounding_box_extent, voxel_size=0.01, particle_radius=0.01,
                record_normals=False, record_velocity = False, max_num_vertices=8000000, max_num_indices=100000000, max_num_particles=9000000,
                 hash_level_dimensions=None):
        # Initialize some constants
        # These constants came from papers and personal experiments
        self.rho = 1000
        self.radius = particle_radius

        # G records the G_i needed for anisotropic kernel
        self.G = ti.Matrix.field(3, 3, ti.f32, shape=max_num_particles)

        # updated kernel center
        self.x_bar = ti.Vector.field(3, ti.f32, shape=max_num_particles)

        # Constants used in Anisotropic Kernel
        self.kr = 4
        self.ks = 1400
        self.kn = 0.5
        self.Nvep = 25
        self.akLambda = 0.9

        # Pre-calculate values
        self.volume = self.radius * self.radius * self.radius * np.pi * 4 / 3
        self.Acoeff = 315 / (64 * np.pi)
        self.diameter = 2 * self.radius
        self.mass = self.rho * self.volume

        self.num_vertices = ti.field(dtype=ti.i32, shape=())
        self.num_indices = ti.field(dtype=ti.i32, shape=())
        # Initialize a map that maps the bottom left corner, which is vertex #3
        # to other vertices in each x, y, z direction
        self.interpolation_table = ti.Vector.field(2, dtype=ti.i32, shape=12)
        self.interpolation_table[0] = ti.Vector([0, 1])
        self.interpolation_table[1] = ti.Vector([1, 2])
        self.interpolation_table[2] = ti.Vector([2, 3])
        self.interpolation_table[3] = ti.Vector([3, 0])
        self.interpolation_table[4] = ti.Vector([4, 5])
        self.interpolation_table[5] = ti.Vector([5, 6])
        self.interpolation_table[6] = ti.Vector([6, 7])
        self.interpolation_table[7] = ti.Vector([7, 4])
        self.interpolation_table[8] = ti.Vector([0, 4])
        self.interpolation_table[9] = ti.Vector([1, 5])
        self.interpolation_table[10] = ti.Vector([2, 6])
        self.interpolation_table[11] = ti.Vector([3, 7])


        # Initialize utility data

        self.mc_tables = _SRBuiltInData()
        self.voxel = _SRVoxel(bounding_box_extent=bounding_box_extent, voxel_size=voxel_size,
                                        particle_radius=particle_radius, level_dimensions=hash_level_dimensions)

        # These store the necessary information for the generated mesh
        self.mesh_vertex = ti.Vector.field(3, dtype=ti.f32, shape=max_num_vertices)
        self.mesh_index = ti.field(dtype=ti.i32, shape=max_num_indices)
        self.vdb_voxel = ti.Vector.field(4, dtype = ti.f32, shape=2)
        self.record_normals = record_normals
        self.record_velocity = record_velocity


        if self.record_normals:
            self.mesh_normal = ti.Vector.field(3, dtype=ti.f32, shape=max_num_vertices)

        if self.record_velocity:
            self.mesh_velocity = ti.Vector.field(3, dtype=ti.f32, shape=max_num_vertices)

        # initialize some utilities infos
        self.reset()

    ##################################################################################
    #                                                                                #
    #                                                                                #
    #                    Implementation for Anisotropic Kernels                      #
    #                                                                                #
    #                                                                                #
    ##################################################################################

    #@brief: the isotropic weighting function
    #@detail:
    #          1 - (||x_i - x_j|| / r_i)^3,    if ||x_i - x_j|| < r_i
    #   wij =
    #          0                          ,    otherwise
    #
    # we are using dx for x_i - x_j, r for r_i
    @ti.func
    def isoweight(self, dx, r):
        norm_dx = dx.norm()
        res = 0.0
        if norm_dx < r:
            res = 1.0 - ti.pow(norm_dx / r, 3.0)
        return res

    #@brief: pre-process data for anisotropic kernel
    @ti.kernel
    def pre_process_data(self, smooth_radius: ti.f32, num_particles: ti.i32, pos: ti.template()):
        smooth_range = ti.ceil(4 * smooth_radius / self.voxel.voxel_size, dtype=ti.i32) + 1
        smooth_offset = smooth_range // 2
        for i in range(1, num_particles):
            x, y, z = self.voxel.get_voxel_index(pos[i])
            # this is sum w_ij over j
            total_weight = 0.0
            x_wi = ti.Matrix([0.0, 0.0, 0.0])

            # We first calculate C_i and x_i^W
            # The range (9, 9, 9) is calculated as the paper suggest we use r_i = 2h_i
            # our current h_i = 2.2 * particle radius
            for dx, dy, dz in ti.ndrange(smooth_range, smooth_range, smooth_range):
                dx -= smooth_offset
                dy -= smooth_offset
                dz -= smooth_offset
                if x + dx < 0 or y + dy < 0 or z + dz < 0:
                    continue
                particle_len = self.voxel.particle_len(x + dx, y + dy, z + dz)
                for j in range(particle_len):
                    particle_id = self.voxel.particle_id(x + dx, y + dy, z + dz, j) - 1
                    if particle_id >= 0:
                        w_ij = self.isoweight(pos[i] - pos[particle_id], 2 * smooth_radius)
                        total_weight += w_ij
                        x_wi += w_ij * pos[particle_id]

            if total_weight != 0:
                x_wi /= total_weight

            self.x_bar[i] = (1 - self.akLambda) * pos[i] + self.akLambda * x_wi

            # Now we start to calculate C_i
            num_neighbor_particles = 0
            C_i = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            for dx, dy, dz in ti.ndrange(smooth_range, smooth_range, smooth_range):
                dx -= smooth_offset
                dy -= smooth_offset
                dz -= smooth_offset
                if x + dx < 0 or y + dy < 0 or z + dz < 0:
                    continue
                particle_len = self.voxel.particle_len(x + dx, y + dy, z + dz)
                for j in range(particle_len):
                    particle_id = self.voxel.particle_id(x + dx, y + dy, z + dz, j) - 1
                    w_ij = self.isoweight(pos[i] - pos[particle_id], 2 * smooth_radius)
                    if w_ij > 0 and particle_id >= 0:
                        num_neighbor_particles += 1
                        dp = pos[particle_id] - x_wi
                        C_i += w_ij * dp @ dp.transpose()

            if total_weight != 0:
                C_i /= total_weight

            # Now we perform SVD to obtain C = R Sigma R^T
            R, Sigma, RT = ti.svd(C_i)

            # Then, we see if we need to modify C_i
            # condition is  sigma_1 >= kr * sigma_d
            if num_neighbor_particles > self.Nvep:
                Sigma[1, 1] = ti.max(Sigma[1, 1], Sigma[0, 0] / self.kr)
                Sigma[2, 2] = ti.max(Sigma[2, 2], Sigma[0, 0] / self.kr)
                Sigma *= self.ks
            else:
                Sigma[0, 0] = self.kn
                Sigma[1, 1] = self.kn
                Sigma[2, 2] = self.kn

            self.G[i] = (1 / smooth_radius) * R @ Sigma.inverse() @ RT.transpose()

            if isnan(self.G[i].determinant()):
                num_neighbor_particles = 0
                for dx, dy, dz in ti.ndrange(smooth_range, smooth_range, smooth_range):
                    dx -= smooth_offset
                    dy -= smooth_offset
                    dz -= smooth_offset
                    if x + dx < 0 or y + dy < 0 or z + dz < 0:
                        continue
                    particle_len = self.voxel.particle_len(x + dx, y + dy, z + dz)
                    for j in range(particle_len):
                        particle_id = self.voxel.particle_id(x + dx, y + dy, z + dz, j) - 1
                        w_ij = self.isoweight(pos[i] - pos[particle_id], 2 * smooth_radius)
                        if w_ij > 0 and particle_id >= 0:
                            num_neighbor_particles += 1
                            dp = pos[particle_id] - x_wi
                            val = dp @ dp.transpose()
                            print("weight: {}, dp: {}, {}, {}, current_id {}, particle_id:{}".format(w_ij, dp[0], dp[1], dp[2], i, particle_id))
                print("neighbors: {}".format(num_neighbor_particles))
                print(C_i)



    #@brief
    # Implementation of the modified cubic spline kernel
    #
    # Traditional cubic spline kernel is
    #
    #        6 * sigma3 * (q^3 - q^2) + 1, q in (0, 0.5]
    # W(q) = 2 * sigma3 * (1 - q)^3      , q in (0.5, 1]
    #        0                          , otherwise
    #
    # where q = ||r|| / h and sigma3 = 8 / pi h^3
    #
    # here we see in our anisotropic version, we have
    # sigma = 8 / pi

    @ti.func
    def cubic_spline_kernel(self, dx, G):
        res = 0.0
        q = (G @ dx).norm()

        if 0 < q <= 0.5:
            res = 48 / np.pi * G.determinant() * (q * q * q - q * q) + 1
        elif 0.5 < q <= 1:
            res = 16 / np.pi * G.determinant() * (1 - q) * (1 - q) * (1 - q) 


        return res

    def generate_smooth_fluid_field(self, smooth_radius, num_particles, pos, velocity = None):
        print("Pre-Processing Data")
        self.pre_process_data(smooth_radius, num_particles, pos)
        print("Filling Fluid Field Data with Velocity Data")
        if self.record_velocity: 
            self.generate_smooth_fluid_field_with_velocity_impl(smooth_radius, num_particles, pos, velocity)
        else:
            self.generate_smooth_fluid_field_impl(smooth_radius, num_particles, pos)

    @ti.kernel
    def generate_smooth_fluid_field_impl(self, smooth_radius: ti.f32, num_particles: ti.i32, pos: ti.template()):
        smooth_range = ti.ceil(4 * smooth_radius / self.voxel.voxel_size, dtype=ti.i32) + 1
        smooth_offset = smooth_range // 2

        for i in range(num_particles):
            x, y, z = self.voxel.get_voxel_index(pos[i])
            for dx, dy, dz in ti.ndrange(smooth_range, smooth_range, smooth_range):
                nx = x + dx - smooth_offset
                ny = y + dy - smooth_offset
                nz = z + dz - smooth_offset
            
                for j in range(8):
                    vx, vy, vz = self.voxel.get_voxel_vertex_position(nx, ny, nz, j)
                    vertex_mapped_position = self.voxel.get_voxel_vertex_mapped_position(nx, ny, nz, j)
                    if vx < 0 or vy < 0 or vz < 0:
                        continue

                    self.voxel.voxel_vertex_value[vx, vy, vz] += self.volume * self.cubic_spline_kernel(self.x_bar[i] - vertex_mapped_position, self.G[i])



    @ti.kernel
    def generate_smooth_fluid_field_with_velocity_impl(self, smooth_radius: ti.f32, num_particles: ti.i32, pos: ti.template(), velocity : ti.template()):
        smooth_range = ti.ceil(4 * smooth_radius / self.voxel.voxel_size, dtype=ti.i32) + 1
        smooth_offset = smooth_range // 2

        contribution_coeff = self.voxel.voxel_size * self.voxel.voxel_size * self.voxel.voxel_size

        for i in range(num_particles):
            x, y, z = self.voxel.get_voxel_index(pos[i])
            v = velocity[i]
            for dx, dy, dz in ti.ndrange(smooth_range, smooth_range, smooth_range):
                nx = x + dx - smooth_offset
                ny = y + dy - smooth_offset
                nz = z + dz - smooth_offset
            
                for j in range(8):
                    vx, vy, vz = self.voxel.get_voxel_vertex_position(nx, ny, nz, j)
                    vertex_mapped_position = self.voxel.get_voxel_vertex_mapped_position(nx, ny, nz, j)
                    if vx < 0 or vy < 0 or vz < 0:
                        continue

                    contribution_factor = self.poly6_weight(pos[i] - vertex_mapped_position, smooth_radius) * contribution_coeff
                    self.voxel.voxel_vertex_value[vx, vy, vz] += self.volume * self.cubic_spline_kernel(self.x_bar[i] - vertex_mapped_position, self.G[i])
                    self.voxel.voxel_vertex_velocity[vx, vy, vz, 0] += v[0] *  contribution_factor
                    self.voxel.voxel_vertex_velocity[vx, vy, vz, 1] += v[1] *  contribution_factor
                    self.voxel.voxel_vertex_velocity[vx, vy, vz, 2] += v[2] *  contribution_factor

    @ti.kernel
    def update_particles(self, num_particles : ti.i32, pos : ti.template()):
        for i in range(num_particles):
            self.voxel.add_particle(pos[i], i + 1)

    ##################################################################################
    #                                                                                #
    #                                                                                #
    #                     Implementation for Normal Kernels                          #
    #                                                                                #
    #                                                                                #
    ##################################################################################

    #@brief:
    # Calculate weighted smoothing kernel, we are using this formula:
    #               --  A(h^2 - ||r||^2)^3    , if 0 <= ||r|| <= h
    #     W(r, h) = |
    #               --  0                     , if ||r|| > h
    #
    # where A = 315/(64 * PI * h^9) in our 3D case
    @ti.func
    def poly6_weight(self, r: ti.template(), h: ti.f32) -> ti.f32:
        rnorm = r.norm()
        res = 0.0
        if rnorm <= h:
            # variable used to reduce repeated calculation
            tmp = h * h
            res = tmp - rnorm * rnorm
            res = res * res * res
            # calculate h^9
            tmp = tmp * tmp * tmp * tmp * h
            res = self.Acoeff * res / tmp
        else:
            res = 0
        return res

    #@brief
    def reset(self):
        self.num_indices[None] = 0
        self.num_vertices[None] = 0
        self.G.fill(0)
        self.x_bar.fill(0)
        self.voxel.deactivate()

    #@brief
    # Return grid index in x, y, z
    @ti.kernel
    def generate_fluid_field(self, smooth_radius: ti.f32, num_particles : ti.i32, pos: ti.template()):
        for i in range(num_particles):
            x, y, z = self.voxel.get_voxel_index(pos[i])
            for dx, dy, dz in ti.ndrange(3, 3, 3):
                nx = x + dx - 1
                ny = y + dy - 1
                nz = z + dz - 1
                if nx < 0 or ny < 0 or nz < 0:
                    continue
                vertex_position = self.voxel.get_mapped_position(nx, ny, nz)
                self.voxel.voxel_vertex_value[nx, ny, nz] += self.volume * self.poly6_weight(pos[i] - vertex_position, smooth_radius)

    @ti.func
    def vertex_interpolate(self, isolevel, p1, p2, val1, val2, eps=0.00001):
        offset = 0.0
        delta = val2 - val1

        if ti.abs(delta) < eps:
            offset = 0.5
        else:
            offset = (isolevel - val1) / delta
        return p1 + offset * (p2 - p1)

    def dual_contouring(self, isolevel, smooth_radius, num_particles, pos, velocity = None, use_smooth_fluid_field = True,
                       record_normals = True):
        print("------------------------------------------------------------")
        print("Resetting Previous Frame Data")
        self.reset()

        print("Updating Particle Positions")
        self.update_particles(num_particles, pos)

        print("Generating Fluid Field")
        if use_smooth_fluid_field:
            self.generate_fluid_field(smooth_radius, num_particles, pos)
            self.generate_smooth_fluid_field(smooth_radius, num_particles, pos, velocity)
        else:
            self.generate_fluid_field(smooth_radius, num_particles, pos)

        self.smooth_sdf(iteration=1, radius=1)

        print("Performing Dual Contouring")
        self.dual_contouring_impl(isolevel, 0.01)

        print("Performing Dual Polygen")
        self.dual_contouring_polygen(isolevel)

        if self.record_normals:
            print("Building Mesh Normal Data")
            self.process_normal()

        if self.record_velocity:
            print("Building Mesh Velocity Data")
            self.process_velocity()

        print("------------------------------------------------------------")


    @ti.func
    def calc_vertex_normal(self, x, y, z):
       norm = ti.Vector([0.0, 0.0, 0.0])
       if x == 0:
           norm[0] = -self.voxel.voxel_vertex_value[x + 1, y, z]
       else:
           norm[0] = self.voxel.voxel_vertex_value[x - 1, y, z] - self.voxel.voxel_vertex_value[x + 1, y, z]
       if y == 0:
           norm[1] = -self.voxel.voxel_vertex_value[x, y + 1, z]
       else:
           norm[1] = self.voxel.voxel_vertex_value[x, y - 1, z] - self.voxel.voxel_vertex_value[x, y + 1, z]
       if z == 0:
           norm[2] = -self.voxel.voxel_vertex_value[x, y, z + 1]
       else:
           norm[2] = self.voxel.voxel_vertex_value[x, y, z - 1] - self.voxel.voxel_vertex_value[x, y, z + 1]
       return self.normalize(norm)


    @ti.func
    def trilinear_interpolate(self, x_d, y_d, z_d, c000, 
                              c001, c010, c100, c011, c101, c110, c111):
        c00 = c000 * (1 - x_d) + c100 * x_d
        c01 = c001 * (1 - x_d) + c101 * x_d
        c10 = c010 * (1 - x_d) + c110 * x_d
        c11 = c011 * (1 - x_d) + c111 * x_d

        c0 = c00 * (1 - y_d) + c10 * y_d
        c1 = c01 * (1 - y_d) + c11 * y_d

        return c0 * (1 - z_d) + c1 * z_d

    
    @ti.func
    def process_normal_at(self, pos: ti.template()):
        x0, y0, z0 = self.voxel.get_voxel_index(pos)
        x_d, y_d, z_d = (pos - self.voxel.get_mapped_position(x0, y0, z0)) / self.voxel.voxel_size

        c000 = self.calc_vertex_normal(x0, y0, z0)
        c100 = self.calc_vertex_normal(x0 + 1, y0, z0)
        c010 = self.calc_vertex_normal(x0, y0 + 1, z0)
        c001 = self.calc_vertex_normal(x0, y0, z0 + 1)
        c110 = self.calc_vertex_normal(x0 + 1, y0 + 1, z0)
        c011 = self.calc_vertex_normal(x0, y0 + 1, z0 + 1)
        c101 = self.calc_vertex_normal(x0 + 1, y0, z0 + 1)
        c111 = self.calc_vertex_normal(x0 + 1, y0 + 1, z0 + 1)

        return self.normalize(self.trilinear_interpolate(x_d, y_d, z_d, c000, c001, c010, c100, 
                                         c011, c101, c110, c111))


    @ti.func
    def get_velocity_at(self, x, y, z):
        return ti.Vector([self.voxel.voxel_vertex_velocity[x, y, z, 0], 
                         self.voxel.voxel_vertex_velocity[x, y, z, 1], 
                         self.voxel.voxel_vertex_velocity[x, y, z, 2]])

    @ti.func
    def process_velocity_at(self, pos: ti.template()):
        x0, y0, z0 = self.voxel.get_voxel_index(pos)
        x_d, y_d, z_d = (pos - self.voxel.get_mapped_position(x0, y0, z0)) / self.voxel.voxel_size

        c000 = self.get_velocity_at(x0, y0, z0)
        c100 = self.get_velocity_at(x0 + 1, y0, z0)
        c010 = self.get_velocity_at(x0, y0 + 1, z0)
        c001 = self.get_velocity_at(x0, y0, z0 + 1)
        c110 = self.get_velocity_at(x0 + 1, y0 + 1, z0)
        c011 = self.get_velocity_at(x0, y0 + 1, z0 + 1)
        c101 = self.get_velocity_at(x0 + 1, y0, z0 + 1)
        c111 = self.get_velocity_at(x0 + 1, y0 + 1, z0 + 1)

        return self.trilinear_interpolate(x_d, y_d, z_d, c000, c001, c010, c100,
                                         c011, c101, c110, c111)

    @ti.kernel
    def process_velocity(self):
        for i in range(self.num_vertices[None]):
            v = self.process_velocity_at(self.mesh_vertex[i])
            self.mesh_velocity[i] = v

    @ti.kernel
    def process_normal(self):
        for i in range(self.num_vertices[None]):
            # Use Tri-linear interpolation to perform the work
            # Use formula here: https://en.wikipedia.org/wiki/Trilinear_interpolation
            # First get the current grid cell index / grid vert bottom corner
            pos = self.mesh_vertex[i]
            x0, y0, z0 = self.voxel.get_voxel_index(pos)
            x_d, y_d, z_d = (pos - self.voxel.get_mapped_position(x0, y0, z0)) / self.voxel.voxel_size

            c000 = self.calc_vertex_normal(x0, y0, z0)
            c100 = self.calc_vertex_normal(x0 + 1, y0, z0)
            c010 = self.calc_vertex_normal(x0, y0 + 1, z0)
            c001 = self.calc_vertex_normal(x0, y0, z0 + 1)
            c110 = self.calc_vertex_normal(x0 + 1, y0 + 1, z0)
            c011 = self.calc_vertex_normal(x0, y0 + 1, z0 + 1)
            c101 = self.calc_vertex_normal(x0 + 1, y0, z0 + 1)
            c111 = self.calc_vertex_normal(x0 + 1, y0 + 1, z0 + 1)

            c00 = c000 * (1 - x_d) + c100 * x_d
            c01 = c001 * (1 - x_d) + c101 * x_d
            c10 = c010 * (1 - x_d) + c110 * x_d
            c11 = c011 * (1 - x_d) + c111 * x_d

            c0 = c00 * (1 - y_d) + c10 * y_d
            c1 = c01 * (1 - y_d) + c11 * y_d

            self.mesh_normal[i] = self.normalize(c0 * (1 - z_d) + c1 * z_d)

    @ti.func
    def normalize(self, v: ti.template()) -> ti.template():
        res = ti.Vector([0.0, 0.0, 0.0])
        if v.norm() != 0:
            res = v / v.norm()
        return res


    @ti.kernel
    def print_nonzero(self, which: ti.i32):
        for i, j, k in self.voxel.voxel_vertex_value:
            if self.voxel.voxel_vertex_value[i, j, k] != 0:
                print("Which {}, Nonzero {}".format(which, self.voxel.voxel_vertex_value[i, j, k]))

    def smooth_sdf(self, iteration = 1, radius = 1):
        sd = self.density_field_stats()
        for i in range(iteration):
            print("Smoothing Iteration {}".format(i))
            self.smooth_sdf_impl(radius, sd)
            self.copy_and_clear()
        
    @ti.func
    def gaussian_blur(self, distance: ti.f32, sd: ti.f32):
        return ti.exp(-distance * distance / (2 * sd * sd)) / ti.pow(2 * np.pi * sd * sd, 1.5)

    @ti.kernel
    def density_field_stats(self) -> ti.f32:
        mean = 0.0
        cnt = 0

        for i, j, k in self.voxel.voxel_vertex_value:
            if self.voxel.voxel_vertex_value[i, j, k] < 0:
                mean += self.voxel.voxel_vertex_value[i, j, k]
                cnt += 1
        mean /= cnt

        variance = 0.0
        for i, j, k in self.voxel.voxel_vertex_value:
            if self.voxel.voxel_vertex_value[i, j, k] < 0:
                variance += (self.voxel.voxel_vertex_value[i, j, k] - mean) * (self.voxel.voxel_vertex_value[i, j, k] - mean)

        sd = ti.sqrt(variance / cnt)

        return sd

    @ti.kernel
    def smooth_sdf_impl(self, radius: ti.i32, sd: ti.f32):
        # TODO: try implement a Gaussian blur

        for i, j, k in self.voxel.voxel_vertex_value:
            # We are indexing into neighbor vetices
            avg = 0.0
            weight = 0.0
            for dx, dy, dz in ti.ndrange(2 * radius + 1, 2 * radius + 1, 2 * radius + 1):
                dx -= radius;
                dy -= radius
                dz -= radius
                x, y, z = ti.Vector([i + dx, j + dy, k + dz])  

                if x < 0 or y < 0 or z < 0:
                    continue

                diff = ti.abs(self.voxel.voxel_vertex_value[i, j, k] - self.voxel.voxel_vertex_value[x, y, z]) 
                gaussian_weight = self.gaussian_blur(diff, sd)
                avg += gaussian_weight * self.voxel.voxel_vertex_value[x, y, z]
                weight += gaussian_weight

            if weight != 0.0:
                avg /= weight

            # print("generating avg / weight: {} / {}, with sd: {}".format(avg, weight, sd))
            self.voxel.voxel_value[i, j, k] = avg


    def erode_sdf(self, iteration = 1):
        for i in range(iteration):
            self.erode_sdf_impl()

    @ti.kernel
    def erode_sdf_impl(self):
        for i, j, k in self.voxel.voxel_vertex_value:
            value = self.voxel.voxel_vertex_value[i, j, k]
            for dx, dy, dz in ti.ndrange(3):
                pass


    def dilate_sdf(self, iteration = 1):
        for i in range(iteration):
            print("Dilating Sdf Iteration #{}".format(i))
            self.dilate_sdf_impl()

    @ti.kernel
    def visit_existing_voxel(self):
        for i, j, k in self.voxel.voxel_vertex_value:
            self.voxel.visit_voxel(i, j, k, 2)

                                
    def copy_and_clear(self):
        self.voxel.voxel_vertex_value_levels[0].deactivate_all()
        self.copy_sdf()
        self.voxel.voxel_value_levels[0].deactivate_all()

    @ti.kernel
    def copy_sdf(self):
        for i, j, k in self.voxel.voxel_value:
            self.voxel.voxel_vertex_value[i, j, k] = self.voxel.voxel_value[i, j, k]
        



    @ti.kernel
    def rasterize_particles(self, num_particles: ti.i32, particle_radius: ti.f32, pos: ti.template()):
        for i in range(num_particles):
            x, y, z = self.voxel.get_voxel_index(pos[i])

            for dx, dy, dz in ti.ndrange(3, 3, 3):
                nx = x + dx - 1
                ny = y + dy - 1
                nz = z + dz - 1
                if nx < 0 or ny < 0 or nz < 0:
                    continue
                
                self.voxel.voxel_vertex_value[nx, ny, nz] = -1



    @ti.kernel
    def dual_contouring_impl(self, isolevel: ti.f32, bias: ti.f32):
        for x, y, z in self.voxel.voxel_vertex_value:
            # Indexing into the (x, y, z) cube
            cube_index = 0
            for vertex_index in range(8):
                if self.voxel.get_voxel_vertex_value(x, y, z, vertex_index) < isolevel:
                    cube_index |= 1 << vertex_index

            # Has intersection with grid
            if self.mc_tables.edge_table[cube_index] != 0:
                A = ti.Matrix(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0]
                    ]
                )
                cnt = 0
                mean_point = ti.Vector([0.0, 0.0, 0.0])
                for w in range(12):
                    if self.mc_tables.edge_table[cube_index] & (1 << w):
                        ind0, ind1 = self.interpolation_table[w]
                        mean_point += self.vertex_interpolate(isolevel,
                            self.voxel.get_voxel_vertex_mapped_position(x, y, z, ind0),
                            self.voxel.get_voxel_vertex_mapped_position(x, y, z, ind1),
                            self.voxel.get_voxel_vertex_value(x, y, z, ind0),
                            self.voxel.get_voxel_vertex_value(x, y, z, ind1)
                        )
                        cnt += 1


                mean_point /= 1.0 * cnt
                cnt = 0

                for w in range(12):
                    if self.mc_tables.edge_table[cube_index] & (1 << w):
                        ind0, ind1 = self.interpolation_table[w]
                        intersect_point = self.vertex_interpolate(isolevel,
                            self.voxel.get_voxel_vertex_mapped_position(x, y, z, ind0),
                            self.voxel.get_voxel_vertex_mapped_position(x, y, z, ind1),
                            self.voxel.get_voxel_vertex_value(x, y, z, ind0),
                            self.voxel.get_voxel_vertex_value(x, y, z, ind1)
                        )
                        n = self.process_normal_at(intersect_point)
                        for j in range(3):
                            A[cnt, j] = n[j]
                        intersect_point -= mean_point
                        np = n.dot(intersect_point)
                        A[cnt, 3] = np
                        cnt += 1


                Q, Ahat = householder_decomp(A) 
                # if (Ahat - Q.transpose() @ A).norm() > 0.01:
                #     print("Error Too Large")

                res = solve_qef(Ahat)
                nx, ny, nz = self.voxel.get_voxel_index(res + mean_point)
                if nx != x or ny != y or nz != z:
                    res = mean_point
                else:
                    res += mean_point

                indv = ti.atomic_add(self.num_vertices[None], 1)
                # Record the position information
                self.voxel.dc_point[x, y, z, 0] = res[0]
                self.voxel.dc_point[x, y, z, 1] = res[1]
                self.voxel.dc_point[x, y, z, 2] = res[2]

                # Record the index information
                self.voxel.dc_point[x, y, z, 3] = indv + 1



    @ti.func
    def get_dual_contour_point(self, ind):
        x = ind[0]
        y = ind[1]
        z = ind[2]
        res = ti.Vector([0.0, 0.0, 0.0])
        res[0] = self.voxel.dc_point[x, y, z, 0]
        res[1] = self.voxel.dc_point[x, y, z, 1]
        res[2] = self.voxel.dc_point[x, y, z, 2]
        return res, int(self.voxel.dc_point[x, y, z, 3])

    # @detail
    # currently we are at i, j, k cube
    # and we are finding cubes that are adjacent to edge with corner index u, v
    @ti.func
    def get_edge_adjacent_cubes(self, i, j, k, u, v):
        #@detail:
        # Grids are arranged so that:
        # Grid Index: [0, 0, 0]      <-------> The grid at lower-left corner
        # Grid Vert Index: [0, 0, 0] <-------> The vertex coordinate at the lower-left corner
        # Each grid has vertex id from 0 - 7 indicating the 8 vertices, and the ids are arrange as:
        #
        #
        #       4 ---------- 5
        #       / |        /|
        #      /  |       / |
        #     7----------6  |
        #     |   |      |  |
        #     |   0------|--1
        #     |  /       | /
        #     | /        |/
        #     3----------2
        # 
        # 3 -> 2 indicates the positive x direction
        # 3 -> 0 indicates the positive y direction
        # 3 -> 7 indicates the positive z direction
        if u > v:
            tmp = v
            v = u
            u = tmp
        
        coord0 = ti.Vector([i, j, k])
        coord1 = coord0
        coord2 = coord0
        coord3 = coord0

        # y + 1, z + 1
        if u == 4 and v == 5:
            coord1 = coord0 + ti.Vector([0, 1, 0])
            coord2 = coord0 + ti.Vector([0, 1, 1])
            coord3 = coord0 + ti.Vector([0, 0, 1])
        # x - 1, y + 1
        elif u == 0 and v == 4:
            coord1 = coord0 + ti.Vector([-1, 0, 0])
            coord2 = coord0 + ti.Vector([-1, 1, 0])
            coord3 = coord0 + ti.Vector([0, 1, 0])
        # x - 1, z + 1
        elif u == 4 and v == 7:
            coord1 = coord0 + ti.Vector([-1, 0, 0])
            coord2 = coord0 + ti.Vector([-1, 0, 1])
            coord3 = coord0 + ti.Vector([0, 0, 1])
        # x - 1, y - 1
        elif u == 3 and v == 7:
            coord1 = coord0 + ti.Vector([0, -1, 0])
            coord2 = coord0 + ti.Vector([-1, -1, 0])
            coord3 = coord0 + ti.Vector([-1, 0, 0])
        # y - 1, z + 1 
        elif u == 6 and v == 7:
            coord1 = coord0 + ti.Vector([0, -1, 0])
            coord2 = coord0 + ti.Vector([0, -1, 1])
            coord3 = coord0 + ti.Vector([0, 0, 1])
        # x + 1, y - 1
        elif u == 2 and v == 6:
            coord1 = coord0 + ti.Vector([0, -1, 0])
            coord2 = coord0 + ti.Vector([1, -1, 0])
            coord3 = coord0 + ti.Vector([1, 0, 0])
        # y - 1, z - 1
        elif u == 2 and v == 3:
            coord1 = coord0 + ti.Vector([0, 0, -1])
            coord2 = coord0 + ti.Vector([0, -1, -1])
            coord3 = coord0 + ti.Vector([0, -1, 0])
        # x + 1, z - 1
        elif u == 1 and v == 2:
            coord1 = coord0 + ti.Vector([1, 0, 0])
            coord2 = coord0 + ti.Vector([1, 0, -1])
            coord3 = coord0 + ti.Vector([0, 0, -1])
        # y + 1, z - 1
        elif u == 0 and v == 1:
            coord1 = coord0 + ti.Vector([0, 0, -1])
            coord2 = coord0 + ti.Vector([0, 1, -1])
            coord3 = coord0 + ti.Vector([0, 1, 0])
        # x + 1, y + 1
        elif u == 1 and v == 5:
            coord1 = coord0 + ti.Vector([0, 1, 0])
            coord2 = coord0 + ti.Vector([1, 1, 0])
            coord3 = coord0 + ti.Vector([1, 0, 0])
        # x + 1, z + 1
        elif u == 5 and v == 6:
            coord1 = coord0 + ti.Vector([0, 0, 1])
            coord2 = coord0 + ti.Vector([1, 0, 1])
            coord3 = coord0 + ti.Vector([1, 0, 0])
        # x - 1, z - 1
        elif u == 0 and v == 3:
            coord1 = coord0 + ti.Vector([0, 0, -1])
            coord2 = coord0 + ti.Vector([-1, 0, -1])
            coord3 = coord0 + ti.Vector([-1, 0, 0])

        return coord0, coord1, coord2, coord3

    @ti.kernel
    def dual_contouring_polygen(self, isolevel: ti.f32):
        for x, y, z in self.voxel.voxel_vertex_value:
            cube_index = 0
            
            for vertex_index in range(8):
                if self.voxel.get_voxel_vertex_value(x, y, z, vertex_index) < isolevel:
                    cube_index |= 1 << vertex_index

            # # Has intersection with grid
            if self.mc_tables.edge_table[cube_index] != 0:
                for w in range(12):
                    if w != 11 and w != 3 and w != 2:
                        continue
                    if self.mc_tables.edge_table[cube_index] & (1 << w):
                        ind0, ind1 = self.interpolation_table[w]
                        dcp1, dcp2, dcp3, dcp4 = self.get_edge_adjacent_cubes(x, y, z, ind0, ind1)

                        p1, i1 = self.get_dual_contour_point(dcp1)
                        p2, i2 = self.get_dual_contour_point(dcp2)
                        p3, i3 = self.get_dual_contour_point(dcp3)
                        p4, i4 = self.get_dual_contour_point(dcp4)


                        cond0 = (cube_index & (1 << ind0)) != 0
                        flip = False
                        if cond0 != (w == 11):
                            flip = True

                        if i1 != 0 and i2 != 0 and i3 != 0 and i4 != 0:
                            i1 -= 1
                            i2 -= 1
                            i3 -= 1
                            i4 -= 1
                            self.mesh_vertex[i1] = p1
                            self.mesh_vertex[i2] = p2
                            self.mesh_vertex[i3] = p3
                            self.mesh_vertex[i4] = p4

                            indi = ti.atomic_add(self.num_indices[None], 3)
                            if not flip:
                                self.mesh_index[indi] = i3
                                self.mesh_index[indi + 1] = i2
                                self.mesh_index[indi + 2] = i1
                            else:
                                self.mesh_index[indi] = i1
                                self.mesh_index[indi + 1] = i2
                                self.mesh_index[indi + 2] = i3

                        
                            indi = ti.atomic_add(self.num_indices[None], 3)
                            if not flip:
                                self.mesh_index[indi] = i4
                                self.mesh_index[indi + 1] = i3
                                self.mesh_index[indi + 2] = i1
                            else:
                                self.mesh_index[indi] = i1
                                self.mesh_index[indi + 1] = i3
                                self.mesh_index[indi + 2] = i4

    @ti.kernel
    def marching_cube_impl(self, isolevel: ti.f32):
        for i, j, k in self.voxel.voxel_vertex_value:
            for neighbor_index in range(8):
                x, y, z = ti.Vector([i, j, k]) - self.voxel.voxel_vertex_offset_map[neighbor_index]
                if x < 0 or y < 0 or z < 0:
                    continue

                if self.voxel.visit_voxel(x, y, z) == 0:
                    cube_index = 0
                    for vertex_index in range(8):
                        if self.voxel.get_voxel_vertex_value(x, y, z, vertex_index) < isolevel:
                            cube_index |= 1 << vertex_index

                    if self.mc_tables.edge_table[cube_index] != 0:
                        tri_vertex_index = ti.Vector([0, 0, 0, 0, 0, 0, 0, 0,
                                                      0, 0, 0, 0, 0, 0, 0, 0])
                        for w in ti.static(range(12)):
                            if self.mc_tables.edge_table[cube_index] & (1 << w):
                                vertex_index = ti.atomic_add(self.num_vertices[None], 1)
                                ind0, ind1 = self.interpolation_table[w]
                                self.mesh_vertex[vertex_index] = self.vertex_interpolate(isolevel,
                                    self.voxel.get_voxel_vertex_mapped_position(x, y, z, ind0),
                                    self.voxel.get_voxel_vertex_mapped_position(x, y, z, ind1),
                                    self.voxel.get_voxel_vertex_value(x, y, z, ind0),
                                    self.voxel.get_voxel_vertex_value(x, y, z, ind1)
                                )
                                tri_vertex_index[w] = vertex_index

                        w = 0
                        while self.mc_tables.tri_table[cube_index, w] != -1:
                            index = ti.atomic_add(self.num_indices[None], 3)
                            self.mesh_index[index + 2] = tri_vertex_index[self.mc_tables.tri_table[cube_index, w]]
                            self.mesh_index[index + 1] = tri_vertex_index[self.mc_tables.tri_table[cube_index, w + 1]]
                            self.mesh_index[index + 0] = tri_vertex_index[self.mc_tables.tri_table[cube_index, w + 2]]
                            w += 3


    def export_point_cloud(self, file_path: str):
        assert file_path[-3:] == "ply", "Output file must be in ply format!"
        pc = o3d.geometry.PointCloud()
        point_data = np.ndarray(shape=(self.num_vertices[None], 3), dtype=np.float64)
        taichi_tools.copy_vertex_field_to_array(self.mesh_vertex, point_data, self.num_vertices[None])
        pc.points = o3d.utility.Vector3dVector(point_data)
        o3d.io.write_point_cloud(file_path, pc)



    def export_mesh(self, file_path: str, output_normal = False, output_velocity = False):
        assert file_path[-3:] == "ply", "Output file must be in ply format!"

        # print("Number of Vertices: {}, Number of Indices: {}".format(self.num_vertices[None], self.num_indices[None]))

        vertex_data = self.mesh_vertex.to_numpy()[:self.num_vertices[None]]
        index_data = np.reshape(self.mesh_index.to_numpy()[:self.num_indices[None]], (-1, 3))
        print("Number of Vertices: {}, Number of Indices: {}".format(vertex_data.shape, index_data.shape))
        # velocity_data = self.mesh_velocity.to_numpy()[:self.num_vertices[None]]

        # vertex_data = np.ndarray(shape=(self.num_vertices[None],3), dtype=np.float64)
        # index_data = np.ndarray(shape=(self.num_indices[None] // 3, 3), dtype=int)
        # taichi_tools.copy_vertex_field_to_array(self.mesh_vertex, vertex_data, self.num_vertices[None])
        # taichi_tools.copy_index_field_to_array(self.mesh_index, index_data, self.num_indices[None])

        ply_exporter = ti.tools.PLYWriter(self.num_vertices[None], self.num_indices[None] // 3)
        ply_exporter.add_vertex_pos(vertex_data[:, 0], vertex_data[:, 1], vertex_data[:, 2])
        ply_exporter.add_faces(index_data)

        if output_normal:
            print("Outputing normals!")
            normal_data = self.mesh_normal.to_numpy(np.float64)[:self.num_vertices[None]]
            # normal_data = np.ndarray(shape=(self.num_vertices[None],3), dtype=np.float64)
            # taichi_tools.copy_vertex_field_to_array(self.mesh_normal, normal_data, self.num_vertices[None])
            ply_exporter.add_vertex_normal(normal_data[:, 0], normal_data[:, 1], normal_data[:, 2])

        if output_velocity:
            print("Outputing velocity!")
            velocity_data = self.mesh_velocity.to_numpy(np.float64)[:self.num_vertices[None]]

            # velocity_data = np.ndarray(shape=(self.num_vertices[None],3), dtype=np.float64)
            # taichi_tools.copy_vertex_field_to_array(self.mesh_velocity, velocity_data, self.num_vertices[None])
            ply_exporter.add_vertex_channel("vx", "float", velocity_data[:, 0])
            ply_exporter.add_vertex_channel("vy", "float", velocity_data[:, 1])
            ply_exporter.add_vertex_channel("vz", "float", velocity_data[:, 2])


        ply_exporter.export(file_path)


