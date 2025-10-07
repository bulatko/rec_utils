import numpy as np

def abs_rel(gt_depth, depth):
    return np.median(gt_depth / depth)
