import numpy as np

def abs_rel(source_depth, target_depth):
    return float(np.median(target_depth / source_depth))
