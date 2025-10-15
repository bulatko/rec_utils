import numpy as np
import os

def read_bin(bin_path):
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"File {bin_path} does not exist")
    
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 6)

def write_bin(bin_path, xyz, rgb):
    if not os.path.exists(bin_path.parent):
        raise FileNotFoundError(f"File {bin_path} does not exist")
    
    rgb = np.clip(rgb, 0, 255)[:, :3]
    if rgb.max() <= 1:
        rgb = (rgb * 255)

    points = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
    points.tofile(bin_path)