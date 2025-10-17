import open3d as o3d
import numpy as np
from typing import Optional, Union
from pathlib import Path


def write_pcd(path: Union[str, Path], vertices: np.ndarray, colors: Optional[np.ndarray], voxel_size: Optional[float] = None) -> None:

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size)
    o3d.io.write_point_cloud(str(path), pcd)