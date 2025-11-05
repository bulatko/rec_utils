from rec_utils.structures import Scene
from pathlib import Path
import numpy as np
import cv2
import os
import open3d as o3d

def export_mask_clustering(scene: Scene, output_dir: Path, voxel_size: float = 0.025, depth_prep_fn = lambda depth: (depth * 1000).astype(np.uint16), confidence_threshold=3, depth_truncation=np.inf):
    output_dir.mkdir(parents=True, exist_ok=True)
    K_color = scene.frames[0].image_intrinsics
    K_depth = scene.frames[0].depth_intrinsics

    intrinsics_path = output_dir / "intrinsic"
    intrinsics_path.mkdir(parents=True, exist_ok=True)
    np.savetxt(intrinsics_path / "intrinsic_color.txt", K_color)
    np.savetxt(intrinsics_path / "intrinsic_depth.txt", K_depth)

    np.savetxt(intrinsics_path / "extrinsic_color.txt", np.eye(4))
    np.savetxt(intrinsics_path / "extrinsic_depth.txt", np.eye(4))

    depth_dir = output_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    color_dir = output_dir / "color"
    color_dir.mkdir(parents=True, exist_ok=True)
    pose_dir = output_dir / "pose"
    pose_dir.mkdir(parents=True, exist_ok=True)
    
    for frame in scene.frames:

        assert frame.image_path is not None, "Image is None"
        assert frame.depth_path is not None or frame.depth is not None, "Depth is None"
        assert frame.pose is not None, "Pose is None"

        img_name = Path(frame.image_path).stem
        image_path = color_dir / f"{img_name}.jpg"
        depth_path = depth_dir / f"{img_name}.png"
        pose_path = pose_dir / f"{img_name}.txt"


        if frame.image_path is not None and frame.image_path.exists():
            try:
                os.symlink(frame.image_path, image_path)
            except FileExistsError:
                pass
        elif frame.image is not None:
            cv2.imwrite(image_path, cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR))
        else:
            raise ValueError(f"Image not found for frame {frame.frame_id}")

        if frame.depth_path is not None and frame.depth_path.exists():
            try:
                os.symlink(frame.depth_path, depth_path)
            except FileExistsError:
                pass
        elif frame.depth is not None:
            depth = frame.masked_depth(confidence_threshold=confidence_threshold)
            depth[depth > depth_truncation] = 0
            
            cv2.imwrite(depth_path, depth_prep_fn(depth))
        else:
            raise ValueError(f"Depth not found for frame {frame.frame_id}")

        np.savetxt(pose_path, frame.pose)

    vertices, colors = scene.get_pcd(confidence_threshold=confidence_threshold, depth_truncation=depth_truncation)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    o3d.io.write_point_cloud(output_dir / f"{scene.id}.ply", pcd)
    
    return np.asarray(pcd.points), np.asarray(pcd.colors)