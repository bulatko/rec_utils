import torch
import trimesh as tm
from pathlib import Path
from typing import Union, Optional
from rec_utils.structures import Scene, Frame
import numpy as np
from rec_utils.structures.utils import adjust_intrinsics
import cv2
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import os



def to_se3_matrix(pvec):
    pose = np.eye(4)
    pose[:3, :3] = R.from_quat(pvec[4:]).as_matrix()
    pose[:3, 3] = pvec[1:4]
    return pose


def load_intrinsic_extrinsic(result, stamps):
    c = np.load(f'{result}/intrinsics.npy')
    intrinsic = o3d.core.Tensor([[c[0], 0, c[2]], [0, c[1], c[3]], [0, 0, 1]], dtype=o3d.core.Dtype.Float64)
    poses = np.loadtxt(f'{result}/traj_full.txt')
    poses = [to_se3_matrix(poses[int(s)]) for s in stamps]
    poses = list(map(lambda x: o3d.core.Tensor(x, dtype=o3d.core.Dtype.Float64), poses))
    return intrinsic, poses

class HiSLAM2Dataset:
    def __init__(self, root_dir: Path, scenes: Union[list[str], Optional[None]]=None):
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        self.root_dir = root_dir
        self.scene_ids = scenes if scenes is not None else [a.stem for a in root_dir.iterdir() if a.is_dir()]
        self.scenes = [None for _ in range(len(self.scene_ids))]

        self.scene_id2index = {scene_id: index for index, scene_id in enumerate(self.scene_ids)}

    def load_scene(self, index):
        if self.scenes[index] is not None:
            return self.scenes[index]
        self.scenes[index] = HiSLAM2Scene(self.root_dir / self.scene_ids[index])
        return self.scenes[index]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.scene_id2index[index]

        
        return self.load_scene(index)


    def __repr__(self):
        return f"HiSLAM2Dataset(root_dir={self.root_dir}, num_scenes={len(self.scenes)})"

class HiSLAM2Scene(Scene):
    def __init__(self, root_dir: Path):
        super().__init__(root_dir)

    def get_frame_list(self):
        self.frames = []
        image_names = os.listdir(os.path.join(self.root_dir, 'renders', 'image_after_opt'))
        stamps = [float(os.path.basename(i)[:-4]) for i in image_names]
        intrinsic, poses = load_intrinsic_extrinsic(self.root_dir, stamps)
        
        for i in range(len(image_names)):
            image_path = os.path.join(self.root_dir, 'renders', 'image_after_opt', os.path.basename(image_names[i]))
            pose = poses[i]
            intrinsics = intrinsic
            confidence = 1.0
            output_params = None

            self.frames.append(HiSLAM2Frame(image_path=image_path, pose=pose, intrinsics=intrinsics, depth_path=depth_path, confidence=confidence, output_params=output_params))

        return self.frames


class HiSLAM2Frame(Frame):
    def __init__(self, image_path, pose, intrinsics, depth_path, confidence, output_params):
        pose = pose.numpy()
        intrinsics = intrinsics.numpy()
        super().__init__(image_path=image_path, pose=pose, depth_intrinsics=intrinsics)
        self.confidence = confidence
        self.output_params = output_params
        self.depth_path = depth_path
        self._depth = None

    @property
    def depth(self):
        if self.depth_path is None:
            return None
        if self._depth is None:
            self._depth = cv2.imread(self.depth_path, cv2.IMREAD_ANYDEPTH) / 6553.5
        return self._depth

    @property
    def frame_id(self):
        return Path(self.image_path).stem
    
    def align(self, inv_pose: np.ndarray, gt_pose: np.ndarray, scale: float):
        if self._depth is None:
            _ = self.depth
        
        super().align(inv_pose, gt_pose, scale)

    def get_pcd(self, confidence_threshold=0, depth_truncation=np.inf):
        pose = np.eye(4)
        if self.pose is not None:
            pose = self.pose
            
        
        if self._image_intrinsics is None and self._depth_intrinsics is None:
            raise ValueError("Image and depth intrinsics are both None")
        image = self.image
        depth = self.depth

        if depth is None:
            raise ValueError("Depth is None")
        if image is None:
            image = np.zeros((depth.shape[0], depth.shape[1], 3))

        intrinsics = self.unscaled_intrinsics
        depth_intrinsics = adjust_intrinsics(intrinsics, (1, 1), self.depth_shape)
        depth_shape = self.depth_shape
        image = cv2.resize(image, (depth_shape[1], depth_shape[0]), interpolation=cv2.INTER_LINEAR)

        y, x = np.where((depth > 0) & (self.confidence > confidence_threshold) & (depth < depth_truncation))
        colors = image[y, x]
        pixel_coords = np.vstack([x, y, np.ones(len(x))])
        normalized_coords = np.linalg.inv(depth_intrinsics)[:3, :3] @ pixel_coords

        z = depth[y, x]

        camera_points_3d = normalized_coords * z
        camera_points_homogeneous = np.vstack([camera_points_3d, np.ones(len(x))])
        world_points_homogeneous = pose @ camera_points_homogeneous

        points = world_points_homogeneous[:3, :] / world_points_homogeneous[3, :]
        points = points.T 

        return points, colors