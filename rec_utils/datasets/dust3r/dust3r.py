import torch
import trimesh as tm
from pathlib import Path
from typing import Union, Optional
from rec_utils.structures import Scene, Frame
import numpy as np
from rec_utils.structures.utils import adjust_intrinsics
import cv2


def load_dust3r_scene(scene_dir):
    output_params = torch.load(scene_dir / 'output_params.pt', weights_only=True)
    scene_params = torch.load(scene_dir / 'scene_params.pt', weights_only=True)

    return output_params, scene_params

class DUSt3RDataset:
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
        self.scenes[index] = DUSt3RScene(self.root_dir / self.scene_ids[index])
        return self.scenes[index]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.scene_id2index[index]
        if isinstance(index, int):
            index = index
            return self.load_scene(index)

        raise ValueError(f"Invalid index type {type(index)}")


    def __repr__(self):
        return f"DUSt3RDataset(root_dir={self.root_dir}, num_scenes={len(self.scenes)})"

class DUSt3RScene(Scene):
    def __init__(self, root_dir: Path):
        super().__init__(root_dir)

    def get_frame_list(self):
        self.frames = []
        output_params, scene_params = load_dust3r_scene(self.root_dir)

        poses = scene_params['poses']
        pts3d = scene_params['pts3d']
        Ks = scene_params['Ks']
        image_names = scene_params['image_files']
        confs = scene_params['im_conf']
        depths = scene_params['depths']
        
        for i in range(len(image_names)):
            frame_id = Path(image_names[i]).stem
            image_path = image_names[i]
            pose = poses[i]
            intrinsics = Ks[i]
            depth = depths[i]
            confidence = confs[i]
            output_params = output_params

            self.frames.append(DUSt3RFrame(image_path=image_path, pose=pose, intrinsics=intrinsics, depth=depth, confidence=confidence, output_params=output_params))

        return self.frames


class DUSt3RFrame(Frame):
    def __init__(self, image_path, pose, intrinsics, depth, confidence, output_params):
        pose = pose.numpy()
        intrinsics = intrinsics.numpy()
        depth = depth.numpy()
        confidence = confidence.numpy()

        super().__init__(image_path=image_path, pose=pose, depth_intrinsics=intrinsics)
        self.confidence = confidence
        self.output_params = output_params
        self._depth = depth

    @property
    def depth(self):
        return self._depth

    @property
    def frame_id(self):
        return Path(self.image_path).stem

    def get_pcd(self, confidence_threshold=0):
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

        y, x = np.where((depth > 0) & (self.confidence > confidence_threshold))
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