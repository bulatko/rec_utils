from rec_utils.structures import Scene, Frame
from rec_utils.structures.utils import adjust_intrinsics
from pathlib import Path
from typing import Union
import numpy as np
import cv2
from rec_utils.datasets.arkit.utils import determine_rotation_from_poses, rotate_pose, rotate_image, process_unscaled_intrinsics
from rec_utils.datasets import DUSt3RDataset, DUSt3RScene, DUSt3RFrame
from rec_utils.datasets.dust3r import load_dust3r_scene

class ARKitDataset:
    def __init__(self, root_dir: Union[str, Path]):
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
            
        self.ids = [a.name for a in root_dir.iterdir() if a.is_dir()]
        self.root_dir = root_dir
        self.scenes = [None for _ in range(len(self.ids))]
        self.scene_id2index = {scene_id: index for index, scene_id in enumerate(self.ids)}
    
    def __len__(self):
        return len(self.scenes)
    
    def load_scene(self, index):
        if self.scenes[index] is not None:
            return self.scenes[index]
        self.scenes[index] = ARKitPosedImagesScene(self.root_dir / self.ids[index])
        return self.scenes[index]

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.scene_id2index[index]
        if isinstance(index, int):
            index = index
            return self.load_scene(index)

        raise ValueError(f"Invalid index type {type(index)}")

    def __repr__(self):
        return f"ARKitDataset(root_dir={self.root_dir}, num_scenes={len(self.scenes)})"



class ARKitPosedImagesScene(Scene):

    def get_frame_list(self):
        self.frames = []
        image_paths = list(self.root_dir.glob("*.jpg"))
        intrinsics = np.loadtxt(self.root_dir / "intrinsic.txt")
        poses = [np.loadtxt(self.root_dir / f"{frame_path.stem}.txt") for frame_path in image_paths]
        rotation_angle = determine_rotation_from_poses(poses)
        self.rotation_angle = rotation_angle
        for image_path in image_paths:
            frame_id = Path(image_path).stem
            color_path = self.root_dir / f"{frame_id}.jpg"
            depth_path = self.root_dir / f"{frame_id}.png"
            pose = np.loadtxt(self.root_dir / f"{frame_id}.txt")

            self.frames.append(ARKitFrame(image_path=color_path, depth_path=depth_path, pose=pose, image_intrinsics=intrinsics, depth_scale=1000.0, rotation_angle=rotation_angle))
        self.frames.sort(key=lambda x: x.frame_id)
        return self.frames


class ARKitFrame(Frame):
    def __init__(self, *args, rotation_angle: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.rotation_angle = rotation_angle
        self._pose = rotate_pose(self._pose, self.rotation_angle)

    @property
    def frame_id(self):
        return Path(self.image_path).stem


    @property
    def image(self):
        if self.image_path is None:
            return None
        if self._image is None:
            image = cv2.imread(self.image_path)
            self.__H, self.__W = image.shape[:2]
            self._image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self._image = rotate_image(self._image, self.rotation_angle)

        return self._image

    @property
    def depth(self):
        if self.depth_path is None:
            return None
        if self._depth is None:
            self._depth = cv2.imread(self.depth_path, cv2.IMREAD_UNCHANGED) / self._depth_scale
            self.__dH, self.__dW = self._depth.shape[:2]
            self._depth = rotate_image(self._depth, self.rotation_angle)

        return self._depth

    
    @property
    def unscaled_intrinsics(self):
        if self._intrinsics is not None:
            return self._intrinsics

        if self._image_intrinsics is None and self._depth_intrinsics is None:
            raise ValueError("Image and depth intrinsics are both None")

        if self._image_intrinsics is not None and self._depth_intrinsics is not None:

            base_image_shape = (self.__H, self.__W)
            base_depth_shape = (self.__dH, self.__dW)
            img2depth = adjust_intrinsics(self._image_intrinsics, base_image_shape, base_depth_shape)
            depth2img = adjust_intrinsics(self._depth_intrinsics, base_depth_shape, base_image_shape)
            if not np.isclose(img2depth, depth2img).all():
                raise ValueError("Image and depth intrinsics are not the same")
            self._intrinsics = adjust_intrinsics(self._image_intrinsics, base_image_shape, (1, 1))
            return self._intrinsics

        if self._image_intrinsics is not None:
            self._intrinsics = adjust_intrinsics(self._image_intrinsics, (self.__H, self.__W), (1, 1))
            self._intrinsics = process_unscaled_intrinsics(self._intrinsics, self.rotation_angle)
            return self._intrinsics
        if self._depth_intrinsics is not None:
            self._intrinsics = adjust_intrinsics(self._depth_intrinsics, (self.__dH, self.__dW), (1, 1))
            self._intrinsics = process_unscaled_intrinsics(self._intrinsics, self.rotation_angle)
            return self._intrinsics

        return self._intrinsics



class ARKitDUSt3RFrame(DUSt3RFrame):
    def __init__(self, image_path, pose, intrinsics, depth, confidence, output_params):
        super().__init__(image_path, pose, intrinsics, depth, confidence, output_params)
        if "rotation_angle" not in output_params:
            raise ValueError("Rotation angle is not in output params")
        self.rotation_angle = output_params["rotation_angle"]

    @property
    def image(self):
        if self.image_path is None:
            return None
        if self._image is None:
            image = cv2.imread(self.image_path)
            self._image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self._image = rotate_image(self._image, self.rotation_angle)

        return self._image

class ARKitDUSt3RDataset(DUSt3RDataset):

    def load_scene(self, index):
        if self.scenes[index] is not None:
            return self.scenes[index]
        self.scenes[index] = ARKitDUSt3RScene(self.root_dir / self.scene_ids[index])
        return self.scenes[index]


class ARKitDUSt3RScene(DUSt3RScene):
    
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

            self.frames.append(ARKitDUSt3RFrame(image_path=image_path, pose=pose, intrinsics=intrinsics, depth=depth, confidence=confidence, output_params=output_params))

        return self.frames