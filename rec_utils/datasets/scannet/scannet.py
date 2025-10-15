from rec_utils.structures import Scene, Frame
from pathlib import Path
import json
from .splits import TRAIN_SPLIT, VAL_SPLIT, ALL_SPLIT
from typing import Union
import numpy as np

SPLIT_MAP = {
    "train": TRAIN_SPLIT,
    "val": VAL_SPLIT,
    "all": ALL_SPLIT
}

class ScanNetDataset:
    def __init__(self, root_dir: Union[str, Path], split: Union[str, list[str]] = "all", posed_images: bool = False):
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        if split not in ["train", "val", "all"]:
            print("try to use custom split")
            self.split = "custom"
            ids = split
            for index in ids:
                if index not in ALL_SPLIT:
                    raise ValueError(f"Scene {index} not found")
            self.ids = ids
        else:
            self.split = split
            self.ids = SPLIT_MAP[split]
        self.posed_images = posed_images
        self.root_dir = root_dir
        self.scenes = [None for _ in range(len(self.ids))]
        self.scene_id2index = {scene_id: index for index, scene_id in enumerate(self.ids)}
    
    def __len__(self):
        return len(self.scenes)
    
    def load_scene(self, index):
        if self.scenes[index] is not None:
            return self.scenes[index]
        if self.posed_images:
            self.scenes[index] = ScanNetPosedImagesScene(self.root_dir / self.ids[index])
        else:
            self.scenes[index] = ScanNetScene(self.root_dir / self.ids[index])
        return self.scenes[index]

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.scene_id2index[index]
            
        return self.load_scene(index)

        

    def __repr__(self):
        return f"ScanNetDataset(root_dir={self.root_dir}, split={self.split}, num_scenes={len(self.scenes)})"



class ScanNetScene(Scene):
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        super().__init__(root_dir)

    def get_frame_list(self):
        self.frames = []
        info_path = self.root_dir / "info.json"
        with open(info_path, "r") as f:
            info = json.load(f)
        color_dir = self.root_dir / "color"
        depth_dir = self.root_dir / "depth"
        intrinsics = info["intrinsics"]
        for frame in info["frames"]:
            frame_id = Path(frame["filename_color"]).stem
            color_path = color_dir / f"{frame_id}.jpg"
            depth_path = depth_dir / f"{frame_id}.png"
            pose = frame["pose"]

            self.frames.append(ScanNetFrame(image_path=color_path, depth_path=depth_path, pose=pose, image_intrinsics=intrinsics, depth_scale=1000.0))
        return self.frames

    def __repr__(self):
        return f"ScanNetScene(root_dir={self.root_dir}, num_frames={len(self.frames)})"


class ScanNetPosedImagesScene(ScanNetScene):

    def get_frame_list(self):
        self.frames = []
        image_paths = self.root_dir.glob("*.jpg")
        intrinsics = np.loadtxt(self.root_dir / "intrinsic.txt")
        for image_path in image_paths:
            frame_id = Path(image_path).stem
            color_path = self.root_dir / f"{frame_id}.jpg"
            depth_path = self.root_dir / f"{frame_id}.png"
            pose = np.loadtxt(self.root_dir / f"{frame_id}.txt")

            self.frames.append(ScanNetFrame(image_path=color_path, depth_path=depth_path, pose=pose, image_intrinsics=intrinsics, depth_scale=1000.0))
        return self.frames

class ScanNetFrame(Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    @property
    def frame_id(self):
        return Path(self.image_path).stem