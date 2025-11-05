from rec_utils.structures import Scene, Frame
from pathlib import Path
from .splits import TRAIN_SPLIT, VAL_SPLIT, ALL_SPLIT, V2_SPLIT
from typing import Union
import os
from .utils import read_cameras_text, read_images_text

SPLIT_MAP = {
    "train": TRAIN_SPLIT,
    "val": VAL_SPLIT,
    "all": ALL_SPLIT,
    'v2': V2_SPLIT
}

class ScanNetPPDataset:
    def __init__(self, root_dir: Union[str, Path], data_dir: Union[str, Path], split: Union[str, list[str]] = "all"):
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        if split not in SPLIT_MAP.keys():
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
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.scenes = [None for _ in range(len(self.ids))]
        self.scene_id2index = {scene_id: index for index, scene_id in enumerate(self.ids)}
    
    def __len__(self):
        return len(self.scenes)
    
    def load_scene(self, index):
        if self.scenes[index] is not None:
            return self.scenes[index]
        self.scenes[index] = ScanNetPPScene(self.root_dir / self.ids[index], self.data_dir / self.ids[index])
        return self.scenes[index]

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.scene_id2index[index]
        
        return self.load_scene(index)


    def __repr__(self):
        return f"ScanNetPPDataset(root_dir={self.root_dir}, split={self.split}, num_scenes={len(self.scenes)})"




class ScanNetPPScene(Scene):
    def __init__(self, root_dir: Path, data_dir: Path):
        self.root_dir = root_dir
        self.data_dir = data_dir
        super().__init__(root_dir)

    def get_frame_list(self):
        cameras = read_cameras_text(self.data_dir / 'iphone' / 'colmap' / "cameras.txt")
        images = read_images_text(self.data_dir / 'iphone' / 'colmap' / "images.txt")
        self.frames = []
        color_dir = self.root_dir / "iphone" / "rgb"
        depth_dir = self.root_dir / "iphone" / "render_depth"

        for frame in images:
            frame_id = Path(frame["frame_id"]).stem
            color_path = color_dir / f"{frame_id}.jpg"
            if not color_path.exists():
                continue
            depth_path = depth_dir / f"{frame_id}.png"
            depth_path = depth_path if depth_path.exists() else None
            pose = frame["pose"]
            intrinsics = cameras[frame["camera_id"]]

            frame = ScanNetPPFrame(image_path=color_path, depth_path=depth_path, pose=pose, depth_scale=1000.0)
            frame._intrinsics = intrinsics

            self.frames.append(frame)
        return self.frames

    def __repr__(self):
        return f"ScanNetPPScene(root_dir={self.root_dir}, num_frames={len(self.frames)})"


class ScanNetPPFrame(Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    @property
    def frame_id(self):
        return self.image_path.stem