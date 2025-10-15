import numpy as np
from rec_utils.functions import qt2pose


def read_images_text(path):
    images = []
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                    tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images.append({
                    "frame_id": image_name.split(".")[0],
                    "camera_id": camera_id,
                    "pose": qt2pose(qvec, tvec)
                })
    return images


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                params = np.array(tuple(map(float, elems[4:])))
                intrinsic = np.eye(3)
                intrinsic[0, 0] = params[0]
                intrinsic[1, 1] = params[1]
                intrinsic[0, 2] = params[2]
                intrinsic[1, 2] = params[3]
                cameras[camera_id] = intrinsic
    return cameras
