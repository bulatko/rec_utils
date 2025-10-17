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
                    "pose": np.linalg.inv(qt2pose(qvec, tvec))
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
                model = elems[1]
                w, h = int(elems[2]), int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                intrinsic = np.eye(4)

                if model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"]:
                    fx, cx, cy = params[:3]
                    fy = fx
                elif model in ["PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"]:
                    fx, fy, cx, cy = params[:4]
                else:
                    raise ValueError(f"Invalid camera model: {model}")

                intrinsic[0, 0] = fx / w
                intrinsic[0, 2] = cx / w
                
                intrinsic[1, 1] = fy / h
                intrinsic[1, 2] = cy / h


                cameras[camera_id] = intrinsic
    return cameras
