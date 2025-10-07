import numpy as np
import cv2
import open3d as o3d
from rec_utils.structures.utils import adjust_intrinsics

class Frame:
    def __init__(self, image_path=None, depth_path=None, pose=None, image_intrinsics=None, depth_intrinsics=None, depth_scale=1.0):
        self.image_path = image_path
        self.depth_path = depth_path
        self._pose = self._preprocess_pose(pose)
        self._image_intrinsics = self._preprocess_intrinsics(image_intrinsics)
        self._depth_intrinsics = self._preprocess_intrinsics(depth_intrinsics)
        self._intrinsics = None
        self._depth_scale = depth_scale
        self._image = None
        self._depth = None

    def _preprocess_pose(self, matrix):
        if matrix is not None:
            matrix = np.asarray(matrix)
            if self._check_matrix_shape(matrix, possible_shapes=[(3, 3), (3, 4), (4, 4)]):
                return self._extend_matrix(matrix)
            else:
                raise ValueError(f"Pose matrix has invalid shape {matrix.shape}")
        return None

    def _preprocess_intrinsics(self, matrix):
        if matrix is not None:
            matrix = np.asarray(matrix)
            if self._check_matrix_shape(matrix, possible_shapes=[(2, 2), (3, 3), (4, 4)]):
                return self._extend_matrix(matrix)
            else:
                raise ValueError(f"Intrinsics matrix has invalid shape {matrix.shape}")
        return None

    @property
    def frame_id(self):
        raise NotImplementedError("Not implemented")

    @property
    def unscaled_intrinsics(self):
        if self._intrinsics is not None:
            return self._intrinsics
        
        if self._image_intrinsics is None and self._depth_intrinsics is None:
            raise ValueError("Image and depth intrinsics are both None")

        if self._image_intrinsics is not None and self._depth_intrinsics is not None:
            img2depth = adjust_intrinsics(self._image_intrinsics, self.image_shape, self.depth_shape)
            depth2img = adjust_intrinsics(self._depth_intrinsics, self.depth_shape, self.image_shape)
            if not np.isclose(img2depth, depth2img).all():
                raise ValueError("Image and depth intrinsics are not the same")
            self._intrinsics = adjust_intrinsics(self._image_intrinsics, self.image_shape, (1, 1))
            return self._intrinsics

        if self._image_intrinsics is not None:
            self._intrinsics = adjust_intrinsics(self._image_intrinsics, self.image_shape, (1, 1))
            return self._intrinsics
        if self._depth_intrinsics is not None:
            self._intrinsics = adjust_intrinsics(self._depth_intrinsics, self.depth_shape, (1, 1))
            return self._intrinsics

        return self._intrinsics

    def align(self, inv_pose: np.ndarray, gt_pose: np.ndarray, scale: float):

        if type(inv_pose) != np.ndarray:
            raise ValueError("Inv pose is not a numpy array")
        if type(gt_pose) != np.ndarray:
            raise ValueError("Gt pose is not a numpy array")
        if type(scale) not in [float, int]:
            raise ValueError("Scale is not a float or int")

        inv_pose = self._preprocess_pose(inv_pose)
        gt_pose = self._preprocess_pose(gt_pose)
        if self._depth is None:
            raise ValueError("Depth is None")
        if self._pose is None:
            raise ValueError("Pose is None")

        self._pose = inv_pose @ self._pose
        self._pose[:3, 3] *= scale
        self._pose = gt_pose @ self._pose
        self._depth = self._depth * scale

    def _check_matrix_shape(self, matrix, possible_shapes):
        if matrix.shape not in possible_shapes:
            return False
        return True

    def _extend_matrix(self, matrix):
        result = np.eye(4)
        if matrix is not None:
            result[:matrix.shape[0], :matrix.shape[1]] = matrix
        return result

    @property
    def image(self):
        if self.image_path is None:
            return None
        if self._image is None:
            image = cv2.imread(self.image_path)
            self._image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self._image

    @property
    def depth(self):
        if self.depth_path is None:
            return None
        if self._depth is None:
            self._depth = cv2.imread(self.depth_path, cv2.IMREAD_UNCHANGED) / self._depth_scale
        return self._depth
    
    @property
    def pose(self):
        if self._pose is None:
            return None
        return self._pose

    def shape(self, item):
        if item is None:
            raise ValueError(f"Item is None")
        return item.shape

    @property
    def image_shape(self):
        return self.shape(self.image)

    @property
    def depth_shape(self):
        return self.shape(self.depth)

    def get_pcd(self):
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

        y, x = np.where(depth > 0)
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

