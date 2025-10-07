from .align_fn import abs_rel
import numpy as np
import cv2

class Aligner:
    def __init__(self, inv_pose, gt_pose, scale):
        self.inv_pose = inv_pose
        self.gt_pose = gt_pose
        self.scale = scale

    @staticmethod
    def build_1p1d(cls, pose, gt_pose, depth, gt_depth, align_fn=abs_rel, min_depth=0.0, max_depth=np.inf):
        mask = (gt_depth > min_depth) & (gt_depth < max_depth)
        scale = align_fn(gt_depth[mask], depth[mask])
        return cls(np.linalg.inv(pose), gt_pose, scale)

    def align_depth(self, depth):
        return depth * self.scale

    def align_pose(self, pose):
        pose = self.inv_pose @ pose
        pose[:3, 3] *= self.scale
        pose = self.gt_pose @ pose
        return pose

