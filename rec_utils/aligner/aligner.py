from .align_fn import abs_rel
import numpy as np
from copy import deepcopy
from rec_utils.structures import Scene
import cv2


class Aligner:
    def __init__(self, inv_pose, gt_pose, scale):
        self.inv_pose = inv_pose
        self.gt_pose = gt_pose
        self.scale = scale

    @classmethod
    def build_1p1d(cls, source_pose, target_pose, source_depth, target_depth, align_fn=abs_rel, min_depth=0.0, max_depth=np.inf):
        mask = (target_depth > min_depth) & (target_depth < max_depth)
        source_depth = cv2.resize(source_depth, (target_depth.shape[1], target_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        scale = align_fn(source_depth[mask], target_depth[mask])
        
        return cls(np.linalg.inv(source_pose), target_pose, scale)

    def align(self, scene: Scene, inplace=False):
        if not inplace:
            scene = deepcopy(scene)
        
        scene.align(self.inv_pose, self.gt_pose, self.scale)
        return scene

def build_aligner_1p1d(source_scene: Scene, target_scene: Scene, align_fn=abs_rel, min_depth=0.0, max_depth=np.inf):
    """
    Builds an aligner for a 1 pose and 1 depth frame alignment.
    Args:
        source_scene: Source scene that should be aligned
        target_scene: Target scene
        align_fn: Alignment function
        min_depth: Minimum depth
        max_depth: Maximum depth
    Returns:
        Aligner
    """
    pose_ids = None
    depth_ids = None


    for source_index in range(len(source_scene)):
        source_frame = source_scene[source_index]
        for target_index in range(len(target_scene)):
            target_frame = target_scene[target_index]
            if source_frame.frame_id == target_frame.frame_id:
                if source_frame.pose is not None and target_frame.pose is not None:
                    pose_ids = (source_index, target_index)
                if source_frame.depth is not None and target_frame.depth is not None:
                    depth_ids = (source_index, target_index)
                if pose_ids is not None and depth_ids is not None:
                    break
        if pose_ids is not None and depth_ids is not None:
            break
    if pose_ids is None or depth_ids is None:
        raise ValueError("No common frames found")

    source_pose_id, target_pose_id = pose_ids
    source_depth_id, target_depth_id = depth_ids

    source_pose = source_scene.poses[source_pose_id]
    target_pose = target_scene.poses[target_pose_id]

    source_depth = source_scene.depths[source_depth_id]
    target_depth = target_scene.depths[target_depth_id]
    
    return Aligner.build_1p1d(source_pose=source_pose, target_pose=target_pose, source_depth=source_depth, target_depth=target_depth, align_fn=align_fn, min_depth=min_depth, max_depth=max_depth)
