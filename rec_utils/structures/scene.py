import numpy as np

class _IndexableProperty:
    """Вспомогательный класс для доступа к свойствам фреймов по индексу"""
    def __init__(self, frames, attr_name):
        self.frames = frames
        self.attr_name = attr_name
    
    def __getitem__(self, index):
        return getattr(self.frames[index], self.attr_name)
    
    def __len__(self):
        return len(self.frames)

    def __repr__(self):
        return f"IndexableProperty(attr_name={self.attr_name})"

class Scene:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.frames = self.get_frame_list()

    def get_frame_list(self):
        raise NotImplementedError("Not implemented")

    def __len__(self):
        return len(self.frames)

    @property
    def images(self):
        return _IndexableProperty(self.frames, 'image')

    @property
    def depths(self):
        return _IndexableProperty(self.frames, 'depth')

    @property
    def poses(self):
        return _IndexableProperty(self.frames, 'pose')

    def _filter_frames(self, condition = lambda x: True):
        self.frames = [frame for frame in self.frames if condition(frame)]

    def filter_valid_poses(self):
        self._filter_frames(lambda x: x.pose is not None)

    def filter_valid_depths(self):
        self._filter_frames(lambda x: x.depth_path is not None or x.depth is not None)

    def align(self, inv_pose, gt_pose, scale):
        for frame in self.frames:
            frame.align(inv_pose, gt_pose, scale)
    
    def __getitem__(self, index):
        return self.frames[index]

    def get_pcd(self, range_frames=None, ignore_error=True, **frame_kwargs):
        if range_frames is None:
            range_frames = range(len(self))
        all_vertices = []
        all_colors = []
        for i in range_frames:
            frame = self.frames[i]
            try:
                vertices, colors = frame.get_pcd(**frame_kwargs)
                all_vertices.append(vertices)
                all_colors.append(colors)
            except ValueError as e:
                if ignore_error:
                    continue
                raise e
        all_vertices = np.concatenate(all_vertices, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        return all_vertices, all_colors