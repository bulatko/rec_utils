from rec_utils.datasets import HiSLAM2Dataset, ScanNetPPDataset
from rec_utils.aligner import build_aligner_1p1d
import open3d as o3d
import random

scannetpp_path = "/home/jovyan/users/lemeshko/processed_data/scannetpp"
hislam2_path = "/home/jovyan/users/kolodiazhnyi/zakharov/Indoor/SLAM/HI-SLAM2/outputs/scannet_pp"

# scene_id = "0a5c013435"
scene_id = "0cf2e9402d"

scannetpp_dataset = ScanNetPPDataset(scannetpp_path)
hislam2_dataset = HiSLAM2Dataset(hislam2_path)

scannetpp_scene = scannetpp_dataset[scene_id]
hislam2_scene = hislam2_dataset[scene_id]

# aligner = build_aligner_1p1d(source_scene=hislam2_scene, target_scene=scannetpp_scene)
# aligned_scene = aligner.align(hislam2_scene, inplace=False)

# how much frames use to build point cloud
frames_range = range(0, len(hislam2_scene), 3)

vertices, colors = hislam2_scene.get_pcd(frames_range)
# vertices, colors = aligned_scene.get_pcd(frames_range)
# sample
random_idxs = random.sample(range(vertices.shape[0]), k=200000)
vertices, colors = vertices[random_idxs], colors[random_idxs]

# save point cloud
out_path = "./pcd.ply"

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices)
pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
pcd = pcd.voxel_down_sample(voxel_size=0.025)
o3d.io.write_point_cloud(out_path, pcd)