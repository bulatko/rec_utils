# RecUtils
Library with usefull instruments for 3D reconstruction


## Installation

```
pip install -e .
```


## Run DUSt3R model

```
from rec_utils.datasets import DUSt3RDataset, ScanNetDataset
from rec_utils.reconstruction.dust3r import DUSt3RModel, run_dust3r_inference, export_dust3r_scene
from pathlib import Path


root_dir = "SCANNET_ROOT_DIR"

dataset = ScanNet(
    root_dir=root_dir,
    split="all",
)


device = "cuda:0"
posed = False # True to pass poses into dust3r
output_dir = Path("OUTPUT_DIR")

image_size = 512
num_images = 40
model = DUSt3RModel.from_pretrained().to(device)


for index in range(len(dataset)):
    scene = dataset[index]

    output_path = output_dir / scene.id

    optimizer = run_dust3r_inference(
        model=model,
        scene=scene,
        image_size=image_size,
        device=device,
        num_images=num_images,
        silent=False,
        posed=args.posed,
    )

    export_dust3r_scene(optimizer=optimizer, output_dir=output_path)

# now you can access dust3r scans via DUSt3RDataset & DUSt3RFrame

dust3r_dataset = DUSt3RDataset(
    root_dir=output_dir
)
len(dust3r_dataset)
# 1512

vertices, colors = dust3r_dataset[0].get_pcd()


```

## Aligning

```
from rec_utils.datasets import DUSt3RDataset, ScanNetDataset
from rec_utils.aligner import build_aligner_1p1d
import open3d as o3d

scannet_path = "PATH_TO_SCANNET_DATASET"
dust3r_path = "PATH_TO_DUST3R_DATASET"

scene_id = "scene0000_00"

scannet_dataset = ScanNetDataset(scnnet_path)
dust3r_dattaset = DUSt3RDataset(dust3r_path)

scannet_scene = scannet_dataset[scene_id]
dust3r_scene = dust3r_dataset[scene_id]

aligner = build_aligner_1p1d(source_scene=dust3r_scene, target_scene=scannet_scene)
aligner.align(dust3r_scene, inplace=True)

# how much frames use to build point cloud
frames_range = range(0, len(dust3r_scene), 3)

vertices, colors = dust3r_scene.get_pcd(frames_range)

# save point cloud

out_path = "./pcd.ply"

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices)
pcd.colors = o3d.utility.Vector3dVector(color / 255.0)
pcd = pcd.voxel_down_sample(voxel_size=0.025)
o3d.io.write_point_cloud(out_path, pcd)
```