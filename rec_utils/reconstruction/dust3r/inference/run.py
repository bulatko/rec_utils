from .grs import get_reconstructed_scene
from ..model import DUSt3RModel
from rec_utils.structures import Scene
from typing import Tuple
import numpy as np
from typing import Union, List
from pathlib import Path
import torch
from ..cloud_opt import PointCloudOptimizer, ModularPointCloudOptimizer, PairViewer


def run_dust3r_inference(
    model: DUSt3RModel, 
    scene: Scene,
    num_images: int,
    image_size: int,  
    device, 
    schedule: str = "linear", 
    niter: int = 300,
    silent: bool = False, 
    scenegraph_type: str = "complete", 
    winsize: int = 1, 
    refid: int = 0, 
    posed: bool = False):

    indices = np.arange(len(scene))
    if len(scene) > num_images:
        indices = np.linspace(0, len(scene) - 1, num_images).astype(int)
    image_files = [str(scene[i].image_path) for i in indices]
    poses = None
    if posed:
        poses = [scene[i].pose for i in indices]

    optimizer = get_reconstructed_scene(model, device, silent, image_size, image_files, schedule, niter, scenegraph_type, winsize, refid, poses)

    return optimizer



def export_dust3r_scene(optimizer: Union[PointCloudOptimizer, ModularPointCloudOptimizer, PairViewer], output_dir: Union[str, Path], output_params=None):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {
        'scene_params': output_dir / 'scene_params.pt',
        'output_params': output_dir / 'output_params.pt'
    }

    # Сохранение только необходимых параметров сцены
    scene_params = {}

    scene_params['image_files'] = optimizer.image_files
    # Позиции камер
    if hasattr(optimizer, 'get_im_poses'):
        scene_params['poses'] = optimizer.get_im_poses().detach().cpu()
    elif hasattr(optimizer, 'poses'):
        scene_params['poses'] = optimizer.poses.detach().cpu() if torch.is_tensor(optimizer.poses) else optimizer.poses

    # Калибровочные матрицы
    if hasattr(optimizer, 'get_intrinsics'):
        scene_params['Ks'] = optimizer.get_intrinsics().detach().cpu()
    elif hasattr(optimizer, 'Ks'):
        scene_params['Ks'] = optimizer.Ks.detach().cpu() if torch.is_tensor(optimizer.Ks) else optimizer.Ks

    # Карты глубины
    if hasattr(optimizer, 'get_depthmaps'):
        depthmaps = optimizer.get_depthmaps()
        scene_params['depths'] = [d.detach().cpu() if torch.is_tensor(d) else d for d in depthmaps]

    # 3D точки
    if hasattr(optimizer, 'get_pts3d'):
        scene_params['pts3d'] = [p.detach().cpu() if torch.is_tensor(p) else p for p in optimizer.get_pts3d()]

    # Карты уверенности
    if hasattr(optimizer, 'im_conf'):
        scene_params['im_conf'] = [c.detach().cpu() if torch.is_tensor(c) else c for c in optimizer.im_conf]

    # Размеры изображений
    if hasattr(optimizer, 'imshapes'):
        scene_params['imshapes'] = optimizer.imshapes



    # Сохранение параметров сцены
    torch.save(scene_params, saved_files['scene_params'])
    torch.save(output_params, saved_files['output_params'])


    