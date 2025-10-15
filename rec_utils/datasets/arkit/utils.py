import numpy as np
import cv2
from typing import Union

def rotate_image(image: Union[np.ndarray, None], rotation_angle: int) -> np.ndarray:
    if image is None:
        return None
    if rotation_angle == 0:
        rotated_img = image
        
    elif rotation_angle == 90:
        rotated_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        
    elif rotation_angle == 180:
        rotated_img = cv2.rotate(image, cv2.ROTATE_180)
        
    elif rotation_angle == 270:
        rotated_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    else:
        raise ValueError(f"Invalid rotation angle: {rotation_angle}. Must be 0, 90, 180, or 270.")

    return rotated_img

def rotate_pose(pose: Union[np.ndarray, None], rotation_angle: int) -> np.ndarray:
    """
    Применяет коррекцию позы при повороте изображения.
    
    Когда изображение поворачивается на угол α, система координат камеры
    должна повернуться на -α (в обратную сторону), чтобы проекция осталась корректной.
    Поэтому применяется инверсия матрицы поворота.
    """
    if pose is None:
        return None
    
    if rotation_angle == 0:
        return pose
    elif rotation_angle not in [90, 180, 270]:
        raise ValueError(f"Invalid rotation angle: {rotation_angle}. Must be 0, 90, 180, or 270.")
    
    angle_rad = np.radians(rotation_angle)  # поворот изображения
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    # Матрица cam_to_rotated: поворот системы координат камеры
    cam_to_rotated = np.array([
        [cos_a, -sin_a, 0, 0],
        [sin_a, cos_a, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    # Применяем инверсию: rotated_to_cam
    # pose_cam_to_world @ rotated_to_cam = pose_rotated_cam_to_world
    rotated_to_cam = np.linalg.inv(cam_to_rotated)
    
    return pose @ rotated_to_cam

def process_unscaled_intrinsics(intrinsics: np.ndarray, rotation_angle) -> np.ndarray:
    """
    Обрабатывает нормализованные интринсики (приведенные к размеру изображения 1x1).
    
    При повороте изображения меняются:
    - фокусные расстояния fx, fy (swap при 90°/270°)
    - центры камеры cx, cy согласно геометрии поворота
    """
    H, W = 1, 1  # нормализованные координаты
    if intrinsics is None:
        return None
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    if rotation_angle == 0:
        new_fx, new_fy = fx, fy
        new_cx, new_cy = cx, cy
    elif rotation_angle == 90:
        new_fx, new_fy = fy, fx
        new_cx = H - cy
        new_cy = cx
    elif rotation_angle == 180:
        new_fx, new_fy = fx, fy
        new_cx = W - cx
        new_cy = H - cy
    elif rotation_angle == 270:
        new_fx, new_fy = fy, fx
        new_cx = cy
        new_cy = W - cx
    else:
        raise ValueError(f"Invalid rotation angle: {rotation_angle}. Must be 0, 90, 180, or 270.")

    intrinsics[0, 0] = new_fx
    intrinsics[1, 1] = new_fy
    intrinsics[0, 2] = new_cx
    intrinsics[1, 2] = new_cy

    return intrinsics

def determine_rotation_from_poses(poses_cam_to_world: list) -> int:
    """
    Автоматически определяет необходимый угол поворота изображений
    на основе анализа поз камер (аналогично ARKitScenes).
    
    Args:
        poses_cam_to_world: список 4x4 матриц camera-to-world
        
    Returns:
        rotation_angle: угол поворота (0, 90, 180, 270)
    """
    
    # Вычисляем средний "up" вектор камеры в мировых координатах
    up_vectors = []
    right_vectors = []
    
    for pose in poses_cam_to_world:
        # Вектор "вверх" камеры в её системе координат: [0, -1, 0, 0]
        up_cam = np.array([0.0, -1.0, 0.0, 0.0])
        up_world = pose @ up_cam
        up_vectors.append(up_world[:3])
        
        # Вектор "вправо" камеры: [1, 0, 0, 0]
        right_cam = np.array([1.0, 0.0, 0.0, 0.0])
        right_world = pose @ right_cam
        right_vectors.append(right_world[:3])
    
    # Усредняем
    up_vec = (np.mean(up_vectors, axis=0))
    right_vec = (np.mean(right_vectors, axis=0))
    up_vec = up_vec / np.linalg.norm(up_vec)
    right_vec = right_vec / np.linalg.norm(right_vec)
    
    # Вектор "вверх" в мировых координатах (обычно Z)
    world_up = np.array([0.0, 0.0, 1.0])
    
    # Углы с вертикалью
    up_angle = np.degrees(np.arccos(np.dot(up_vec, world_up)))
    right_angle = np.degrees(np.arccos(np.dot(right_vec, world_up)))
    
    # Определяем ориентацию
    if abs(up_angle - 90) < abs(right_angle - 90):
        # "Вверх" камеры близок к горизонтали -> ландшафтная ориентация
        if right_angle > 90:
            return 90  # LEFT -> поворот на 90°
        else:
            return 270  # RIGHT -> поворот на 270°
    else:
        # "Вправо" камеры близок к горизонтали -> портретная ориентация
        if up_angle > 90:
            return 180  # DOWN -> поворот на 180°
        else:
            return 0  # UP -> без поворота