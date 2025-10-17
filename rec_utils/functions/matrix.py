import numpy as np
from typing import Optional

def check_matrix_shape(matrix: np.ndarray, possible_shapes: list[tuple[int, int]]) -> bool:
    if matrix.shape not in possible_shapes:
        return False
    return True

def extend_matrix(matrix: Optional[np.ndarray] = None, size: int = 4) -> np.ndarray:
    result = np.eye(size)
    if matrix is not None:
        result[:matrix.shape[0], :matrix.shape[1]] = matrix
    return result