def adjust_intrinsics(matrix, source_shape, target_shape):
    if matrix is None:
        raise ValueError("Matrix is None")
    if source_shape is None or target_shape is None:
        raise ValueError("Source and target shapes have to be not None")
    
    h_scale = target_shape[0] / source_shape[0]
    w_scale = target_shape[1] / source_shape[1]

    res = matrix.copy()
    res[0] *= w_scale
    res[1] *= h_scale

    return res