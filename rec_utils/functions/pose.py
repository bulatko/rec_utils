import numpy as np


def qvec2rotmat(qvec, mode="wxyz"):
    assert len(qvec) == 4, "qvec must be of length 4"
    assert len(mode) == 4, "mode must be of length 4"
    for letter in 'wxyz':
        if letter not in mode:
            raise ValueError(f"Invalid mode {mode}, unsupported letter {letter}")

    q = {}

    q['x'] = qvec[mode.index('x')]
    q['y'] = qvec[mode.index('y')]
    q['z'] = qvec[mode.index('z')]
    q['w'] = qvec[mode.index('w')]
    qvec = [q['w'], q['x'], q['y'], q['z']]
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def qt2pose(qvec, tvec, mode="wxyz"):
    pose = np.eye(4)
    pose[:3, :3] = np.array(qvec2rotmat(qvec, mode))
    pose[:3, 3] = tvec
    return pose