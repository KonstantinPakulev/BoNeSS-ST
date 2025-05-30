import numpy as np

from source.utils.common_utils import rad2deg


def to_bearing_vector(nic_kp: np.ndarray) -> np.ndarray:
    return normalize(to_homogeneous(nic_kp))


"""
Projective transformations
"""

def to_normalized_image_coordinates(kp: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    return to_cartesian(np.transpose(np.linalg.inv(intrinsics) @ np.transpose(to_homogeneous(kp), (1, 0)), (1, 0)))


def to_homogeneous(kp: np.ndarray) -> np.ndarray:
    return np.concatenate([kp, np.ones((kp.shape[0], 1))], axis=-1)


def to_cartesian(kp: np.ndarray) -> np.ndarray:
    return kp[:, :-1] / kp[:, -1, None]


"""
General math operations
"""

def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=-1), a_min=1e-8, a_max=None)[..., None]


def angle_mat(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """
    :param R1: 3 x 3
    :param R2: 3 x 3
    :return: angle in degrees
    """
    c = (np.trace(R1.T @ R2) - 1) / 2
    return rad2deg(np.arccos(np.clip(c, a_min=-1.0, a_max=1.0)))


def angle_vec(v1: np.ndarray, v2: np.ndarray, degrees: bool = True) -> np.ndarray:
    """
    :param v1: B x N x C, unit vector
    :param v2: B x N x C, unit vector
    :return: B, angle between vectors
    """
    angle = np.arccos(np.clip((v1 * v2).sum(-1), a_min=-1.0, a_max=1.0))

    if degrees:
        return rad2deg(angle)
    else:
        return angle
