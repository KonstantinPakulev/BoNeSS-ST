import numpy as np

import torch

from torch.nn import functional as F


def normalize_coord(grid, w, h, align_corners=False):
    """
    :param grid: B x H x W x 2
    """

    # Make a copy to avoid in-place modification
    norm_grid = grid.clone()

    if align_corners:
        # If norm-grid values are top-left corners of pixels
        norm_grid[:, :, :, 0] = norm_grid[:, :, :, 0] / (w - 1) * 2 - 1
        norm_grid[:, :, :, 1] = norm_grid[:, :, :, 1] / (h - 1) * 2 - 1

    else:
        # If norm-grid values are centers of pixels
        norm_grid[:, :, :, 0] = norm_grid[:, :, :, 0] / w * 2 - 1
        norm_grid[:, :, :, 1] = norm_grid[:, :, :, 1] / h * 2 - 1

    return norm_grid


def create_coordinate_grid(shape, center=True, scale_factor=1.0):
    """
    :param shape: (b, _, h, w) :type tuple
    :param scale_factor: float
    :param center: bool
    :return B x H x W x 2; x, y orientation of coordinates located in center of pixels :type torch.tensor, float
    """
    b, _, h, w = shape

    grid_y, grid_x = torch.meshgrid([torch.arange(h), torch.arange(w)])

    grid_x = grid_x.float().unsqueeze(-1)
    grid_y = grid_y.float().unsqueeze(-1)
    grid = torch.cat([grid_x, grid_y], dim=-1)  # H x W x 2

    # Each coordinate represents the location of the center of a pixel
    if center:
        grid += 0.5

    grid *= scale_factor

    return grid.unsqueeze(0).repeat(b, 1, 1, 1)


def grid_sample(t, grid):
    """
    :param t: B x C x H_in x W_in
    :param grid: B x H_out x W_out x 2; Grid have have (x,y) coordinates orientation
    :return B x C x H_out x W_out
    """
    norm_grid = normalize_coord(grid, t.shape[3], t.shape[2])

    return F.grid_sample(t, norm_grid, mode='bilinear')


def sample_tensor(t, kp,
                  mode='bilinear',
                  align_corners=False):
    """
    :param t: B x C x H x W
    :param kp: B x N x 2
    :param mode: str
    :param align_corners: bool
    :return B x N x C
    """
    kp_grid = normalize_coord(kp[:, :, [1, 0]].unsqueeze(1), t.shape[3], t.shape[2], align_corners)
    kp_t = F.grid_sample(t, kp_grid, mode=mode).squeeze(2).permute(0, 2, 1)

    return kp_t


def get_rect(shape, offset=0):
    """
    :param shape: (b, c, h, w) or B x 3
    :param region_size: int
    :param offset: int
    :return 4 or B x 4
    """
    if torch.is_tensor(shape):
        b = shape.shape[0]

        rect = torch.ones(b, 4) * offset
        rect[:, 2] = shape[:, 1] - 1 - offset
        rect[:, 3] = shape[:, 2] - 1 - offset

        return rect

    else:
        return offset, offset, shape[-2] - 1 - offset, shape[-1] - 1 - offset


def get_rect_mask(points, rect):
    """
    :param points: ... x 2; (y, x) orientation
    :param rect: (y, x, h, w) or B x 4
    :return:
    """
    if torch.is_tensor(rect):
        return (points[..., 0] >= rect[:, None, 0]) & \
               (points[..., 1] >= rect[:, None, 1]) & \
               (points[..., 0] <= rect[:, None, 2]) & \
               (points[..., 1] <= rect[:, None, 3])

    else:
        y, x, h, w = rect

        return points[..., 0].ge(y) & \
               points[..., 1].ge(x) & \
               points[..., 0].le(h) & \
               points[..., 1].le(w)
    

def gather_kp(kp, idx):
    return torch.gather(kp, 1, idx.unsqueeze(-1).repeat(1, 1, 2))


def get_offset(image, transformed_image):
    return torch.div(torch.tensor(image.shape[:2]) - torch.tensor(transformed_image.shape[1:]), 2, rounding_mode='floor').to(torch.float32)


"""
Kernel functions
"""


def apply_kernel(t, kernel, **params):
    """
    :param t: N x 1 x H x W
    :param kernel: 1 x 1 x ks x ks
    :return: N x 1 x H x W
    """
    t = F.conv2d(t, weight=kernel, padding=kernel.shape[2] // 2, **params)

    return t


def apply_gaussian_filter(score, kernel_size, cov):
    """
    :param score: N x 1 x H x W
    :param kernel_size: kernel size
    :param cov: covariance
    """
    if cov == 0:
        raise NotImplementedError

    else:
        gauss_kernel = get_gaussian_kernel(kernel_size, cov).to(score.device)

        score = apply_kernel(score, gauss_kernel)

        return score


def apply_box_filter(t, kernel_size):
    kernel = torch.ones(1, 1, kernel_size, kernel_size).to(t.device)

    return apply_kernel(t, kernel)


def apply_erosion_filter(t, kernel_size):
    kernel = torch.ones(1, 1, kernel_size, kernel_size).to(t.device)
    return apply_kernel(t, kernel) == (kernel_size**2)


"""
Gradient calculation utils
"""


def get_sobel_kernel(kernel_size, device, transpose=False):
    patch_coord = create_coordinate_grid((1, 1, kernel_size, kernel_size)) - kernel_size / 2

    kernel = patch_coord[..., 1 if transpose else 0] / (patch_coord ** 2).sum(dim=-1).clamp(min=1e-8)

    return kernel.unsqueeze(0).to(device)


def get_grad_kernels(device):
    kernel = torch.tensor([[0, 0, 0],
                           [-0.5, 0, 0.5],
                           [0, 0, 0]], dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)

    return kernel, kernel.permute(0, 1, 3, 2)


def get_hess_kernel(device):
    dxdx_kernel = torch.tensor([[0, 0, 0],
                                [1, -2, 1],
                                [0, 0, 0]], dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)

    dydy_kernel = torch.tensor([[0, 1, 0],
                                [0, -2, 0],
                                [0, 1, 0]], dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)

    dxdy_kernel = 0.25 * torch.tensor([[1, 0, -1],
                                       [0, 0, 0],
                                       [-1, 0, 1]], dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)

    return dxdx_kernel, dydy_kernel, dxdy_kernel


def get_second_moment_matrix(image_gray,
                             sobel_size,
                             window_size, window_cov):
    b, c, h, w = image_gray.shape

    dx_kernel = get_sobel_kernel(sobel_size, image_gray.device)
    dy_kernel = get_sobel_kernel(sobel_size, image_gray.device, transpose=True)

    dx = apply_kernel(image_gray, dx_kernel)
    dx2 = dx * dx

    dy = apply_kernel(image_gray, dy_kernel)
    dy2 = dy * dy

    dxdy = dx * dy

    dI = torch.stack([apply_gaussian_filter(dy2, window_size, window_cov),
                      apply_gaussian_filter(dxdy, window_size, window_cov),
                      apply_gaussian_filter(dxdy, window_size, window_cov),
                      apply_gaussian_filter(dx2, window_size, window_cov)], dim=-1).view(b, c, h, w, 2, 2)

    return dI


"""
Support utils
"""


def get_trace(t):
    """
    :param t: ... x 2 x 2
    :return: ...
    """
    return t.diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)


def get_eigen_values(t):
    """
    :param t: ... x 2 x 2
    :return ... x 2
    """
    tr = get_trace(t)
    d = tr ** 2 - 4 * t.det()

    d_mask = (d > 0) | (d == 0 & (tr > 0))

    sqrt_d = torch.sqrt(d * d_mask.float())

    eig_val = torch.stack([(tr + sqrt_d) / 2 * d_mask.float(),
                           (tr - sqrt_d) / 2 * d_mask.float()], dim=-1)

    return eig_val


def get_gaussian_kernel(patch_size, cov):
    patch_coord = create_coordinate_grid((1, 1, patch_size, patch_size))
    patch_center = torch.tensor([patch_size / 2, patch_size / 2]).view(1, 1, 1, 1, 2)

    diff = patch_coord - patch_center

    ll_pg = torch.exp(-0.5 * (diff.unsqueeze(-2) @ diff.unsqueeze(-1)).squeeze(-1).squeeze(-1) / cov)
    ll_pg = ll_pg / ll_pg.sum(dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)

    return ll_pg


def rad2deg(radians: float) -> float:
    return radians * 180 / np.pi


def deg2rad(degrees: float) -> float:
    return degrees / 180 * np.pi


def shoelace_area(points):
    """
    :param points: B x N x K x 2
    """
    return 0.5 * (points[..., 1] * torch.roll(points[..., 0], 1, dims=-1) -
                  torch.roll(points[..., 1], 1, dims=-1) * points[..., 0]).sum(dim=-1).abs()


def qvec2rotmat(qvec):
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


"""
Numpy functions
"""

def round_up2odd(v):
    return int(np.ceil(v)) // 2 * 2 + 1


def round_down2even(v):
    return int(np.floor(v)) // 2 * 2


"""
Projective transformation utils
"""


def project(point: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    :param point: B x N x 2
    :param h: B x 3 x 3
    :return: B x N x 2
    """
    point = to_homogeneous(point).permute(0, 2, 1)  # B x 3 x N
    p_point = (h @ point).permute(0, 2, 1)  # B x N x 3

    return to_cartesian(p_point).view(point.shape[0], -1, 2)


def to_homogeneous(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    :param t: Shape B x N x 2 or B x H x W x 3, :type torch.tensor, float
    :param dim: dimension along which to concatenate
    """
    if dim == -1:
        index = len(t.shape) - 1
    else:
        index = dim

    shape = t.shape[:index] + t.shape[index + 1:]
    ones = torch.ones(shape).unsqueeze(dim).float().to(t.device)
    t = torch.cat((t, ones), dim=dim)

    return t


def to_cartesian(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    :param t: Shape B x N x 3 or B x H x W x 4, :type torch.tensor, float
    :param dim: dimension along which to normalize
    """
    index = torch.tensor([t.shape[dim] - 1]).to(t.device)
    t = t / torch.index_select(t, dim=dim, index=index).clamp(min=1e-8)

    index = torch.arange(t.shape[dim] - 1).to(t.device)
    t = torch.index_select(t, dim=dim, index=index)

    return t
