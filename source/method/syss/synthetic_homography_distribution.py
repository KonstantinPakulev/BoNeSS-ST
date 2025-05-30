import torch
import numpy as np
import enum
from source.utils.common_utils import round_up2odd, project


class SamplingStrategy(enum.IntEnum):
    DEFAULT = 0
    ASYMMETRIC = 1


class Homography:

    def __init__(self, h12):
        self.h12 = h12
        self.h21 = torch.inverse(h12)

    @property
    def m(self):
        return self.h12.shape[0]

    def project12(self, point):
        """
        :param point: B x N x 2; (y, x) orientation
        :return: B x N x 2; (y, x) orientation
        """
        return project(point.flip([-1]), self.h12).flip([-1])
    
    def project21(self, point):
        """
        :param point: B x N x 2; (y, x) orientation
        :return: B x N x 2; (y, x) orientation
        """
        return project(point.flip([-1]), self.h21).flip([-1])


class SyntheticHomographyDistribution:

    def __init__(self, beta, patch_size, sampling_strategy: SamplingStrategy):
        self._beta = beta
        self._patch_scaled_size = round_up2odd(patch_size * self._beta)
        self._sampling_strategy = sampling_strategy

        self._max_log = np.emath.logn(2, self._beta)


    def _get_displaced_points_default(self, m, device):
        outer_edge = 0.5 / (np.sqrt(self._beta))
        outer_border = 0.5 - outer_edge

        point2 = torch.tensor([[outer_border, outer_border],
                                [outer_border, 1 - outer_border],
                                [1 - outer_border, 1 - outer_border],
                                [1 - outer_border, outer_border]], dtype=torch.float32, device=device).unsqueeze(0).repeat(m, 1, 1)
                                
        left_disp = outer_edge - 0.5 / (2 ** (torch.rand(m, device=device) * self._max_log))
        right_disp = 0.5 / (2 ** (torch.rand(m, device=device) * self._max_log)) - outer_edge

        persp_disp = outer_edge - 0.5 / (2 ** (torch.rand(m, device=device) * self._max_log))

        point2 += torch.stack([left_disp, persp_disp, 
                               left_disp, -persp_disp, 
                               right_disp, persp_disp, 
                               right_disp, -persp_disp], dim=-1).view(m, 4, 2)
        
        return point2
    
    def _get_displaced_points_asymmetric(self, m, device):
        disp = 0.5 - 2 ** (-torch.rand(m, 4, device=device) * self._max_log - 1)

        point2 = torch.stack([disp[:, 0], disp[:, 2], 
                              disp[:, 0], 1 - disp[:, 2], 
                              1 - disp[:, 1], 1 - disp[:, 3], 
                              1 - disp[:, 1], disp[:, 3]], dim=-1).view(m, 4, 2)
        
        return point2
    
    def sample(self, m, device):
        point1 = torch.tensor([[0., 0.],
                                [0., 1.],
                                [1., 1.],
                                [1., 0.]], dtype=torch.float32, device=device).unsqueeze(0).repeat(m, 1, 1)
        
        if self._sampling_strategy == SamplingStrategy.DEFAULT:
            point2 = self._get_displaced_points_default(m, device)

        elif self._sampling_strategy == SamplingStrategy.ASYMMETRIC:
            point2 = self._get_displaced_points_asymmetric(m, device)

        else:
            raise ValueError(f"Invalid sampling strategy: {self._sampling_strategy}")
            
        point1 *= self.patch_scaled_size
        point2 *= self.patch_scaled_size

        M = torch.stack([torch.cat([point1, torch.ones(m, 4, 1, device=device),
                                    torch.zeros(m, 4, 3, device=device),
                                    -point1 * point2[..., 0, None]], dim=-1),
                        torch.cat([torch.zeros(m, 4, 3, device=device),
                                    point1, torch.ones(m, 4, 1, device=device),
                                    -point1 * point2[..., 1, None]], dim=-1)],
                        dim=-2).view(m, 8, 8)
        b = point2.view(m, 8, 1)

        h = torch.linalg.lstsq(M, b).solution.squeeze(-1)
        h = torch.cat([h, torch.ones(m, 1, device=device)], dim=-1).view(m, 3, 3)

        return Homography(h)
    
    @property
    def beta(self):
        return self._beta
    
    @property
    def patch_scaled_size(self):
        return self._patch_scaled_size
    
    @property
    def border_size(self):
        return self._patch_scaled_size
