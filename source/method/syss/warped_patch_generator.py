import torch

from source.utils.endpoint_utils import create_patch_grid, sample_tensor_patch

from source.method.syss.keypoint_projection_generator import KeypointProjectionGenerator
from source.method.syss.synthetic_homography_distribution import SyntheticHomographyDistribution


class WarpedPatchGenerator:

    def __init__(self, h: SyntheticHomographyDistribution, patch_scaled_size: int, device: torch.device):
        self.h = h
        self.patch_scaled_size = patch_scaled_size

        self.center1 = torch.tensor([patch_scaled_size // 2 + 0.5], device=device).\
                                    view(1, 1, 1).\
                                    repeat(self.h.m, 1, 2)
        
        self.p_center1 = self.h.project12(self.center1)
        self.center2 = self.p_center1.long() + 0.5
    
    def generate_warped_patches(self, image, kp, patch_size):
        b = image.shape[0]
        n = kp.shape[1]
        bn = b * n
        nm = n * self.h.m
        ext_patch_size = patch_size + 2

        center2_patch_grid = self._create_center2_patch_grid(ext_patch_size)
        
        center2_p_patch_grid = (self.h.project21(center2_patch_grid).view(self.h.m, 1, -1, 2)
                                - self.center1.unsqueeze(-2)
                                + kp.reshape(1, bn, 1, 2)). \
            view(self.h.m, b, n, -1, 2). \
            permute(1, 2, 0, 3, 4). \
            reshape(b, nm, -1, 2)

        w_patch = sample_tensor_patch(image, 
                                      center2_p_patch_grid, 
                                      ext_patch_size).\
            permute(0, 1, 3, 2)
        
        del center2_patch_grid, center2_p_patch_grid

        return self._correct_grid_displacement(w_patch, patch_size, ext_patch_size, n)
    
    def generate_warp_visualization_patches(self, image, kp, patch_size):
        b, c = image.shape[:2]
        n = kp.shape[1]
        bn = b * n
        nm = n * self.h.m
        ext_patch_size = patch_size + 2

        center2_patch_grid = self._create_center2_patch_grid(ext_patch_size)

        center2_p_patch_grid = self.h.project21(center2_patch_grid).\
            view(self.h.m, 1, 1, -1, 2).\
            repeat(1, b, n, 1, 1).\
            view(self.h.m, b, n, -1, 2). \
            permute(1, 2, 0, 3, 4). \
            view(bn, self.h.m, -1, 2)
        
        kp_scaled_size_patch = sample_tensor_patch(image, kp, self.patch_scaled_size).\
            permute(0, 1, 3, 2).\
            view(bn, c, self.patch_scaled_size, self.patch_scaled_size)
        
        w_vis_patch = sample_tensor_patch(kp_scaled_size_patch,
                                          center2_p_patch_grid,
                                          ext_patch_size).\
            permute(0, 1, 3, 2).\
            reshape(b, nm, c, -1)
        
        del center2_patch_grid, kp_scaled_size_patch, center2_p_patch_grid

        return self._correct_grid_displacement(w_vis_patch, patch_size, ext_patch_size, n)
    
    def get_keypoint_projection_generator(self, image, kp, patch_size):
        return KeypointProjectionGenerator(self.generate_warped_patches(image, kp, patch_size),
                                           patch_size,
                                           self.center1, self.p_center1, 
                                           self.h)

    def _create_center2_patch_grid(self, ext_patch_size):
        return create_patch_grid(self.center2.permute(1, 0, 2), ext_patch_size). \
            permute(1, 0, 2, 3). \
            view(self.h.m, -1, 2)
    
    def _correct_grid_displacement(self, w_patch, patch_size, ext_patch_size, n):
        b, nm, c = w_patch.shape[:3]
        bnm = b * nm

        grid_disp = (torch.tensor([ext_patch_size // 2 + 0.5], device=w_patch.device).
                     view(1, 1, 1).
                     repeat(self.h.m, 1, 2)
                     + (self.p_center1 - self.center2)). \
            view(1, 1, self.h.m, 1, 2). \
            repeat(b, n, 1, 1, 1). \
            view(bnm, 1, 2)
            
        w_patch = sample_tensor_patch(w_patch.reshape(bnm, c, ext_patch_size, ext_patch_size),
                                      grid_disp,
                                      patch_size). \
            permute(0, 1, 3, 2).\
            view(b, n, self.h.m, c, patch_size, patch_size)
        
        return w_patch
