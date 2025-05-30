from source.utils.visualization.miscellaneous import torch2cv
from source.utils.visualization.drawing import draw_image, draw_keypoint
from source.utils.endpoint_utils import sample_tensor_patch


class KeypointPatchPainter:

    def __init__(self, image, kp):
        self.image = image
        self.kp = kp

        self.kp_patch_dict = {}

    def get_image(self, batch_idx):
        return torch2cv(self.image[batch_idx])
    
    def get_keypoints(self, batch_idx, kp_idx, patch_h_size):
        kp = self.kp[batch_idx, kp_idx]

        diff_mask = ((self.kp[batch_idx] - kp.unsqueeze(0)).abs() < patch_h_size).sum(dim=-1) == 2

        return (self.kp[batch_idx, diff_mask] - kp.unsqueeze(0)).numpy() + patch_h_size
    
    def get_keypoint_patch(self, batch_idx, kp_idx, patch_size):
        if patch_size not in self.kp_patch_dict:
            b, k = self.kp.shape[:2]
            c = self.image.shape[1]
            
            self.kp_patch_dict[patch_size] = sample_tensor_patch(self.image, self.kp, patch_size).\
                view(b, k, patch_size, patch_size, c).\
                permute(0, 1, 4, 2, 3)
        
        return torch2cv(self.kp_patch_dict[patch_size][batch_idx, kp_idx])
    
    def draw_keypoint_patch(self, axis, batch_idx, kp_idx, patch_size):
        kp_patch = self.get_keypoint_patch(batch_idx, kp_idx, patch_size)

        draw_image(axis, kp_patch)
