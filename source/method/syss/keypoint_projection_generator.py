import torch

from source.method.syss.synthetic_homography_distribution import Homography

from source.method.syss.patch_detector import PatchDetector
from source.utils.endpoint_utils import mask_border, nms, localize_keypoints_via_taylor


class KeypointProjectionGenerator:

    def __init__(self, 
                 w_patch: torch.Tensor,
                 patch_size: int,
                 center1: torch.Tensor, p_center1: torch.Tensor,
                 h: Homography):
        
        self.w_patch = w_patch

        self.patch_size = patch_size

        self.center1 = center1
        self.p_center1 = p_center1

        self.h = h
    
    def generate_keypoint_projections(self, patch_detector: PatchDetector) -> torch.Tensor:
        b, n, m = self.w_patch.shape[:3]
        bn = b * n

        kp_meas = patch_detector.get_keypoint_measurements(self.w_patch).view(-1, 1, 2)
        kp_meas = (kp_meas
                   - 0.5
                   - self.patch_size // 2
                   + self.p_center1.repeat(bn, 1, 1)). \
            view(b, n, m, 2). \
            permute(2, 0, 1, 3). \
            reshape(m, -1, 2)
                
        kp_proj = self.h.project21(kp_meas)
        kp_proj = (kp_proj - self.center1).view(m, b, n, 2)

        return kp_proj
    
    def generate_keypoint_measurements(self, patch_detector: PatchDetector):
        w_patch_score = patch_detector.get_patch_score(self.w_patch)
        cand_kp_meas, cand_kp_meas_base_detector_score = patch_detector.get_candidate_keypoint_measurements(w_patch_score)

        b, n, m = w_patch_score.shape[:3]
        bnm = b * n * m

        w_patch_score = mask_border(w_patch_score, patch_detector.border_size, -1)
        w_patch_nms_score = nms(w_patch_score.view(bnm, 1, self.patch_size, self.patch_size), patch_detector.nms_size).\
                                view(b, n, m, 1, self.patch_size, self.patch_size)
        
        cand_kp_meas_nms_mask = cand_kp_meas_base_detector_score != 0

        kp_meas_inc, kp_meas_inc_sing_mask, kp_meas_inc_cond_num_mask = \
            localize_keypoints_via_taylor(cand_kp_meas.view(bnm, 1, 2), 
                                          w_patch_score.view(bnm, 1, self.patch_size, self.patch_size), 
                                          mask_increment=False)
        
        kp_meas_inc = kp_meas_inc.view(b, n, m, 2)
        kp_meas_inc_sing_mask = kp_meas_inc_sing_mask.view(b, n, m)
        kp_meas_inc_cond_num_mask = kp_meas_inc_cond_num_mask.view(b, n, m)

        return w_patch_score, w_patch_nms_score, cand_kp_meas, cand_kp_meas_nms_mask, kp_meas_inc_sing_mask, kp_meas_inc_cond_num_mask, kp_meas_inc
