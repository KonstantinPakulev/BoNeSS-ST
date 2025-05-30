import enum
import torch
from typing import Tuple

from source.method.syss.base_detectors.base_detector import BaseDetector
from source.method.syss.base_detectors.instantiate_base_detector import instantiate_base_detector

from source.utils.endpoint_utils import mask_border, flat2grid, nms, localize_keypoints_via_taylor
from source.utils.common_utils import round_down2even


class UnreliableMeasurementType(enum.IntEnum):
    NO = 0
    NMS = 1
    SING = 2
    COND_NUM = 3


class PatchDetector:

    @staticmethod
    def from_config(cfg):
        base_detector = instantiate_base_detector(cfg.base_detector)

        return PatchDetector(base_detector, UnreliableMeasurementType(cfg.unreliable_measurement_type))
    
    def __init__(self, base_detector: BaseDetector, unrel_meas_type: UnreliableMeasurementType):
        self._base_detector = base_detector
        self.unrel_meas_type = unrel_meas_type

        if unrel_meas_type == UnreliableMeasurementType.NO:
            self._patch_size = self.nms_size + base_detector.border_size * 2

        else:
            self._patch_size = self.nms_size + round_down2even(self.nms_size) + base_detector.border_size * 2

    @property
    def localize(self):
        return self._base_detector.localize
    
    @property
    def base_detector(self):
        return self._base_detector
    
    @property
    def nms_size(self):
        return self._base_detector.nms_size
    
    @property
    def border_size(self):
        return self._base_detector.border_size
    
    @property
    def patch_size(self):
        return self._patch_size
    
    def get_patch_score(self, patch: torch.Tensor) -> torch.Tensor:
        b, n, m = patch.shape[:3]

        patch = patch.view(-1, 1, self.patch_size, self.patch_size)

        return self._base_detector.get_score(patch).view(b, n, m, 1, self.patch_size, self.patch_size)
    
    def get_candidate_keypoint_measurements(self, patch_score: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n, m = patch_score.shape[:3]
        bnm = b * n * m

        if self.unrel_meas_type == UnreliableMeasurementType.NO:
            patch_score = mask_border(patch_score, self.border_size, -1)

        else:
            nms_border_size = (self.patch_size - self.nms_size) // 2

            patch_score = mask_border(patch_score, self.border_size, -1)
            patch_nms_score = nms(patch_score.view(bnm, 1, self.patch_size, self.patch_size), self.nms_size)
            patch_nms_score = mask_border(patch_nms_score, nms_border_size, -1)
        
        cand_kp_meas_score, flat_cand_kp_meas = patch_nms_score.view(bnm, -1).max(dim=-1)
        cand_kp_meas = flat2grid(flat_cand_kp_meas, self.patch_size) + 0.5

        return cand_kp_meas.view(b, n, m, 2), cand_kp_meas_score.view(b, n, m)
    
    def get_keypoint_measurements(self, patch):
        patch_score = self.get_patch_score(patch)

        # num_zeros = (patch == 0.0).sum(dim=(3, 4, 5))
        # print(num_zeros.nonzero())
        
        cand_kp_meas, cand_kp_meas_score = self.get_candidate_keypoint_measurements(patch_score)

        b, n, m = patch.shape[:3]
        bnm = b * n * m
        
        if self.base_detector.localize or (self.unrel_meas_type > UnreliableMeasurementType.NO):
            kp_meas_inc, kp_meas_inc_sing_mask, kp_meas_inc_cond_num_mask = \
                localize_keypoints_via_taylor(cand_kp_meas.view(bnm, 1, 2), patch_score.view(bnm, 1, self.patch_size, self.patch_size))
            
            if self.unrel_meas_type > UnreliableMeasurementType.NO:
                kp_meas_inc_mask = cand_kp_meas_score.view(bnm, 1) != 0.0

            if self.unrel_meas_type >= UnreliableMeasurementType.SING:
                kp_meas_inc_mask &= kp_meas_inc_sing_mask

            if self.unrel_meas_type == UnreliableMeasurementType.COND_NUM:
                kp_meas_inc_mask &= kp_meas_inc_cond_num_mask

            nms_border_size = (self.patch_size - self.nms_size) // 2
            top_left = torch.tensor([nms_border_size, nms_border_size], device=patch.device) + 0.5

            kp_meas_inc_mask = kp_meas_inc_mask.float()

            cand_kp_meas = (cand_kp_meas.view(bnm, 2) * kp_meas_inc_mask + top_left.unsqueeze(0) * (1 - kp_meas_inc_mask)).view(b, n, m, 2)
            kp_meas_inc = (kp_meas_inc.squeeze(1) * kp_meas_inc_mask).view(b, n, m, 2)

        if self.base_detector.localize:
            return cand_kp_meas + kp_meas_inc

        else:
            return cand_kp_meas
