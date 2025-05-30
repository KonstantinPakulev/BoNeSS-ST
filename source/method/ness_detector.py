import torch
import torch.nn as nn

from dataclasses import dataclass

from source.method.nn.backbones import UNet
from source.method.syss.base_detectors.instantiate_base_detector import instantiate_base_detector
from source.method.syss.base_detectors.base_detector import BaseDetector

from source.utils.endpoint_utils import flat2grid, localize_keypoints_via_taylor


@dataclass
class KeypointSamples:
    kp: torch.tensor
    cand_kp: torch.tensor
    kp_inc_sing_mask: torch.tensor
    kp_inc_cond_num_mask: torch.tensor
    cand_kp_mask: torch.tensor
    cand_kp_noise_mask: torch.tensor
    kp_neur_beta_eme: torch.tensor


class NeSSDetector(nn.Module):

    @staticmethod
    def from_config(detector_config):
        neur_beta_eme_model = UNet.from_config(detector_config.model)
        base_detector = instantiate_base_detector(detector_config.base_detector)
        
        return NeSSDetector(neur_beta_eme_model, base_detector)
    
    def __init__(self, neur_beta_eme_model: nn.Module, base_detector: BaseDetector):
        super().__init__()
        self.neur_beta_eme_model = neur_beta_eme_model
        self.base_detector = base_detector
    
    def forward(self, image):
        return self.neur_beta_eme_model(image)[-1].clamp(min=0.0)
    
    def get_keypoints(self, image, grayscale_image, n):
        neur_beta_eme = self.forward(image)

        base_detector_score = self.base_detector.get_score(grayscale_image)
        base_detector_extrema_mask = self.base_detector.get_extrema_mask(base_detector_score)

        ness_extrema_map = (-neur_beta_eme).exp() * base_detector_extrema_mask.float()

        b, _, _, w = image.shape

        flat_cand_kp = ness_extrema_map.view(b, -1).topk(n, dim=-1)[1]
        cand_kp = flat2grid(flat_cand_kp, w) + 0.5

        if self.base_detector.localize:
            kp_inc = localize_keypoints_via_taylor(cand_kp, base_detector_score)[0]
            kp = cand_kp + kp_inc

        else:
            kp = cand_kp

        return kp
    
    def get_keypoint_samples(self, image, grayscale_image, n, border_size, salient_thresh=None, noise_thresh=None):
        neur_beta_eme = self.forward(image)

        self.base_detector.border_size = border_size
        
        base_detector_score = self.base_detector.get_score(grayscale_image)
        base_detector_extrema_mask = self.base_detector.get_extrema_mask(base_detector_score)

        if salient_thresh is not None or noise_thresh is not None:
            if salient_thresh is not None:
                salient_mask = base_detector_score > salient_thresh

            else:
                salient_mask = torch.zeros_like(base_detector_score, dtype=torch.bool)
            
            if noise_thresh is not None:
                noise_mask = base_detector_score < noise_thresh
            
            else:
                noise_mask = torch.zeros_like(base_detector_score, dtype=torch.bool)
                        
            base_detector_extrema_mask = base_detector_extrema_mask & (salient_mask | noise_mask)

        ness_extrema_map = (-neur_beta_eme.detach()).exp() * base_detector_extrema_mask.float()

        b, _, _, w = image.shape

        cand_kp_ness, flat_cand_kp = ness_extrema_map.view(b, -1).topk(n, dim=-1)
        cand_kp = flat2grid(flat_cand_kp, w) + 0.5

        if self.base_detector.localize:
            kp_inc, kp_inc_sing_mask, kp_inc_cond_num_mask = localize_keypoints_via_taylor(cand_kp, base_detector_score)
            kp = cand_kp + kp_inc
        
        else:
            kp = cand_kp
            kp_inc_sing_mask = torch.ones(b, n, dtype=torch.bool)
            kp_inc_cond_num_mask = torch.ones(b, n, dtype=torch.bool)

        cand_kp_mask = cand_kp_ness > 0.0

        if noise_thresh is not None:
            cand_kp_noise_mask = (base_detector_score.view(b, -1).gather(-1, flat_cand_kp) < noise_thresh) & cand_kp_mask

        else:
            cand_kp_noise_mask = torch.zeros(b, n, dtype=torch.bool) & cand_kp_mask
        
        kp_neur_beta_eme = neur_beta_eme.view(b, -1).gather(-1, flat_cand_kp)

        kp_samples = KeypointSamples(
            kp = kp,
            cand_kp = cand_kp,
            kp_inc_sing_mask = kp_inc_sing_mask,
            kp_inc_cond_num_mask = kp_inc_cond_num_mask,
            cand_kp_mask = cand_kp_mask,
            cand_kp_noise_mask = cand_kp_noise_mask,
            kp_neur_beta_eme = kp_neur_beta_eme
        )

        return kp_samples
