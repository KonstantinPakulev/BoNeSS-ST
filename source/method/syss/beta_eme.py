import enum
import torch
import math

from source.method.syss.patch_detector import PatchDetector

from source.utils.common_utils import get_trace, get_eigen_values


class Bound(enum.IntEnum):
    NONE = 0
    SQRT_TRACE = 1
    TRACE = 2
    SPECTRAL = 3


class BetaEME:

    @staticmethod
    def from_config(cfg):
        return BetaEME(Bound(cfg.bound), cfg.use_decomposition)

    def __init__(self, bound: Bound, use_decomposition: bool):
        self.bound = bound
        self.use_decomposition = use_decomposition
        
    def calculate(self, kp_proj: torch.Tensor):
        if self.bound == Bound.NONE:
            if self.use_decomposition:
                raise ValueError("No decomposition is available for this option.")
                                
            else:
                return kp_proj.norm(dim=-1).mean(dim=0)

        elif self.bound == Bound.SQRT_TRACE:
            return self._calculate_trace_bound(kp_proj).sqrt()

        elif self.bound == Bound.TRACE:
            return self._calculate_trace_bound(kp_proj)
        
        elif Bound.SPECTRAL:
            if self.use_decomposition:
                kp_proj_cov = (kp_proj.unsqueeze(-1) @ kp_proj.unsqueeze(-2)).sum(dim=0) / kp_proj.shape[0]

                return 2 * get_eigen_values(kp_proj_cov).max(dim=-1)[0]
            
            else:
                raise ValueError("Only decomposition is available for this option.")

        else:
            raise ValueError(f"Unknown option: {self.bound}.")
    
    def replace_with_max_value(self, kp_beta_eme: torch.Tensor, kp_noise_mask: torch.Tensor, 
                               patch_detector: PatchDetector, beta: float):
        max_beta_eme = self._calculate_max_beta_eme(patch_detector, beta)

        return kp_beta_eme.masked_fill(kp_noise_mask, max_beta_eme)
        
    def _calculate_trace_bound(self, kp_proj: torch.Tensor):
        if self.use_decomposition:
            kp_proj_mean = kp_proj.mean(dim=0).unsqueeze(0)

            diff = kp_proj - kp_proj_mean
            kp_proj_cov = (diff.unsqueeze(-1) @ diff.unsqueeze(-2)).sum(dim=0) / (kp_proj.shape[0] - 1)

            delta = kp_proj_mean.squeeze(0)

            return get_trace(kp_proj_cov) + (delta * delta).sum(dim=-1)

        
        else:
            return (kp_proj.norm(dim=-1) ** 2).mean(dim=0)

    def _calculate_max_beta_eme(self, patch_detector: PatchDetector, beta: float):
        inc = 0.5 if patch_detector.localize else 0.0

        max_beta_eme = (patch_detector.nms_size // 2 + inc) * math.sqrt(2) * beta

        if self.bound in [Bound.NONE, Bound.SQRT_TRACE]:
            return max_beta_eme
        
        elif self.bound in [Bound.TRACE, Bound.SPECTRAL]:
            return max_beta_eme ** 2
        
        else:
            raise ValueError(f"Unknown option: {self.bound}.")
