import torch

from source.baselines.r2d2.r2d2.nets.patchnet import Quad_L2Net_ConfCFS
from source.baselines.r2d2.r2d2.extract import NonMaxSuppression


class R2D2(Quad_L2Net_ConfCFS):

    def __init__(self, reliability_thr=0.7, repeatability_thr=0.7):
        super().__init__()
        self.reliability_thr = reliability_thr
        self.repeatability_thr = repeatability_thr
    
    def get_keypoints(self, image: torch.Tensor, n: int):
        res = self.forward([image])

        reliability = res['reliability'][0]
        repeatability = res['repeatability'][0]

        detector = NonMaxSuppression(rel_thr=self.reliability_thr, rep_thr=self.repeatability_thr)

        y, x = detector(**res)
        rel = reliability[0, 0, y, x]
        rep = repeatability[0, 0, y, x]
        
        kp = torch.stack([y, x], dim=-1)
        kp_score = rel * rep

        n = min(n, kp_score.shape[0])

        topk_idx = kp_score.topk(n)[1]
        kp = kp[topk_idx] + 0.5

        return kp.unsqueeze(0).float()
