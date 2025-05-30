import torch

from types import SimpleNamespace

from source.baselines.superpoint.superpoint_pretrained_network.demo_superpoint import SuperPointNet as BaseSuperPoint
from source.baselines.superpoint.superpoint.superpoint_pytorch import SuperPoint as SuperPointPytorch, batched_nms, select_top_k_keypoints


class SuperPoint(BaseSuperPoint):

    def __init__(self):
        super().__init__()
        self.conf = SimpleNamespace(**SuperPointPytorch.default_conf)
        self.stride = 8
    
    def get_keypoints(self, grayscale_image: torch.Tensor, n: int):
        semi, _ = self.forward(grayscale_image)

        # Decode the detection scores
        scores = torch.nn.functional.softmax(semi, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            b, h * self.stride, w * self.stride
        )
        scores = batched_nms(scores, self.conf.nms_radius)

         # Discard keypoints near the image borders
        if self.conf.remove_borders:
            pad = self.conf.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1
        
        # Extract keypoints
        scores = scores.squeeze(0)
        idxs = torch.where(scores > self.conf.detection_threshold)
        
        keypoints_all = torch.stack(idxs[-2:], dim=-1).float()
        scores_all = scores[idxs]

        kp = select_top_k_keypoints(keypoints_all, scores_all, n)[0] + 0.5

        return kp.unsqueeze(0)
