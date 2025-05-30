import torch
from torch.nn.functional import normalize

from source.baselines.disk.disk.disk.model import DISK as BaseDISK

from source.utils.endpoint_utils import sample_tensor


class DISK(BaseDISK):

    def get_keypoints(self, image: torch.Tensor, n: int):
        heatmap = self._split(self.unet(image))[1]

        keypoints  = self.detector.nms(heatmap, n)

        kp = torch.stack([k.xys[..., [1, 0]] for k in keypoints]).float() + 0.5
        
        return kp

    def get_descriptors(self, image: torch.Tensor, kp: torch.Tensor):
        desc_map = self._split(self.unet(image))[0]

        kp_desc = sample_tensor(desc_map, kp)
        kp_desc = normalize(kp_desc, dim=-1)

        return kp_desc
        
    def get_features(self, image: torch.Tensor, n: int):
        desc_map, heatmap = self._split(self.unet(image))

        keypoints  = self.detector.nms(heatmap, n)
        features = [kps.merge_with_descriptors(desc_map[i]) for i, kps in enumerate(keypoints)]

        kp = torch.stack([f.kp[..., [1, 0]] for f in features])
        kp_desc = torch.stack([f.desc for f in features])
        
        return kp, kp_desc
