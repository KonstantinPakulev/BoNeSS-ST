import torch

from source.baselines.keynet.keynet.model.network import KeyNet as BaseKeyNet
from source.baselines.keynet.keynet.model.extraction_tools import remove_borders
from source.baselines.keynet.keynet.model.modules import NonMaxSuppression
from source.baselines.keynet.keynet.model.config_files.keynet_configs import keynet_config


class KeyNet(BaseKeyNet):

    def __init__(self):
        self.config = keynet_config['KeyNet_default_config']
        super().__init__(self.config)
    
    def get_keypoints(self, grayscale_image: torch.Tensor, n: int):
        score = self.forward(grayscale_image)
        score = remove_borders(score, borders=15)

        kp = NonMaxSuppression(nms_size=self.config['nms_size'])(score)
        kp_score = score[0, 0, kp[0], kp[1]]

        sorted_kp_score, indices = torch.sort(kp_score, descending=True)
        indices = indices[torch.where(sorted_kp_score > 0.)]
        kp = kp[:, indices[:n]].t() + 0.5

        return kp.unsqueeze(0).float()
