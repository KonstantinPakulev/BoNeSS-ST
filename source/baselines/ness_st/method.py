import torch

from omegaconf import DictConfig

from source.baselines.ness_st.ness_st.standalone.ICCV2023.nessst import NeSSST as BaseNeSSST


class NeSSST(BaseNeSSST):
    
    def __init__(self, nms_size=5):
        super().__init__()
        self.nms_size = nms_size
    
    def get_keypoints(self, image: torch.Tensor, grayscale_image: torch.Tensor, n: int):
        return self.__call__(image, grayscale_image, self.nms_size, n)
