import torch

from source.baselines.rekd.rekd.config import get_config
from source.baselines.rekd.rekd.training.model.REKD import REKD as BaseREKD
from source.baselines.rekd.rekd.utils.geometry_tools import apply_nms, get_point_coordinates, remove_borders


class REKD(BaseREKD):

    def __init__(self):
        self.args = get_config(jupyter=True)
        super().__init__(self.args, None)

    def get_keypoints(self, grayscale_image: torch.Tensor, n: int):
        im_scores = self.forward(grayscale_image)[0].cpu().numpy()
        im_scores = remove_borders(im_scores[0,0,:,:], borders=self.args.border_size)
        im_scores = apply_nms(im_scores, self.args.nms_size)
        
        points = get_point_coordinates(im_scores, num_points=n, order_coord='yxsr')[..., :2] + 0.5

        return torch.tensor(points, dtype=torch.float32, device=grayscale_image.device).unsqueeze(0)
