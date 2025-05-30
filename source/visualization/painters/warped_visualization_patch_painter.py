
from source.utils.visualization.miscellaneous import torch2cv
from source.utils.visualization.drawing import draw_image, draw_patch_border

class WarpedVisualizationPatchPainter:

    def __init__(self, 
                 w_vis_patch):
        self.w_vis_patch = w_vis_patch
    
    def get_warp_visualization_patch(self, batch_idx, kp_idx, h_idx):
        return torch2cv(self.w_vis_patch[batch_idx, kp_idx, h_idx])
    
    def draw_warp_visualization_patch(self, axis, batch_idx, kp_idx, h_idx):
        w_vis_patch = self.get_warp_visualization_patch(batch_idx, kp_idx, h_idx)

        draw_image(axis, w_vis_patch)
