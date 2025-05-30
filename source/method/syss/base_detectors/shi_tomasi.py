from source.method.syss.base_detectors.base_detector import BaseDetector

from source.utils.common_utils import get_eigen_values, get_second_moment_matrix


class ShiTomasi(BaseDetector):

    @staticmethod
    def from_config(cfg):
         return ShiTomasi(cfg.sobel_size, cfg.window_size, cfg.window_cov,
                                  cfg.nms_size, cfg.localize,
                                  cfg.get('border_size', None))
    
    def __init__(self,
                sobel_size=3, window_size=3, window_cov=2,
                nms_size=5, localize=True, border_size=None):
            super().__init__(nms_size, localize, border_size)
            self.sobel_size = sobel_size
            self.window_size = window_size
            self.window_cov = window_cov
    
    def _calculate_border_size(self):
        return max(self.sobel_size, self.window_size) // 2 + self.sobel_size // 2

    def get_score(self, grayscale_image):
        smm = get_second_moment_matrix(grayscale_image,
                                       self.sobel_size, self.window_size, self.window_cov)
        
        st_score, _ = get_eigen_values(smm).min(dim=-1)

        return st_score
