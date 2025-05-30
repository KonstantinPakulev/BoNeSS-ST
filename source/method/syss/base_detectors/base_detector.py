from abc import ABC, abstractmethod

from source.utils.endpoint_utils import nms, mask_border, select_kp, localize_keypoints_via_taylor


class BaseDetector(ABC):

    def __init__(self, 
                 nms_size, localize, border_size):
        self.nms_size = nms_size
        self.localize = localize
        self._border_size = border_size
    
    @property
    def border_size(self):
        if self._border_size is None:
            self._border_size = self._calculate_border_size()

        return self._border_size
    
    @border_size.setter
    def border_size(self, value):
        self._border_size = value
    
    @abstractmethod
    def _calculate_border_size(self):
        ...

    @abstractmethod
    def get_score(self, grayscale_image):
        ...

    def get_extrema_mask(self, score):
        nms_score = nms(score, self.nms_size)
        nms_score = mask_border(nms_score, self.border_size)

        return nms_score > 0
    
    def get_keypoints(self, grayscale_image, n):
        score = self.get_score(grayscale_image)

        cand_kp = select_kp(score, 
                            self.nms_size, n,
                            border_size=self.border_size)
                
        if self.localize:
            return cand_kp + localize_keypoints_via_taylor(cand_kp, score)[0]
        
        else:
            return cand_kp
