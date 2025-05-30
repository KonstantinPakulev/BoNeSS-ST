import torch
import cv2 
import numpy as np


class SIFT:
    
    def __init__(self, nOctaveLayers=3, contrastThreshold=0.03, edgeThreshold=10, sigma=1.6):
        self.cv_sift = cv2.SIFT_create(nOctaveLayers=nOctaveLayers, 
                                       contrastThreshold=contrastThreshold, 
                                       edgeThreshold=edgeThreshold, 
                                       sigma=sigma)
        
    def get_keypoints(self, grayscale_image: torch.Tensor, n: int):
        cv_kp_list = self.cv_sift.detect((grayscale_image.squeeze().cpu().numpy() * 255).astype(np.uint8), None)

        kp_score = np.array([cv_kp.response for cv_kp in cv_kp_list])
        sorted_idx = np.argsort(kp_score)[::-1]

        kp = np.array([cv_kp_list[idx].pt for idx in sorted_idx])

        unique_kp, unique_idx = np.unique(kp, axis=0, return_index=True)
        
        kp = unique_kp[np.argsort(unique_idx)[:n]] + 0.5

        return torch.from_numpy(kp).to(grayscale_image.device).float().flip(-1).unsqueeze(0)
