import torch
import numpy as np

from source.evaluation.l2_matcher import L2Matcher, get_mutual_match_mask

from source.utils.common_utils import project, gather_kp

"""
Metrics
"""


def repeatability(kp1: torch.Tensor, kp2: torch.Tensor,
                  offset1: torch.Tensor, offset2: torch.Tensor,
                  kp_mask1: torch.Tensor, kp_mask2: torch.Tensor,
                  h: torch.Tensor, 
                  image_dims1: torch.Tensor, image_dims2: torch.Tensor,
                  max_px_err_thr: int) -> np.ndarray:
    """
    Calculate repeatability of keypoints.

    Args:
        kp1: Keypoints (x, y) in the first image. B x N x 2
        kp2: Keypoints (x, y) in the second image. B x N x 2
        offset1: The offset of the first image from the original image after transforms. B x 2
        offset2: The offset of the second image from the original image after transforms. B x 2
        image_dims1: Dimensions (w, h) of the first image after transforms. B x 2
        image_dims2: Dimensions (w, h) of the second image after transforms. B x 2
    """
    # Project keypoints to the other image
    w_kp1 = project(kp1, h)
    w_kp2 = project(kp2, torch.inverse(h))

    # Determine which keypoints are within the image boundaries of the image
    shared_kp_mask1 = is_inside_image_dims(w_kp1, offset2, image_dims2) & kp_mask1
    shared_kp_mask2 = is_inside_image_dims(w_kp2, offset1, image_dims1) & kp_mask2

    dist = torch.cdist(w_kp1, kp2)

    nn_idx1 = dist.argmin(dim=-1)
    nn_idx2 = dist.argmin(dim=-2)

    mutual_match_mask = get_mutual_match_mask(nn_idx1, nn_idx2)
    shared_match_mask = shared_kp_mask1 & torch.gather(shared_kp_mask2.float(), -1, nn_idx1).bool() & mutual_match_mask

    nn_kp2 = gather_kp(kp2, nn_idx1)
    proj_err = (nn_kp2 - w_kp1).norm(dim=-1)

    err_thr_list = np.linspace(1, max_px_err_thr, num=max_px_err_thr)

    rep = torch.zeros((kp1.shape[0], max_px_err_thr))

    num_shared_kp = torch.minimum(shared_kp_mask1.sum(dim=-1), shared_kp_mask2.sum(dim=-1)).clamp(min=1e-8)


    for i, thr in enumerate(err_thr_list):
        num_correct_proj = (proj_err.le(thr) & shared_match_mask).float().sum(dim=-1)

        rep[:, i] = num_correct_proj / num_shared_kp

    return rep.cpu().numpy()


def mean_matching_accuracy(kp1, kp2,
                           kp_desc1, kp_desc2,
                           kp_mask1, kp_mask2,
                           h,
                           max_px_err_thr):
    matcher = L2Matcher()

    matches = matcher.match(kp_desc1, kp_desc2,
                            kp_mask1, kp_mask2)

    nn_kp2 = matches.gather_nn(kp2)

    w_kp1 = project(kp1, h)
    proj_err = (nn_kp2 - w_kp1).norm(dim=-1)

    err_thr_list = np.linspace(1, max_px_err_thr, num=max_px_err_thr)

    mma = torch.zeros((kp1.shape[0], max_px_err_thr))

    for i, thr in enumerate(err_thr_list):
        num_correct_mm = (proj_err.le(thr) & matches.mutual_nn_match_mask1).float().sum(dim=-1)
        num_mm = matches.mutual_nn_match_mask1.sum(dim=-1).clamp(min=1e-8)

        mma[:, i] = num_correct_mm / num_mm
    
    return mma.cpu().numpy()


"""
Utils
"""

def is_inside_image_dims(kp, offset, image_dims):
    return kp.ge(offset).float().prod(dim=-1).bool() & \
           kp.lt(image_dims + offset).float().prod(dim=-1).bool()
