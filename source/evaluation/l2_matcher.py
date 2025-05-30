import torch

from source.utils.common_utils import gather_kp

class L2Matcher:

    def match(self, kp_desc1, kp_desc2, kp_mask1=None, kp_mask2=None):
        desc_dist = torch.cdist(kp_desc1, kp_desc2)

        if kp_mask1 is not None and kp_mask2 is not None:
            desc_dist *= kp_mask1.unsqueeze(-1)
            desc_dist *= kp_mask2.unsqueeze(1)

            desc_dist += 20 * (~kp_mask1.unsqueeze(-1))
            desc_dist += 20 * (~kp_mask2.unsqueeze(1))

        nn_desc_value1, nn_desc_idx1 = desc_dist.topk(dim=-1, k=2, largest=False)
        nn_desc_value2, nn_desc_idx2 = desc_dist.topk(dim=-2, k=2, largest=False)

        mutual_nn_match_mask1 = get_mutual_match_mask(nn_desc_idx1[..., 0], nn_desc_idx2[:, 0, :])

        if kp_mask1 is not None:
            mutual_nn_match_mask1 &= kp_mask1

        if kp_mask2 is not None:
            mutual_nn_match_mask1 &= torch.gather(kp_mask2.float(), -1, nn_desc_idx1[..., 0]).bool()

        return Matches(mutual_nn_match_mask1, nn_desc_idx1[..., 0], nn_desc_value1, nn_desc_value2)


class Matches:

    def __init__(self, mutual_nn_match_mask1, nn_desc_idx1, nn_desc_value1, nn_desc_value2):
        self._mutual_nn_match_mask1 = mutual_nn_match_mask1
        self.nn_desc_idx1 = nn_desc_idx1

        self.nn_desc_value1 = nn_desc_value1
        self.nn_desc_value2 = nn_desc_value2
    
    def get_match_mask(self, ratio_test_thr):
        desc_ratio1 = self.nn_desc_value1[..., 0] / self.nn_desc_value1[..., 1]
        desc_ratio2 = self.nn_desc_value2[:, 0, :] / self.nn_desc_value2[:, 1, :]
        
        nn_desc_ratio2 = torch.gather(desc_ratio2, -1, self.nn_desc_idx1)

        match_mask1 = self._mutual_nn_match_mask1 & (desc_ratio1 <= ratio_test_thr) & (nn_desc_ratio2 <= ratio_test_thr)

        return match_mask1
    
    @property
    def mutual_nn_match_mask1(self):
        return self._mutual_nn_match_mask1
    
    def gather_nn(self, kp2):
        return gather_kp(kp2, self.nn_desc_idx1)


"""
Support utils
"""


def get_mutual_match_mask(nn_idx1, nn_idx2):
    idx = torch.arange(0, nn_idx1.shape[1]).repeat(nn_idx1.shape[0], 1).to(nn_idx1.device)

    nn_idx = torch.gather(nn_idx2, -1, nn_idx1)

    return nn_idx == idx
