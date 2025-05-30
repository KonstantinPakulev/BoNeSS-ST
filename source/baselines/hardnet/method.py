import torch
import torch.nn as nn

from source.baselines.hardnet.hardnet.code.Utils import L2Norm

from source.utils.endpoint_utils import sample_tensor_patch


class HardNetPS(nn.Module):
    
    def __init__(self, patch_size=32):
        super(HardNetPS, self).__init__()
        self.patch_size = patch_size

        self.features = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = True),
        nn.BatchNorm2d(32, affine=True),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = True),
        nn.BatchNorm2d(32, affine=True),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = True),
        nn.BatchNorm2d(64, affine=True),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = True),
        nn.BatchNorm2d(64, affine=True),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = True),
        nn.BatchNorm2d(128, affine=True),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = True),
        nn.BatchNorm2d(128, affine=True),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=8, bias = True)
    )
            
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)
            
    def get_descriptors(self, grayscale_image: torch.Tensor, kp: torch.Tensor):
        kp_patch = sample_tensor_patch(grayscale_image, kp, self.patch_size).\
            squeeze(-1).\
            view(-1, 1, self.patch_size, self.patch_size)
        
        kp_desc = self.forward(kp_patch).unsqueeze(0)

        return kp_desc
