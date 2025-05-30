import torch
import torch.nn as nn


class ConvLayer(nn.Sequential):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, bias=True):
        padding = (kernel_size - 1) // 2

        conv = [nn.Conv2d(in_channels, out_channels,
                          kernel_size=kernel_size, padding=padding, bias=bias)]

        super().__init__(*conv)


class DownLayer(nn.Sequential):

    def __init__(self, in_channels, out_channels,
                 is_first=False):
        super().__init__()

        if is_first:
            conv = [ConvLayer(in_channels, out_channels),
                    ConvLayer(out_channels, out_channels)]

        else:
            conv = [nn.BatchNorm2d(in_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    ConvLayer(in_channels, out_channels),
                    ConvLayer(out_channels, out_channels)]

        super().__init__(*conv)


class UpLayer(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up_conv = nn.Sequential(nn.BatchNorm2d(in_channels),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(in_channels, in_channels,
                                                        kernel_size=2, stride=2))

        cat_channels = in_channels + skip_channels

        self.conv = nn.Sequential(ConvLayer(cat_channels, cat_channels),
                                  ConvLayer(cat_channels, out_channels))

    def forward(self, x, x_skip):
        x_up = self.up_conv(x)
        x_cat = torch.cat([x_up, x_skip], dim=1)

        return self.conv(x_cat)
