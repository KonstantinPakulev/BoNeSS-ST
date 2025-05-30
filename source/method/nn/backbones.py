import torch.nn as nn

from source.method.nn.blocks import DownLayer, UpLayer


class UNet(nn.Module):

    @staticmethod
    def from_config(unet_config):
        return UNet(unet_config.input_channels,
                    unet_config.down,
                    unet_config.get('up', []))

    def __init__(self, input_channels,
                 down, up):
        super().__init__()
        self.down_layers = nn.ModuleList()

        down = [input_channels] + list(down)

        for i, (d_in, d_out) in enumerate(zip(down[:-1], down[1:])):
            self.down_layers.append(DownLayer(d_in, d_out, i == 0))

        self.up_layers = nn.ModuleList()

        up = [down[-1]] + list(up)
        skip = down[-2:0:-1]

        for i, (u_in, u_skip, u_out) in enumerate(zip(up[:-1], skip, up[1:])):
            self.up_layers.append(UpLayer(u_in, u_skip, u_out))

    def forward(self, image):
        features = [image]

        for layer in self.down_layers:
            features.append(layer(features[-1]))

        skip_features = features[-2:0:-1]

        for layer, s_f in zip(self.up_layers, skip_features):
            features.append(layer(features[-1], s_f))

        return features


class VGG(nn.Module):

    @staticmethod
    def from_config(module_config):
        vgg_config = module_config.vgg
        return VGG(vgg_config.input_channels,
                   vgg_config.down)

    def __init__(self, input_channels,
                 down):
        super().__init__()
        self.down_layers = nn.ModuleList()

        down = [input_channels] + list(down)

        for i, (d_in, d_out) in enumerate(zip(down[:-1], down[1:])):
            self.down_layers.append(DownLayer(d_in, d_out,
                                              i == 0))

    def forward(self, x):
        for layer in self.down_layers:
            x = layer(x)

        return x
