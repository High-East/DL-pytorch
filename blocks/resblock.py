from torch import Tensor
import torch.nn as nn

from .basicblock import ConvSamePad2d


class ResBasicBlock(nn.Module):
    # expansion: int = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int or tuple,
            stride: int = 1,
            groups: int = 1,
            norm_layer=None,
            activation: str = 'ReLU',
            # downsample: bool = False
            # dilation: int = 1,
    ):
        super(ResBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = ConvSamePad2d(in_channels, out_channels, kernel_size, stride, groups, bias=False)
        self.norm1 = norm_layer(out_channels)
        self.activation = getattr(nn, activation)(inplace=True)
        self.conv2 = ConvSamePad2d(out_channels, out_channels, kernel_size, stride, groups, bias=False)
        self.norm2 = norm_layer(out_channels)

        self.shortcut = nn.Sequential()

        if (stride >= 2) or (in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                norm_layer(out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if hasattr(self, 'shortcut'):
            identity = self.shortcut(x)
        else:
            identity = x

        out += identity
        out = self.activation(out)

        return out


"""
References
- https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
