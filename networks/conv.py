from torch import nn
from torch.nn.utils import spectral_norm as spectral_norm_fn

class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            spec_norm=False,
            batch_norm=False,
            activation=None
        ):

        super(ConvBlock, self).__init__()

        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        if spec_norm:
            conv = spectral_norm_fn(conv)

        layers = [conv]

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if activation is not None:
            layers.append(activation)

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class ConvTransBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            batch_norm=False,
            activation=None
        ):

        super(ConvTransBlock, self).__init__()

        conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        layers = [conv]

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if activation is not None:
            layers.append(activation)

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)