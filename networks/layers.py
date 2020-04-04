from .conv import ConvBlock, ConvTransBlock

class Conv(ConvBlock):

    def __init__(self, in_channels, out_channels, activation):
        super(Conv, self).__init__(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            spec_norm=False,
            batch_norm=True,
            activation=activation)



class HalfConv(ConvBlock):

    def __init__(self, in_channels, out_channels, spec_norm, batch_norm, activation):
        super(HalfConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            spec_norm=spec_norm,
            batch_norm=batch_norm,
            activation=activation)

class FlattenConv(ConvBlock):

    def __init__(self, in_channels, out_channels, spec_norm, batch_norm, activation):
        super(FlattenConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=1,
            padding=0,
            spec_norm=spec_norm,
            batch_norm=batch_norm,
            activation=activation)


class DoubleConvTranspose(ConvTransBlock):

    def __init__(self, in_channels, out_channels, batch_norm, activation):
        super(DoubleConvTranspose, self).__init__(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            batch_norm=batch_norm,
            activation=activation)

class ProjectConv(ConvTransBlock):

    def __init__(self, in_channels, out_channels, activation):
        super(ProjectConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=1,
            padding=0,
            batch_norm=True,
            activation=activation)