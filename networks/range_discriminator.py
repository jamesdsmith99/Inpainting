from torch import nn

from .layers import FlattenConv, HalfConv

class RangeDiscriminator(nn.Module):

    '''
    DCGAN discriminator with spec norm outputs in the range [0, output_range]
    '''

    def __init__(self, output_range=1, f=64):
        super(RangeDiscriminator, self).__init__()

        self.range = output_range

        conv1 =    HalfConv(  1,   f, spec_norm=True, batch_norm=True,  activation=nn.LeakyReLU(0.2, True)) # 64x64 ⟶ 32x32
        conv2 =    HalfConv(  f, 2*f, spec_norm=True, batch_norm=True,  activation=nn.LeakyReLU(0.2, True)) # 32x32 ⟶ 16x16
        conv3 =    HalfConv(2*f, 4*f, spec_norm=True, batch_norm=True,  activation=nn.LeakyReLU(0.2, True)) # 16x16 ⟶ 8x8
        conv4 =    HalfConv(4*f, 8*f, spec_norm=True, batch_norm=True,  activation=nn.LeakyReLU(0.2, True)) # 8x8   ⟶ 4x4
        
        self.model = nn.Sequential(conv1, conv2, conv3, conv4)

        self.conv_out = FlattenConv(8*f,   1, spec_norm=True, batch_norm=False, activation=nn.Sigmoid())    # 4x4   ⟶ 1x1


    def forward(self, x):
        x = self.model(x)
        return self.range * self.conv_out(x)