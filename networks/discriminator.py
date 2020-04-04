from torch import nn

from .layers import FlattenConv, HalfConv

class Discriminator(nn.Module):

    '''
    DCGAN discriminator with spec norm
    '''

    def __init__(self, f=64):
        super(Discriminator, self).__init__()

        conv1 =    HalfConv(  1,   f, spec_norm=True, batch_norm=True,  activation=nn.LeakyReLU(0.2, True)) # 64x64 ⟶ 32x32
        conv2 =    HalfConv(  f, 2*f, spec_norm=True, batch_norm=True,  activation=nn.LeakyReLU(0.2, True)) # 32x32 ⟶ 16x16
        conv3 =    HalfConv(2*f, 4*f, spec_norm=True, batch_norm=True,  activation=nn.LeakyReLU(0.2, True)) # 16x16 ⟶ 8x8
        conv4 =    HalfConv(4*f, 8*f, spec_norm=True, batch_norm=True,  activation=nn.LeakyReLU(0.2, True)) # 8x8   ⟶ 4x4
        conv5 = FlattenConv(8*f,   1, spec_norm=True, batch_norm=False, activation=nn.Sigmoid())            # 4x4   ⟶ 1x1

        self.model = nn.Sequential(conv1, conv2, conv3, conv4, conv5)

    def forward(self, x):
        return self.model(x)