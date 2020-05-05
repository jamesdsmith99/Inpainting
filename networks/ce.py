import torch
from torch import nn

from .layers import DoubleConvTranspose, FlattenConv, HalfConv, ProjectConv

class CE(nn.Module):

    def __init__(self, f=32):
        super(CE, self).__init__()

        self.conv_down1 = HalfConv(  1,    f, spec_norm=False, batch_norm=True, activation=nn.ReLU(True)) # 64x64 ⟶ 32x32
        self.conv_down2 = HalfConv(  f,  2*f, spec_norm=False, batch_norm=True, activation=nn.ReLU(True)) # 32x32 ⟶ 16x16
        self.conv_down3 = HalfConv(2*f,  4*f, spec_norm=False, batch_norm=True, activation=nn.ReLU(True)) # 16x16 ⟶ 8x8
        self.conv_down4 = HalfConv(4*f,  8*f, spec_norm=False, batch_norm=True, activation=nn.ReLU(True)) # 8x8   ⟶ 4x4
        
        self.conv_down5 = FlattenConv(8*f, 16*f, spec_norm=False, batch_norm=True, activation=nn.ReLU(True)) # 4x4 ⟶ 1x1

        self.conv_up1 = ProjectConv(16*f, 16*f, nn.ReLU(inplace=True))                                    # 1x1   ⟶ 4x4
        
        self.conv_up2 = DoubleConvTranspose(16*f, 8*f, batch_norm=True, activation=nn.ReLU(True))   # 4x4   ⟶ 8x8
        self.conv_up3 = DoubleConvTranspose( 8*f, 4*f, batch_norm=True, activation=nn.ReLU(True))   # 8x8   ⟶ 16x16
        self.conv_up4 = DoubleConvTranspose( 4*f, 2*f, batch_norm=True, activation=nn.ReLU(True))   # 16x16 ⟶ 32x32
        self.conv_up5 = DoubleConvTranspose( 2*f,   f, batch_norm=True, activation=nn.ReLU(True))   # 32x32 ⟶ 64x64

        self.conv_last = nn.Conv2d(f, 1, kernel_size=3, stride=1, padding=1)                              # 64x64 ⟶ 64x64
        self.tanh = nn.Tanh()

        self.layers = torch.nn.Sequential(
            self.conv_down1,
            self.conv_down2,
            self.conv_down3,
            self.conv_down4,
            self.conv_down5,
            self.conv_up1,
            self.conv_up2,
            self.conv_up3,
            self.conv_up4,
            self.conv_up5,
            self.conv_last,
            self.tanh
        )
    
    def forward(self, x):
        return self.layers(x)


