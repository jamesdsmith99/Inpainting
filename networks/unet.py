import torch
from torch import nn

from .layers import DoubleConvTranspose, FlattenConv, HalfConv, ProjectConv

class UNet(nn.Module):

    def __init__(self, f=32):
        super(UNet, self).__init__()

        self.conv_down1 = HalfConv(  1,    f, spec_norm=False, batch_norm=True, activation=nn.ReLU(True)) # 64x64 ⟶ 32x32
        self.conv_down2 = HalfConv(  f,  2*f, spec_norm=False, batch_norm=True, activation=nn.ReLU(True)) # 32x32 ⟶ 16x16
        self.conv_down3 = HalfConv(2*f,  4*f, spec_norm=False, batch_norm=True, activation=nn.ReLU(True)) # 16x16 ⟶ 8x8
        self.conv_down4 = HalfConv(4*f,  8*f, spec_norm=False, batch_norm=True, activation=nn.ReLU(True)) # 8x8   ⟶ 4x4
        
        self.conv_down5 = FlattenConv(8*f, 16*f, spec_norm=False, batch_norm=True, activation=nn.ReLU(True)) # 4x4 ⟶ 1x1

        self.conv_up1 = ProjectConv(16*f, 16*f, nn.ReLU(inplace=True))                                    # 1x1   ⟶ 4x4
        
        self.conv_up2 = DoubleConvTranspose(8*f + 16*f, 8*f, batch_norm=True, activation=nn.ReLU(True))   # 4x4   ⟶ 8x8
        self.conv_up3 = DoubleConvTranspose(4*f +  8*f, 4*f, batch_norm=True, activation=nn.ReLU(True))   # 8x8   ⟶ 16x16
        self.conv_up4 = DoubleConvTranspose(2*f +  4*f, 2*f, batch_norm=True, activation=nn.ReLU(True))   # 16x16 ⟶ 32x32
        self.conv_up5 = DoubleConvTranspose(  f +  2*f,   f, batch_norm=True, activation=nn.ReLU(True))   # 32x32 ⟶ 64x64

        self.conv_last = nn.Conv2d(f, 1, kernel_size=3, stride=1, padding=1)                              # 64x64 ⟶ 64x64
        self.tanh = nn.Tanh()

    def up(self, layer, x, skip):
        x = torch.cat([x, skip], dim=1)
        return layer(x)
    
    def forward(self, x):
        skipA = self.conv_down1(x)
        skipB = self.conv_down2(skipA)
        skipC = self.conv_down3(skipB)
        skipD = self.conv_down4(skipC)

        x = self.conv_down5(skipD)
        x = self.conv_up1(x)

        x = self.up(self.conv_up2, x, skipD)
        x = self.up(self.conv_up3, x, skipC)
        x = self.up(self.conv_up4, x, skipB)
        x = self.up(self.conv_up5, x, skipA)

        x = self.conv_last(x)
        return self.tanh(x)


