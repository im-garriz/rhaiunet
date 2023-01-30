import torch
from torch import nn
from dropblock import DropBlock2D, LinearScheduler
import torch.nn.functional as F

try:
    from common.utils import *
    from models.pooling_layers import HartleyPool2d, HybridPooling
except:
    pass

class DSC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DSC, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=(kernel_size-1)//2, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out 


class InceptionBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, pooling='max'):
        super().__init__()
        
        self.activation = nn.SiLU(inplace=True)

        self.convd_k1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        self.convd_k3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        self.convd_k5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        if pooling == 'max':
            self.pooling = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                DSC(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )
        elif pooling == 'avg':
            self.pooling = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )
        
    def forward(self, x):

        k1 = self.convd_k1(x)
        k3 = self.convd_k3(x)
        k5 = self.convd_k5(x)
        k7 = self.pooling(x)
        
        out = k1 + k3 + k5 + k7

        return out
    
class ResidualInceptionBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, dropout='none', dropout_p=0.04, block_size=1):
        super().__init__()
        
        self.dropout = dropout
        
        mid_channels = out_channels
        
        if dropout == 'dropout':
            self.dropout_layer = nn.Dropout(p=dropout_p)
        elif dropout == 'dropblock':
            self.dropout_layer = LinearScheduler(
                DropBlock2D(block_size=block_size, drop_prob=0.),
                start_value=0.,
                stop_value=dropout_p,
                nr_steps=100
            )
        
        self.subblock1 = InceptionBlock(in_channels, mid_channels)
        self.subblock2 = InceptionBlock(mid_channels, out_channels)
        
        self.convd_k1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        
    def forward(self, x):
        
        out = self.subblock1(x)
        out = self.subblock2(out)
        
        if self.dropout == 'dropout' or self.dropout == 'dropblock':
            return self.dropout_layer(out + self.convd_k1(x))
        else:
            return out + self.convd_k1(x)
        
        
class UnetSharp(nn.Module):

    def __init__(self, input_channels=1, nc=16, img_size=128, pooling="Max", dropout='dropout', block_size=3):
        super().__init__()

        dropout_ps = [0.075, 0.027, 0.006, 0.002, 0.075]
        dropout_ps = [0.01, 0.075, 0.027, 0.006, 0.002, 0.075]
        dropblock_sizes = [17, 11, 5, 3]
        
        self.pooling = pooling
        
        if self.pooling == 'Max':
            self.pool = nn.MaxPool2d(2, 2)
        elif self.pooling == 'Hartley':
            self.hartley_pool_x1_0 = HartleyPool2d(img_size//2)
            self.hartley_pool_x2_0 = HartleyPool2d(img_size//4)
            self.hartley_pool_x3_0 = HartleyPool2d(img_size//8)
            self.hartley_pool_x4_0 = HartleyPool2d(img_size//16)
        elif self.pooling == 'Hybrid':
            self.hybrid_pool_x1_0 = HybridPooling(img_size//2, nc)
            self.hybrid_pool_x2_0 = HybridPooling(img_size//4, nc*2)
            self.hybrid_pool_x3_0 = HybridPooling(img_size//8, nc*4)
            self.hybrid_pool_x4_0 = HybridPooling(img_size//16, nc*8)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = ResidualInceptionBlock(input_channels, nc, nc, dropout=dropout, dropout_p=dropout_ps[0], block_size=dropblock_sizes[0])
        self.conv1_0 = ResidualInceptionBlock(nc, nc*2, nc*2, dropout=dropout, dropout_p=dropout_ps[1], block_size=dropblock_sizes[1])
        self.conv2_0 = ResidualInceptionBlock(nc*2, nc*4, nc*4, dropout=dropout, dropout_p=dropout_ps[2], block_size=dropblock_sizes[2])
        self.conv3_0 = ResidualInceptionBlock(nc*4, nc*8, nc*8, dropout=dropout, dropout_p=dropout_ps[3], block_size=dropblock_sizes[3])
        self.conv4_0 = ResidualInceptionBlock(nc*8, nc*16, nc*16, dropout=dropout, dropout_p=dropout_ps[4])

        self.conv0_1 = ResidualInceptionBlock(nc+nc*2, nc, nc, dropout=dropout, dropout_p=dropout_ps[0], block_size=dropblock_sizes[0])
        self.conv1_1 = ResidualInceptionBlock(nc*2+nc*4, nc*2, nc*2, dropout=dropout, dropout_p=dropout_ps[1], block_size=dropblock_sizes[1])
        self.conv2_1 = ResidualInceptionBlock(nc*4+nc*8, nc*4, nc*4, dropout=dropout, dropout_p=dropout_ps[2], block_size=dropblock_sizes[2])
        self.conv3_1 = ResidualInceptionBlock(nc*8+nc*16, nc*8, nc*8, dropout=dropout, dropout_p=dropout_ps[3], block_size=dropblock_sizes[3])

        self.conv0_2 = ResidualInceptionBlock(nc*2+nc*2, nc, nc, dropout=dropout, dropout_p=dropout_ps[0], block_size=dropblock_sizes[0])
        self.conv1_2 = ResidualInceptionBlock(nc*2*2+nc*4, nc*2, nc*2, dropout=dropout, dropout_p=dropout_ps[1], block_size=dropblock_sizes[1])
        self.conv2_2 = ResidualInceptionBlock(nc*4*2+nc*8, nc*4, nc*4, dropout=dropout, dropout_p=dropout_ps[2], block_size=dropblock_sizes[2])

        self.conv0_3 = ResidualInceptionBlock(nc*3+nc*2, nc, nc, dropout=dropout, dropout_p=dropout_ps[0], block_size=dropblock_sizes[0])
        self.conv1_3 = ResidualInceptionBlock(nc*2*3+nc*4, nc*2, nc*2, dropout=dropout, dropout_p=dropout_ps[1], block_size=dropblock_sizes[1])

        self.conv0_4 = ResidualInceptionBlock(nc*4+nc*2, nc, nc, dropout=dropout, dropout_p=dropout_ps[0], block_size=dropblock_sizes[0])

        self.final = DSC(nc, 1, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        if self.pooling == 'Max':
            x1_0 = self.conv1_0(self.pool(x0_0))
        elif self.pooling == 'Hartley':
            x1_0 = self.conv1_0(self.hartley_pool_x1_0(x0_0))
        elif self.pooling == 'Hybrid':
            x1_0 = self.conv1_0(self.hybrid_pool_x1_0(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        
        if self.pooling == 'Max':
            x2_0 = self.conv2_0(self.pool(x1_0))
        elif self.pooling == 'Hartley':
            x2_0 = self.conv2_0(self.hartley_pool_x2_0(x1_0))
        elif self.pooling == 'Hybrid':
            x2_0 = self.conv2_0(self.hybrid_pool_x2_0(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        
        if self.pooling == 'Max':
            x3_0 = self.conv3_0(self.pool(x2_0))
        elif self.pooling == 'Hartley':
            x3_0 = self.conv3_0(self.hartley_pool_x3_0(x2_0))
        elif self.pooling == 'Hybrid':
            x3_0 = self.conv3_0(self.hybrid_pool_x3_0(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        
        if self.pooling == 'Max':
            x4_0 = self.conv4_0(self.pool(x3_0))
        elif self.pooling == 'Hartley':
            x4_0 = self.conv4_0(self.hartley_pool_x4_0(x3_0))
        elif self.pooling == 'Hybrid':
            x4_0 = self.conv4_0(self.hybrid_pool_x4_0(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output

    def init_weights(self):
        self.apply(weights_init)
        
############################################################################


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, pooling, out_size, dropout='none', dropout_p=0., block_size=3):
        super().__init__()
        
        if pooling == 'Max':
            self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            #ResidualInceptionBlock(in_channels, out_channels, dropout, dropout_p, block_size)
            RDAUNET_EncoderResidualUnit(in_channels, out_channels)
        )
        elif pooling == 'Hartley':
            self.maxpool_conv = nn.Sequential(
            HartleyPool2d(out_size),
            ResidualInceptionBlock(in_channels, out_channels)
        )
        elif pooling == 'Hybrid':
            self.maxpool_conv = nn.Sequential(
            HybridPooling(out_size, in_channels),
            ResidualInceptionBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    

class RDAUNET_EncoderResidualUnit(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(RDAUNET_EncoderResidualUnit, self).__init__()

        self.in_channels, self.out_channels, self.stride = in_channels, out_channels, stride

        self.sequence = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):

        return self.sequence(x) + self.downsample(x)
    
    
class RDAUNET_DecoderResidualUnit(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(RDAUNET_DecoderResidualUnit, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels

        self.sequence = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):

        return self.sequence(x) + self.shortcut(x)
    
class _Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, pooling, out_size, dropout='none', dropout_p=0., block_size=3):
        super().__init__()
        
        if pooling == 'Max':
            self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            RDAUNET_EncoderResidualUnit(in_channels, out_channels)
        )
        elif pooling == 'Hartley':
            self.maxpool_conv = nn.Sequential(
            HartleyPool2d(out_size),
            RDAUNET_EncoderResidualUnit(in_channels, out_channels)
        )
        elif pooling == 'Hybrid':
            self.maxpool_conv = nn.Sequential(
            HybridPooling(out_size, in_channels),
            RDAUNET_EncoderResidualUnit(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ResidualInceptionBlock(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = ResidualInceptionBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):

    def __init__(self, g_n_channels, h_n_channels, ksize=3):
        super(AttentionGate, self).__init__()

        self.conv1_g = nn.Sequential(
            nn.Conv2d(g_n_channels, g_n_channels, kernel_size=ksize, padding=(ksize-1)//2),
            nn.BatchNorm2d(g_n_channels)
        )
        
        self.conv1_h = nn.Sequential(
            nn.Conv2d(h_n_channels, g_n_channels, kernel_size=ksize, padding=(ksize-1)//2),
            nn.BatchNorm2d(g_n_channels)
        )

        self.relu = nn.ReLU(inplace=True)

        self.conv_out = nn.Sequential(
            nn.Conv2d(g_n_channels, g_n_channels, kernel_size=ksize, padding=(ksize-1)//2),
            nn.BatchNorm2d(g_n_channels),
            nn.Sigmoid()
        )
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, g, h):
        
        h = self.upsample(h)

        x = self.relu(self.conv1_g(g) + self.conv1_h(h))

        alpha = self.conv_out(x)

        return alpha * g
    
    

class UNetSharp2(nn.Module):
    
    """
    Esta es la final
    
    """
    def __init__(self, nc=32, n_channels=1, n_classes=1, pooling='Max', size=128, dropout='none', dropout_p=0., block_size=3):
        super(UNetSharp2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes


        self.inc = DoubleConv(n_channels, nc)
        self.down1 = Down(nc, nc*2, pooling, size//2, dropout, dropout_p, block_size)
        self.down2 = Down(nc*2, nc*4, pooling, size//4, dropout, dropout_p, block_size)
        self.down3 = Down(nc*4, nc*8, pooling, size//8, dropout, dropout_p, block_size)
        self.down4 = Down(nc*8, nc*16, pooling, size//16, dropout, dropout_p, block_size)
        
        self.inc2 = DoubleConv(n_channels, nc)
        self.down12 = Down(nc, nc*2, pooling, size//2, dropout, dropout_p, block_size)
        self.down22 = Down(nc*2, nc*4, pooling, size//4, dropout, dropout_p, block_size)
        self.down32 = Down(nc*4, nc*8, pooling, size//8, dropout, dropout_p, block_size)
        self.down42 = Down(nc*8, nc*16, pooling, size//16, dropout, dropout_p, block_size)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.up1 = ResidualInceptionBlock(2*nc*8 + nc*16, nc*8, dropout, dropout_p, block_size)
        self.up2 = ResidualInceptionBlock(2*nc*4 + nc*8, nc*4, dropout, dropout_p, block_size)
        self.up3 = ResidualInceptionBlock(2*nc*2 + nc*4, nc*2, dropout, dropout_p, block_size)
        self.up4 = ResidualInceptionBlock(2*nc + nc*2, nc, dropout, dropout_p, block_size)
        
        """
        self.up1 = RDAUNET_DecoderResidualUnit(2*nc*8 + nc*16, nc*8)
        self.up2 = RDAUNET_DecoderResidualUnit(2*nc*4 + nc*8, nc*4)
        self.up3 = RDAUNET_DecoderResidualUnit(2*nc*2 + nc*4, nc*2)
        self.up4 = RDAUNET_DecoderResidualUnit(2*nc + nc*2, nc)
        """
        """
        
        """
        
        ksize=3
        
        self.attention1 = AttentionGate(2*nc*8, nc*16, ksize=ksize)
        self.attention2 = AttentionGate(2*nc*4, nc*8, ksize=ksize)
        self.attention3 = AttentionGate(2*nc*2, nc*4, ksize=ksize)
        self.attention4 = AttentionGate(2*nc, nc*2, ksize=ksize)
        
        self.outc = nn.Conv2d(nc, 1, kernel_size=1)

    def init_weights(self):

        self.apply(weights_init)

    def forward(self, x, x_orig):

        # size x size x 1
        x1 = self.inc(x)
        self.l1 = torch.clone(x1)
        # size x size x nc
        x2 = self.down1(x1)
        self.l2 = torch.clone(x2)
        # size/2 x size/2 x nc*2
        x3 = self.down2(x2)
        self.l3 = torch.clone(x3)
        # size/4 x size/4 x nc*4
        x4 = self.down3(x3)
        self.l4 = torch.clone(x4)
        # size/8 x size/8 x nc*8
        x5 = self.down4(x4)
        self.l5 = torch.clone(x5)
        # size/16 x size/16 x nc*16
        
        
        # size x size x 1
        x12 = self.inc2(x_orig)
        # size x size x nc
        x22 = self.down12(x12)
        # size/2 x size/2 x nc*2
        x32 = self.down22(x22)
        # size/4 x size/4 x nc*4
        x42 = self.down32(x32)
        # size/8 x size/8 x nc*8
        
        
        z = self.attention1(torch.cat([x4, x42], axis=1), x5)
        #z = self.attention1(x4, x5)
        y = self.upsample(x5)
        x = self.up1(torch.cat([z, y], axis=1))
        self.l6 = torch.clone(x)
        
        z = self.attention2(torch.cat([x3, x32], axis=1), x)
        #z = self.attention2(x3, x)
        y = self.upsample(x)
        x = self.up2(torch.cat([z, y], axis=1))
        self.l7 = torch.clone(x)
        
        z = self.attention3(torch.cat([x2, x22], axis=1), x)
        #z = self.attention3(x2, x)
        y = self.upsample(x)
        x = self.up3(torch.cat([z, y], axis=1))
        self.l8 = torch.clone(x)
        
        z = self.attention4(torch.cat([x1, x12], axis=1), x)
        #z = self.attention4(x1, x)
        y = self.upsample(x)
        x = self.up4(torch.cat([z, y], axis=1))
        self.l9 = torch.clone(x)
        
        return self.outc(x)
    
    def get_outputs_per_layer(self):
        
        return self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7, self.l8, self.l9,
    
    
class _UNetSharp2(nn.Module):
    

    def __init__(self, nc=32, n_channels=1, n_classes=1, pooling='Max', size=128, dropout='none', dropout_p=0., block_size=3):
        super(UNetSharp2, self).__init__()
        

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, nc)
        self.down1 = Down(nc, nc*2, pooling, size//2, dropout, dropout_p, block_size)
        self.down2 = Down(nc*2, nc*4, pooling, size//4, dropout, dropout_p, block_size)
        self.down3 = Down(nc*4, nc*8, pooling, size//8, dropout, dropout_p, block_size)
        self.down4 = Down(nc*8, nc*16, pooling, size//16, dropout, dropout_p, block_size)
        
        self.inc2 = DoubleConv(n_channels, nc)
        self.down12 = Down(nc, nc*2, pooling, size//2, dropout, dropout_p, block_size)
        self.down22 = Down(nc*2, nc*4, pooling, size//4, dropout, dropout_p, block_size)
        self.down32 = Down(nc*4, nc*8, pooling, size//8, dropout, dropout_p, block_size)
        self.down42 = Down(nc*8, nc*16, pooling, size//16, dropout, dropout_p, block_size)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.up1 = RDAUNET_DecoderResidualUnit(2*nc*8 + nc*16, nc*8)
        self.up2 = RDAUNET_DecoderResidualUnit(2*nc*4 + nc*8, nc*4)
        self.up3 = RDAUNET_DecoderResidualUnit(2*nc*2 + nc*4, nc*2)
        self.up4 = RDAUNET_DecoderResidualUnit(2*nc + nc*2, nc)
   
        self.attention1 = AttentionGate(2*nc*8, nc*16)
        self.attention2 = AttentionGate(2*nc*4, nc*8)
        self.attention3 = AttentionGate(2*nc*2, nc*4)
        self.attention4 = AttentionGate(2*nc, nc*2)
        
        self.outc = nn.Conv2d(nc, 1, kernel_size=1)

    def init_weights(self):

        self.apply(weights_init)

    def forward(self, x, x_orig):

        # size x size x 1
        x1 = self.inc(x)
        self.l1 = torch.clone(x1)
        # size x size x nc
        x2 = self.down1(x1)
        self.l2 = torch.clone(x2)
        # size/2 x size/2 x nc*2
        x3 = self.down2(x2)
        self.l3 = torch.clone(x3)
        # size/4 x size/4 x nc*4
        x4 = self.down3(x3)
        self.l4 = torch.clone(x4)
        # size/8 x size/8 x nc*8
        x5 = self.down4(x4)
        self.l5 = torch.clone(x5)
        # size/16 x size/16 x nc*16
        
        
        # size x size x 1
        x12 = self.inc2(x_orig)
        # size x size x nc
        x22 = self.down12(x12)
        # size/2 x size/2 x nc*2
        x32 = self.down22(x22)
        # size/4 x size/4 x nc*4
        x42 = self.down32(x32)
        # size/8 x size/8 x nc*8
        
        
        z = self.attention1(torch.cat([x4, x42], axis=1), x5)
        #z = self.attention1(x4, x5)
        y = self.upsample(x5)
        x = self.up1(torch.cat([z, y], axis=1))
        self.l6 = torch.clone(x)
        
        z = self.attention2(torch.cat([x3, x32], axis=1), x)
        #z = self.attention2(x3, x)
        y = self.upsample(x)
        x = self.up2(torch.cat([z, y], axis=1))
        self.l7 = torch.clone(x)
        
        z = self.attention3(torch.cat([x2, x22], axis=1), x)
        #z = self.attention3(x2, x)
        y = self.upsample(x)
        x = self.up3(torch.cat([z, y], axis=1))
        self.l8 = torch.clone(x)
        
        z = self.attention4(torch.cat([x1, x12], axis=1), x)
        #z = self.attention4(x1, x)
        y = self.upsample(x)
        x = self.up4(torch.cat([z, y], axis=1))
        self.l9 = torch.clone(x)
        
        return self.outc(x)
    
    def get_outputs_per_layer(self):
        
        return self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7, self.l8, self.l9, 
    

    
class UNetSharp_sinNP(nn.Module):
    def __init__(self, nc=32, n_channels=1, n_classes=1, pooling='Max', size=128, dropout='none', dropout_p=0., block_size=3):
        super(UNetSharp_sinNP, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, nc)
        self.down1 = Down(nc, nc*2, pooling, size//2, dropout, dropout_p, block_size)
        self.down2 = Down(nc*2, nc*4, pooling, size//4, dropout, dropout_p, block_size)
        self.down3 = Down(nc*4, nc*8, pooling, size//8, dropout, dropout_p, block_size)
        self.down4 = Down(nc*8, nc*16, pooling, size//16, dropout, dropout_p, block_size)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.up1 = ResidualInceptionBlock(nc*8 + nc*16, nc*8, dropout, dropout_p, block_size)
        self.up2 = ResidualInceptionBlock(nc*4 + nc*8, nc*4, dropout, dropout_p, block_size)
        self.up3 = ResidualInceptionBlock(nc*2 + nc*4, nc*2, dropout, dropout_p, block_size)
        self.up4 = ResidualInceptionBlock(nc + nc*2, nc, dropout, dropout_p, block_size)
        
        
        self.attention1 = AttentionGate(nc*8, nc*16)
        self.attention2 = AttentionGate(nc*4, nc*8)
        self.attention3 = AttentionGate(nc*2, nc*4)
        self.attention4 = AttentionGate(nc, nc*2)
        
        self.outc = nn.Conv2d(nc, 1, kernel_size=1)

    def init_weights(self):

        self.apply(weights_init)

    def forward(self, x):

        # size x size x 1
        x1 = self.inc(x)
        self.l1 = torch.clone(x1)
        # size x size x nc
        x2 = self.down1(x1)
        self.l2 = torch.clone(x2)
        # size/2 x size/2 x nc*2
        x3 = self.down2(x2)
        self.l3 = torch.clone(x3)
        # size/4 x size/4 x nc*4
        x4 = self.down3(x3)
        self.l4 = torch.clone(x4)
        # size/8 x size/8 x nc*8
        x5 = self.down4(x4)
        self.l5 = torch.clone(x5)
        # size/16 x size/16 x nc*16
        
        z = self.attention1(x4, x5)
        #z = self.attention1(x4, x5)
        y = self.upsample(x5)
        x = self.up1(torch.cat([z, y], axis=1))
        self.l6 = torch.clone(x)
        
        z = self.attention2(x3, x)
        #z = self.attention2(x3, x)
        y = self.upsample(x)
        x = self.up2(torch.cat([z, y], axis=1))
        self.l7 = torch.clone(x)
        
        z = self.attention3(x2,  x)
        #z = self.attention3(x2, x)
        y = self.upsample(x)
        x = self.up3(torch.cat([z, y], axis=1))
        self.l8 = torch.clone(x)
        
        z = self.attention4(x1, x)
        #z = self.attention4(x1, x)
        y = self.upsample(x)
        x = self.up4(torch.cat([z, y], axis=1))
        self.l9 = torch.clone(x)
        
        return self.outc(x)
        

if __name__ == '__main__':
    
    x = torch.rand([8, 1, 128, 128])
    unet = UnetSharp()
    a = unet(x)
    print(a.shape)
    