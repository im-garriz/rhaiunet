import torch
import torch.nn as nn
from models.segmentors.pooling_layers import HartleyPool2d, HybridPooling
from models.segmentors.gated_conv import GatedConv2d

try:
    from common.utils import *
    
except:
    pass


class DSC(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1):
        super(DSC, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=ksize,
                                   padding=(ksize-1)//2, stride=stride, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        
        out = self.depthwise(x)
        out = self.pointwise(out)
        
        return out
    
    
class Upsample(nn.Module):
    
    def __init__(self, in_channels):
        super(Upsample, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        
    def forward(self, x):
        return self.up(x)
        


class InceptionConvolutionLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, dims):
        super(InceptionConvolutionLayer, self).__init__()
        
        self.dsc_5 = nn.Sequential(
            DSC(in_channels, in_channels, 5),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.dsc_3 = nn.Sequential(
            DSC(in_channels, in_channels, 3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.dsc_1 = nn.Sequential(
            DSC(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.hyb_pool = nn.Sequential(
            HybridPooling(dims, in_channels, 2, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.last_dsc = nn.Sequential(
            DSC(in_channels*4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        
        out1 = self.dsc_5(x)
        out2 = self.dsc_3(x)
        out3 = self.dsc_1(x)
        out4 = self.hyb_pool(x)
        
        out = self.last_dsc(torch.cat([out1, out2, out3, out4], axis=1))

        return out
        

class ResidualInceptionLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, dims):
        super(ResidualInceptionLayer, self).__init__()
        
        self.inception_conv1 = InceptionConvolutionLayer(in_channels, out_channels, dims)
        self.inception_conv2 = InceptionConvolutionLayer(out_channels, out_channels, dims)
        self.skip_conv = DSC(in_channels, out_channels, 1)
        
    def forward(self, x):
        
        out = self.inception_conv1(x)
        out = self.inception_conv2(out)
        
        x = self.skip_conv(x)
        
        return out + x
    
    
class CrossSpatialAttentionFilter(nn.Module):
    
    def __init__(self, enc_l_channels, enc_lm1_channels, dec_lm1_channels, enc_l_dims):
        super(CrossSpatialAttentionFilter, self).__init__()
        
        self.gating1 = nn.Sequential(
            DSC(enc_l_channels, enc_l_channels, 1, stride=2)
        )
        self.gating2= nn.Sequential(
            DSC(enc_lm1_channels, enc_l_channels, 1, stride=2)
        )
        self.gating3 = GatedConv2d(dec_lm1_channels, enc_l_channels, 1)
        
        self.dsc1_1 = DSC(enc_l_channels, enc_l_channels, 1)
        self.dsc2_1 = DSC(enc_l_channels, enc_l_channels, 1)
        
        # ADD
        
        self.dsc1_2 = nn.Sequential(
            DSC(enc_l_channels, enc_l_channels, 1),
            nn.ReLU(inplace=True)
        )
        self.dsc2_2 = nn.Sequential(
            DSC(enc_l_channels, 1, 1),
            Upsample(1)
        )
        
        # PROD
        
        self.last = nn.Sequential(
            nn.BatchNorm2d(enc_l_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, enc_l, enc_lm1, dec_lm1):
        
        x1 = self.dsc1_1(self.gating1(enc_l))
        x2 = self.dsc2_1(self.gating2(enc_lm1))
        x3 = self.gating3(dec_lm1)
        
        x = x1 + x2 + x3
        
        x = self.dsc1_2(x)
        x = self.dsc2_2(x)
        
        x = x * enc_l
        
        out = self.last(x)
        
        return out
        
class RCA_IUNET(nn.Module):
    
    def __init__(self, img_channels, nc, dims=128):
        super(RCA_IUNET, self).__init__()
        
        self.downsampling_residual_inception_block1 = ResidualInceptionLayer(img_channels, nc, dims)
        self.downsampling_residual_inception_block2 = ResidualInceptionLayer(nc, nc*2, dims//2)
        self.downsampling_residual_inception_block3 = ResidualInceptionLayer(nc*2, nc*4, dims//4)
        self.downsampling_residual_inception_block4 = ResidualInceptionLayer(nc*4, nc*8, dims//8)
        
        self.bottleneck_residual_inception_block = ResidualInceptionLayer(nc*8, nc*16, dims//16)
        
        self.upsampling_residual_inception_block4 = ResidualInceptionLayer(nc*8 + nc*16, nc*8, dims//8)
        self.upsampling_residual_inception_block3 = ResidualInceptionLayer(nc*4 + nc*8, nc*4, dims//4)
        self.upsampling_residual_inception_block2 = ResidualInceptionLayer(nc*2 + nc * 4, nc*2, dims//2)
        self.upsampling_residual_inception_block1 = ResidualInceptionLayer(nc + nc*2, img_channels, dims)
        
        
        self.pool1 = HybridPooling(dims//2, nc)
        self.pool2 = HybridPooling(dims//4, nc*2)
        self.pool3 = HybridPooling(dims//8, nc*4)
        self.pool4 = HybridPooling(dims//16, nc*8)
        
        self.up1 = Upsample(nc*2)
        self.up2 = Upsample(nc*4)
        self.up3 = Upsample(nc*8)
        self.up4 = Upsample(nc*16)
        
        self.cross_spatial_attention_filter1 = CrossSpatialAttentionFilter(nc, img_channels, nc*2, dims)
        self.cross_spatial_attention_filter2 = CrossSpatialAttentionFilter(nc*2, nc, nc*4, dims//2)
        self.cross_spatial_attention_filter3 = CrossSpatialAttentionFilter(nc*4, nc*2, nc*8, dims//4)
        self.cross_spatial_attention_filter4 = CrossSpatialAttentionFilter(nc*8, nc*4, nc*16, dims//8)
        
    def forward(self, x):
        
        d_rsi1_output = self.downsampling_residual_inception_block1(x)
        d_rsi2_output = self.downsampling_residual_inception_block2(self.pool1(d_rsi1_output))
        d_rsi3_output = self.downsampling_residual_inception_block3(self.pool2(d_rsi2_output))
        d_rsi4_output = self.downsampling_residual_inception_block4(self.pool3(d_rsi3_output))
        
        bottleneck_rsi_output = self.bottleneck_residual_inception_block(self.pool4(d_rsi4_output))
        
        csa4_output = self.cross_spatial_attention_filter4(d_rsi4_output, self.pool3(d_rsi3_output), bottleneck_rsi_output)
        e_rsi4_output = self.upsampling_residual_inception_block4(torch.cat([self.up4(bottleneck_rsi_output), csa4_output],axis=1))
        
        csa3_output = self.cross_spatial_attention_filter3(d_rsi3_output, self.pool2(d_rsi2_output), e_rsi4_output)
        e_rsi3_output = self.upsampling_residual_inception_block3(torch.cat([self.up3(e_rsi4_output), csa3_output], axis=1))

        csa2_output = self.cross_spatial_attention_filter2(d_rsi2_output, self.pool1(d_rsi1_output), e_rsi3_output)
        e_rsi2_output = self.upsampling_residual_inception_block2(torch.cat([self.up2(e_rsi3_output), csa2_output], axis=1))

        csa1_output = self.cross_spatial_attention_filter1(d_rsi1_output, x, e_rsi2_output)
        out = self.upsampling_residual_inception_block1(torch.cat([self.up1(e_rsi2_output), csa1_output], axis=1))
        
        return out
    
    def init_weights(self):

        self.apply(weights_init)
        
        
        
if __name__ == '__main__':
    
    def get_n_params(model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp
    
    x = torch.rand([8, 1, 128, 128])
    model = RCA_IUNET(1, nc=16)
    a = model(x)
    print(a.shape)
    print(f"RCA IUNET: {get_n_params(model)}")
    """
    model2 = UnetSharp(nc=16, pooling='Hybrid')
    a = model2(x)
    print(a.shape)
    print(f"Unet#: {get_n_params(model2)}")
    
    x1 = torch.rand([8, 16, 128, 128])
    x2= torch.rand([8, 8, 128, 128])
    x3 = torch.rand([8, 64, 64, 64])
    gate = CrossSpatialAttentionFilter(16, 8, 64)
    a = gate(x1, x2, x3)
    
    print(a.shape)"""
    