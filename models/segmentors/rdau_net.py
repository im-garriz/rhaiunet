import torch
import torch.nn as nn
import torchvision
from common.utils import *


class EncoderResidualUnit(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(EncoderResidualUnit, self).__init__()

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


class DilationConVolutionsChain(nn.Module):

    def __init__(self):

        super(DilationConVolutionsChain, self).__init__()

        self.dilated_conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.dilated_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.dilated_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4, bias=False)
        self.dilated_conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=8, dilation=8, bias=False)
        self.dilated_conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=16, dilation=16, bias=False)
        self.dilated_conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=32, dilation=32, bias=False)


    def forward(self, x):

        dconv1 = self.dilated_conv1(x)
        dconv2 = self.dilated_conv2(dconv1)
        dconv3 = self.dilated_conv3(dconv2)
        dconv4 = self.dilated_conv4(dconv3)
        dconv5 = self.dilated_conv5(dconv4)
        dconv6 = self.dilated_conv6(dconv5)

        return dconv6 + dconv5 + dconv4 + dconv3 + dconv2 + dconv1


class AttentionGate(nn.Module):

    def __init__(self, g_n_channels, h_n_channels):
        super(AttentionGate, self).__init__()

        self.conv1_g = nn.Conv2d(g_n_channels, g_n_channels, kernel_size=1, stride=1, padding=0)
        self.conv1_h = nn.Conv2d(h_n_channels, g_n_channels, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(g_n_channels, g_n_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(g_n_channels, g_n_channels, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, g, h):

        x = self.relu(self.conv1_g(g) + self.conv1_h(h))

        x = self.conv2(x)
        x = self.conv3(x)

        alpha = self.sigmoid(x)

        return alpha + g


class DecoderResidualUnit(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(DecoderResidualUnit, self).__init__()

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

class RDAU_NET(nn.Module):

    def __init__(self):
        super(RDAU_NET, self).__init__()

        self.downsampling_conv1 = EncoderResidualUnit(in_channels=1, out_channels=32, stride=1)
        self.downsampling_conv2 = EncoderResidualUnit(in_channels=32, out_channels=64, stride=2)
        self.downsampling_conv3 = EncoderResidualUnit(in_channels=64, out_channels=128, stride=2)
        self.downsampling_conv4 = EncoderResidualUnit(in_channels=128, out_channels=256, stride=2)
        self.downsampling_conv5 = EncoderResidualUnit(in_channels=256, out_channels=512, stride=2)
        self.downsampling_conv6 = EncoderResidualUnit(in_channels=512, out_channels=512, stride=2)

        self.dilated_conv_block = DilationConVolutionsChain()

        self.upsample_x2 = nn.Sequential(
            # Cambiar a upconv
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(inplace=True)
        )

        self.attention_gate5 = AttentionGate(512, 256)
        self.attention_gate4 = AttentionGate(256, 512)
        self.attention_gate3 = AttentionGate(128, 256)
        self.attention_gate2 = AttentionGate(64, 128)
        self.attention_gate1 = AttentionGate(32, 64)

        self.decoder_residual_block5 = DecoderResidualUnit(768, 512)
        self.decoder_residual_block4 = DecoderResidualUnit(768, 256)
        self.decoder_residual_block3 = DecoderResidualUnit(384, 128)
        self.decoder_residual_block2 = DecoderResidualUnit(192, 64)
        self.decoder_residual_block1 = DecoderResidualUnit(96, 32)

        self.last_conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.Hardsigmoid()
        )

    def init_weights(self):

        self.apply(weights_init)

    def forward(self, x):

        downsampled1 = self.downsampling_conv1(x)
        downsampled2 = self.downsampling_conv2(downsampled1)
        downsampled3 = self.downsampling_conv3(downsampled2)
        downsampled4 = self.downsampling_conv4(downsampled3)
        downsampled5 = self.downsampling_conv5(downsampled4)
        downsampled6 = self.downsampling_conv6(downsampled5)

        dilation_convs_output = self.dilated_conv_block(downsampled6)

        upsampled_dilation_convs_output = self.upsample_x2(dilation_convs_output)
        att_gate5_out = self.attention_gate5(downsampled5, upsampled_dilation_convs_output)
        concat5 = torch.cat((att_gate5_out, upsampled_dilation_convs_output), 1)
        upsampled5 = self.decoder_residual_block5(concat5)

        h = self.upsample_x2(upsampled5)
        att_gate4_out = self.attention_gate4(downsampled4, h)
        concat4 = torch.cat((att_gate4_out, h), 1)
        upsampled4 = self.decoder_residual_block4(concat4)

        h = self.upsample_x2(upsampled4)
        att_gate3_out = self.attention_gate3(downsampled3, h)
        concat3 = torch.cat((att_gate3_out, h), 1)
        upsampled3 = self.decoder_residual_block3(concat3)

        h = self.upsample_x2(upsampled3)
        att_gate2_out = self.attention_gate2(downsampled2, h)
        concat2 = torch.cat((att_gate2_out, h), 1)
        upsampled2 = self.decoder_residual_block2(concat2)

        h = self.upsample_x2(upsampled2)
        att_gate1_out = self.attention_gate1(downsampled1, h)
        concat1 = torch.cat((att_gate1_out, h), 1)
        upsampled1 = self.decoder_residual_block1(concat1)

        return self.last_conv(upsampled1)
