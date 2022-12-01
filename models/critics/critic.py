import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from common.utils import weights_init


class Critic1(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.main_module = nn.Sequential(
            # Image (Cx128x128)
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2, inplace=True),

            # State (128x64x64)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x32x32)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x16x16)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.MaxPool2d(2)
            )

            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0),
            #nn.Sigmoid()
        )

    def init_weights(self):

        self.apply(weights_init)

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

class Critic2(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.main_module = nn.Sequential(
            # Image (Cx128x128)
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(0.2, inplace=True),

            # State (128x64x64)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x32x32)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x16x16)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.MaxPool2d(2)
            )

            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=4, stride=1, padding=0),
            #nn.Sigmoid()
        )

    def init_weights(self):

        self.apply(weights_init)

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

class Critic3_CT_WGAN(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.block1 = nn.Sequential(
            # Image (Cx128x128)
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2, inplace=True),)

        self.block2 = nn.Sequential(
            # State (128x64x64)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),)

        self.block3 = nn.Sequential(
            # State (256x32x32)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),)

        self.block4 = nn.Sequential(
            # State (512x16x16)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.MaxPool2d(2)
            )

            # output of main module --> State (1024x4x4)

        self.linear = nn.Sequential(nn.Linear(1024, 1)) 

    def init_weights(self):

        self.apply(weights_init)

    def forward(self, x, dropout=0.0, intermediate_output=False):
        x = self.block1(x)# State (128x64x64)
        x = F.dropout(x, training=True, p=dropout)
        x = self.block2(x)# State (256x32x32)
        x = F.dropout(x, training=True, p=dropout)
        x = self.block3(x)# State (512x16x16)
        x = F.dropout(x, training=True, p=dropout)
        x = self.block4(x)# State (1024x4x4)
        x = F.dropout(x, training=True, p=dropout)

        x = x.view(x.size(0), 1024, -1)
        x = x.mean(dim=2)

        out = self.linear(x)

        if intermediate_output:
            return out, x # y is the D_(.), intermediate layer given in paper.

        return out