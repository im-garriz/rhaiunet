import torch
import torch.nn as nn
from torch.autograd import Function
import math
from torch.nn.modules.utils import _pair


def _spectral_crop(input, oheight, owidth, _torch=True):

    cutoff_freq_h = math.ceil(oheight / 2)
    cutoff_freq_w = math.ceil(owidth / 2)

    if oheight % 2 == 1:
        if owidth % 2 == 1:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):]
            bottom_left = input[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w]
            bottom_right = input[:, :, -(cutoff_freq_h-1):, -(cutoff_freq_w-1):]
        else:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            bottom_left = input[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w]
            bottom_right = input[:, :, -(cutoff_freq_h-1):, -cutoff_freq_w:]
    else:
        if owidth % 2 == 1:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):]
            bottom_left = input[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            bottom_right = input[:, :, -cutoff_freq_h:, -(cutoff_freq_w-1):]
        else:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            bottom_left = input[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            bottom_right = input[:, :, -cutoff_freq_h:, -cutoff_freq_w:]

    if _torch:
        top_combined = torch.cat((top_left, top_right), dim=-1)
        bottom_combined = torch.cat((bottom_left, bottom_right), dim=-1)
        all_together = torch.cat((top_combined, bottom_combined), dim=-2)
    else:
        top_combined = np.concatenate((top_left, top_right), axis=-1)
        bottom_combined = np.concatenate((bottom_left, bottom_right), axis=-1)
        all_together = np.concatenate((top_combined, bottom_combined), axis=-2)

    return all_together

def _spectral_pad(input, output, oheight, owidth):
    cutoff_freq_h = math.ceil(oheight / 2)
    cutoff_freq_w = math.ceil(owidth / 2)

    pad = torch.zeros_like(input)

    if oheight % 2 == 1:
        if owidth % 2 == 1:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):] = output[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):]
            pad[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w] = output[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w]
            pad[:, :, -(cutoff_freq_h-1):, -(cutoff_freq_w-1):] = output[:, :, -(cutoff_freq_h-1):, -(cutoff_freq_w-1):]
        else:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -cutoff_freq_w:] = output[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            pad[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w] = output[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w]
            pad[:, :, -(cutoff_freq_h-1):, -cutoff_freq_w:] = output[:, :, -(cutoff_freq_h-1):, -cutoff_freq_w:]
    else:
        if owidth % 2 == 1:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):] = output[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):]
            pad[:, :, -cutoff_freq_h:, :cutoff_freq_w] = output[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            pad[:, :, -cutoff_freq_h:, -(cutoff_freq_w-1):] = output[:, :, -cutoff_freq_h:, -(cutoff_freq_w-1):]
        else:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -cutoff_freq_w:] = output[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            pad[:, :, -cutoff_freq_h:, :cutoff_freq_w] = output[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            pad[:, :, -cutoff_freq_h:, -cutoff_freq_w:] = output[:, :, -cutoff_freq_h:, -cutoff_freq_w:]	

    return pad	

def DiscreteHartleyTransform(input):

    fft = torch.fft.fft2(input)
    dht = fft.real - fft.imag
    return dht



class SpectralPoolingFunction(Function):
    @staticmethod
    def forward(ctx, input, oheight, owidth):
        ctx.oh = oheight
        ctx.ow = owidth
        ctx.save_for_backward(input)

        # Hartley transform by RFFT
        dht = DiscreteHartleyTransform(input)
        # frequency cropping
        all_together = _spectral_crop(dht, oheight, owidth)
        #all_together = dht[:, :, oheight//2:3*oheight//2, oheight//2:3*oheight//2]
        # inverse Hartley transform
        dht = DiscreteHartleyTransform(all_together)
        return dht

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables

        # Hartley transform by RFFT
        dht = DiscreteHartleyTransform(grad_output)
        # frequency padding
        grad_input = _spectral_pad(input, dht, ctx.oh, ctx.ow)
        # inverse Hartley transform
        grad_input = DiscreteHartleyTransform(grad_input)
        return grad_input, None, None

class HartleyPool2d(nn.Module):
    def __init__(self, pool_size):
        super(HartleyPool2d, self).__init__()
        self.h, self.w = _pair(pool_size)
    def forward(self, input):
        return SpectralPoolingFunction.apply(input, self.h, self.w)
    
class HybridPooling(nn.Module):
    
    def __init__(self, pool_size, n_channels, ksize=2, stride=2):
        super(HybridPooling, self).__init__()
        
        self.hartley_pool = HartleyPool2d(pool_size)
        if stride == 2:
            self.max_pool = nn.MaxPool2d(ksize, stride)
        else:
            self.max_pool = nn.MaxPool2d(ksize, stride, 1, 2)
            
        self.conv_k1 = nn.Conv2d(n_channels*2, n_channels, kernel_size=1)
            
    def forward(self, x):
    
        spectral_pool = self.hartley_pool(x)
        max_pool = self.max_pool(x)
    
        concatenated = torch.cat([spectral_pool, max_pool], axis=1)
        out = self.conv_k1(concatenated)
        return out


if __name__ == '__main__':
    
    pass

