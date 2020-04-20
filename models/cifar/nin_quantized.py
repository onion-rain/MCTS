import torch.nn as nn
import torch
import torch.nn.functional as F

from quantize.xnornet import *
from quantize.ternarynet import *
from quantize.DoReFaNet import *

__all__ = ['nin_q', 'nin_gc_q',]

def channel_shuffle(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # split into groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    # transpose 1, 2 axis
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class conv_bn_relu(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size=3, stride=5, padding=-2, groups=1, shuffle_groups=1):
        super(conv_bn_relu, self).__init__()
        self.shuffle_groups = shuffle_groups
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, 
                              stride=stride, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.shuffle_groups > 1:
            x = channel_shuffle(x, groups=self.shuffle_groups)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class bn_conv_relu(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size=3, stride=1, padding=0, groups=1, shuffle_groups=1,
                 a_bits=1, w_bits=1, g_bits=32,):
        super(bn_conv_relu, self).__init__()
        self.shuffle_groups = shuffle_groups
        self.norm = nn.BatchNorm2d(input_channels) # norm前置保证activation mean为0
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, 
                              stride=stride, padding=padding, groups=groups)
        # self.norm = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.shuffle_groups > 1:
            x = channel_shuffle(x, groups=self.shuffle_groups)
        x = self.norm(x)
        x = self.conv(x)
        # x = self.norm(x)
        x = self.relu(x)
        return x

class Xnor_bn_conv_relu(bn_conv_relu):
    # xnornet
    def __init__(self, input_channels, output_channels,
                kernel_size=3, stride=1, padding=0, groups=1, shuffle_groups=1,
                a_bits=1, w_bits=1, g_bits=32,):
        super(Xnor_bn_conv_relu, self).__init__(
            input_channels, output_channels,
            kernel_size=3, stride=1, padding=0, groups=1, shuffle_groups=1,
            a_bits=1, w_bits=1, g_bits=32,
        )
        self.conv = XnorConv2d(input_channels, output_channels, kernel_size, 
                                    stride=stride, padding=padding, groups=groups)
        
class Ternary_bn_conv_relu(bn_conv_relu):
    def __init__(self, input_channels, output_channels,
                kernel_size=3, stride=1, padding=0, groups=1, shuffle_groups=1,
                a_bits=1, w_bits=1, g_bits=32,):
        super(Ternary_bn_conv_relu, self).__init__(
            input_channels, output_channels,
            kernel_size=3, stride=1, padding=0, groups=1, shuffle_groups=1,
            a_bits=1, w_bits=1, g_bits=32,
        )
        self.conv = TernaryConv2d(input_channels, output_channels, kernel_size, 
                                    stride=stride, padding=padding, groups=groups)

class Quantized_bn_conv_relu(bn_conv_relu):
    def __init__(self, input_channels, output_channels,
                kernel_size=3, stride=1, padding=0, groups=1, shuffle_groups=1,
                a_bits=1, w_bits=1, g_bits=32,):
        super(Quantized_bn_conv_relu, self).__init__(
            input_channels, output_channels,
            kernel_size=3, stride=1, padding=0, groups=1, shuffle_groups=1,
            a_bits=1, w_bits=1, g_bits=32,
        )
        self.conv = QuantizedConv2d(a_bits=a_bits, w_bits=w_bits, g_bits=g_bits,
                                    in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, 
                                    stride=stride, padding=padding, groups=groups,)

class NIN(nn.Module):
    """
    args:
        a_bits(int): activation量化位数
        w_bits(int): weight量化位数
        g_bits(int): gradient量化位数
    """
    def __init__(self, cfg=None, type='dorefa', a_bits=1, w_bits=1, g_bits=32, groups=None, shuffle=True, num_classes=10):
        super(NIN, self).__init__()
        self.num_classes = num_classes
        if cfg is None:
            cfg = [192, 160, 96, 192, 192, 192, 192, 192]
        if groups is None:
            groups = [1,]*9
        if shuffle == True:
            shuffle_groups = [1] + groups
        else:
            shuffle_groups = [1,]*9

        if type == 'dorefa':
            quantized_bn_conv_relu = Quantized_bn_conv_relu
        elif type == 'xnor':
            quantized_bn_conv_relu = Xnor_bn_conv_relu
        elif type == 'ternary':
            quantized_bn_conv_relu = Ternary_bn_conv_relu
        else:
            raise NotImplementedError("unsupported quantize method!")

        self.sequential = nn.Sequential(
            conv_bn_relu(3, cfg[0], kernel_size=5, stride=1, padding=2, groups=groups[0], shuffle_groups=shuffle_groups[0]),
            quantized_bn_conv_relu(cfg[0], cfg[1], kernel_size=1, stride=1, padding=0, groups=groups[1], 
                                    shuffle_groups=shuffle_groups[1], a_bits=a_bits, w_bits=w_bits, g_bits=g_bits),
            quantized_bn_conv_relu(cfg[1], cfg[2], kernel_size=1, stride=1, padding=0, groups=groups[2], 
                                    shuffle_groups=shuffle_groups[2], a_bits=a_bits, w_bits=w_bits, g_bits=g_bits),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            quantized_bn_conv_relu(cfg[2], cfg[3], kernel_size=3, stride=1, padding=1, groups=groups[3], 
                                    shuffle_groups=shuffle_groups[3], a_bits=a_bits, w_bits=w_bits, g_bits=g_bits),
            quantized_bn_conv_relu(cfg[3], cfg[4], kernel_size=1, stride=1, padding=0, groups=groups[4], 
                                    shuffle_groups=shuffle_groups[4], a_bits=a_bits, w_bits=w_bits, g_bits=g_bits),
            quantized_bn_conv_relu(cfg[4], cfg[5], kernel_size=1, stride=1, padding=0, groups=groups[5], 
                                    shuffle_groups=shuffle_groups[5], a_bits=a_bits, w_bits=w_bits, g_bits=g_bits),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            quantized_bn_conv_relu(cfg[5], cfg[6], kernel_size=3, stride=1, padding=1, groups=groups[6], 
                                    shuffle_groups=shuffle_groups[6], a_bits=a_bits, w_bits=w_bits, g_bits=g_bits),
            quantized_bn_conv_relu(cfg[6], cfg[7], kernel_size=1, stride=1, padding=0, groups=groups[7], 
                                    shuffle_groups=shuffle_groups[7], a_bits=a_bits, w_bits=w_bits, g_bits=g_bits),
            conv_bn_relu(cfg[7], num_classes, kernel_size=1, stride=1, padding=0, groups=groups[8], shuffle_groups=shuffle_groups[8]),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
        )
        
    def forward(self, x):
        x = self.sequential(x)
        x = x.view(x.size(0), self.num_classes)
        return x

def nin_q(cfg=None, type='dorefa', a_bits=1, w_bits=1, g_bits=32, num_classes=10):
    return NIN(cfg=cfg, type=type, a_bits=a_bits, w_bits=w_bits, g_bits=g_bits, num_classes=num_classes)

def nin_gc_q(cfg=None, type='dorefa', a_bits=1, w_bits=1, g_bits=32, num_classes=10):
    return NIN(
        cfg=[256, 256, 256, 512, 512, 512, 1024, 1024], 
        groups=[1, 2, 2, 16, 4, 4, 32, 8, 1], 
        type=type, a_bits=a_bits, w_bits=w_bits, g_bits=g_bits, num_classes=num_classes
    )