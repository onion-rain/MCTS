import torch.nn as nn
import torch
import torch.nn.functional as F

from quantification.xnornet import *

__all__ = ['nin_xnornet', 'nin_gc_xnornet']

class conv_bn_relu(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size=3, stride=5, padding=-2, groups=1):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, 
                              stride=stride, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class Binary_conv_bn_relu(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size=3, stride=1, padding=0, groups=1):
        super(Binary_conv_bn_relu, self).__init__()
        self.norm = nn.BatchNorm2d(input_channels)
        self.active = BinaryActive()
        self.conv = BinarizeConv2d(input_channels, output_channels, kernel_size, 
                                    stride=stride, padding=padding, groups=groups)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.active(x)
        x = self.conv(x)
        x = self.relu(x)
        return x

class NIN(nn.Module):
    def __init__(self, cfg=None, groups=None, num_classes=10):
        super(NIN, self).__init__()
        self.num_classes = num_classes
        if cfg is None:
            cfg = [192, 160, 96, 192, 192, 192, 192, 192]
        if groups is None:
            groups = [1,]*9
        
        self.sequential = nn.Sequential(
            conv_bn_relu(3, cfg[0], kernel_size=5, stride=1, padding=2, groups=groups[0]),
            Binary_conv_bn_relu(cfg[0], cfg[1], kernel_size=1, stride=1, padding=0, groups=groups[1]),
            Binary_conv_bn_relu(cfg[1], cfg[2], kernel_size=1, stride=1, padding=0, groups=groups[2]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            Binary_conv_bn_relu(cfg[2], cfg[3], kernel_size=5, stride=1, padding=2, groups=groups[3]),
            Binary_conv_bn_relu(cfg[3], cfg[4], kernel_size=1, stride=1, padding=0, groups=groups[4]),
            Binary_conv_bn_relu(cfg[4], cfg[5], kernel_size=1, stride=1, padding=0, groups=groups[5]),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            Binary_conv_bn_relu(cfg[5], cfg[6], kernel_size=3, stride=1, padding=1, groups=groups[6]),
            Binary_conv_bn_relu(cfg[6], cfg[7], kernel_size=1, stride=1, padding=0, groups=groups[7]),
            conv_bn_relu(cfg[7], num_classes, kernel_size=1, stride=1, padding=0, groups=groups[8]),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
        )
        
    def forward(self, x):
        x = self.sequential(x)
        x = x.view(x.size(0), self.num_classes)
        return x

def nin_xnornet(cfg=None, num_classes=10):
    return NIN(cfg=cfg, num_classes=num_classes)

def nin_gc_xnornet(cfg=None, num_classes=10):
    return NIN(cfg=[256, 256, 256, 512, 512, 512, 1024, 1024], groups=[1, 2, 2, 16, 4, 4, 32, 8, 1], num_classes=num_classes)