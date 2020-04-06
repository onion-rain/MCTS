import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from quantification.binary import *

__all__ = ['resnet20_binary', 'resnet32_binary', 'resnet44_binary', 'resnet56_binary', 'resnet110_binary']

# def conv7x7(in_channels, out_channels, stride=1):
#     """7x7 convolution with padding"""
#     return BinarizeConv2d(in_channels, out_channels, kernel_size=7, stride=stride, bias=False)

# def conv3x3(in_channels, out_channels, stride=1, groups=1, padding=1):
#     """3x3 convolution with padding"""
#     return BinarizeConv2d(in_channels, out_channels, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False)

# def conv1x1(in_channels, out_channels, stride=1):
#     """1x1 convolution"""
#     return BinarizeConv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

# def activation(input):
#     return F.hardtanh(input, inplace=True)

def conv7x7(in_channels, out_channels, stride=1):
    """7x7 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, bias=False)

def conv3x3(in_channels, out_channels, stride=1, groups=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=1)

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

def activation(input):
    return F.relu(input, inplace=True)

class first_conv(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(first_conv, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.norm1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = activation(self.norm1(out))
        return out


class Basicneck(nn.Module):
    
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(Basicneck, self).__init__()

        self.conv1 = conv1x1(in_channels, mid_channels, stride=1)
        self.norm1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = conv3x3(mid_channels, out_channels, stride=stride)
        self.norm2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.conv_shortcut = conv1x1(in_channels, out_channels, stride=stride)

    def forward(self, x):
        # shortcut
        shortcut = x
        if hasattr(self, 'conv_shortcut'):
            shortcut = self.conv_shortcut(shortcut)

        # conv1
        residual = self.conv1(x)
        residual = activation(self.norm1(residual))

        # conv2
        residual = self.conv2(residual)

        out = activation(self.norm2(residual + shortcut))
        return out


class Bottleneck(nn.Module):
    
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(in_channels, mid_channels, stride=1)
        self.norm1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = conv3x3(mid_channels, mid_channels, stride=stride)
        self.norm2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = conv1x1(mid_channels, out_channels, stride=1)
        self.norm3 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.conv_shortcut = conv1x1(in_channels, out_channels, stride=stride)

    def forward(self, x):
        # shortcut
        shortcut = x
        if hasattr(self, 'conv_shortcut'):
            shortcut = self.shortcut(shortcut)

        # conv1
        residual = self.conv1(x)
        residual = activation(self.norm1(residual))

        # conv2
        residual = self.conv2(residual)
        residual = activation(self.norm2(residual))

        # conv3
        residual = self.conv3(residual)

        out = activation(self.norm3(residual + shortcut))
        return out


class ResNet_cifar_binary(nn.Module):

    def __init__(self, block, stage_repeat=[3, 3, 3], num_classes=1000):
        """
        args:
            stage_repeat(list)：每个stage重复的block数
            num_classes(int)：分类数
            gene(list): 存储每层卷积的输入输出通道，用来构造剪之后的网络，前部表示output_scale_ids，后部表示mid_scale_ids
                        若gene为None则构造原始resnet
        """
        super(ResNet_cifar_binary, self).__init__()

        self.num_classes = num_classes
        self.stage_repeat = stage_repeat

        # stage_channels = [64, 128, 256, 512, 2048] # 原始每层stage的输出通道数
        stage_channels = [16, 64, 128, 256] # 原始每层stage的输出通道数
        assert len(stage_channels)-1 == len(stage_repeat)
        
        output_channels = [stage_channels[0]]
        for i in range(1, len(stage_channels)):
            output_channels += [stage_channels[i],]*stage_repeat[i-1]
        
        self.features = nn.ModuleList()

        self.features.append(first_conv(3, output_channels[0], stride=1))

        mid_channels = []
        for i in range(1, len(stage_channels)):
            mid_channels += [int(stage_channels[i]/4),]*stage_repeat[i-1]

        block_num = 1
        for stage in range(len(stage_repeat)):
            if stage == 0:
                self.features.append(block(output_channels[block_num-1], mid_channels[block_num-1], output_channels[block_num], stride=1))
                block_num += 1
            else:
                self.features.append(block(output_channels[block_num-1], mid_channels[block_num-1], output_channels[block_num], stride=2))
                block_num += 1

            for i in range(1, stage_repeat[stage]):
                self.features.append(block(output_channels[block_num-1], mid_channels[block_num-1], output_channels[block_num], stride=1))
                block_num +=1

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 输出尺寸为1*1
        self.classifier = nn.Linear(stage_channels[-1], num_classes)


    def forward(self, x):

        for idx, block in enumerate(self.features):
            x = block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        
def resnet20_binary(cfg=None, num_classes=10):
    return ResNet_cifar_binary(Basicneck, [3, 3, 3], num_classes)

def resnet32_binary(cfg=None, num_classes=10):
    return ResNet_cifar_binary(Basicneck, [5, 5, 5], num_classes)

def resnet44_binary(gene=None, num_classes=10):
    return ResNet_cifar_binary(Basicneck, [7, 7, 7], num_classes)

def resnet56_binary(gene=None, num_classes=10):
    return ResNet_cifar_binary(Bottleneck, [6, 6, 6], num_classes)

def resnet110_binary(gene=None, num_classes=10):
    return ResNet_cifar_binary(Bottleneck, [12, 12, 12], num_classes)


