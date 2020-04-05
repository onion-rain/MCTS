import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv7x7(in_channels, out_channels, stride=1):
    """7x7 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, bias=False)

def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class first_conv(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(first_conv, self).__init__()

        self.conv1 = conv7x7(in_channels, out_channels, stride=stride)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(F.relu(out, inplace=True))
        out = self.maxpool(out)
        return out


class Basicneck(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(Basicneck, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.norm2 = nn.BatchNorm2d(mid_channels)

        if stride != 1 or in_channels != out_channels:
            self.conv_shortcut = conv1x1(in_channels, out_channels, stride=stride)
            self.norm_shortcut = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # shortcut
        shortcut = x
        if hasattr(self, 'shortcut'):
            shortcut = self.conv_shortcut(shortcut)
            shortcut = self.norm_shortcut(shortcut)

        # conv1
        residual = self.conv1(x)
        residual = F.hardtanh(self.norm1(residual, inplace=True))

        # conv2
        residual = self.conv2(residual)
        residual = F.hardtanh(self.norm2(residual, inplace=True))

        out = residual + shortcut
        out = F.relu(out, inplace=True)
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
            self.shortcut = conv1x1(in_channels, out_channels, stride=stride)
            self.norm_shortcut = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # shortcut
        shortcut = x
        if hasattr(self, 'shortcut'):
            shortcut = self.shortcut(shortcut)
            shortcut = self.norm_shortcut(shortcut)

        # conv1
        residual = self.conv1(x)
        residual = self.norm1(F.relu(residual, inplace=True))

        # conv2
        residual = self.conv2(residual)
        residual = self.norm2(F.relu(residual, inplace=True))

        # conv3
        residual = self.conv3(residual)
        residual = self.norm3(residual)

        out = residual + shortcut
        out = F.relu(out, inplace=True)
        return out


class ResNet_cifar_binary(nn.Module):

    def __init__(self, stage_repeat=[3, 4, 6, 3], num_classes=1000):
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

        stage_channels = [64, 128, 256, 512, 2048] # 原始每层stage的输出通道数
        
        output_channels = [stage_channels[0]]
        for i in range(1, len(stage_channels)):
            output_channels += [stage_channels[i],]*stage_repeat[i-1]
        mid_channels = []
        for i in range(1, len(stage_channels)):
            mid_channels += [int(stage_channels[i]/4),]*stage_repeat[i-1]

        self.features = nn.ModuleList()

        # first conv(stage0)
        self.features.append(first_conv(3, output_channels[0], stride=2))

        block_num = 1
        for stage in range(len(stage_repeat)):
            if stage == 0:
                self.features.append(Bottleneck(output_channels[block_num-1], mid_channels[block_num-1], output_channels[block_num], stride=1))
                block_num += 1
            else:
                self.features.append(Bottleneck(output_channels[block_num-1], mid_channels[block_num-1], output_channels[block_num], stride=2))
                block_num += 1

            for i in range(1, stage_repeat[stage]):
                self.features.append(Bottleneck(output_channels[block_num-1], mid_channels[block_num-1], output_channels[block_num], stride=1))
                block_num +=1

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 输出尺寸为1*1
        self.classifier = nn.Linear(stage_channels[4], num_classes)


    def forward(self, x):

        for idx, block in enumerate(self.features):
            x = block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        
# 这些应该是basicblock。。懒得写了
# def resnet18_prunednet(cfg=None, num_classes=1000):
#     return ResNet_Prunednet([2, 2, 2, 2], num_classes, gene=cfg)


# def resnet34_prunednet(cfg=None, num_classes=1000):
#     return ResNet_Prunednet([3, 4, 6, 3], num_classes, gene=cfg)


def resnet50_prunednet(gene=None, num_classes=1000):
    return ResNet_Prunednet([3, 4, 6, 3], num_classes, gene=gene)


def resnet101_prunednet(gene=None, num_classes=1000):
    return ResNet_Prunednet([3, 4, 23, 3], num_classes, gene=gene)


def resnet152_prunednet(gene=None, num_classes=1000):
    return ResNet_Prunednet([3, 8, 36, 3], num_classes, gene=gene)

