import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import channel_selection, shortcut_package

# cs means channel select
# 用于cifar数据集stage = 3


__all__ = ['resnet20_cs', 'resnet32_cs', 'resnet44_cs', 'resnet56_cs', 'resnet110_cs', 'resnet1202_cs']


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class Basicblock(nn.Module):
    def __init__(self, in_channels, out_channels, cfg, stride=1):
        # 此处in_channels不能用cfg[0]代替，因为此处与其他block相接，不能直接删除通道，要用channel_selection来选择通道
        super(Basicblock, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.select = channel_selection(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(cfg[0], cfg[1], stride=1)
        
        self.norm2 = nn.BatchNorm2d(cfg[1])
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg[1], out_channels, stride=stride)


        if stride != 1 or in_channels != out_channels:
            """用一个1*1的卷积核缩小特征尺寸"""
            self.shortcut = shortcut_package(conv1x1(in_channels, out_channels, stride=stride))
        else:
            self.shortcut = shortcut_package(nn.Sequential())

    def forward(self, x):
        shortcut = self.shortcut(x)

        residual = self.norm1(x)
        residual = self.select(residual)
        residual = self.relu1(residual)
        residual = self.conv1(residual)
        
        residual = self.norm2(residual)
        residual = self.relu2(residual)
        residual = self.conv2(residual)

        out = residual + shortcut
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, cfg, stride=1):
        # 此处in_channels, out_channels不能用cfg[0],cfg[3]代替，因为此处与其他block相接，不能直接删除通道，
        # in_channels要用channel_selection来选择通道,out_channels要与原网络相同
        super(Bottleneck, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.select = channel_selection(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(cfg[0], cfg[1], stride=1)
        
        self.norm2 = nn.BatchNorm2d(cfg[1])
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg[1], cfg[2], stride=stride)

        self.norm3 = nn.BatchNorm2d(cfg[2])
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(cfg[2], out_channels, stride=1)


        if stride != 1 or in_channels != out_channels:
            """用一个1*1的卷积核缩小特征尺寸"""
            self.shortcut = shortcut_package(conv1x1(in_channels, out_channels, stride=stride))
        else:
            self.shortcut = shortcut_package(nn.Sequential())

    def forward(self, x):
        shortcut = self.shortcut(x)

        residual = self.norm1(x)
        residual = self.select(residual)
        residual = self.relu1(residual)
        residual = self.conv1(residual)
        
        residual = self.norm2(residual)
        residual = self.relu2(residual)
        residual = self.conv2(residual)

        residual = self.norm3(residual)
        residual = self.relu3(residual)
        residual = self.conv3(residual)

        out = residual + shortcut
        return out


class ResNet_cs(nn.Module):

    def __init__(self, block, depth, cfg=None, num_classes=10):
        """
        args:
            block: 构造块，Bottleneck或者Basicblock
            depth: 深度，conv层数
            cfg：每个bn层的通道数
            num_classes：分类数
        """
        super(ResNet_cs, self).__init__()
        self.num_classes = num_classes
        if block is Bottleneck:
            assert (depth - 2) % 9 == 0, 'depth should be 9n+2'
            n = (depth - 2) // 9 # 每个stage里block数
            self.nconv = 3 # 每个block conv层数
            self.cfg_original = [[16, 16, 16], [64, 16, 16]*(n-1), [64, 32, 32], [128, 32, 32]*(n-1), [128, 64, 64], [256, 64, 64]*(n-1), [256]]
        elif block is Basicblock:
            assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
            n = (depth - 2) // 6 # 每个stage里block数
            self.nconv = 2 # 每个block conv层数
            self.cfg_original = [[16, 16], [64, 16]*(n-1), [64, 32], [128, 32]*(n-1), [128, 64], [256, 64]*(n-1), [256]]
        else:
            print("undefined block")
            exit(1)

        self.block_count = 0
        if cfg is None:
            cfg = self.cfg_original
            cfg = [item for sub_list in cfg for item in sub_list]
        # 展平
        self.cfg_original = [item for sub_list in self.cfg_original for item in sub_list]

        self.conv1 = conv3x3(3, 16, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.stage_1 = self._make_layer(block, n, cfg=cfg[0:self.nconv*n], stride=1)
        self.stage_2 = self._make_layer(block, n, cfg=cfg[self.nconv*n:2*self.nconv*n], stride=2)
        self.stage_3 = self._make_layer(block, n, cfg=cfg[2*self.nconv*n:3*self.nconv*n], stride=2)
        self.norm2 = nn.BatchNorm2d(self.cfg_original[-1])
        self.select = channel_selection(self.cfg_original[-1])
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1)) # 输出尺寸为1*1

        self.classifier = nn.Linear(cfg[-1], num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, num_blocks, cfg, stride):
        layers = []
        layers.append(block(self.cfg_original[self.block_count], 
                            self.cfg_original[self.block_count+self.nconv],
                            cfg=cfg[0:3], 
                            stride=stride)) # 每个stage只有第一个block的stride不为1来改变特征图大小
        self.block_count += self.nconv
        for i in range(1, num_blocks-1):
            layers.append(block(self.cfg_original[self.block_count], 
                                self.cfg_original[self.block_count+self.nconv],
                                cfg=cfg[self.nconv*i : self.nconv*(i+1)], 
                                stride=1))
            self.block_count += self.nconv
        layers.append(block(self.cfg_original[self.block_count], 
                            self.cfg_original[self.block_count+self.nconv],
                            cfg=cfg[self.nconv*(num_blocks-1):], 
                            stride=1))
        self.block_count += self.nconv
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x) # 32x32(cifar)

        x = self.stage_1(x) # 32x32(cifar)
        x = self.stage_2(x) # 16x16(cifar)
        x = self.stage_3(x) # 8x8(cifar)

        x = self.norm2(x)
        x = self.select(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1) # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def resnet20_cs(cfg=None, **kwargs):
    depth = 20
    return ResNet_cs(Basicblock, depth, cfg, **kwargs)

def resnet32_cs(cfg=None, **kwargs):
    depth = 32
    return ResNet_cs(Basicblock, depth, cfg, **kwargs)

def resnet44_cs(cfg=None, **kwargs):
    depth = 44
    return ResNet_cs(Basicblock, depth, cfg, **kwargs)

def resnet56_cs(cfg=None, **kwargs):
    depth = 56
    return ResNet_cs(Basicblock, depth, cfg, **kwargs)

def resnet110_cs(cfg=None, **kwargs):
    depth = 110
    return ResNet_cs(Basicblock, depth, cfg, **kwargs)

def resnet1202_cs(cfg=None, **kwargs):
    depth = 1202
    return ResNet_cs(Basicblock, depth, cfg, **kwargs)