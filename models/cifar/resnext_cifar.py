import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['resnext29_2x64d', 'resnext29_4x64d', 'resnext29_8x64d', 'resnext29_32x4d', 'resnext29_8x4d', 
            'resnext29_8x16d']


class Bottleneck(nn.Module):
    '''Grouped convolution block.'''

    def __init__(self, in_channels, cardinality=32, bottleneck_width=4, expansion=2, stride=1):
        super(Bottleneck, self).__init__()
        group_width = cardinality * bottleneck_width
        self.expansion = expansion

        self.conv_reduce = nn.Conv2d(in_channels, group_width, kernel_size=1, bias=False)
        self.norm_reduce = nn.BatchNorm2d(group_width)
        self.relu_reduce = nn.ReLU(inplace=True)

        self.conv_conv = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.norm_conv = nn.BatchNorm2d(group_width)
        self.relu_conv = nn.ReLU(inplace=True)

        self.conv_expand = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
        self.norm_expand = nn.BatchNorm2d(self.expansion*group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*group_width:
            self.shortcut.add_module('shortcut_conv', nn.Conv2d(in_channels, self.expansion*group_width, kernel_size=1, stride=stride, bias=False))
            self.shortcut.add_module('shortcut_norm', nn.BatchNorm2d(self.expansion*group_width))

        

    def forward(self, x):
        shortcut = self.shortcut(x)

        bottleneck = self.conv_reduce(x)
        bottleneck = self.norm_reduce(bottleneck)
        bottleneck = self.relu_reduce(bottleneck)

        bottleneck = self.conv_conv(bottleneck)
        bottleneck = self.norm_conv(bottleneck)
        bottleneck = self.relu_conv(bottleneck)

        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.norm_expand(bottleneck)

        out = shortcut + bottleneck
        out = F.relu(out, inplace=True)
        return out


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, expansion, num_classes=10):
        """ Constructor
        Args:
            num_blocks: 每层的bottleneck构造块数目
            cardinality: 每个bottleneck构造块中分组卷积分组数
            bottleneck_width: 每个bottleneck构造块中分组卷积每组的3x3卷积的输出通道数
            expansion: 每个bottleneck构造块中分组卷积每组的conv_expand通道扩张倍数
            num_classes: 最后全连接层的输出通道数
        """
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.num_classes = num_classes
        self.in_channels = 64

        # First convolution
        self.conv_0 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.norm_0 = nn.BatchNorm2d(64)
        self.relu_0 = nn.ReLU(inplace=True)

        # Each bottleneck block
        self.stage_1 = self._make_layer(Bottleneck, num_blocks[0], expansion, stride=1)
        self.stage_2 = self._make_layer(Bottleneck, num_blocks[1], expansion, stride=2)
        self.stage_3 = self._make_layer(Bottleneck, num_blocks[2], expansion, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 输出尺寸为1*1
        self.classifier = nn.Linear(cardinality*bottleneck_width*expansion*4, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, num_blocks, expansion, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            blocks.append(block(self.in_channels, self.cardinality, self.bottleneck_width, expansion, stride))
            self.in_channels = expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2 # 每层bottleneck_width加倍，每层out_channels=bottleneck_width*expansion*cardinality因此也加倍
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.norm_0(x)
        x = self.relu_0(x) # 32x32(cifar)

        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x) # 8x8(cifar)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def resnext29_2x64d(**kwargs):
    return ResNeXt(num_blocks=[3,3,3], cardinality=2, bottleneck_width=64, expansion=2, **kwargs)

def resnext29_4x64d(**kwargs):
    return ResNeXt(num_blocks=[3,3,3], cardinality=4, bottleneck_width=64, expansion=2, **kwargs)

def resnext29_8x64d(**kwargs):
    return ResNeXt(num_blocks=[3,3,3], cardinality=8, bottleneck_width=64, expansion=2, **kwargs)

def resnext29_32x4d(**kwargs):
    return ResNeXt(num_blocks=[3,3,3], cardinality=32, bottleneck_width=4, expansion=2, **kwargs)

def resnext29_8x4d(**kwargs):
    return ResNeXt(num_blocks=[3,3,3], cardinality=8, bottleneck_width=4, expansion=2, **kwargs)






def resnext29_8x16d(**kwargs):
    return ResNeXt(num_blocks=[3,3,3], cardinality=8, bottleneck_width=16, expansion=2, **kwargs)