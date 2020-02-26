import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']



class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
  
    def __init__(self, in_channels, out_channels, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.norm2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                采用0填充来拟补深度
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        input=x[:, :, ::2, ::2], 
                        pad=(
                            0, 0, # 最后一维
                            0, 0, # 倒数第二维
                            out_channels//4, out_channels//4 # 倒数第三维
                        ), 
                        mode="constant", 
                        value=0
                    )
                )
            elif option == 'B':
                """
                用一个1*1的卷积核缩小特征尺寸
                """
                self.shortcut = nn.Sequential()
                self.shortcut.add_module('shortcut_conv', conv1x1(in_channels, self.expansion * out_channels, stride=stride))
                self.shortcut.add_module('shortcut_norm', nn.BatchNorm2d(self.expansion * out_channels))

    def forward(self, x):
        shortcut = self.shortcut(x)

        bottleneck = self.conv1(x)
        bottleneck = self.norm1(out)
        bottleneck = self.relu1(out)

        bottleneck = self.conv2(out)
        bottleneck = self.norm2(out)

        out = bottleneck + shortcut
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()

        self.in_channels = 16 # 初始化stage_1的in_channels

        self.conv1 = conv3x3(3, 16, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.stage_1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.stage_2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.stage_3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 输出尺寸为1*1
        self.classifier = nn.Linear(64, num_classes)

        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion # 更新in_channels
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) # 32x32(cifar)

        x = self.stage_1(x) # 32x32(cifar)
        x = self.stage_2(x) # 16x16(cifar)
        x = self.stage_3(x) # 8x8(cifar)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def resnet20(**kwargs):
    return ResNet(BasicBlock, [3, 3, 3], **kwargs)


def resnet32(**kwargs):
    return ResNet(BasicBlock, [5, 5, 5], **kwargs)


def resnet44(**kwargs):
    return ResNet(BasicBlock, [7, 7, 7], **kwargs)


def resnet56(**kwargs):
    return ResNet(BasicBlock, [9, 9, 9], **kwargs)


def resnet110(**kwargs):
    return ResNet(BasicBlock, [18, 18, 18], **kwargs)


def resnet1202(**kwargs):
    return ResNet(BasicBlock, [200, 200, 200], **kwargs)
