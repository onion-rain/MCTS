import torch
import torch.nn as nn
import torch.nn.functional as F

from . import shortcut_package

# 用于cifar数据集stage = 3

__all__ = ['resnet18_meta', 'resnet34_meta', 'resnet50_meta', 'resnet101_meta', 'resnet152_meta',]


channel_scale = [] # 随机压缩率生成池
for i in range(31):
    channel_scale += [(10 + i * 3)/100]

def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class first_conv(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc11 = nn.Linear(1, 32)
        self.fc12 = nn.Linear(32, self.out_channels * self.in_channels * 7 * 7)
        # self.conv1 = conv1x1(self.in_channels, self.out_channels, stride=stride)
        self.norm1 = nn.BatchNorm2d(out_channels)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x, scale_cfg):

        in_channels = self.in_channels
        out_channels = int(self.out_channels * scale_cfg)
        block_cfg = torch.FloatTensor([in_channels, out_channels]).to(x.device)

        # conv1
        conv1_param = F.relu(self.fc11(block_cfg))
        conv1_param = F.relu(self.fc12(conv1_param))
        conv1_param = conv1_param.view_as(self.out_channels, self.in_channels, 1, 1)
        out = F.conv2d(x, conv1_param[:out_channels, :in_channels, :, :], 
                            bias=None, stride=self.stride, padding=1, groups=1)
        out = self.norm1(F.relu(out))
        
        out = self.maxpool(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()

        self.in_channels = in_channels
        self.mid_channels = int(out_channels/self.expansion)
        self.out_channels = out_channels

        self.fc11 = nn.Linear(3, 32)
        self.fc12 = nn.Linear(32, self.mid_channels * self.in_channels * 1 * 1)
        # self.conv1 = conv1x1(self.in_channels, self.mid_channels, stride=1)
        self.norm1 = nn.BatchNorm2d(mid_channels)

        self.fc21 = nn.Linear(3, 32)
        self.fc22 = nn.Linear(32, self.mid_channels * self.mid_channels * 3 * 3)
        # self.conv2 = conv3x3(self.mid_channels, self.mid_channels, stride=stride)
        self.norm2 = nn.BatchNorm2d(self.mid_channels)

        self.fc31 = nn.Linear(3, 32)
        self.fc32 = nn.Linear(32, self.out_channels * self.mid_channels * 1 * 1)
        # self.conv3 = conv1x1(self.mid_channels, self.out_channels, stride=1)
        self.norm3 = nn.BatchNorm2d(self.out_channels)


        if stride != 1 or in_channels != out_channels:
            """用一个1*1的卷积核缩小特征尺寸"""
            self.shortcut = shortcut_package(conv1x1(in_channels, out_channels, stride=stride))
        else:
            self.shortcut = shortcut_package(nn.Sequential())

    def forward(self, x, scale_cfg):
        # shortcut
        shortcut = self.shortcut(x)

        in_channels = int(self.in_channels * scale_cfg[0])
        mid_channels = int(self.mid_channels * scale_cfg[1])
        out_channels = int(self.out_channels * scale_cfg[2])
        block_cfg = torch.FloatTensor([in_channels, mid_channels, out_channels]).to(x.device)

        # conv1
        conv1_param = F.relu(self.fc11(block_cfg))
        conv1_param = F.relu(self.fc12(conv1_param))
        conv1_param = conv1_param.view_as(self.mid_channels, self.in_channels, 1, 1)
        residual = F.conv2d(x, conv1_param[:mid_channels, :in_channels, :, :], 
                            bias=None, stride=1, padding=1, groups=1)
        residual = self.norm1(F.relu(residual))

        # conv2
        conv2_param = F.relu(self.fc21(block_cfg))
        conv2_param = F.relu(self.fc22(conv2_param))
        conv2_param = conv2_param.view_as(self.mid_channels, self.mid_channels, 3, 3)
        residual = F.conv2d(residual, conv2_param[:mid_channels, :mid_channels, :, :], 
                            bias=None, stride=self.stride, padding=1, groups=1)
        residual = self.norm2(F.relu(residual))

        # conv3
        conv3_param = F.relu(self.fc31(block_cfg))
        conv3_param = F.relu(self.fc22(conv3_param))
        conv3_param = conv3_param.view_as(self.out_channels, self.mid_channels, 1, 1)
        residual = F.conv2d(residual, conv3_param[:out_channels, :mid_channels, :, :], 
                            bias=None, stride=1, padding=1, groups=1)
        residual = self.norm3(F.relu(residual))

        out = residual + shortcut
        out = F.relu(out)
        return out


class ResNet_meta(nn.Module):

    def __init__(self, cfg=[,], num_classes=10):
        """
        args:
            cfg(list)：每个stage重复的block数
            num_classes(int)：分类数
        """
        super(ResNet_meta, self).__init__()
        self.num_classes = num_classes
        stage_channels = [64, 128, 256, 512, 2048] # 每层stage的输出通道数
        
        self.features = nn.ModuleList()

        # first conv
        self.features.append(first_conv(3, stage_channels[0], stride=2))

        # stage1
        self.features.append(Bottleneck(stage_channels[0], stage_channels[1], stride=1, is_downsample=True))
        for i in range(1, stage_repeat[0]):
            self.features.append(Bottleneck(stage_channels[1], stage_channels[1]))

        #stage2
        self.features.append(Bottleneck(stage_channels[1], stage_channels[2], stride=2, is_downsample=True))
        for i in range(1, stage_repeat[1]):
            self.features.append(Bottleneck(stage_channels[2], stage_channels[2]))

        #stage3
        self.features.append(Bottleneck(stage_channels[2], stage_channels[3], stride=2, is_downsample=True))
        for i in range(1, stage_repeat[2]):
            self.features.append(Bottleneck(stage_channels[3], stage_channels[3]))

        #stage4
        self.features.append(Bottleneck(stage_channels[3], stage_channels[4], stride=2, is_downsample=True))
        for i in range(1, stage_repeat[3]):
            self.features.append(Bottleneck(stage_channels[4], stage_channels[4]))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 输出尺寸为1*1
        self.classifier = nn.Linear(stage_channels[4], num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



def resnet18_meta(**kwargs):
    return ResNet_meta([2, 2, 2, 2], **kwargs)


def resnet34_meta(**kwargs):
    return ResNet_meta([3, 4, 6, 3], **kwargs)


def resnet50_meta(**kwargs):
    return ResNet_meta([3, 4, 6, 3], **kwargs)


def resnet101_meta(**kwargs):
    return ResNet_meta([3, 4, 23, 3], **kwargs)


def resnet152_meta(**kwargs):
    return ResNet_meta([3, 8, 36, 3], **kwargs)