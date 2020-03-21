import torch
import torch.nn as nn
import torch.nn.functional as F


# 用于cifar数据集stage = 3

__all__ = ['ResNet_meta', 'resnet18_meta', 'resnet34_meta', 'resnet50_meta', 'resnet101_meta', 'resnet152_meta',]



def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class first_conv(nn.Module):
    def __init__(self, in_channels, out_channels, channel_scales, stride=1):
        super(first_conv, self).__init__()

        self.channel_scales = channel_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.fc11 = nn.Linear(1, 32)
        self.fc12 = nn.Linear(32, self.out_channels * self.in_channels * 7 * 7)
        # self.conv1 = conv1x1(self.in_channels, self.out_channels, stride=stride)
        # self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm1 = nn.ModuleList()
        for scale in channel_scales: # 所有可能的通道数都造一个bn层
            self.norm1.append(nn.BatchNorm2d(int(self.out_channels*scale), affine=False))


        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x, scale_id):
        
        # self.in_channels为原始通道数，in_channels为pruned通道数
        in_channels = self.in_channels
        out_channels = int(self.out_channels * self.channel_scales[scale_id])
        block_cfg = torch.FloatTensor([out_channels]).to(x.device)

        # conv1
        conv1_param = F.relu(self.fc11(block_cfg))
        conv1_param = F.relu(self.fc12(conv1_param))
        conv1_param = conv1_param.view(self.out_channels, self.in_channels, 7, 7)
        out = F.conv2d(x, conv1_param[:out_channels, :in_channels, :, :], 
                            bias=None, stride=self.stride, padding=1, groups=1)
        out = self.norm1[scale_id](F.relu(out))
        
        out = self.maxpool(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, channel_scales, stride=1):
        super(Bottleneck, self).__init__()

        self.channel_scales = channel_scales
        self.in_channels = in_channels
        self.mid_channels = int(out_channels/self.expansion)
        self.out_channels = out_channels
        self.stride = stride

        self.fc11 = nn.Linear(3, 32)
        self.fc12 = nn.Linear(32, self.mid_channels * self.in_channels * 1 * 1)
        # self.conv1 = conv1x1(self.in_channels, self.mid_channels, stride=1)
        # self.norm1 = nn.BatchNorm2d(self.mid_channels)
        self.norm1 = nn.ModuleList()
        for scale in channel_scales: # 所有可能的通道数都造一个bn层
            self.norm1.append(nn.BatchNorm2d(int(self.mid_channels*scale), affine=False))

        self.fc21 = nn.Linear(3, 32)
        self.fc22 = nn.Linear(32, self.mid_channels * self.mid_channels * 3 * 3)
        # self.conv2 = conv3x3(self.mid_channels, self.mid_channels, stride=stride)
        # self.norm2 = nn.BatchNorm2d(self.mid_channels)
        self.norm2 = nn.ModuleList()
        for scale in channel_scales: # 所有可能的通道数都造一个bn层
            self.norm2.append(nn.BatchNorm2d(int(self.mid_channels*scale), affine=False))

        self.fc31 = nn.Linear(3, 32)
        self.fc32 = nn.Linear(32, self.out_channels * self.mid_channels * 1 * 1)
        # self.conv3 = conv1x1(self.mid_channels, self.out_channels, stride=1)
        # self.norm3 = nn.BatchNorm2d(self.out_channels)
        self.norm3 = nn.ModuleList()
        for scale in channel_scales: # 所有可能的通道数都造一个bn层
            self.norm3.append(nn.BatchNorm2d(int(self.out_channels*scale), affine=False))

        self.fc_shortcut1 = nn.Linear(3, 32)
        self.fc_shortcut2 = nn.Linear(32, self.out_channels * self.in_channels * 1 * 1)
        # self.shortcut = conv1x1(in_channels, out_channels, stride=stride)
        # self.norm_shortcut = nn.BatchNorm2d(self.out_channels)
        self.norm_shortcut = nn.ModuleList()
        for scale in channel_scales: # 所有可能的通道数都造一个bn层
            self.norm_shortcut.append(nn.BatchNorm2d(int(self.out_channels*scale), affine=False))

    def forward(self, x, scale_ids):

        in_channels = int(self.in_channels * self.channel_scales[scale_ids[0]])
        mid_channels = int(self.mid_channels * self.channel_scales[scale_ids[1]])
        out_channels = int(self.out_channels * self.channel_scales[scale_ids[2]])
        block_cfg = torch.FloatTensor([in_channels, mid_channels, out_channels]).to(x.device)

        # shortcut
        shortcut = x
        if in_channels != out_channels or self.stride !=1:
            conv_shortcut_param = F.relu(self.fc_shortcut1(block_cfg))
            conv_shortcut_param = F.relu(self.fc_shortcut2(conv_shortcut_param))
            conv_shortcut_param = conv_shortcut_param.view(self.out_channels, self.in_channels, 1, 1)
            shortcut = F.conv2d(shortcut, conv_shortcut_param[:out_channels, :in_channels, :, :], 
                                bias=None, stride=self.stride, padding=0, groups=1)
            shortcut = self.norm_shortcut[scale_ids[2]](F.relu(shortcut))

        # conv1
        conv1_param = F.relu(self.fc11(block_cfg))
        conv1_param = F.relu(self.fc12(conv1_param))
        conv1_param = conv1_param.view(self.mid_channels, self.in_channels, 1, 1)
        residual = F.conv2d(x, conv1_param[:mid_channels, :in_channels, :, :], 
                            bias=None, stride=1, padding=0, groups=1)
        residual = self.norm1[scale_ids[1]](F.relu(residual))

        # conv2
        conv2_param = F.relu(self.fc21(block_cfg))
        conv2_param = F.relu(self.fc22(conv2_param))
        conv2_param = conv2_param.view(self.mid_channels, self.mid_channels, 3, 3)
        residual = F.conv2d(residual, conv2_param[:mid_channels, :mid_channels, :, :], 
                            bias=None, stride=self.stride, padding=1, groups=1)
        residual = self.norm2[scale_ids[1]](F.relu(residual))

        # conv3
        conv3_param = F.relu(self.fc31(block_cfg))
        conv3_param = F.relu(self.fc32(conv3_param))
        conv3_param = conv3_param.view(self.out_channels, self.mid_channels, 1, 1)
        residual = F.conv2d(residual, conv3_param[:out_channels, :mid_channels, :, :], 
                            bias=None, stride=1, padding=0, groups=1)
        residual = self.norm3[scale_ids[2]](F.relu(residual))

        out = residual + shortcut
        out = F.relu(out)
        return out


class ResNet_meta(nn.Module):

    def __init__(self, stage_repeat=[2, 2, 2, 2], num_classes=1000, cfg=None):
        """
        args:
            stage_repeat(list)：每个stage重复的block数
            num_classes(int)：分类数
            TODO cfg(list): 存储每层卷积的输入输出通道，用来构造剪之后的网络
        """
        super(ResNet_meta, self).__init__()

        stage_channels = [64, 128, 256, 512, 2048] # 每层stage的输出通道数
        self.num_classes = num_classes
        self.stage_repeat = stage_repeat
        self.channel_scales = [] # 随机压缩率摇奖池
        for i in range(31):
            self.channel_scales.append((10 + i * 3)/100)

        self.features = nn.ModuleList()

        # first conv(stage0)
        self.features.append(first_conv(3, stage_channels[0], self.channel_scales, stride=2))

        # stage1
        self.features.append(Bottleneck(stage_channels[0], stage_channels[1], self.channel_scales, stride=1))
        for i in range(1, stage_repeat[0]):
            self.features.append(Bottleneck(stage_channels[1], stage_channels[1], self.channel_scales, stride=1))

        #stage2
        self.features.append(Bottleneck(stage_channels[1], stage_channels[2], self.channel_scales, stride=2))
        for i in range(1, stage_repeat[1]):
            self.features.append(Bottleneck(stage_channels[2], stage_channels[2], self.channel_scales, stride=1))

        #stage3
        self.features.append(Bottleneck(stage_channels[2], stage_channels[3], self.channel_scales, stride=2,))
        for i in range(1, stage_repeat[2]):
            self.features.append(Bottleneck(stage_channels[3], stage_channels[3], self.channel_scales, stride=1))

        #stage4
        self.features.append(Bottleneck(stage_channels[3], stage_channels[4], self.channel_scales, stride=2))
        for i in range(1, stage_repeat[3]):
            self.features.append(Bottleneck(stage_channels[4], stage_channels[4], self.channel_scales, stride=1))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 输出尺寸为1*1
        self.classifier = nn.Linear(stage_channels[4], num_classes)

        # for m in self.modules():
        #     if isinstance(m, (nn.Linear, nn.Conv2d)):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x, output_scale_ids=None, mid_scale_ids=None):
        if output_scale_ids is None:
            output_scale_ids = [-1,]*(sum(self.stage_repeat)+2)
            mid_scale_ids = [-1,]*sum(self.stage_repeat)
        for idx, block in enumerate(self.features):
            if idx == 0:
                x = block(x, output_scale_ids[idx])
            else:
                x = block(x, [output_scale_ids[idx-1], 
                              mid_scale_ids[idx-1], 
                              output_scale_ids[idx]])

        x = self.avgpool(x)
        x = torch.flatten(x, 1) # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



def resnet18_meta(cfg=None, num_classes=1000):
    return ResNet_meta([2, 2, 2, 2], num_classes, cfg=cfg)


def resnet34_meta(cfg=None, num_classes=1000):
    return ResNet_meta([3, 4, 6, 3], num_classes, cfg=cfg)


def resnet50_meta(cfg=None, num_classes=1000):
    return ResNet_meta([3, 4, 6, 3], num_classes, cfg=cfg)


def resnet101_meta(cfg=None, num_classes=1000):
    return ResNet_meta([3, 4, 23, 3], num_classes, cfg=cfg)


def resnet152_meta(cfg=None, num_classes=1000):
    return ResNet_meta([3, 8, 36, 3], num_classes, cfg=cfg)