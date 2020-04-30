import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from quantize.xnornet import *
from quantize.ternarynet import *
from quantize.DoReFaNet import *

__all__ = ['resnet20_q', 'resnet32_q', 'resnet44_q', 'resnet56_q', 'resnet110_q']

def activation(input):
    return F.relu(input, inplace=True)

class first_conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(first_conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                            padding=1, groups=1, bias=False, dilation=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        out = self.conv1(x)
        out = activation(self.norm1(out))
        return out


class Basicneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, mid_channels, out_channels, stride=1,
                                a_bits=1, w_bits=1, g_bits=32,):
        super(Basicneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride,
                            padding=1, groups=1, bias=False, dilation=1)
        self.norm1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1,
                            padding=1, groups=1, bias=False, dilation=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
    def forward(self, x):
        # shortcut
        shortcut = x
        if hasattr(self, 'conv_shortcut'):
            shortcut = self.conv_shortcut(shortcut)
        # conv1
        residual = self.conv1(x)
        residual = activation(self.norm1(residual))
        # conv2
        residual = self.norm2(self.conv2(residual))
        out = activation(residual + shortcut)
        return out

class Bottleneck(nn.Module):
    expansion = 2
    def __init__(self, in_channels, mid_channels, out_channels, stride=1,
                                a_bits=1, w_bits=1, g_bits=32,):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride,
                            padding=1, groups=1, bias=False, dilation=1)
        self.norm2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
    def forward(self, x):
        # shortcut
        shortcut = x
        if hasattr(self, 'conv_shortcut'):
            shortcut = self.conv_shortcut(shortcut)
        # conv1
        residual = self.conv1(x)
        residual = self.norm1(activation(residual))
        # conv2
        residual = self.conv2(residual)
        residual = self.norm2(activation(residual))
        # conv3
        residual = activation(self.conv3(residual))
        out = self.norm3(residual + shortcut)
        return out

class Xnor_Basicneck(Basicneck):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1,
                                a_bits=1, w_bits=1, g_bits=32,):
        super(Xnor_Basicneck, self).__init__(
            in_channels, mid_channels, out_channels, stride=stride,
            a_bits=1, w_bits=1, g_bits=32,
        )
        self.conv1 = XnorConv2d(in_channels, mid_channels, kernel_size=3, stride=stride, 
                                padding=1, groups=1, bias=False, dilation=1)
        self.conv2 = XnorConv2d(mid_channels, out_channels, kernel_size=3, stride=1, 
                                padding=1, groups=1, bias=False, dilation=1)
        if hasattr(self, 'conv_shortcut'):
            self.conv_shortcut = XnorConv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class Xnor_Bottleneck(Bottleneck):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1,
                                a_bits=1, w_bits=1, g_bits=32,):
        super(Xnor_Bottleneck, self).__init__(
            in_channels, mid_channels, out_channels, stride=stride,
            a_bits=1, w_bits=1, g_bits=32,
        )
        self.conv1 = XnorConv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.conv2 = XnorConv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, 
                                padding=1, groups=1, bias=False, dilation=1)
        self.conv3 = XnorConv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        if hasattr(self, 'conv_shortcut'):
            self.conv_shortcut = XnorConv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class Ternary_Basicneck(Basicneck):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1,
                                a_bits=1, w_bits=1, g_bits=32,):
        super(Ternary_Basicneck, self).__init__(
            in_channels, mid_channels, out_channels, stride=stride,
            a_bits=1, w_bits=1, g_bits=32,
        )
        self.conv1 = TernaryConv2d(in_channels, mid_channels, kernel_size=3, stride=stride, 
                                padding=1, groups=1, bias=False, dilation=1)
        self.conv2 = TernaryConv2d(mid_channels, out_channels, kernel_size=3, stride=1, 
                                padding=1, groups=1, bias=False, dilation=1)
        if hasattr(self, 'conv_shortcut'):
            self.conv_shortcut = TernaryConv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class Ternary_Bottleneck(Bottleneck):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1,
                                a_bits=1, w_bits=1, g_bits=32,):
        super(Ternary_Bottleneck, self).__init__(
            in_channels, mid_channels, out_channels, stride=stride,
            a_bits=1, w_bits=1, g_bits=32,
        )
        self.conv1 = TernaryConv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.conv2 = TernaryConv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, 
                                padding=1, groups=1, bias=False, dilation=1)
        self.conv3 = TernaryConv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        if hasattr(self, 'conv_shortcut'):
            self.conv_shortcut = TernaryConv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class Quantized_Basicneck(Basicneck):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1,
                                a_bits=1, w_bits=1, g_bits=32,):
        super(Quantized_Basicneck, self).__init__(
            in_channels, mid_channels, out_channels, stride=stride,
            a_bits=1, w_bits=1, g_bits=32,
        )
        self.conv1 = QuantizedConv2d(a_bits=a_bits, w_bits=w_bits, g_bits=g_bits,
                in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=stride, 
                padding=1, groups=1, bias=False, dilation=1)
        self.conv2 = QuantizedConv2d(a_bits=a_bits, w_bits=w_bits, g_bits=g_bits,
                in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, 
                padding=1, groups=1, bias=False, dilation=1)
        if hasattr(self, 'conv_shortcut'):
            self.conv_shortcut = QuantizedConv2d(a_bits=a_bits, w_bits=w_bits, g_bits=g_bits,
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False)

class Quantized_Bottleneck(Bottleneck):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1,
                                a_bits=1, w_bits=1, g_bits=32,):
        super(Quantized_Bottleneck, self).__init__(
            in_channels, mid_channels, out_channels, stride=stride,
            a_bits=1, w_bits=1, g_bits=32,
        )
        self.conv1 = QuantizedConv2d(a_bits=a_bits, w_bits=w_bits, g_bits=g_bits,
                in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, bias=False)
        self.conv2 = QuantizedConv2d(a_bits=a_bits, w_bits=w_bits, g_bits=g_bits,
                in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=stride, 
                padding=1, groups=1, bias=False, dilation=1)
        self.conv3 = QuantizedConv2d(a_bits=a_bits, w_bits=w_bits, g_bits=g_bits,
                in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        if hasattr(self, 'conv_shortcut'):
            self.conv_shortcut = QuantizedConv2d(a_bits=a_bits, w_bits=w_bits, g_bits=g_bits,
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False)

class ResNet_cifar(nn.Module):
    """
    args:
        stage_repeat(list)：每个stage重复的block数
        num_classes(int)：分类数
        a_bits(int): activation量化位数
        w_bits(int): weight量化位数
        g_bits(int): gradient量化位数
    """
    def __init__(self, block, stage_repeat=[3, 3, 3], a_bits=1, w_bits=1, g_bits=32, num_classes=1000):
        super(ResNet_cifar, self).__init__()

        self.num_classes = num_classes
        self.stage_repeat = stage_repeat

        stage_channels = [16, 64, 128, 256] # 原始每层stage的输出通道数
        assert len(stage_channels)-1 == len(stage_repeat)
        
        output_channels = [stage_channels[0]]
        for i in range(1, len(stage_channels)):
            output_channels += [stage_channels[i],]*stage_repeat[i-1]
        
        self.features = nn.ModuleList()

        self.features.append(first_conv(3, output_channels[0], stride=1))

        mid_channels = []
        for i in range(1, len(stage_channels)):
            mid_channels += [int(stage_channels[i]/block.expansion),]*stage_repeat[i-1]
        
        block_num = 1
        for stage in range(len(stage_repeat)):
            if stage == 0:
                self.features.append(block(output_channels[block_num-1], mid_channels[block_num-1], output_channels[block_num], stride=1,
                                            a_bits=a_bits, w_bits=w_bits, g_bits=g_bits))
                block_num += 1
            else:
                self.features.append(block(output_channels[block_num-1], mid_channels[block_num-1], output_channels[block_num], stride=2,
                                            a_bits=a_bits, w_bits=w_bits, g_bits=g_bits))
                block_num += 1

            for i in range(1, stage_repeat[stage]):
                self.features.append(block(output_channels[block_num-1], mid_channels[block_num-1], output_channels[block_num], stride=1,
                                            a_bits=a_bits, w_bits=w_bits, g_bits=g_bits))
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
        
def resnet20_q(type='dorefa', a_bits=1, w_bits=1, g_bits=32, num_classes=10):
    if type == 'none':
        block = Basicneck
    elif type == 'dorefa':
        block = Quantized_Basicneck
    elif type == 'xnor':
        block = Xnor_Basicneck
    elif type == 'ternary':
        block = Ternary_Basicneck
    else:
        raise NotImplementedError("unsupported quantize method!")
    return ResNet_cifar(block, [3, 3, 3], 
        a_bits=a_bits, w_bits=w_bits, g_bits=g_bits, num_classes=num_classes)

def resnet32_q(type='dorefa', a_bits=1, w_bits=1, g_bits=32, num_classes=10):
    if type == 'none':
        block = Basicneck
    elif type == 'dorefa':
        block = Quantized_Basicneck
    elif type == 'xnor':
        block = Xnor_Basicneck
    elif type == 'ternary':
        block = Ternary_Basicneck
    else:
        raise NotImplementedError("unsupported quantize method!")
    return ResNet_cifar(block, [5, 5, 5], 
        a_bits=a_bits, w_bits=w_bits, g_bits=g_bits, num_classes=num_classes)

def resnet44_q(type='dorefa', a_bits=1, w_bits=1, g_bits=32, num_classes=10):
    if type == 'none':
        block = Basicneck
    elif type == 'dorefa':
        block = Quantized_Basicneck
    elif type == 'xnor':
        block = Xnor_Basicneck
    elif type == 'ternary':
        block = Ternary_Basicneck
    else:
        raise NotImplementedError("unsupported quantize method!")
    return ResNet_cifar(block, [7, 7, 7], 
        a_bits=a_bits, w_bits=w_bits, g_bits=g_bits, num_classes=num_classes)

def resnet56_q(type='dorefa', a_bits=1, w_bits=1, g_bits=32, num_classes=10):
    if type == 'none':
        block = Bottleneck
    elif type == 'dorefa':
        block = Quantized_Bottleneck
    elif type == 'xnor':
        block = Xnor_Bottleneck
    elif type == 'ternary':
        block = Ternary_Bottleneck
    else:
        raise NotImplementedError("unsupported quantize method!")
    return ResNet_cifar(block, [6, 6, 6], 
        a_bits=a_bits, w_bits=w_bits, g_bits=g_bits, num_classes=num_classes)

def resnet110_q(type='dorefa', a_bits=1, w_bits=1, g_bits=32, num_classes=10):
    if type == 'none':
        block = Bottleneck
    elif type == 'dorefa':
        block = Quantized_Bottleneck
    elif type == 'xnor':
        block = Xnor_Bottleneck
    elif type == 'ternary':
        block = Ternary_Bottleneck
    else:
        raise NotImplementedError("unsupported quantize method!")
    return ResNet_cifar(block, [12, 12, 12], 
        a_bits=a_bits, w_bits=w_bits, g_bits=g_bits, num_classes=num_classes)
