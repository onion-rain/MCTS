import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161',
            'densenet_cifar1', 'densenet_cifar2']


class _DenseLayer_BasicBlock(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer_BasicBlock, self).__init__()
        self.drop_rate = drop_rate
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, growth_rate, kernel_size=3, 
                               padding=1, bias=False)

    def forward(self, pre_features):
        new_features = torch.cat(pre_features, 1)
        new_features = self.norm1(new_features)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseLayer_Bottleneck(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer_Bottleneck, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size*growth_rate, 
                                kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size*growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size*growth_rate, growth_rate,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, pre_features):
        new_features = torch.cat(pre_features, 1)
        new_features = self.norm1(new_features)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        new_features = self.norm2(new_features)
        new_features = self.relu2(new_features)
        new_features = self.conv2(new_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, block, num_layers, num_input_features, growth_rate, 
                    bn_size, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = block(
                num_input_features=num_input_features + i*growth_rate, 
                growth_rate=growth_rate, 
                bn_size=bn_size, 
                drop_rate=drop_rate,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features] # 此列表保存各层输出的特征图
        for name, layer in self.named_children():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    """
    转换层，用于连接相邻的DenseBlock
    """
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                                kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
                        (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, block, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            denseblock = _DenseBlock(
                block=block,
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.features.add_module('denseblock%d' % (i + 1), denseblock)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.norm5 = nn.BatchNorm2d(num_features)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1) # x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


def densenet121(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """
    return DenseNet(_DenseLayer_Bottleneck, 32, (6, 12, 24, 16), 64, **kwargs)


def densenet161(**kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return DenseNet(_DenseLayer_Bottleneck, 48, (6, 12, 36, 24), 96, **kwargs)


def densenet169(**kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """
    return DenseNet(_DenseLayer_Bottleneck, 32, (6, 12, 32, 32), 64, **kwargs)


def densenet201(**kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """
    return DenseNet(_DenseLayer_Bottleneck, 32, (6, 12, 48, 32), 64, **kwargs)


def densenet_cifar1(**kwargs):
    return DenseNet(_DenseLayer_Bottleneck, 12, (32, 32, 32), 24, **kwargs)


def densenet_cifar2(**kwargs):
    return DenseNet(_DenseLayer_Bottleneck, 40, (62, 62, 62), 80, **kwargs)
    