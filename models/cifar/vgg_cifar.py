import torch
import torch.nn as nn


__all__ = [
    'VGG_cifar', 'vgg_cfg',
    'vgg11_cifar', 'vgg11_bn_cifar', 
    'vgg13_cifar', 'vgg13_bn_cifar', 
    'vgg16_cifar', 'vgg16_bn_cifar',
    'vgg19_cifar', 'vgg19_bn_cifar',
]


def make_layers(structure, batch_norm=False):
    layers = []
    in_channels = 3
    for v in structure:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG_cifar(nn.Module):

    def __init__(self, structure, batch_norm, num_classes=1000, init_weights=True):
        super(VGG_cifar, self).__init__()
        self.features = make_layers(structure, batch_norm=batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(structure[-1], num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



structures = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


def vgg11_cifar(structure=None, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    if structure is not None:
        structure = structures['A']
    return VGG_cifar(structure, False, **kwargs)


def vgg11_bn_cifar(structure=None, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    if structure is not None:
        structure = structures['A']
    return VGG_cifar(structure, True, **kwargs)


def vgg13_cifar(structure=None, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    if structure is not None:
        structure = structures['B']
    return VGG_cifar(structure, False, **kwargs)


def vgg13_bn_cifar(structure=None, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    if structure is not None:
        structure = structures['B']
    return VGG_cifar(structure, True, **kwargs)


def vgg16_cifar(structure=None, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    if structure is not None:
        structure = structures['D']
    return VGG_cifar(structure, False, **kwargs)


def vgg16_bn_cifar(structure=None, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    if structure is not None:
        structure = structures['D']
    return VGG_cifar(structure, True, **kwargs)


def vgg19_cifar(structure=None, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    if structure is not None:
        structure = structures['E']
    return VGG_cifar(structure, False, **kwargs)


def vgg19_bn_cifar(structure=None, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    if structure is not None:
        structure = structures['E']
    return VGG_cifar(structure, True, **kwargs)

def vgg_cfg(structure=None, **kwargs):
    """默认VGG19BN，可通过structure更改结构"""
    if structure is None:
        structure = structures['E']
    return VGG_cifar(structure, True, **kwargs)