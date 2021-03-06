import torch
import torch.nn as nn


__all__ = [
    'VGG_cifar',
    'vgg11_cifar', 'vgg11_bn_cifar', 
    'vgg13_cifar', 'vgg13_bn_cifar', 
    'vgg16_cifar', 'vgg16_bn_cifar',
    'vgg19_cifar', 'vgg19_bn_cifar',
]


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1) # baseline为bias=true版本
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG_cifar(nn.Module):

    def __init__(self, cfg, batch_norm, num_classes=1000, init_weights=True):
        super(VGG_cifar, self).__init__()
        self.num_classes = num_classes
        self.features = make_layers(cfg, batch_norm=batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(cfg[-1], num_classes)
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



cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


def vgg11_cifar(cfg=None, **kwargs):
    if cfg is None:
        cfg = cfgs['A']
    return VGG_cifar(cfg, False, **kwargs)


def vgg11_bn_cifar(cfg=None, **kwargs):
    if cfg is None:
        cfg = cfgs['A']
    return VGG_cifar(cfg, True, **kwargs)


def vgg13_cifar(cfg=None, **kwargs):
    if cfg is None:
        cfg = cfgs['B']
    return VGG_cifar(cfg, False, **kwargs)


def vgg13_bn_cifar(cfg=None, **kwargs):
    if cfg is None:
        cfg = cfgs['B']
    return VGG_cifar(cfg, True, **kwargs)


def vgg16_cifar(cfg=None, **kwargs):
    if cfg is None:
        cfg = cfgs['D']
    return VGG_cifar(cfg, False, **kwargs)


def vgg16_bn_cifar(cfg=None, **kwargs):
    if cfg is None:
        cfg = cfgs['D']
    return VGG_cifar(cfg, True, **kwargs)


def vgg19_cifar(cfg=None, **kwargs):
    if cfg is None:
        cfg = cfgs['E']
    return VGG_cifar(cfg, False, **kwargs)


def vgg19_bn_cifar(cfg=None, **kwargs):
    if cfg is None:
        cfg = cfgs['E']
    return VGG_cifar(cfg, True, **kwargs)

if __name__ == "__main__":
    model = vgg16_bn_cifar()
    print("end")