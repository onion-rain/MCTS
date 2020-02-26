import torch
import torch.nn as nn

__all_ = ['test_model']

class test(nn.Module):
    def __init__(self, num_classes=10):
        super(test, self).__init__()
        self.conv = nn.Conv2d(3, 100, kernel_size=1, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(100, num_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def test_model():
    return test()