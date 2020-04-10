import torch
import torch.nn.functional as F

__all__ = ["BinaryActive", "BinarizeConv2d"]

class binary_active(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        input = input.sign()
        return input
    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        # STE
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class BinaryActive(torch.nn.Module):
    def __init__(self):
        super(BinaryActive, self).__init__()
 
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return binary_active.apply(input)


class binary_weight(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    @staticmethod
    def forward(self, input):
        return input.sign()
    @staticmethod
    def backward(self, grad_output):
        # STE, 由于preprocess做过mean center和clamp, weight已经∈[-1, 1]
        return grad_output.clone()

def preprocess(weight):
    # mean center
    s = weight.size()
    mean = weight.data.mean(1, keepdim=True)
    weight = weight.data.sub(mean)
    # clamp
    weight.data = weight.data.clamp(-1.0, 1.0)
    return weight

def bianrize(weight):
    weight = preprocess(weight)
    # alpha = torch.mean(weight.abs(), (3, 2, 1), keepdim=True)
    n = weight[0].nelement()
    size = weight.size()
    alpha = weight.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(size)
    weight.data = binary_weight.apply(weight.data)
    weight.data = weight.data.mul(alpha)
    return weight.data

class BinaryWeight(torch.nn.Module):
    def __init__(self):
        super(BinaryWeight, self).__init__()
 
    def forward(self, weight):
        # See the autograd section for explanation of what happens here.
        weight.data = bianrize(weight)
        return weight


class BinarizeConv2d(torch.nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.binary_weight = BinaryWeight()

    def forward(self, input):
        self.weight = self.binary_weight(self.weight)
        out = F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        return out
