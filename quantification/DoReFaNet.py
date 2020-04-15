import torch
import torch.nn.functional as F

__all__ = ["QuantizedConv2d"]

# TODO gradient量化未实现

def affine(input):
    # range变为[0, 1]
    max_abs = input.abs().max()
    if max_abs > 0:
        output = (input/max_abs + 1) / 2
    else:
        output = input
    return output

class Round(torch.autograd.Function):
    # torch.round自动求导导数为0
    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output
    @staticmethod
    def backward(self, grad_input):
        return grad_input.clone()

def quantize(input, bits):
    scale = float(2**bits - 1)
    output = Round.apply(input*scale) / scale
    return output


#  .d88b.  db    db  .d8b.  d8b   db d888888b d888888b d88888D d88888b          .d8b.   .o88b. d888888b d888888b db    db  .d8b.  d888888b d888888b  .d88b.  d8b   db 
# .8P  Y8. 88    88 d8' `8b 888o  88 `~~88~~'   `88'   YP  d8' 88'             d8' `8b d8P  Y8 `~~88~~'   `88'   88    88 d8' `8b `~~88~~'   `88'   .8P  Y8. 888o  88 
# 88    88 88    88 88ooo88 88V8o 88    88       88       d8'  88ooooo         88ooo88 8P         88       88    Y8    8P 88ooo88    88       88    88    88 88V8o 88 
# 88    88 88    88 88~~~88 88 V8o88    88       88      d8'   88~~~~~         88~~~88 8b         88       88    `8b  d8' 88~~~88    88       88    88    88 88 V8o88 
# `8P  d8' 88b  d88 88   88 88  V888    88      .88.    d8' db 88.             88   88 Y8b  d8    88      .88.    `8bd8'  88   88    88      .88.   `8b  d8' 88  V888 
#  `Y88'Y8 ~Y8888P' YP   YP VP   V8P    YP    Y888888P d88888P Y88888P C88888D YP   YP  `Y88P'    YP    Y888888P    YP    YP   YP    YP    Y888888P  `Y88P'  VP   V8P 

def quantize_activation(activation, bits):
    if bits == 32:
        return activation
    # 作者说他assume这之前那个有界的激活函数已经将activation限制在[0, 1]了, 咱也不知道作者咋设计的网络，只好自己clamp了
    # activation = torch.clamp(activation*0.1, 0, 1)
    activation = torch.tanh(activation)
    activation = affine(activation)
    activation = quantize(activation, bits)
    return activation


#  .d88b.  db    db  .d8b.  d8b   db d888888b d888888b d88888D d88888b         db   d8b   db d88888b d888888b  d888b  db   db d888888b .d8888. 
# .8P  Y8. 88    88 d8' `8b 888o  88 `~~88~~'   `88'   YP  d8' 88'             88   I8I   88 88'       `88'   88' Y8b 88   88 `~~88~~' 88'  YP 
# 88    88 88    88 88ooo88 88V8o 88    88       88       d8'  88ooooo         88   I8I   88 88ooooo    88    88      88ooo88    88    `8bo.   
# 88    88 88    88 88~~~88 88 V8o88    88       88      d8'   88~~~~~         Y8   I8I   88 88~~~~~    88    88  ooo 88~~~88    88      `Y8b. 
# `8P  d8' 88b  d88 88   88 88  V888    88      .88.    d8' db 88.             `8b d8'8b d8' 88.       .88.   88. ~8~ 88   88    88    db   8D 
#  `Y88'Y8 ~Y8888P' YP   YP VP   V8P    YP    Y888888P d88888P Y88888P C88888D  `8b8' `8d8'  Y88888P Y888888P  Y888P  YP   YP    YP    `8888Y' 

def quantize_weight(weight, bits):
    if bits == 32:
        return weight
    weight = torch.tanh(weight)
    weight = affine(weight)
    weight = quantize(weight, bits)
    weight = 2 * weight -1 # range扩展为[-1, 1]
    return weight


#  .d88b.  db    db  .d8b.  d8b   db d888888b d888888b d88888D d88888b d8888b.          .o88b.  .d88b.  d8b   db db    db .d888b. d8888b. 
# .8P  Y8. 88    88 d8' `8b 888o  88 `~~88~~'   `88'   YP  d8' 88'     88  `8D         d8P  Y8 .8P  Y8. 888o  88 88    88 VP  `8D 88  `8D 
# 88    88 88    88 88ooo88 88V8o 88    88       88       d8'  88ooooo 88   88         8P      88    88 88V8o 88 Y8    8P    odD' 88   88 
# 88    88 88    88 88~~~88 88 V8o88    88       88      d8'   88~~~~~ 88   88         8b      88    88 88 V8o88 `8b  d8'  .88'   88   88 
# `8P  d8' 88b  d88 88   88 88  V888    88      .88.    d8' db 88.     88  .8D         Y8b  d8 `8b  d8' 88  V888  `8bd8'  j88.    88  .8D 
#  `Y88'Y8 ~Y8888P' YP   YP VP   V8P    YP    Y888888P d88888P Y88888P Y8888D' C88888D  `Y88P'  `Y88P'  VP   V8P    YP    888888D Y8888D' 

class QuantizedConv2d(torch.nn.Conv2d):
    """
    args:
        a_bits(int): activation量化位数
        w_bits(int): weight量化位数
        TODO g_bits(int): gradient量化位数
    """
    def __init__(self, a_bits=1, w_bits=1, g_bits=32, *kargs, **kwargs):
        super(QuantizedConv2d, self).__init__(*kargs, **kwargs)
        self.a_bits = a_bits
        self.w_bits = w_bits
        self.g_bits = g_bits

    def forward(self, input):
        quantized_activation = quantize_activation(input, self.a_bits)
        quantized_weight = quantize_weight(self.weight, self.w_bits)
        out = F.conv2d(quantized_activation, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        return out


#  .d88b.  db    db  .d8b.  d8b   db d888888b d888888b d88888D d88888b d8888b.         db      d888888b d8b   db d88888b  .d8b.  d8888b. 
# .8P  Y8. 88    88 d8' `8b 888o  88 `~~88~~'   `88'   YP  d8' 88'     88  `8D         88        `88'   888o  88 88'     d8' `8b 88  `8D 
# 88    88 88    88 88ooo88 88V8o 88    88       88       d8'  88ooooo 88   88         88         88    88V8o 88 88ooooo 88ooo88 88oobY' 
# 88    88 88    88 88~~~88 88 V8o88    88       88      d8'   88~~~~~ 88   88         88         88    88 V8o88 88~~~~~ 88~~~88 88`8b   
# `8P  d8' 88b  d88 88   88 88  V888    88      .88.    d8' db 88.     88  .8D         88booo.   .88.   88  V888 88.     88   88 88 `88. 
#  `Y88'Y8 ~Y8888P' YP   YP VP   V8P    YP    Y888888P d88888P Y88888P Y8888D' C88888D Y88888P Y888888P VP   V8P Y88888P YP   YP 88   YD

class QuantizedLinear(torch.nn.Conv2d):
    """
    args:
        a_bits(int): activation量化位数
        w_bits(int): weight量化位数
        TODO g_bits(int): gradient量化位数
    """
    def __init__(self, a_bits=1, w_bits=1, g_bits=32, *kargs, **kwargs):
        super(QuantizedLinear, self).__init__(*kargs, **kwargs)
        self.a_bits = a_bits
        self.w_bits = w_bits
        self.g_bits = g_bits

    def forward(self, input):
        quantized_activation = quantize_activation(input, self.a_bits)
        quantized_weight = quantize_weight(self.weight, self.w_bits)
        out = F.linear(quantized_activation, quantized_weight, self.bias)
        return out
