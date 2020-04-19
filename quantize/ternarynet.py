import torch
import torch.nn.functional as F

__all__ = ["TernaryConv2d"]

# 三值化

class Ternarize(torch.autograd.Function):
    # torch.round自动求导导数为0
    @staticmethod
    def forward(self, input):
        E = torch.mean(input.abs(), (3, 2, 1), keepdim=True)
        delta = 0.7 * E # threshold
        input = torch.sign((input + delta).sign() + (input - delta).sign())
        return input, delta
    @staticmethod
    def backward(self, grad_input, grad_delta):
        # STE
        return grad_input.clone()

def ternarize(input):
    return Ternarize.apply(input)

# d888888b d88888b d8888b. d8b   db  .d8b.  d8888b. d888888b d88888D d88888b          .d8b.   .o88b. d888888b d888888b db    db  .d8b.  d888888b d888888b  .d88b.  d8b   db 
# `~~88~~' 88'     88  `8D 888o  88 d8' `8b 88  `8D   `88'   YP  d8' 88'             d8' `8b d8P  Y8 `~~88~~'   `88'   88    88 d8' `8b `~~88~~'   `88'   .8P  Y8. 888o  88 
#    88    88ooooo 88oobY' 88V8o 88 88ooo88 88oobY'    88       d8'  88ooooo         88ooo88 8P         88       88    Y8    8P 88ooo88    88       88    88    88 88V8o 88 
#    88    88~~~~~ 88`8b   88 V8o88 88~~~88 88`8b      88      d8'   88~~~~~         88~~~88 8b         88       88    `8b  d8' 88~~~88    88       88    88    88 88 V8o88 
#    88    88.     88 `88. 88  V888 88   88 88 `88.   .88.    d8' db 88.             88   88 Y8b  d8    88      .88.    `8bd8'  88   88    88      .88.   `8b  d8' 88  V888 
#    YP    Y88888P 88   YD VP   V8P YP   YP 88   YD Y888888P d88888P Y88888P C88888D YP   YP  `Y88P'    YP    Y888888P    YP    YP   YP    YP    Y888888P  `Y88P'  VP   V8P

def ternarize_activation(input):
    return ternarize(input)


# d888888b d88888b d8888b. d8b   db  .d8b.  d8888b. d888888b d88888D d88888b         db   d8b   db d88888b d888888b  d888b  db   db d888888b 
# `~~88~~' 88'     88  `8D 888o  88 d8' `8b 88  `8D   `88'   YP  d8' 88'             88   I8I   88 88'       `88'   88' Y8b 88   88 `~~88~~' 
#    88    88ooooo 88oobY' 88V8o 88 88ooo88 88oobY'    88       d8'  88ooooo         88   I8I   88 88ooooo    88    88      88ooo88    88    
#    88    88~~~~~ 88`8b   88 V8o88 88~~~88 88`8b      88      d8'   88~~~~~         Y8   I8I   88 88~~~~~    88    88  ooo 88~~~88    88    
#    88    88.     88 `88. 88  V888 88   88 88 `88.   .88.    d8' db 88.             `8b d8'8b d8' 88.       .88.   88. ~8~ 88   88    88    
#    YP    Y88888P 88   YD VP   V8P YP   YP 88   YD Y888888P d88888P Y88888P C88888D  `8b8' `8d8'  Y88888P Y888888P  Y888P  YP   YP    YP    

def ternarize_weight(input):
    return ternarize(input)

def get_alpha(weight, threshold):
    weight_abs = weight.abs()
    mask_gt = weight_abs.gt(threshold)
    mask_le = weight_abs.le(threshold)
    weight_abs[mask_le] = 0
    weight_abs_threshold_sum = torch.sum(weight_abs, (3, 2, 1), keepdim=True)
    nonzero_num = torch.sum(mask_le, (3, 2, 1), keepdim=True)
    alpha = weight_abs_threshold_sum / nonzero_num
    return alpha


# d888888b d88888b d8888b. d8b   db  .d8b.  d8888b. d888888b d88888D d88888b          .o88b.  .d88b.  d8b   db db    db .d888b. d8888b. 
# `~~88~~' 88'     88  `8D 888o  88 d8' `8b 88  `8D   `88'   YP  d8' 88'             d8P  Y8 .8P  Y8. 888o  88 88    88 VP  `8D 88  `8D 
#    88    88ooooo 88oobY' 88V8o 88 88ooo88 88oobY'    88       d8'  88ooooo         8P      88    88 88V8o 88 Y8    8P    odD' 88   88 
#    88    88~~~~~ 88`8b   88 V8o88 88~~~88 88`8b      88      d8'   88~~~~~         8b      88    88 88 V8o88 `8b  d8'  .88'   88   88 
#    88    88.     88 `88. 88  V888 88   88 88 `88.   .88.    d8' db 88.             Y8b  d8 `8b  d8' 88  V888  `8bd8'  j88.    88  .8D 
#    YP    Y88888P 88   YD VP   V8P YP   YP 88   YD Y888888P d88888P Y88888P C88888D  `Y88P'  `Y88P'  VP   V8P    YP    888888D Y8888D' 

class TernaryConv2d(torch.nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(TernaryConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        ternary_input, _ = ternarize_activation(input)
        ternary_weight, threshold = ternarize_weight(self.weight)

        alpha = get_alpha(self.weight, threshold) # scaling factor
        ternary_weight = ternary_weight.mul(alpha)

        out = F.conv2d(ternary_input, ternary_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        return out
