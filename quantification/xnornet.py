import torch
import torch.nn.functional as F

__all__ = ["BinaryConv2d"]

# d8888b. d888888b d8b   db  .d8b.  d8888b. d888888b d88888D d88888b          .d8b.   .o88b. d888888b d888888b db    db  .d8b.  d888888b d888888b  .d88b.  d8b   db 
# 88  `8D   `88'   888o  88 d8' `8b 88  `8D   `88'   YP  d8' 88'             d8' `8b d8P  Y8 `~~88~~'   `88'   88    88 d8' `8b `~~88~~'   `88'   .8P  Y8. 888o  88 
# 88oooY'    88    88V8o 88 88ooo88 88oobY'    88       d8'  88ooooo         88ooo88 8P         88       88    Y8    8P 88ooo88    88       88    88    88 88V8o 88 
# 88~~~b.    88    88 V8o88 88~~~88 88`8b      88      d8'   88~~~~~         88~~~88 8b         88       88    `8b  d8' 88~~~88    88       88    88    88 88 V8o88 
# 88   8D   .88.   88  V888 88   88 88 `88.   .88.    d8' db 88.             88   88 Y8b  d8    88      .88.    `8bd8'  88   88    88      .88.   `8b  d8' 88  V888 
# Y8888P' Y888888P VP   V8P YP   YP 88   YD Y888888P d88888P Y88888P C88888D YP   YP  `Y88P'    YP    Y888888P    YP    YP   YP    YP    Y888888P  `Y88P'  VP   V8P 

class Binarize_activation(torch.autograd.Function):
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

def binarize_activation(input):
    return Binarize_activation.apply(input)


# d8888b. d888888b d8b   db  .d8b.  d8888b. d888888b d88888D d88888b         db   d8b   db d88888b d888888b  d888b  db   db d888888b 
# 88  `8D   `88'   888o  88 d8' `8b 88  `8D   `88'   YP  d8' 88'             88   I8I   88 88'       `88'   88' Y8b 88   88 `~~88~~' 
# 88oooY'    88    88V8o 88 88ooo88 88oobY'    88       d8'  88ooooo         88   I8I   88 88ooooo    88    88      88ooo88    88    
# 88~~~b.    88    88 V8o88 88~~~88 88`8b      88      d8'   88~~~~~         Y8   I8I   88 88~~~~~    88    88  ooo 88~~~88    88    
# 88   8D   .88.   88  V888 88   88 88 `88.   .88.    d8' db 88.             `8b d8'8b d8' 88.       .88.   88. ~8~ 88   88    88    
# Y8888P' Y888888P VP   V8P YP   YP 88   YD Y888888P d88888P Y88888P C88888D  `8b8' `8d8'  Y88888P Y888888P  Y888P  YP   YP    YP  

class Binarize_weight(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        return input.sign()
    @staticmethod
    def backward(self, grad_output):
        # STE, 由于weight_preprocess做过mean center和clamp, weight已经∈[-1, 1]
        return grad_output.clone()

def binarize_weight(input):
    return Binarize_weight.apply(input)

def affine(input):
    # range变为[-1, 1]
    # mean center
    mean = input.mean(1, keepdim=True)
    input = input.sub(mean)
    # clamp
    # TODO 直接clamp损失比较大，考虑先缩放一下
    input = input.clamp(-1.0, 1.0)
    return input

def get_alpha(weight):
    element_num = weight[0].nelement()
    size = weight.size()
    alpha = weight.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(element_num).expand(size)
    return alpha


# d8888b. d888888b d8b   db  .d8b.  d8888b. db    db          .o88b.  .d88b.  d8b   db db    db .d888b. d8888b. 
# 88  `8D   `88'   888o  88 d8' `8b 88  `8D `8b  d8'         d8P  Y8 .8P  Y8. 888o  88 88    88 VP  `8D 88  `8D 
# 88oooY'    88    88V8o 88 88ooo88 88oobY'  `8bd8'          8P      88    88 88V8o 88 Y8    8P    odD' 88   88 
# 88~~~b.    88    88 V8o88 88~~~88 88`8b      88            8b      88    88 88 V8o88 `8b  d8'  .88'   88   88 
# 88   8D   .88.   88  V888 88   88 88 `88.    88            Y8b  d8 `8b  d8' 88  V888  `8bd8'  j88.    88  .8D 
# Y8888P' Y888888P VP   V8P YP   YP 88   YD    YP    C88888D  `Y88P'  `Y88P'  VP   V8P    YP    888888D Y8888D' 
                                                                                                              
class BinaryConv2d(torch.nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinaryConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        binary_input = binarize_activation(input)
        binary_weight = binarize_weight(self.weight)
        alpha = get_alpha(self.weight) # scaling factor
        binary_weight = binary_weight.mul(alpha)

        out = F.conv2d(binary_input, binary_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        return out
