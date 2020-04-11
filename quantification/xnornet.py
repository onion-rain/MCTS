import torch
import torch.nn.functional as F

__all__ = ["BinaryActive", "BinarizeConv2d"]

# d8888b. d888888b d8b   db  .d8b.  d8888b. db    db          .d8b.   .o88b. d888888b d888888b db    db d88888b 
# 88  `8D   `88'   888o  88 d8' `8b 88  `8D `8b  d8'         d8' `8b d8P  Y8 `~~88~~'   `88'   88    88 88'     
# 88oooY'    88    88V8o 88 88ooo88 88oobY'  `8bd8'          88ooo88 8P         88       88    Y8    8P 88ooooo 
# 88~~~b.    88    88 V8o88 88~~~88 88`8b      88            88~~~88 8b         88       88    `8b  d8' 88~~~~~ 
# 88   8D   .88.   88  V888 88   88 88 `88.    88            88   88 Y8b  d8    88      .88.    `8bd8'  88.     
# Y8888P' Y888888P VP   V8P YP   YP 88   YD    YP    C88888D YP   YP  `Y88P'    YP    Y888888P    YP    Y88888P 

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



# d8888b. d888888b d8b   db  .d8b.  d8888b. d888888b d88888D d88888b         db   d8b   db d88888b d888888b  d888b  db   db d888888b 
# 88  `8D   `88'   888o  88 d8' `8b 88  `8D   `88'   YP  d8' 88'             88   I8I   88 88'       `88'   88' Y8b 88   88 `~~88~~' 
# 88oooY'    88    88V8o 88 88ooo88 88oobY'    88       d8'  88ooooo         88   I8I   88 88ooooo    88    88      88ooo88    88    
# 88~~~b.    88    88 V8o88 88~~~88 88`8b      88      d8'   88~~~~~         Y8   I8I   88 88~~~~~    88    88  ooo 88~~~88    88    
# 88   8D   .88.   88  V888 88   88 88 `88.   .88.    d8' db 88.             `8b d8'8b d8' 88.       .88.   88. ~8~ 88   88    88    
# Y8888P' Y888888P VP   V8P YP   YP 88   YD Y888888P d88888P Y88888P C88888D  `8b8' `8d8'  Y88888P Y888888P  Y888P  YP   YP    YP  

class binarize_weight(torch.autograd.Function):
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
    mean = weight.mean(1, keepdim=True)
    weight = weight.sub(mean)
    # clamp
    weight = weight.clamp(-1.0, 1.0)
    return weight
    # # mean center
    # mean = weight.data.mean(1, keepdim=True)
    # weight = weight.data.sub(mean)
    # # clamp
    # weight = weight.data.clamp(-1.0, 1.0)
    # return weight

def bianrize(weight):
    element_num = weight[0].nelement()
    size = weight.size()
    alpha = weight.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(element_num).expand(size)
    weight = binarize_weight.apply(weight)
    weight = weight.mul(alpha)
    return weight

# class BinarizeWeight(torch.nn.Module):

#     def __init__(self):
#         super(BinarizeWeight, self).__init__()
 
#     def forward(self, weight):
#         # See the autograd section for explanation of what happens here.
#         weight = preprocess(weight)
#         weight = bianrize(weight)
#         return weight


class BinarizeConv2d(torch.nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        # self.binarize = BinarizeWeight()

    def forward(self, input):
        # binary_weight = self.binarize(self.weight)
        binary_weight = preprocess(self.weight) # 网络中保存的weiht为全精度，每次前向传播再进行二值化
        binary_weight = bianrize(binary_weight)
        out = F.conv2d(input, binary_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        return out
