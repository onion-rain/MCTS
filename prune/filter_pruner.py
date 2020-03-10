import torch
import numpy as np
import copy


class FilterPruner(object):
    """
    TODO 待测试
    args:
        model(torch.nn.Module): 模型
        arch(str): 模型名，用于加载剪枝后新的网络结构
        prune_percent(list): 剪枝率(default: 0.5)
        device: 设备(default: 'cpu')
        original_cfg(list): 原模型结构(default: None)
        target_cfg(list): 目标模型结构(default: None)。若threshold为'cfg'，则此项必填
        p (int, float, inf, -inf, 'fro', 'nuc', optional): the order of norm. Default: ``'fro'``
            The following norms can be calculated:

            =====  ============================  ==========================
            ord    matrix norm                   vector norm
            =====  ============================  ==========================
            None   Frobenius norm                2-norm
            'fro'  Frobenius norm                --
            'nuc'  nuclear norm                  --
            Other  as vec norm when dim is None  sum(abs(x)**ord)**(1./ord)
            =====  ============================  ==========================
            
    """
    def __init__(self, model, arch=None, prune_percent=[0.5,], device='cpu',
                    original_cfg=None, target_cfg=None, p="fro"):

        self.device = device
        self.arch = arch
        self.prune_percent = prune_percent
        self.original_model = copy.deepcopy(model).to(device)
        self.original_cfg = original_cfg
        self.target_cfg = target_cfg
        self.pruned_cfg = None
        self.p = p


    def extract_conv_weights(self):
        """提取所有conv层 weights(tensor)存到self.conv_weights_list, len()=layers_num， 
            tensor.shape=[filters_num, weights_num]"""
        self.conv_filters_num = [] # 各层filter数量
        self.conv_weights_list = [] # 保存各层权重的tensor
        for module in self.original_model.modules():
            if isinstance(module, torch.nn.Conv2d):
                layer_weights = module.weight.data.clone()
                filters_weights = layer_weights.view(layer_weights.size()[0], -1) # 从卷积核维度以下一维展开
                self.conv_weights_list.append(filters_weights)
                self.conv_filters_num.append(filters_weights.size()[0])


    def simple_prune(self, prune_percent=None):
        """仅将权值归零"""
        if prune_percent is not None:
            self.prune_percent = prune_percent

        self.simple_pruned_model = copy.deepcopy(self.original_model).to(self.device)
        self.conv_threshold = []
        conv_pruned_num = 0
        conv_original_num = 0
        index = 0
        for layer_index, module in enumerate(self.simple_pruned_model.modules()):
            if isinstance(module, torch.nn.Conv2d):
                original_filters_num = module.weight.data.shape[0] # 当前层filter总数
                pruned_filters_num = int(original_filters_num*self.prune_percent[0])
                remain_filters_num = original_filters_num-pruned_filters_num
                if remain_filters_num == 0:
                    error_str = 'Prune Error: layer' + str(layer_index) + ": " + module._get_name() + ': there is no remain nonzero_weight! turn down the prune_percent!'
                    print(error_str)
                    raise

                filter_weight_num = module.weight.data.shape[1] * module.weight.data.shape[2] * module.weight.data.shape[3]

                weight_copy = module.weight.data.clone()
                weight_copy_flat = weight_copy.view(weight_copy.size()[0], -1)
                conv_norm = torch.norm(weight_copy_flat, self.p, 1) # lp norm
                sorted_conv_norm, sorted_filter_index = torch.sort(conv_norm)
                keep_filters_index = sorted_filter_index.cpu().numpy()[::-1][:remain_filters_num]
                
                mask = torch.zeros(original_filters_num * filter_weight_num).to(self.device)
                for i in range(len(keep_filters_index)):
                    mask[keep_filters_index[i]*filter_weight_num : (keep_filters_index[i]+1)*filter_weight_num] = 1
                mask = mask.view_as(module.weight.data)
                module.weight.data.mul_(mask)

                # 更新一些统计数据
                conv_threshold_index = int(len(sorted_conv_norm) * self.prune_percent[0])
                self.conv_threshold.append(sorted_conv_norm[conv_threshold_index].item())
                conv_original_num += original_filters_num
                conv_pruned_num += pruned_filters_num
                # print('layer index: {:<5} total filters: {:<10} remaining filters: {:<10}'.
                #     format(layer_index, original_filters_num, remain_filters_num))
                index += 1

        # print('{:<30}'.format('==> prune conv threshold: '), end='')
        # print('{}'.format([round(i,4) for i in self.conv_threshold]))

        print(index)

        self.conv_prune_ratio = conv_pruned_num/conv_original_num
        print('{:<30}  {:.4f}%'.format('==> prune conv ratio: ', self.conv_prune_ratio*100))


    def prune(self, prune_percent=None):
        """构造新的模型结构"""
        if prune_percent is not None:
            self.prune_percent = prune_percent


        


