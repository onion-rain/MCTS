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
        threshold_scope(str): （default: 'model'）
            可选: 
                'model': 全局统一阈值
                'layer': 每层独立阈值
                'cfg': 每层根据目标结构独立阈值
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
    def __init__(self, model, arch=None, prune_percent=[0.5,], device='cpu', threshold_scope='model',
                    original_cfg=None, target_cfg=None, p="fro"):

        self.device = device
        self.arch = arch
        self.prune_percent = prune_percent
        self.prune_object = prune_object
        self.original_model = copy.deepcopy(model).to(device)
        self.original_cfg = original_cfg
        self.target_cfg = target_cfg
        self.pruned_cfg = None
        self.p = p


    def extract_conv_weights(self):
        """提取所有conv层 weights(tensor)存到self.conv_weights_list, len()=layers_num， 
            tensor.shape=[filters_num, weights_num]"""
        self.conv_weights_list = [] # 保存各层权重的tensor
        for module in self.original_model.modules():
            if isinstance(module, torch.nn.Conv2d):
                layer_weights = module.weight.data.clone()
                filters_weights = layer_weights.view(layer_weights.size()[0], -1) # 从卷积核维度以下一维展开
                self.conv_weights_list = self.conv_weights_list.append(filters_weights)
        


    def get_threshold(self):
        """整个网络共用一个阈值"""
        extract_conv_weights()
        self.conv_threshold = []

        conv_weights = torch.cat(conv_weights_list) # 合并成一个tensor
        conv_norm = torch.norm(self.conv_weights, self.p, 1) # lp norm
        sorted_conv_norm, _ = torch.sort(conv_norm)
        conv_threshold_index = int(len(sorted_conv_norm) * self.prune_percent[0])
        self.conv_threshold.append(sorted_conv_norm[conv_threshold_index])
        print('{:<30}  {:0.4e}'.format('==> prune conv threshold: ', self.conv_threshold))


    def get_thresholds(self):
        """每层单独计算阈值"""
        extract_conv_weights()
        self.conv_threshold = []

        for layer_index in range(len(self.conv_weights_list)):
            conv_norm = torch.norm(self.conv_weights_list[layer_index], self.p, 1) # lp norm
            sorted_conv_norm, _ = torch.sort(conv_norm)
            conv_threshold_index = int(len(sorted_conv_norm) * self.prune_percent[layer_index])
            self.conv_threshold.append(sorted_conv_norm[conv_threshold_index])

        print('{:<30}  {:0.4e}'.format('==> prune conv threshold: ', self.conv_threshold))


    def simple_prune(self, prune_percent=None):
        """仅将权值归零"""
        if prune_percent is not None:
            self.prune_percent = prune_percent

        # 计算阈值
        if len(self.prune_percent) > 1:
            self.get_thresholds()
        else: self.get_threshold()

        # 小于阈值归零处理
        self.pruned_model = copy.deepcopy(self.original_model).to(self.device)
        pruned_num = 0
        for layer_index, module in enumerate(self.pruned_model.modules()):
            if isinstance(module, torch.nn.Conv2d):
                weight_copy = module.weight.data.clone()
                mask = weight_copy.abs().gt(self.threshold).float().to(self.device) # torch.gt(a, b): a>b为1否则为0
                remain_weight = int(torch.sum(mask))
                if remain_weight == 0:
                    error_str = 'Prune Error: layer' + str(layer_index) + ": " + module._get_name() + ': there is no remain nonzero_weight! turn down the prune_percent!'
                    print(error_str)
                    # raise
                conv_pruned_num += (mask.numel() - remain_weight)
                module.weight.data.mul_(mask)
                # print('layer index: {:<5} total weights: {:<10} remaining weights: {:<10}'.
                #     format(layer_index, mask.numel(), remain_weight))

        self.conv_prune_ratio = conv_pruned_num/len(self.conv_weights)

        print('{:<30}  {:.4f}%'.format('==> prune conv ratio: ', self.conv_prune_ratio*100))


    def prune(self, prune_percent=None):
        """构造新的模型结构"""
        if prune_percent is not None:
            self.prune_percent = prune_percent

        # 计算阈值
        if len(self.prune_percent) > 1:
            self.get_thresholds()
        else:
            self.get_threshold

        


