import torch
import numpy as np
import copy


class WeightPruner(object):
    """
    args:
        model(torch.nn.Module): 模型
        prune_percent(float): 剪枝率(default: 0.5)
        device: 设备(default: torch.device('cpu'))
        prune_object(str): 剪枝目标：可选'conv'、'fc'、'all'(default: ['conv',])
    例：
        pruner = WeightPruner(
            model=self.model, 
            prune_percent=self.config.prune_percent, 
            device=self.device, 
            prune_object=['conv', 'fc']
        )
        pruner.prune()
    """
    def __init__(self, model, prune_percent=0.5, device=torch.device('cpu'), prune_object=['conv', 'fc']):

        self.device = device
        self.prune_percent = prune_percent
        self.prune_object = prune_object
        self.original_model = copy.deepcopy(model).to(self.device)
        self.pruned_model = None
        self.original_model.eval()

    def extract_conv_weights(self):
        """提取所有层 weights(tensor)存到self.conv_weights_list, len()=layers_num， 
            对于conv层：tensor.shape=[filters_num, weights_num]
            对于fc层：tensor.shape=[in_channels*out_channels]"""
        self.conv_weights_num = 0
        self.fc_weights_num = 0
        self.conv_weights_list = [] # 保存各层权重的tensor
        self.fc_weights_list = [] # 保存各层权重的tensor
        for module in self.original_model.modules():
            if isinstance(module, torch.nn.Conv2d) and 'conv' in self.prune_object:
                layer_weights = module.weight.data.clone().cpu()
                filters_weights = layer_weights.view(layer_weights.size()[0], -1) # 从卷积核维度以下一维展开
                self.conv_weights_list.append(filters_weights)
                self.conv_weights_num += module.weight.data.numel()
            elif isinstance(module, torch.nn.Linear) and 'fc' in self.prune_object:
                layer_weights = module.weight.data.clone().cpu()
                fc_weights = layer_weights.view(-1) # 从层维度以下一维展开
                self.fc_weights_list.append(fc_weights)
                self.fc_weights_num += module.weight.data.numel()


    def get_threshold(self):
        """整个网络共用一个阈值"""
        self.extract_conv_weights()
        self.conv_threshold = []
        self.fc_threshold = []

        conv_weights_abs = torch.zeros(self.conv_weights_num)
        fc_weights_abs = torch.zeros(self.fc_weights_num)

        index = 0
        if 'conv' in self.prune_object:
            for weights in self.conv_weights_list:
                size = weights.numel()
                conv_weights_abs[index : (index+size)] = weights.flatten(0).abs().clone()
                index += size
            sorted_conv_weights_abs = np.sort(conv_weights_abs)
            conv_threshold_index = int(len(sorted_conv_weights_abs) * self.prune_percent)
            self.conv_threshold.append(sorted_conv_weights_abs[conv_threshold_index])
            print('{:<30}  {:0.4e}'.format('==> prune conv threshold: ', self.conv_threshold[0]))

        index = 0
        if 'fc' in self.prune_object:
            for weights in self.fc_weights_list:
                size = weights.numel()
                fc_weights_abs[index : (index+size)] = weights.flatten(0).abs().clone()
                index += size
            sorted_fc_weights_abs = np.sort(fc_weights_abs)
            fc_threshold_index = int(len(sorted_fc_weights_abs) * self.prune_percent)
            self.fc_threshold.append(sorted_fc_weights_abs[fc_threshold_index])
            print('{:<30}  {:0.4e}'.format('==> prune fc threshold: ', self.fc_threshold[0]))


    def prune(self, model=None, prune_percent=None):
        """仅将权值归零"""
        if model is not None:
            self.original_model = model
        if prune_percent is not None:
            self.prune_percent = prune_percent
        self.original_model.eval()

        # 计算阈值
        self.get_threshold()

        # 小于阈值归零处理
        self.pruned_model = copy.deepcopy(self.original_model).to(self.device)
        conv_pruned_num = 0
        fc_pruned_num = 0
        for layer_index, module in enumerate(self.pruned_model.modules()):
            if isinstance(module, torch.nn.Conv2d) and 'conv' in self.prune_object:
                weight_copy = module.weight.data.clone()
                mask = weight_copy.abs().gt(self.conv_threshold[0]).float().to(self.device) # torch.gt(a, b): a>b为1否则为0
                remain_weight = int(torch.sum(mask))
                if remain_weight == 0:
                    error_str = 'Prune Error: layer' + str(layer_index) + ": " + module._get_name() + ': there is no remain nonzero_weight! turn down the prune_percent!'
                    raise Exception(error_str)
                conv_pruned_num += (mask.numel() - remain_weight)
                module.weight.data.mul_(mask)
                # print('layer index: {:<5} total weights: {:<10} remaining weights: {:<10}'.
                #     format(layer_index, mask.numel(), remain_weight))
            if isinstance(module, torch.nn.Linear) and 'fc' in self.prune_object:
                weight_copy = module.weight.data.clone()
                mask = weight_copy.abs().gt(self.fc_threshold[0]).float().to(self.device) # torch.gt(a, b): a>b为1否则为0
                remain_weight = int(torch.sum(mask))
                if remain_weight == 0:
                    error_str = 'Prune Error: layer' + str(layer_index) + ": " + module._get_name() + ': there is no remain nonzero_weight! turn down the prune_percent!'
                    raise Exception(error_str)
                fc_pruned_num += (mask.numel() - remain_weight)
                module.weight.data.mul_(mask)
                # print('layer index: {:<5} total weights: {:<10} remaining weights: {:<10}'.
                #     format(layer_index, mask.numel(), remain_weight))
        prune_ratio = {}
        if 'conv' in self.prune_object:
            self.conv_prune_ratio = conv_pruned_num/self.conv_weights_num
            prune_ratio["conv_prune_ratio"] = self.conv_prune_ratio 
            print('{:<30}  {:.4f}%'.format('==> prune conv ratio: ', self.conv_prune_ratio*100))
        if 'fc' in self.prune_object:
            self.fc_prune_ratio = fc_pruned_num/self.fc_weights_num
            prune_ratio["fc_prune_ratio"] = self.fc_prune_ratio 
            print('{:<30}  {:.4f}%'.format('==> prune fc ratio: ', self.fc_prune_ratio*100))
        return self.pruned_model, 0, prune_ratio



