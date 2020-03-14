import torch
import numpy as np
import copy

import models

class FilterPruner(object):
    """
    FIXME simple prune 和 prune 得到的模型精度不同
    TODO 适配其他模型
    args:
        model(torch.nn.Module): 模型
        arch(str): 模型名，用于加载剪枝后新的网络结构
        device: 设备(default: 'cpu')
        prune_percent(list): 剪枝率(default: 0.5) 若len(prune_percent) == 1 则所有层使用相同剪枝比例
        target_cfg(list): 目标模型结构(default: None)优先级高于prune_percent
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
    例：
        pruner = FilterPruner(
            model=self.model,
            device=self.device,
            arch=self.config.arch,
            prune_percent=[self.config.prune_percent],
            # target_cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256],
            p=1,
        )
        pruner.simple_prune()
        pruner.prune()
    注：simple_prune()可单独使用，但要prune，必须先simple_prune()
    """
    def __init__(self, model, arch=None, prune_percent=[0.5,], device='cpu',
                    target_cfg=None, p="fro"):

        self.device = device
        self.arch = arch
        self.prune_percent = prune_percent
        self.original_model = copy.deepcopy(model).to(device)
        self.target_cfg = target_cfg
        self.pruned_cfg = []
        self.pruned_cfg_mask = []
        self.p = p
        self.original_model.eval() # 验证模式


    def simple_prune(self, model=None, prune_percent=None):
        """仅将权值归零"""
        if model is not None:
            self.original_model = copy.deepcopy(model).to(self.device)
            self.original_model.eval()
            self.simple_pruned_model = model # 若传入模型则直接在模型上剪，不然还得更新optimizer
        else:
            self.simple_pruned_model = simple_pruned_model(self.original_model).to(self.device)
        self.simple_pruned_model.eval()

        if prune_percent is not None:
            self.prune_percent = prune_percent
        if len(self.prune_percent) == 1:
            p = self.prune_percent[0]
            self.prune_percent = []
            for layer_index, module in enumerate(self.simple_pruned_model.modules()):
                if isinstance(module, torch.nn.Conv2d):
                    self.prune_percent.append(p)
                elif isinstance(module, torch.nn.MaxPool2d):
                    self.prune_percent.append(0)

        self.pruned_cfg = []
        self.conv_threshold = []
        conv_pruned_num = 0
        conv_original_num = 0
        index = 0
        for layer_index, module in enumerate(self.simple_pruned_model.modules()):
            if isinstance(module, torch.nn.Conv2d):
                original_filters_num = module.weight.data.shape[0] # 当前层filter总数
                if self.target_cfg is not None:
                    remain_filters_num = self.target_cfg[index]
                    pruned_filters_num = original_filters_num - remain_filters_num
                else:
                    pruned_filters_num = int(original_filters_num*self.prune_percent[index])
                    remain_filters_num = original_filters_num-pruned_filters_num
                self.pruned_cfg.append(remain_filters_num)

                filter_weight_num = module.weight.data.shape[1] * module.weight.data.shape[2] * module.weight.data.shape[3]

                # 排序计算保留filter的索引
                weight_copy = module.weight.data.clone()
                weight_copy_flat = weight_copy.view(weight_copy.size()[0], -1)
                conv_norm = weight_copy_flat.norm(self.p, 1) # lp norm
                sorted_conv_norm, sorted_filter_index = torch.sort(conv_norm)
                keep_filters_index = sorted_filter_index.cpu().numpy()[::-1][:remain_filters_num]
                
                # 创建mask与模型相乘
                mask = torch.zeros(original_filters_num * filter_weight_num).to(self.device)
                for i in range(len(keep_filters_index)):
                    mask[keep_filters_index[i]*filter_weight_num : (keep_filters_index[i]+1)*filter_weight_num] = 1
                mask = mask.view_as(module.weight.data)
                module.weight.data.mul_(mask)
                
                # 用于从旧模型向新模型恢复weight
                cfg_mask = torch.zeros(original_filters_num).to(self.device)
                cfg_mask[keep_filters_index.tolist()] = 1
                self.pruned_cfg_mask.append(cfg_mask)

                # 更新一些统计数据
                # conv_threshold_index = pruned_filters_num
                # self.conv_threshold.append(sorted_conv_norm[conv_threshold_index].item())
                conv_original_num += original_filters_num
                conv_pruned_num += pruned_filters_num
                # print('layer index: {:<5} total filters: {:<10} remaining filters: {:<10}'.
                #     format(layer_index, original_filters_num, remain_filters_num))
                index += 1
            elif isinstance(module, torch.nn.MaxPool2d):
                self.pruned_cfg.append('M')
                index += 1

        # print('{:<30}'.format('==> prune conv threshold: '), end='')
        # print('{}'.format([round(i,4) for i in self.conv_threshold]))
        # print('{:<30}  {:.4f}'.format('==> conv layer num: ', index))
        self.conv_prune_ratio = conv_pruned_num/conv_original_num
        print('{:<30}  {:.4f}%'.format('==> prune conv ratio: ', self.conv_prune_ratio*100))
        print('{}'.format(self.pruned_cfg))
        return self.simple_pruned_model, self.pruned_cfg


    def prune(self):
        """构造新的模型结构"""
        print('{:<30}  {:<8}'.format('==> creating new model: ', self.arch))
        self.pruned_model = models.__dict__[self.arch](cfg=self.pruned_cfg, num_classes=self.original_model.num_classes) # 根据cfg构建新的model
        self.pruned_model.to(self.device) # 模型转移到设备上

        self.pruned_model.eval()

        self.weight_recover_vgg(self.pruned_cfg_mask, self.simple_pruned_model, self.pruned_model)
        return self.pruned_model, self.pruned_cfg


    def weight_recover_vgg(self, cfg_mask, original_model, pruned_model):
        """根据 cfg_mask 将 original_model 的权重恢复到结构化剪枝后的 pruned_model """
        # 将参数复制到新模型
        layer_id_in_cfg = 0
        conv_in_channels_mask = torch.ones(3)
        conv_out_channels_mask = cfg_mask[layer_id_in_cfg]
        for [module0, module1] in zip(original_model.modules(), pruned_model.modules()):
            if isinstance(module0, torch.nn.Conv2d):
                idx0 = np.squeeze(np.argwhere(np.asarray(conv_in_channels_mask.cpu().numpy()))) # 从掩模计算出需要保留的权重下标
                idx1 = np.squeeze(np.argwhere(np.asarray(conv_out_channels_mask.cpu().numpy())))
                # print('conv: in channels: {:d}, out chennels:{:d}'.format(idx0.shape[0], idx1.shape[0]))
                # module0.weight.data[卷积核个数，深度，长，宽]

                # 当通道数只保留一个时，idx维度为2，元素却只有一个，此时需要降维到一维
                # 否则module0.weight.data[:, idx, :, :]会报错：IndexError: too many indices for array
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))

                w = module0.weight.data[:, idx0, :, :].clone() # 剪输入通道
                w = w[idx1, :, :, :].clone() # 剪输出通道
                module1.weight.data = w.clone()
            elif isinstance(module0, torch.nn.BatchNorm2d):
                idx1 = np.squeeze(np.argwhere(np.asarray(conv_out_channels_mask.cpu().numpy()))) # np.argwhere()返回非零元素下标
                # 将应保留的权值复制到新模型中
                module1.weight.data = module0.weight.data[idx1].clone()
                module1.bias.data = module0.bias.data[idx1].clone()
                module1.running_mean = module0.running_mean[idx1].clone()
                module1.running_var = module0.running_var[idx1].clone()
                # 下一层
                conv_in_channels_mask = conv_out_channels_mask.clone()
                layer_id_in_cfg += 1
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    conv_out_channels_mask = cfg_mask[layer_id_in_cfg]
            elif isinstance(module0, torch.nn.Linear):
                # 调整全连接层输入通道数
                idx0 = np.squeeze(np.argwhere(np.asarray(conv_in_channels_mask.cpu().numpy())))
                module1.weight.data = module0.weight.data[:, idx0].clone() # module0.weight.data[输出通道数，输入通道数]
                # print("full connection: in channels: {:d}, out channels: {:d}".format(idx0.shape[0], module0.weight.data.shape[0]))
                break # 仅调整第一层全连接层

        


