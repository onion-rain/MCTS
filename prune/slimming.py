import torch
import numpy as np
import copy

import models
from models.cifar.slimming_utils import *

class Slimming(object):
    """
    TODO 做成像channel pruner那种
    """
    def __init__(self, arch, model, device, slim_percent, gpu_idx_list):
        self.arch = arch
        self.model = model
        self.device = device
        self.slim_percent = slim_percent
        self.gpu_idx_list = gpu_idx_list
        self.num_classes = self.model.num_classes

    def prune(self):
        if "vgg" in self.arch:
            return self.vgg_slim()
        elif "resnet" in self.arch:
            return self.resnet_slim()
        else:
            raise NotImplemented

    def simple_slim(self, slim_percent=None):
        """
        直接从原模型中将小于阈值的bn层weights置零，不会改变params和flops
        Args:
            slim_percent: 设置slim阈值，若为None则从self.slim_percent获取slim阈值
                Default: None
        """
        self.simple_slimmed_model = copy.deepcopy(self.model).to(self.device)
        # 提取所有bn weights
        bn_abs_weghts = torch.zeros(0).to(self.device)
        for module in self.simple_slimmed_model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                bn_abs_weghts = torch.cat([bn_abs_weghts, module.weight.data.abs().clone()], 0)
        
        # 计算slim阈值
        bn_abs_weights_sorted, _ = torch.sort(bn_abs_weghts)
        if slim_percent == None:
            slim_percent = self.slim_percent
        threshold_index = int(len(bn_abs_weghts) * slim_percent)
        self.threshold = bn_abs_weights_sorted[threshold_index]
        print('{:<30}  {:0.4e}'.format('==> slim threshold: ', self.threshold))
        
        # slim
        slimmed_num = 0
        for layer_index, module in enumerate(self.simple_slimmed_model.modules()):
            if isinstance(module, torch.nn.BatchNorm2d):
                weight_copy = module.weight.data.clone()
                mask = weight_copy.abs().gt(self.threshold).float().to(self.device) # torch.gt(a, b): a>b为1否则为0
                if torch.sum(mask) == 0:
                    error_str = 'Slim Error: layer' + str(layer_index) + ": " + module._get_name() + ': remain_out_channels = 0! turn down the slim_percent!'
                    raise Exception(error_str)
                slimmed_num += (mask.shape[0] - torch.sum(mask))
                module.weight.data.mul_(mask)
                module.bias.data.mul_(mask)
                # print('layer index: {:3d} \t total channel: {:4d} \t remaining channel: {:4d}'.
                #     format(layer_index, mask.shape[0], int(torch.sum(mask))))

        self.slim_ratio = slimmed_num/len(bn_abs_weghts)
        print('{:<30}  {:.4f}%'.format('==> slim ratio: ', self.slim_ratio*100))


    def resnet_slim(self, slim_percent=None):
        """
        删掉低于阈值的bn层，构建新的模型，可以降低params和flops
        Args:
            slim_percent: 设置slim阈值，若为None则从self.slim_percent获取slim阈值
                Default: None
        """
        original_model = copy.deepcopy(self.model).to(self.device)
        # 提取所有bn weights
        bn_abs_weghts = torch.zeros(0).to(self.device)
        for module in original_model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                bn_abs_weghts = torch.cat([bn_abs_weghts, module.weight.data.abs().clone()], 0)

        # 计算slim阈值
        bn_abs_weights_sorted, _ = torch.sort(bn_abs_weghts)
        if slim_percent == None:
            slim_percent = self.slim_percent
        threshold_index = int(len(bn_abs_weghts) * slim_percent)
        self.threshold = bn_abs_weights_sorted[threshold_index]
        print('{:<30}  {:0.4e}'.format('==> slim threshold: ', self.threshold))

        # 计算slim之后的模型结构
        slimmed_num = 0
        cfg = [] # 每层mask非零元素和，即新模型各层通道数列表
        cfg_mask = [] # 每层mask
        for layer_index, module in enumerate(original_model.modules()):
            if isinstance(module, torch.nn.BatchNorm2d):
                weight_copy = module.weight.data.clone()
                mask = weight_copy.abs().gt(self.threshold).float().to(self.device) # torch.gt(a, b): a>b为1否则为0
                if torch.sum(mask) == 0:
                    error_str = 'Slim Error: layer' + str(layer_index) + ": " + module._get_name() + ': remain_out_channels = 0! turn down the slim_percent!'
                    # print(error_str)
                    raise Exception(error_str)


                # print(weight_copy)##############################################



                slimmed_num += (mask.shape[0] - torch.sum(mask))
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())
                print('layer index: {:3d} \t total channel: {:4d} \t remaining channel: {:4d}'.
                    format(layer_index, mask.shape[0], int(torch.sum(mask))))
            elif isinstance(module, torch.nn.MaxPool2d):
                cfg.append('M')
        print(cfg)
        # print(cfg_mask)
        
        # 构建新model
        print('{:<30}  {:<8}'.format('==> creating new model: ', self.arch))
        self.slimmed_model = models.__dict__[self.arch](cfg=cfg, num_classes=self.num_classes) # 根据cfg构建新的model
        if len(self.gpu_idx_list) > 1:
            self.slimmed_model = torch.nn.DataParallel(self.slimmed_model, device_ids=self.gpu_idx_list)
        self.slimmed_model.to(self.device) # 模型转移到设备上
        self.slim_ratio = slimmed_num/len(bn_abs_weghts)
        print('{:<30}  {:.4f}%'.format('==> slim ratio: ', self.slim_ratio*100))
        # print(self.slimmed_model)
        # torch.save(self.slimmed_model, 'slimmed_model.pth')
        # exit(0)

        # 将参数复制到新模型
        layer_id_in_cfg = 0 # 用来更新mask
        conv_count = 0 # 用来标记conv序号，来判断卷积层位置以判断是否剪输入输出通道
        conv_in_channels_mask = torch.ones(3)
        conv_out_channels_mask = cfg_mask[layer_id_in_cfg]
        # 其实就是block之间的通道不能剪，用channel selection代替，其他都可以剪

        original_modules = list(original_model.modules())
        slimmed_modules = list(self.slimmed_model.modules())
        # 此处不能用下面表达式，因为它给出的module顺序瞎几把乱出
        for layer_index, [module0, module1] in enumerate(zip(original_model.modules(), self.slimmed_model.modules())):
            if isinstance(module0, shortcut_package):
                # 虽然也属于conv2d，不过这儿先拿出来好处理
                if isinstance(module0, torch.nn.Conv2d):
                    module1.weight.data = module0.weight.data.clone()
            elif isinstance(module0, torch.nn.Conv2d):
                if conv_count == 0: # 整个网络第一层卷积不剪
                    module1.weight.data = module0.weight.data.clone()
                    conv_count += 1
                elif isinstance(original_modules[layer_index-1], torch.nn.ReLU) and not isinstance(original_modules[layer_index+1], shortcut_package):
                    # 上一层是relu, 下一层不是shortcut, bottleneck内部的卷积
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
                    # 剪
                    w = module0.weight.data[:, idx0, :, :].clone() # 剪输入通道
                    w = w[idx1, :, :, :].clone() # 剪输出通道
                    module1.weight.data = w.clone()
                elif isinstance(original_modules[layer_index+1], shortcut_package):
                    # block最后一个卷积，输出通道数要和原模型一样，只剪输入通道
                    idx0 = np.squeeze(np.argwhere(np.asarray(conv_in_channels_mask.cpu().numpy()))) # 从掩模计算出需要保留的权重下标
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    w = module0.weight.data[:, idx0, :, :].clone() # 剪输入通道
                    module1.weight.data = w.clone()
            elif isinstance(module0, torch.nn.BatchNorm2d):
                idx1 = np.squeeze(np.argwhere(np.asarray(conv_out_channels_mask.cpu().numpy()))) # np.argwhere()返回非零元素下标
                if isinstance(original_modules[layer_index+1], channel_selection):
                    # 下层是channel_selection，也就是说这是bottleneck的第一层，不剪
                    module1.weight.data = module0.weight.data.clone()
                    module1.bias.data = module0.bias.data.clone()
                    module1.running_mean = module0.running_mean.clone()
                    module1.running_var = module0.running_var.clone()
                    # 我们要剪channel_selection层
                    module2 = slimmed_modules[layer_index+1]
                    module2.indexes.data.zero_() # 全归零
                    module2.indexes.data[idx1] = 1.0 # 选择性置一
                else:
                    # 将应保留的权值复制到新模型中
                    module1.weight.data = module0.weight.data[idx1].clone()
                    module1.bias.data = module0.bias.data[idx1].clone()
                    module1.running_mean = module0.running_mean[idx1].clone()
                    module1.running_var = module0.running_var[idx1].clone()
                # 下一层
                layer_id_in_cfg += 1
                conv_in_channels_mask = conv_out_channels_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    conv_out_channels_mask = cfg_mask[layer_id_in_cfg]
            elif isinstance(module0, torch.nn.Linear):
                # 调整全连接层输入通道数
                idx0 = np.squeeze(np.argwhere(np.asarray(conv_in_channels_mask.cpu().numpy())))
                module1.weight.data = module0.weight.data[:, idx0].clone() # module0.weight.data[输出通道数，输入通道数]
                # print("full connection: in channels: {:d}, out channels: {:d}".format(idx0.shape[0], module0.weight.data.shape[0]))
                break # 仅调整第一层全连接层

        return self.slimmed_model, cfg, self.slim_ratio


    def vgg_slim(self, slim_percent=None):
        """
        删掉低于阈值的bn层，构建新的模型，可以降低params和flops
        Args:
            slim_percent: 设置slim阈值，若为None则从self.slim_percent获取slim阈值
                Default: None
        """
        original_model = copy.deepcopy(self.model).to(self.device)
        # 提取所有bn weights
        bn_abs_weghts = torch.zeros(0).to(self.device)
        for module in original_model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                bn_abs_weghts = torch.cat([bn_abs_weghts, module.weight.data.abs().clone()], 0)

        # 计算slim阈值
        bn_abs_weights_sorted, _ = torch.sort(bn_abs_weghts)
        if slim_percent == None:
            slim_percent = self.slim_percent
        threshold_index = int(len(bn_abs_weghts) * slim_percent)
        self.threshold = bn_abs_weights_sorted[threshold_index]
        print('{:<30}  {:0.4e}'.format('==> slim threshold: ', self.threshold))

        # 计算slim之后的模型结构
        slimmed_num = 0
        cfg = [] # 每层mask非零元素和，即新模型各层通道数列表
        cfg_mask = [] # 每层mask
        for layer_index, module in enumerate(original_model.modules()):
            if isinstance(module, torch.nn.BatchNorm2d):
                weight_copy = module.weight.data.clone()
                mask = weight_copy.abs().gt(self.threshold).float().to(self.device) # torch.gt(a, b): a>b为1否则为0
                if torch.sum(mask) == 0:
                    error_str = 'Slim Error: layer' + str(layer_index) + ": " + module._get_name() + ': remain_out_channels = 0! turn down the slim_percent!'
                    # print(error_str)
                    raise Exception(error_str)


                # print(weight_copy)##############################################



                slimmed_num += (mask.shape[0] - torch.sum(mask))
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())
                # print('layer index: {:3d} \t total channel: {:4d} \t remaining channel: {:4d}'.
                #     format(layer_index, mask.shape[0], int(torch.sum(mask))))
            elif isinstance(module, torch.nn.MaxPool2d):
                cfg.append('M')
        print(cfg)
        # print(cfg_mask)
        
        # 构建新model
        print('{:<30}  {:<8}'.format('==> creating new model: ', self.arch))
        self.slimmed_model = models.__dict__[self.arch](cfg=cfg, num_classes=self.num_classes) # 根据cfg构建新的model
        if len(self.gpu_idx_list) > 1:
            self.slimmed_model = torch.nn.DataParallel(self.slimmed_model, device_ids=self.gpu_idx_list)
        self.slimmed_model.to(self.device) # 模型转移到设备上
        self.slim_ratio = slimmed_num/len(bn_abs_weghts)
        print('{:<30}  {:.4f}%'.format('==> slim ratio: ', self.slim_ratio*100))
        # print(self.slimmed_model)

        # torch.save(self.slimmed_model, 'slimmed_model.pth')
        # exit(0)

        # 将参数复制到新模型
        layer_id_in_cfg = 0
        conv_in_channels_mask = torch.ones(3)
        conv_out_channels_mask = cfg_mask[layer_id_in_cfg]
        for [module0, module1] in zip(original_model.modules(), self.slimmed_model.modules()):
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

        return self.slimmed_model, cfg, self.slim_ratio
