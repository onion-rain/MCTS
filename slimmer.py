# import ssl
# #全局取消证书验证
# ssl._create_default_https_context = ssl._create_unverified_context

import torch
from utils.visualize import Visualizer
from tqdm import tqdm
from torch.nn import functional as F
import torchvision as tv
import time
import os
import random
import numpy as np
import copy

from config import Configuration
import models
from utils import accuracy, print_model_parameters, AverageMeter, get_path

class Slimmer(object):
    """
    导入的模型必须有BN层，
    并且事先进行稀疏训练，
    并且全连接层前要将左右特征图池化为1x1，即最终卷积输出的通道数即为全连接层输入通道数
    """
    def __init__(self, config=None, vis=None, **kwargs):
        print("| ----------------- Initializing Slimmer ----------------- |")
        if config == None:
            self.config = Configuration()
            self.config.update_config(kwargs) # 解析参数更新默认配置
            if self.config.check_config(): raise # 检测路径、设备是否存在
        else: self.config = config

        if len(self.config.gpu_idx_list) > 0:
            self.device = torch.device('cuda:{}'.format(min(self.config.gpu_idx_list))) # 起始gpu序号
            print('{:<30}  {:<8}'.format('==> chosen GPU index: ', self.config.gpu_idx))
        else:
            self.device = torch.device('cpu')
            print('{:<30}  {:<8}'.format('==> device: ', 'CPU'))

        # Random Seed
        if self.config.random_seed is None:
            self.config.random_seed = random.randint(1, 10000)
        random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

        #data
        print('{:<30}  {:<8}'.format('==> Preparing dataset: ', self.config.dataset))
        if self.config.dataset is "cifar10":
            self.num_classes = 10
        elif self.config.dataset is "cifar100":
            self.num_classes = 100
        elif self.config.dataset is "imagenet":
            self.num_classes = 1000
        else: 
            print("Dataset undefined")
            exit()

        # load model
        print('{:<30}  {:<8}'.format('==> creating model: ', self.config.arch))
        print('{:<30}  {:<8}'.format('==> loading model: ', self.config.load_model_path if self.config.load_model_path != None else 'None'))
        self.model = models.__dict__[self.config.arch](num_classes=self.num_classes) # 从models中获取名为config.arch的model
        if len(self.config.gpu_idx_list) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config.gpu_idx_list)
        self.model.to(self.device) # 模型转移到设备上
        if self.config.load_model_path: # 加载目标模型参数
            # self.model.load_state_dict(torch.load(self.config.load_model_path, map_location=self.device))
            checkpoint = torch.load(self.config.load_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        # print(self.model)
        # print_model_parameters(self.model)


    def run(self):

        self.slim()

        # # save last model
        # if self.config.save_model_path is None:
        #     self.config.save_model_path = "slimmed_" + self.config.load_model_path
        # if len(self.config.gpu_idx_list) > 1:
        #     torch.save(self.model.module.state_dict(), self.config.save_model_path)
        # else: torch.save(self.model.state_dict(), self.config.save_model_path)

    def simple_slim(self, slim_percent=None):
        """
        直接从原模型中将小于阈值的bn层weights置零，不会改变params和flops
        Args:
            slim_percent: 设置slim阈值，若为None则从self.config.slim_percent获取slim阈值
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
            slim_percent = self.config.slim_percent
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
                    print(error_str)
                    raise
                slimmed_num += (mask.shape[0] - torch.sum(mask))
                module.weight.data.mul_(mask)
                module.bias.data.mul_(mask)
                # print('layer index: {:3d} \t total channel: {:4d} \t remaining channel: {:4d}'.
                #     format(layer_index, mask.shape[0], int(torch.sum(mask))))

        self.slim_ratio = slimmed_num/len(bn_abs_weghts)
        print('{:<30}  {:.4f}%'.format('==> slim ratio: ', self.slim_ratio*100))


    def slim(self, slim_percent=None):
        """
        删掉低于阈值的bn层，构建新的模型，可以降低params和flops
        Args:
            slim_percent: 设置slim阈值，若为None则从self.config.slim_percent获取slim阈值
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
            slim_percent = self.config.slim_percent
        threshold_index = int(len(bn_abs_weghts) * slim_percent)
        self.threshold = bn_abs_weights_sorted[threshold_index]
        print('{:<30}  {:0.4e}'.format('==> slim threshold: ', self.threshold))

        # 计算slim之后的模型结构
        slimmed_num = 0
        cfg_structure = [] # 每层mask非零元素和，即新模型各层通道数列表
        cfg_mask = [] # 每层mask
        for layer_index, module in enumerate(original_model.modules()):
            if isinstance(module, torch.nn.BatchNorm2d):
                weight_copy = module.weight.data.clone()
                mask = weight_copy.abs().gt(self.threshold).float().to(self.device) # torch.gt(a, b): a>b为1否则为0
                if torch.sum(mask) == 0:
                    error_str = 'Slim Error: layer' + str(layer_index) + ": " + module._get_name() + ': remain_out_channels = 0! turn down the slim_percent!'
                    print(error_str)
                    raise


                # print(weight_copy)##############################################



                slimmed_num += (mask.shape[0] - torch.sum(mask))
                cfg_structure.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())
                # print('layer index: {:3d} \t total channel: {:4d} \t remaining channel: {:4d}'.
                #     format(layer_index, mask.shape[0], int(torch.sum(mask))))
            elif isinstance(module, torch.nn.MaxPool2d):
                cfg_structure.append('M')
        print(cfg_structure)
        # print(cfg_mask)
        
        # 构建新model
        print('{:<30}  {:<8}'.format('==> creating new model: ', self.config.arch))
        self.slimmed_model = models.__dict__[self.config.arch](structure=cfg_structure, num_classes=self.num_classes) # 根据cfg构建新的model
        if len(self.config.gpu_idx_list) > 1:
            self.slimmed_model = torch.nn.DataParallel(self.slimmed_model, device_ids=self.config.gpu_idx_list)
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

        return cfg_structure


if __name__ == "__main__":
    slimmer = Slimmer(
        model='nin',
        dataset="cifar10",
        gpu_idx = "5", # choose gpu
        random_seed=2,
        load_model_path="checkpoints/cifar10_nin_epoch123_acc90.83.pth",
        # num_workers = 5, # 使用多进程加载数据
        slim_percent=0.5,
    )
    slimmer.run()
    print("end")