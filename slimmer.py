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
import argparse

from tester import Tester
from config import Configuration
import models
from utils import *
from models.cifar.utils import channel_selection, shortcut_package

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

class Slimmer(object):
    """
    导入的模型必须有BN层，
    并且事先进行稀疏训练，
    并且全连接层前要将左右特征图池化为1x1，即最终卷积输出的通道数即为全连接层输入通道数
    """
    def __init__(self, **kwargs):

        print("| ----------------- Initializing Slimmer ----------------- |")

        self.config = Configuration()
        self.config.update_config(kwargs) # 解析参数更新默认配置
        if self.config.check_config(): raise # 检测路径、设备是否存在
        print('{:<30}  {:<8}'.format('==> num_workers: ', self.config.num_workers))

        self.suffix = get_suffix(self.config)
        print('{:<30}  {:<8}'.format('==> suffix: ', self.suffix))

        # 更新一些默认标志
        self.best_acc1 = 0
        self.checkpoint = None

        # device
        if len(self.config.gpu_idx_list) > 0:
            self.device = torch.device('cuda:{}'.format(min(self.config.gpu_idx_list))) # 起始gpu序号
            print('{:<30}  {:<8}'.format('==> chosen GPU index: ', self.config.gpu_idx))
        else:
            self.device = torch.device('cpu')
            print('{:<30}  {:<8}'.format('==> device: ', 'CPU'))

        # Random Seed 
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(self.config.random_seed)
        torch.backends.cudnn.deterministic = True

        # step1: data
        _, self.val_dataloader, self.num_classes = get_dataloader(self.config)

        # step2: model
        print('{:<30}  {:<8}'.format('==> creating arch: ', self.config.arch))
        if self.config.arch.startswith("vgg"):
            self.slim = self.vgg_slim
        elif self.config.arch.startswith("resnet"):
            self.slim = self.resnet_slim
        cfg = None
        if self.config.resume_path != '': # 断点续练hhh
            checkpoint = torch.load(self.config.resume_path, map_location=self.device)
            print('{:<30}  {:<8}'.format('==> resuming from: ', self.config.resume_path))
            if self.config.refine: # 根据cfg加载剪枝后的模型结构
                cfg=checkpoint['cfg']
                print(cfg)
        else: 
            print("你剪枝不加载模型剪锤子??")
            exit(0)
        self.model = models.__dict__[self.config.arch](cfg=cfg, num_classes=self.num_classes) # 从models中获取名为config.model的model
        if len(self.config.gpu_idx_list) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config.gpu_idx_list)
        self.model.to(self.device) # 模型转移到设备上
        # if checkpoint is not None:
        self.best_acc1 = checkpoint['best_acc1']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("{:<30}  {:<8}".format('==> checkpoint trained epoch: ', checkpoint['epoch']))
        print("{:<30}  {:<8}".format('==> checkpoint best acc1: ', checkpoint['best_acc1']))


        # print(self.model)
        # exit(0)

        # step6: valuator
        self.vis = None
        val_config_dic = {
            'model': self.model,
            'dataloader': self.val_dataloader,
            'device': self.device,
            'vis': self.vis,
            'seed': self.config.random_seed
        }
        self.valuator = Tester(val_config_dic)


    def run(self):

        print("")
        print("| -------------------- original model -------------------- |")
        self.valuator.test(self.model)
        print_flops_params(self.valuator.model, self.config.dataset)
        # print_model_parameters(self.valuator.model)

        # print("")
        # print("| ----------------- simple slimming model ---------------- |")
        # self.simple_slim()
        # self.valuator.test(self.simple_slimmed_model)
        # print_flops_params(self.valuator.model, self.config.dataset)
        # # print_nonzeros(self.valuator.model)

        # # save slimmed model
        # name = ('simpel_slimmed_ratio' + str(self.config.slim_percent)
        #         + '_' + self.config.dataset 
        #         + "_" + self.config.arch
        #         + self.suffix)
        # if len(self.config.gpu_idx_list) > 1:
        #     state_dict = self.simple_slimmed_model.module.state_dict()
        # else: state_dict = self.simple_slimmed_model.state_dict()
        # path = save_checkpoint({
        #     'ratio': self.slim_ratio,
        #     'model_state_dict': state_dict,
        #     'best_acc1': self.valuator.top1_acc.avg,
        # }, file_root='checkpoints/', file_name=name)
        # print('{:<30}  {}'.format('==> simple slimmed model save path: ', path))

        print("")
        print("| -------------------- slimming model -------------------- |")
        cfg = self.slim()
        self.valuator.test(self.slimmed_model)
        print_flops_params(self.valuator.model, self.config.dataset)

        # save slimmed model
        name = ('slimmed_ratio' + str(self.config.slim_percent)
                + '_' + self.config.dataset
                + "_" + self.config.arch
                + self.suffix)
        if len(self.config.gpu_idx_list) > 1:
            state_dict = self.slimmed_model.module.state_dict()
        else: state_dict = self.slimmed_model.state_dict()
        path = save_checkpoint({
            'cfg': cfg,
            'ratio': self.slim_ratio,
            'model_state_dict': state_dict,
            'best_acc1': self.valuator.top1_acc.avg,
        }, file_root='checkpoints/slimmed/', file_name=name)
        print('{:<30}  {}'.format('==> slimmed model save path: ', path))


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


    def resnet_slim(self, slim_percent=None):
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
        cfg = [] # 每层mask非零元素和，即新模型各层通道数列表
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
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())
                print('layer index: {:3d} \t total channel: {:4d} \t remaining channel: {:4d}'.
                    format(layer_index, mask.shape[0], int(torch.sum(mask))))
            elif isinstance(module, torch.nn.MaxPool2d):
                cfg.append('M')
        print(cfg)
        # print(cfg_mask)
        
        # 构建新model
        print('{:<30}  {:<8}'.format('==> creating new model: ', self.config.arch))
        self.slimmed_model = models.__dict__[self.config.arch](cfg=cfg, num_classes=self.num_classes) # 根据cfg构建新的model
        if len(self.config.gpu_idx_list) > 1:
            self.slimmed_model = torch.nn.DataParallel(self.slimmed_model, device_ids=self.config.gpu_idx_list)
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

        return cfg


    def vgg_slim(self, slim_percent=None):
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
        cfg = [] # 每层mask非零元素和，即新模型各层通道数列表
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
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())
                # print('layer index: {:3d} \t total channel: {:4d} \t remaining channel: {:4d}'.
                #     format(layer_index, mask.shape[0], int(torch.sum(mask))))
            elif isinstance(module, torch.nn.MaxPool2d):
                cfg.append('M')
        print(cfg)
        # print(cfg_mask)
        
        # 构建新model
        print('{:<30}  {:<8}'.format('==> creating new model: ', self.config.arch))
        self.slimmed_model = models.__dict__[self.config.arch](cfg=cfg, num_classes=self.num_classes) # 根据cfg构建新的model
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

        return cfg


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='network slimmer')
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH', default='vgg19_bn_cifar',
                        choices=models.ALL_MODEL_NAMES,
                        help='model architecture: ' +
                        ' | '.join(name for name in models.ALL_MODEL_NAMES) +
                        ' (default: resnet18)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--workers', type=int, default=10, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--gpu', type=str, default='0', metavar='gpu_idx',
                        help='training GPU index(default:"0",which means use GPU0')
    parser.add_argument('--resume', dest='resume_path', type=str, default='',
                        metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--refine', action='store_true',
                        help='refine from pruned model, use construction to build the model')

    parser.add_argument('--slim-percent', type=float, default=0.7, metavar='N',
                        help='slim percent(default: 0.7)')

    args = parser.parse_args()


    slimmer = Slimmer(
        arch=args.arch,
        dataset=args.dataset,
        num_workers = args.workers, # 使用多进程加载数据
        gpu_idx = args.gpu, # choose gpu
        resume_path=args.resume_path,
        refine=args.refine,

        slim_percent=args.slim_percent,
    )
    slimmer.run()
    print("end")

    # slimmer = slimmer(
    #     model='nin',
    #     dataset="cifar10",
    #     gpu_idx = "5", # choose gpu
    #     load_model_path="checkpoints/cifar10_nin_epoch123_acc90.83.pth",
    #     # num_workers = 5, # 使用多进程加载数据
    #     slim_percent=0.5,
    # )
    # slimmer.run()
    # print("end")