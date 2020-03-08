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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

class Pruner(object):
    """
    导入的模型必须有BN层，
    并且事先进行稀疏训练，
    并且全连接层前要将左右特征图池化为1x1，即最终卷积输出的通道数即为全连接层输入通道数
    """
    def __init__(self, **kwargs):

        print("| ----------------- Initializing Pruner ----------------- |")

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
            self.prune = self.prune_vgg
        elif self.config.arch.startswith("resnet"):
            self.prune = self.prune_resnet
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

        print("")
        print("| -------------------- pruning model -------------------- |")
        cfg = self.prune()
        self.valuator.test(self.pruned_model)
        print_flops_params(self.valuator.model, self.config.dataset)

        # save pruned model
        name = ('pruned'
                + '_' + self.config.dataset 
                + "_" + self.config.arch
                + self.suffix)
        if len(self.config.gpu_idx_list) > 1:
            state_dict = self.pruned_model.module.state_dict()
        else: state_dict = self.pruned_model.state_dict()
        path = save_checkpoint({
            'cfg': cfg,
            'ratio': self.prune_ratio,
            'model_state_dict': state_dict,
            'best_acc1': self.valuator.top1_acc.avg,
        }, file_root='checkpoints/l1_filter_pruned/', file_name=name)
        print('{:<30}  {}'.format('==> pruned model save path: ', path))


    def prune_vgg(self):
        """
        删掉低于阈值的bn层，构建新的模型，可以降低params和flops
        """
        original_model = copy.deepcopy(self.model).to(self.device)
        # print(self.original_model)
        original_filters_num = 0
        pruned_filters_num = 0
        # cfg 取自 Pruning Filters For Efficient ConvNets Table2中所示结构
        cfg = [32, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256]
        cfg_mask = []
        # 提取所有bn weights
        layer_id = 0
        for module in original_model.modules():
            if isinstance(module, torch.nn.Conv2d):
                filters_num = module.weight.data.shape[0] # 当前层filter总数
                filters_weights = module.weight.data.clone().cpu().numpy() # filter_weight.shape: [filters_num, c, h, w]
                filters_weights_L1 = np.sum(np.fabs(filters_weights), axis=(1, 2, 3)) # 计算所有卷积核的L1范数
                l2b_filters_index = np.argsort(filters_weights_L1) # 从小到大排序
                keep_filters_index = l2b_filters_index[::-1][:cfg[layer_id]] # 筛选保留的filter下标
                
                mask = torch.zeros(filters_num)
                mask[keep_filters_index.tolist()] = 1
                cfg_mask.append(mask)

                original_filters_num += filters_num
                pruned_filters_num += (filters_num - torch.sum(mask).numpy())
                layer_id += 1 # vgg结构每个conv后面都有一个bn层，所以可以用conv代替bn层计数
            elif isinstance(module, torch.nn.MaxPool2d):
                layer_id += 1
        print(cfg)
        # print(cfg_mask)
        self.prune_ratio = pruned_filters_num/original_filters_num
        print('{:<30}  {:.4f}%'.format('==> prune ratio: ', self.prune_ratio*100))

        # 构建新model
        print('{:<30}  {:<8}'.format('==> creating new model: ', self.config.arch))
        self.pruned_model = models.__dict__[self.config.arch](cfg=cfg, num_classes=self.num_classes) # 根据cfg构建新的model
        if len(self.config.gpu_idx_list) > 1:
            self.pruned_model = torch.nn.DataParallel(self.pruned_model, device_ids=self.config.gpu_idx_list)
        self.pruned_model.to(self.device) # 模型转移到设备上
        # print(self.pruned_model)

        # torch.save(self.pruned_model, 'pruned_model.pth')
        # exit(0)
        self.weight_recover_vgg(cfg_mask, original_model, self.pruned_model)
        return cfg


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



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='network pruner')
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

    args = parser.parse_args()


    pruner = Pruner(
        arch=args.arch,
        dataset=args.dataset,
        num_workers = args.workers, # 使用多进程加载数据
        gpu_idx = args.gpu, # choose gpu
        resume_path=args.resume_path,
        refine=args.refine,
    )
    pruner.run()
    print("end")
