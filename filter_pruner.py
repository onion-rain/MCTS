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
from prune.filter_pruner import FilterPruner
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

        # step1: data
        _, self.val_dataloader, self.num_classes = get_dataloader(self.config)

        # step2: model
        print('{:<30}  {:<8}'.format('==> creating arch: ', self.config.arch))
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

        self.vis = None

        # step5: pruner
        self.pruner = FilterPruner(
            model=self.model,
            device=self.device,
            arch=self.config.arch,
            prune_percent=[self.config.prune_percent],
            # target_cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256],
            p=1,
        )

        # step6: valuator
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
        print("| -----------------simple pruning model ------------------ |")
        self.pruner.simple_prune()
        self.valuator.test(self.pruner.simple_pruned_model)
        print_flops_params(self.valuator.model, self.config.dataset)

        print("")
        print("| -------------------- pruning model -------------------- |")
        self.pruner.prune()
        self.valuator.test(self.pruner.pruned_model)
        print_flops_params(self.valuator.model, self.config.dataset)

        # # save pruned model
        # name = ('weight_pruned' + str(self.config.prune_percent) 
        #         + '_' + self.config.dataset 
        #         + "_" + self.config.arch
        #         + self.suffix)
        # if len(self.config.gpu_idx_list) > 1:
        #     state_dict = self.pruned_model.module.state_dict()
        # else: state_dict = self.pruned_model.state_dict()
        # path = save_checkpoint({
        #     # 'cfg': cfg,
        #     'ratio': self.prune_ratio,
        #     'model_state_dict': state_dict,
        #     'best_acc1': self.valuator.top1_acc.avg,
        # }, file_root='checkpoints/weight_pruned/', file_name=name)
        # print('{:<30}  {}'.format('==> pruned model save path: ', path))



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
    parser.add_argument('--prune-percent', type=float, default=0.5, 
                        help='percentage of weight to prune')
    parser.add_argument('--prune-object', type=str, metavar='object', default='all',
                        help='prune object: "conv", "fc", "all"(default: , "all")')

    args = parser.parse_args()


    pruner = Pruner(
        arch=args.arch,
        dataset=args.dataset,
        num_workers = args.workers, # 使用多进程加载数据
        gpu_idx = args.gpu, # choose gpu
        resume_path=args.resume_path,
        refine=args.refine,

        prune_percent=args.prune_percent,
        prune_object=args.prune_object,
    )
    pruner.run()
    print("end")