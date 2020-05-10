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

import models
from utils import *
from traintest import *
from prune.weight_pruner import WeightPruner

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"


class Pruner(object):
    """
    导入的模型必须有BN层，
    并且全连接层前要将特征图池化为1x1
    """
    def __init__(self, **kwargs):

        self.config = Configuration()
        self.config.update_config(kwargs) # 解析参数更新默认配置
        assert self.config.check_config() == 0
        # sys.stdout = Logger(self.config.log_path)
        print("| ----------------- Initializing Pruner ----------------- |")
        print('{:<30}  {:<8}'.format('==> num_workers: ', self.config.num_workers))
        
        # 更新一些默认标志
        self.best_acc1 = 0
        self.checkpoint = None

        # suffix
        self.suffix = suffix_init(self.config)
        # device
        self.device = device_init(self.config)
        # Random Seed 
        seed_init(self.config)
        # data
        self.train_dataloader, self.val_dataloader, self.num_classes = dataloader_init(self.config)
        # model
        self.model, self.cfg, checkpoint = model_init(self.config, self.device, self.num_classes)
        
        self.criterion = torch.nn.CrossEntropyLoss()

        # resume
        if checkpoint is not None:
            if 'epoch' in checkpoint.keys():
                self.start_epoch = checkpoint['epoch'] + 1 # 保存的是已经训练完的epoch，因此start_epoch要+1
                print("{:<30}  {:<8}".format('==> checkpoint trained epoch: ', checkpoint['epoch']))
                if checkpoint['epoch'] > -1:
                    vis_clear = False # 不清空visdom已有visdom env里的内容
            if 'best_acc1' in checkpoint.keys():
                self.best_acc1 = checkpoint['best_acc1']
                print("{:<30}  {:<8}".format('==> checkpoint best acc1: ', checkpoint['best_acc1']))
            # self.checkpoint = checkpoint
        
        self.vis = None

        print()
        print('{:<30}  {:<8}'.format('==> prune_ratio: ', self.config.prune_percent))
        print('{:<30}  {:<8}'.format('==> prune_object: ', self.config.prune_object))

        # step5: pruner
        if self.config.prune_object == 'all':
            self.config.prune_object = ['conv', 'fc']
        self.pruner = WeightPruner(
            model=self.model, 
            prune_percent=self.config.prune_percent, 
            device=self.device, 
            prune_object=self.config.prune_object,
        )

        # step6: valuator
        self.valuator = Tester(
            dataloader=self.val_dataloader,
            device=self.device,
            criterion=self.criterion,
            vis=self.vis,
        )


    def run(self):

        print("")
        print("| -------------------- original model -------------------- |")
        print_flops_params(self.model, self.config.dataset)
        # self.valuator.test(self.model)

        print("")
        print("| -------------------- pruning model -------------------- |")
        self.pruned_model, pruned_ratio = self.pruner.prune(self.model)
        print_flops_params(self.pruned_model, self.config.dataset)
        self.valuator.test(self.pruned_model, epoch=0)

        self.best_acc1 = self.valuator.top1_acc.avg
        print("{}{}".format("best_acc1: ", self.best_acc1))
        save pruned model
        name = ('weight_pruned' + str(self.config.prune_percent) 
                + '_' + self.config.dataset 
                + "_" + self.config.arch
                + self.suffix)
        if len(self.config.gpu_idx_list) > 1:
            state_dict = self.pruned_model.module.state_dict()
        else: state_dict = self.pruned_model.state_dict()
        save_dict = {
            'arch': self.config.arch,
            'ratio': pruned_ratio,
            'model_state_dict': state_dict,
            'best_acc1': self.best_acc1,
        }
        if self.cfg is not None:
            save_dict['cfg'] = self.cfg
        checkpoint_path = save_checkpoint(save_dict, file_root='checkpoints/', file_name=name)
        print("{}{}".format("checkpoint_path: ", checkpoint_path))



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
    parser.add_argument('--gpu', type=str, default='',
                        help='training GPU index(default:"",which means use CPU')
    parser.add_argument('--resume', dest='resume_path', type=str, default='',
                        metavar='PATH', help='path to latest train checkpoint (default: '')')
    parser.add_argument('--refine', action='store_true',
                        help='refine from pruned model, use construction to build the model')
    parser.add_argument('--prune_percent', type=float, default=0.5, 
                        help='percentage of weight to prune')
    parser.add_argument('--prune_object', type=str, metavar='object', default='all',
                        help='prune object: "conv", "fc", "all"(default: , "all")')
    parser.add_argument('--log_path', type=str, default='logs/log.txt',
                        help='default: logs/log.txt')

    parser.add_argument('--json', type=str, default='',
                        help='json configuration file path(default: '')')

    args = parser.parse_args()

    if args.json != '':
        json_path = os.path.join(args.json)
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        params = Params_json(json_path)
        for key in params.dict:
            args.__dict__[key] = params.dict[key]

    pruner = Pruner(
        arch=args.arch,
        dataset=args.dataset,
        num_workers = args.workers, # 使用多进程加载数据
        gpu_idx = args.gpu, # choose gpu
        resume_path=args.resume_path,
        refine=args.refine,
        log_path=args.log_path,

        prune_percent=args.prune_percent,
        prune_object=args.prune_object,
    )
    pruner.run()
    print("end")
