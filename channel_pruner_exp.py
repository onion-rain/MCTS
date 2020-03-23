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
import datetime

from tester import Tester
from prune.channel_pruner import *
import models
from utils import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

# python channel_pruner_exp.py --arch vgg16_bn_cifar --gpu 5 --resume checkpoints/baseline/cifar10_vgg16_bn_cifar_best.pth.tar --prune-percent 0.3 --lp-norm 2 

print(torch.__version__)

class Pruner(object):
    """
    TODO 由于trainer类大改，本类某些函数可能个已过期
    """

    def __init__(self, **kwargs):

        print("| ----------------- Initializing Pruner ----------------- |")

        self.config = Configuration()
        self.config.update_config(kwargs) # 解析参数更新默认配置
        if self.config.check_config(): raise # 检测路径、设备是否存在
        print('{:<30}  {:<8}'.format('==> num_workers: ', self.config.num_workers))
        print('{:<30}  {:<8}'.format('==> batch_size: ', self.config.batch_size))
        print('{:<30}  {:<8}'.format('==> max_epoch: ', self.config.max_epoch))
        print('{:<30}  {:<8}'.format('==> lr_scheduler milestones: ', str([self.config.max_epoch*0.5, self.config.max_epoch*0.75])))
        
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
        self.train_dataloader, self.val_dataloader, self.num_classes = dataloader_div_init(self.config, val_num=50)
        # model
        self.model, self.cfg, checkpoint = model_init(self.config, self.device, self.num_classes)
        
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
            if 'optimizer_state_dict' in checkpoint.keys():
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'best_acc1' in checkpoint.keys():
                self.best_acc1 = checkpoint['best_acc1']
                print("{:<30}  {:<8}".format('==> checkpoint best acc1: ', checkpoint['best_acc1']))
        
        self.vis = None

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

        # print("")
        # print("| -------------------- original model -------------------- |")
        # self.valuator.test(self.model)
        # print_flops_params(self.model, self.config.dataset)
        # # print_model_parameters(self.valuator.model)


        print("")
        print("| -------------------- pruning model -------------------- |")
        Thinet_prune(
            self.model, 
            self.config.prune_percent, 
            self.val_dataloader, 
            self.device, 
            self.config.method, 
            self.config.lp_norm
        )
        self.valuator.test(self.model)
        print_flops_params(self.model, self.config.dataset)

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
                        help='training GPU idx(default:"0",which means use GPU0')
    parser.add_argument('--resume', dest='resume_path', type=str, default='',
                        metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--refine', action='store_true',
                        help='refine from pruned model, use construction to build the model')
    parser.add_argument('--prune-percent', type=float, default=0.5, 
                        help='percentage of weight to prune(default: 0.5)')
    parser.add_argument('--lp-norm', '-lp', dest='lp_norm', type=int, default=2, 
                        help='the order of norm(default: 2)')
    parser.add_argument('--method', type=str, default='greedy', 
                        help="'greedy': select one contributed to the smallest next feature after another\
                            'lasso': select pruned channels by lasso regression\
                            'random': randomly select")

    args = parser.parse_args()


    pruner = Pruner(
        arch=args.arch,
        dataset=args.dataset,
        num_workers = args.workers, # 使用多进程加载数据
        gpu_idx = args.gpu, # choose gpu
        resume_path=args.resume_path,
        refine=args.refine,

        prune_percent=args.prune_percent,
        lp_norm=args.lp_norm,
        method=args.method,
    )
    pruner.run()
    print("end")
