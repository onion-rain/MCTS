# import ssl
# #全局取消证书验证
# ssl._create_default_https_context = ssl._create_unverified_context

import torch
# from tqdm import tqdm
from torch.nn import functional as F
# import torchvision as tv
import numpy as np
import time
import os
import random
import datetime
import argparse
import sys

import models
from traintest import *
from utils import *
from prune.meta_searcher import PrunednetSearcher

import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
# fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh
# sys.stdout = open('logs/log.txt','w')

class MetaSearcher(object):

    def __init__(self, **kwargs):

        self.config = Configuration()
        self.config.update_config(kwargs) # 解析参数更新默认配置
        sys.stdout = Logger(self.config.log_path)
        print("| ----------------- Initializing meta searcher ----------------- |")
        if self.config.check_config(): raise # 检测路径、设备是否存在
        print('{:<30}  {:<8}'.format('==> num_workers: ', self.config.num_workers))
        print('{:<30}  {:<8}'.format('==> batch_size: ', self.config.batch_size))

        # 更新一些默认标志
        self.start_epoch = 0
        self.best_acc1 = 0
        self.checkpoint = None
        vis_clear = True

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
        
        # criterion and optimizer
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion_smooth = CrossEntropyLabelSmooth(self.num_classes, 0.1)

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self.optimizer,
            milestones=[self.config.max_epoch*0.25, self.config.max_epoch*0.75], 
            gamma=0.1,
            last_epoch=self.start_epoch-1, # 我的训练epoch从1开始，而pytorch要通过当前epoch是否等于0判断是不是resume
        )

        # resume
        assert checkpoint is not None
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

        # visdom
        self.vis, self.vis_interval = visdom_init(self.config, self.suffix, vis_clear)

        # step6: searcher
        self.searcher = PrunednetSearcher(
            self.model, 
            self.train_dataloader, 
            self.val_dataloader,
            self.criterion_smooth, 
            self.device, 
            self.vis, 
            self.config.max_flops,
        )

    def run(self):
        print("")
        self.searcher.search()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='meta Prunednet search')
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH', default='resnet_meta',
                        choices=models.ALL_MODEL_NAMES,
                        help='model architecture: ' +
                        ' | '.join(name for name in models.ALL_MODEL_NAMES) +
                        ' (default: resnet_meta)')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='training dataset (default: imagenet)')
    parser.add_argument('--workers', type=int, default=20, metavar='N',
                        help='number of data loading workers (default: 20)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=32, metavar='N',
                        help='number of epochs to train (default: 32)')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-1, 
                        metavar='LR', help='initial learning rate (default: 1e-1)')
    parser.add_argument('--weight-decay', '-wd', dest='weight_decay', type=float,
                        default=1e-4, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='training GPU index(default:"0",which means use GPU0')
    parser.add_argument('--deterministic', '--det', action='store_true',
                    help='Ensure deterministic execution for re-producible results.')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    # parser.add_argument('--valuate', action='store_true',
    #                     help='valuate each training epoch')
    parser.add_argument('--resume', dest='resume_path', type=str, default='',
                        metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--refine', action='store_true',
                        help='refine from pruned model, use construction to build the model')
    parser.add_argument('--usr-suffix', type=str, default='',
                        help='usr_suffix(default:"", means no suffix)')
    parser.add_argument('--log-path', type=str, default='logs/log.txt',
                        help='default: logs/log.txt')

    parser.add_argument('--visdom', dest='visdom', action='store_true',
                        help='visualize the training process using visdom')
    parser.add_argument('--vis-env', type=str, default='', metavar='ENV',
                        help='visdom environment (default: "", which means env is automatically set to args.dataset + "_" + args.arch)')
    parser.add_argument('--vis-legend', type=str, default='', metavar='LEGEND',
                        help='refine from pruned model (default: "", which means env is automatically set to args.arch)')
    parser.add_argument('--vis-interval', type=int, default=50, metavar='N',
                        help='visdom plot interval batchs (default: 50)')

    parser.add_argument('--flops', dest='max_flops', type=float, default=800, 
                        metavar='Flops', help='The maximum amount of computation that can be tolerated(default: 800)')
    args = parser.parse_args()

    # debug用
    # args.workers = 0


    MetaSearcher = MetaSearcher(
        arch=args.arch,
        dataset=args.dataset,
        num_workers = args.workers, # 使用多进程加载数据
        batch_size=args.batch_size,
        max_epoch=args.epochs,
        lr=args.lr,
        gpu_idx = args.gpu, # choose gpu
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        deterministic=args.deterministic,
        # valuate=args.valuate,
        resume_path=args.resume_path,
        refine=args.refine,
        usr_suffix=args.usr_suffix,

        visdom = args.visdom, # 使用visdom可视化训练过程
        vis_env=args.vis_env,
        vis_legend=args.vis_legend,
        vis_interval=args.vis_interval,

        max_flops=args.max_flops,
    )
    MetaSearcher.run()
    print("end")

