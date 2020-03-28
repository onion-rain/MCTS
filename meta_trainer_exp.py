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

import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"
# fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh

# train
# python meta_trainer_exp.py --arch resnet50_pruningnet --dataset imagenet --batch-size 100 --epochs 32 --gpu 3 --valuate --visdom

# retrain
# python meta_trainer_exp.py --arch resnet50_prunednet --dataset imagenet --search-resume checkpoints/MetaPruneSearch_31_resnet50_pruningnet_checkpoint.pth.tar --epochs 60 --gpu 3 --valuate --visdom --log-path logs/resnet50_prunednet_candidate0.txt --candidate 0



class MetaTrainer(object):
    """
    以model arch结尾字符区分pruningnet还是prunednet
    """
    def __init__(self, **kwargs):

        self.config = Configuration()
        self.config.update_config(kwargs) # 解析参数更新默认配置
        sys.stdout = Logger(self.config.log_path)
        print("| ----------------- Initializing meta trainer ----------------- |")
        assert self.config.check_config() == 0
        print('{:<30}  {:<8}'.format('==> num_workers: ', self.config.num_workers))
        print('{:<30}  {:<8}'.format('==> batch_size: ', self.config.batch_size))
        print('{:<30}  {:<8}'.format('==> max_epoch: ', self.config.max_epoch))
        print('{:<30}  {:<8}'.format('==> lr_scheduler milestones: ', str([self.config.max_epoch*0.25, self.config.max_epoch*0.5, self.config.max_epoch*0.75])))

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
        # data & model
        if self.config.arch.endswith('pruningnet'):
            self.train_dataloader, self.val_dataloader, self.num_classes = dataloader_div_init(self.config, val_num=50)
            self.model, self.cfg, checkpoint = model_init(self.config, self.device, self.num_classes)


        # TODO 简化！！！！！！！！！
        elif self.config.arch.endswith('prunednet'): 
            # data
            self.train_dataloader, self.val_dataloader, self.num_classes = dataloader_init(self.config)
            # model
            checkpoint = None
            self.gene = None
            if self.config.resume_path != '': # 断点续练hhh
                assert self.config.search_resume_path == '' # resume_path, search_resume_path二选一
                checkpoint = torch.load(self.config.resume_path, map_location=self.device)
                assert checkpoint['arch'] == self.config.arch
                print('{:<30}  {:<8}'.format('==> resuming from: ', self.config.resume_path))
                self.gene = checkpoint['gene']
            elif self.config.search_resume_path != '': # 从search结果中选取gene
                search_checkpoint = torch.load(self.config.search_resume_path, map_location=self.device)
                candidates = search_checkpoint['candidates']
                print('{:<30}  {:<8}'.format('==> candidate index: ', self.config.candidate_idx))
                self.gene = candidates[self.config.candidate_idx]
            print(self.gene)
            self.model = models.__dict__[self.config.arch](num_classes=self.num_classes, gene=self.gene).to(self.device)


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
        if checkpoint is not None:
            if 'model_state_dict' in checkpoint.keys():
                self.model.load_state_dict(checkpoint['model_state_dict'])
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

        # step6: trainer
        if self.config.arch.endswith('pruningnet'):
            self.trainer = PruningnetTrainer(
                self.model, 
                self.train_dataloader, 
                self.criterion_smooth, 
                self.optimizer, 
                self.device, 
                self.vis, 
                self.vis_interval,
                self.lr_scheduler,
            )
        elif self.config.arch.endswith('prunednet'):
            self.trainer = PrunednetTrainer(
                self.model, 
                self.train_dataloader, 
                self.criterion_smooth, 
                self.optimizer, 
                self.device, 
                self.vis, 
                self.vis_interval,
                self.lr_scheduler,
            )


        # step6: valuator
        self.valuator = None
        if self.config.valuate is True:
            if self.config.arch.endswith('pruningnet'):
                self.valuator = PruningnetTester(
                    dataloader=self.val_dataloader,
                    device=self.device,
                    criterion=self.criterion,
                    vis=self.vis,
                )
            elif self.config.arch.endswith('prunednet'):
                self.valuator = PrunednetTester(
                    dataloader=self.val_dataloader,
                    device=self.device,
                    criterion=self.criterion,
                    vis=self.vis,
                )


    def run(self):

        print("")
        start_time = datetime.datetime.now()
        name = (self.config.dataset + "_" + self.config.arch + self.suffix)
        print_flops_params(model=self.model, dataset=self.config.dataset)

        # initial test
        if self.valuator is not None:
            self.valuator.test(self.model, epoch=self.start_epoch-1)
        print_bar(start_time, self.config.arch, self.config.dataset)
        print("")
        for epoch in range(self.start_epoch, self.config.max_epoch):
            # train & valuate
            self.trainer.train(epoch=epoch)
            if self.valuator is not None:
                self.valuator.test(self.model, epoch=epoch)
            print_bar(start_time, self.config.arch, self.config.dataset)
            print("")
            
            # save checkpoint
            if self.valuator is not None:
                is_best = self.valuator.top1_acc.avg > self.best_acc1
                self.best_acc1 = max(self.valuator.top1_acc.avg, self.best_acc1)
            else:
                is_best = self.trainer.top1_acc.avg > self.best_acc1
                self.best_acc1 = max(self.top1_acc.avg, self.best_acc1)
            if len(self.config.gpu_idx_list) > 1:
                state_dict = self.model.module.state_dict()
            else: state_dict = self.model.state_dict()
            save_dict = {
                'arch': self.config.arch,
                'epoch': epoch,
                'model_state_dict': state_dict,
                'best_acc1': self.best_acc1,
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            if self.config.arch.endswith('prunednet'):
                save_dict['gene'] = self.gene
            save_checkpoint(save_dict, is_best=is_best, epoch=None, file_root='checkpoints/', file_name=name)
        print("{}{}".format("best_acc1: ", self.best_acc1))

   



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='meta Pruningnet train')

    add_trainer_arg_parser(parser)
    add_visdom_arg_parser(parser)
    
    # retrain
    parser.add_argument('--search-resume', dest='search_resume_path', type=str, default='',
                        metavar='PATH', help='path to latest checkpoint (default: '')')
    parser.add_argument('--candidate', dest='candidate_idx', type=int, default=0,
                        metavar='N', help='candidates index choose (default: 0(the best))')
    args = parser.parse_args()

    # debug用
    # args.workers = 0


    MetaTrainer = MetaTrainer(
        arch=args.arch,
        dataset=args.dataset,
        num_workers = args.workers,
        batch_size=args.batch_size,
        max_epoch=args.epochs,
        lr=args.lr,
        gpu_idx = args.gpu,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        deterministic=args.deterministic,
        valuate=args.valuate,
        resume_path=args.resume_path,
        refine=args.refine,
        usr_suffix=args.usr_suffix,
        log_path=args.log_path,
        # 使用visdom可视化训练过程
        visdom = args.visdom, 
        vis_env=args.vis_env,
        vis_legend=args.vis_legend,
        vis_interval=args.vis_interval,
        #retrain
        search_resume_path=args.search_resume_path, 
        candidate_idx=args.candidate_idx,
    )
    MetaTrainer.run()
    print("end")

