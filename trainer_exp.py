# import ssl
# #全局取消证书验证
# ssl._create_default_https_context = ssl._create_unverified_context

import torch
from utils.visualize import Visualizer
from tqdm import tqdm
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
from utils import *
from traintest import *

import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)

# import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
# fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh
# ssh -L 8097:nico1:8097 tiaoban -p 2222

class TrainerExp(object):
    
    def __init__(self, **kwargs):

        self.config = Configuration()
        self.config.update_config(kwargs) # 解析参数更新默认配置
        assert self.config.check_config() == 0
        # sys.stdout = Logger(self.config.log_path)
        print("| ----------------- Initializing Trainer ----------------- |")
        print('{:<30}  {:<8}'.format('==> num_workers: ', self.config.num_workers))
        print('{:<30}  {:<8}'.format('==> batch_size: ', self.config.batch_size))
        print('{:<30}  {:<8}'.format('==> max_epoch: ', self.config.max_epoch))
        print('{:<30}  {:<8}'.format('==> lr_scheduler milestones: ', str([self.config.max_epoch*0.5, self.config.max_epoch*0.75])))

        # 更新一些默认标志
        self.start_epoch = 0
        self.best_acc1 = 0
        self.checkpoint = None
        vis_clear = True
        # if self.config.test_only:
        #     assert self.config.resume_path != ''

        # suffix
        self.suffix = suffix_init(self.config)
        # device
        self.device = device_init(self.config)
        # Random Seed 
        seed_init(self.config)
        # data
        # self.train_dataloader, self.val_dataloader, self.num_classes = dataloader_div_init(self.config, val_num=50)
        self.train_dataloader, self.val_dataloader, self.num_classes = dataloader_init(self.config)
        # model
        self.model, self.cfg, checkpoint = model_init(self.config, self.device, self.num_classes)
        
        # criterion and optimizer
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        # self.optimizer = torch.optim.Adam(
        #     params=self.model.parameters(),
        #     lr=self.config.lr,
        #     # momentum=self.config.momentum,
        #     weight_decay=self.config.weight_decay,
        # )
        
        self.criterion = torch.nn.CrossEntropyLoss()

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self.optimizer,
            milestones=[self.config.max_epoch*0.5, self.config.max_epoch*0.75], 
            gamma=0.1,
            last_epoch=self.start_epoch-1,
        )

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
                
        # visdom
        self.vis, self.vis_interval = visdom_init(self.config, self.suffix, vis_clear)

        if self.config.sr:
            self.sr = self.config.sr
        
        if self.config.sr:
            self.trainer = SlimmerTrainer(
                self.model, 
                self.train_dataloader, 
                self.criterion, 
                self.optimizer, 
                self.device, 
                self.sr_lambda, 
                self.vis, 
                self.vis_interval,
                self.lr_scheduler,
            )
        elif self.config.binarynet:
            self.trainer = BinaryTrainer(
                self.model, 
                self.train_dataloader, 
                self.criterion, 
                self.optimizer, 
                self.device, 
                self.vis, 
                self.vis_interval,
                self.lr_scheduler,
            )
        else:
            self.trainer = Trainer(
                self.model, 
                self.train_dataloader, 
                self.criterion, 
                self.optimizer, 
                self.device, 
                self.vis, 
                self.vis_interval,
                self.lr_scheduler,
            )

        # step6: valuator
        self.valuator = None
        if self.config.valuate == True:
            self.valuator = Tester(
                dataloader=self.val_dataloader,
                device=self.device,
                criterion=self.criterion,
                vis=self.vis,
            )
            
        if self.config.arch.endswith('dorefanet'):
            print()
            print('{:<30}  {:<8}'.format('==> acitvation_bits: ', self.config.a_bits))
            print('{:<30}  {:<8}'.format('==> weight_bits: ',     self.config.w_bits))
            print('{:<30}  {:<8}'.format('==> gradient_bits: ',   self.config.g_bits))
        

    def run(self):

        print("")
        start_time = datetime.datetime.now()
        name = (self.config.dataset + "_" + self.config.arch + self.suffix)
        # get_model_flops(self.model, dataset=self.config.dataset, pr=True)
        print_flops_params(model=self.model, dataset=self.config.dataset)

        # initial test
        if self.valuator is not None:
            self.valuator.test(self.model, epoch=self.start_epoch-1)
        print_bar(start_time, self.config.arch, self.config.dataset, self.best_acc1)
        if self.config.test_only:
            return
        print("")
        for epoch in range(self.start_epoch, self.config.max_epoch):
            # train & valuate
            self.trainer.train(epoch=epoch)
            if self.valuator is not None:
                self.valuator.test(self.model, epoch=epoch)
                
                is_best = self.valuator.top1_acc.avg > self.best_acc1
                self.best_acc1 = max(self.valuator.top1_acc.avg, self.best_acc1)
            else:
                is_best = self.trainer.top1_acc.avg > self.best_acc1
                self.best_acc1 = max(self.trainer.top1_acc.avg, self.best_acc1)

            print_bar(start_time, self.config.arch, self.config.dataset, self.best_acc1)
            print("")
            
            # save checkpoint
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
            if self.cfg is not None:
                save_dict['cfg'] = self.cfg
            checkpoint_path = save_checkpoint(save_dict, is_best=is_best, epoch=None, file_root='checkpoints/', file_name=name)
        
        print("{}{}".format("best_acc1: ", self.best_acc1))
        print("{}{}".format("checkpoint_path: ", checkpoint_path))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='network trainer')

    add_trainer_arg_parser(parser)

    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                        help='train with channel sparsity regularization')
    parser.add_argument('--srl', dest='sr_lambda', type=float, default=1e-4,
                        help='scale sparse rate (default: 1e-4), suggest 1e-4 for vgg, 1e-5 for resnet/densenet')

    parser.add_argument('--binarynet', dest='binarynet', action='store_true',
                        help='train binarynet')

    parser.add_argument('--a_bits', dest='a_bits', type=int, default=1,
                        help='activation quantization bits(default: 1)')
    parser.add_argument('--w_bits', dest='w_bits', type=int, default=1,
                        help='weight quantization bits(default: 1)')
    parser.add_argument('--g_bits', dest='g_bits', type=int, default=32,
                        help='gradient quantization bits(default: 32)')


    add_visdom_arg_parser(parser)
    args = parser.parse_args()

    # debug用
    # args.workers = 0


    trainer_exp = TrainerExp(
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
        valuate=args.valuate,
        resume_path=args.resume_path,
        refine=args.refine,
        usr_suffix=args.usr_suffix,
        log_path=args.log_path,
        test_only=args.test_only,

        sr=args.sr,
        sr_lambda=args.sr_lambda,

        binarynet=args.binarynet,

        a_bits=args.a_bits,
        w_bits=args.w_bits,
        g_bits=args.g_bits,

        visdom = args.visdom, # 使用visdom可视化训练过程
        vis_env=args.vis_env,
        vis_legend=args.vis_legend,
        vis_interval=args.vis_interval,
    )
    trainer_exp.run()
    print("end")

