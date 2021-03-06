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
from distill.distill import *

import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)

# import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh
# ssh -L 8097:nico1:8097 tiaoban -p 2222

class DistillerExp(object):
    
    def __init__(self, **kwargs):

        self.config = Configuration()
        self.config.update_config(kwargs) # 解析参数更新默认配置
        assert self.config.check_config() == 0
        # sys.stdout = Logger(self.config.log_path)
        print("| ----------------- Initializing Trainer ----------------- |")
        print('{:<30}  {:<8}'.format('==> num_workers: ', self.config.num_workers))
        print('{:<30}  {:<8}'.format('==> batch_size: ', self.config.batch_size))
        print('{:<30}  {:<8}'.format('==> max_epoch: ', self.config.max_epoch))
        milestones = [self.config.max_epoch*0.5, self.config.max_epoch*0.75]\
                        if self.config.milestones == ''\
                        else sting2list(self.config.milestones)
        print('{:<30}  {:<8}'.format('==> lr_scheduler milestones: ', str(milestones)))

        # 更新一些默认标志
        self.start_epoch = 0
        self.best_acc1 = 0
        self.checkpoint = None
        vis_clear = True

        # suffix
        # self.config.suffix_usr += "_distill"
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
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.kd_criterion = kd_loss(self.config.kd_alpha, self.config.kd_temperature)

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self.optimizer,
            milestones=[self.config.max_epoch*0.5, self.config.max_epoch*0.75]\
                        if self.config.milestones == ''\
                        else sting2list(self.config.milestones), 
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

        print("")
        self.teacher_model, self.teacher_cfg, teacher_checkpoint = teacher_model_init(self.config, self.device, self.num_classes)
        print("{:<30}  {:<8}".format('==> teacher model best acc1: ', teacher_checkpoint['best_acc1']))
                
        # visdom
        self.vis, self.vis_interval = visdom_init(self.config, self.suffix, vis_clear)
        
        self.trainer = DistillerTrainer(
            self.model, 
            self.teacher_model, 
            self.train_dataloader, 
            self.kd_criterion, 
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
        

    def run(self):

        print("")
        start_time = datetime.datetime.now()
        name = (self.config.dataset + "_" + self.config.arch + self.suffix)
        # get_model_flops(self.model, dataset=self.config.dataset, pr=True)
        print_flops_params(model=self.model, dataset=self.config.dataset)

        # initial test
        if self.valuator is not None:
            self.valuator.test(self.model, epoch=self.start_epoch-1)
        # print_bar(start_time, self.config.arch, self.config.dataset, self.best_acc1)
            print_bar_name(start_time, name, self.best_acc1)
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

            # print_bar(start_time, self.config.arch, self.config.dataset, self.best_acc1)
            print_bar_name(start_time, name, self.best_acc1)
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
    add_visdom_arg_parser(parser)

    parser.add_argument('--kd_teacher_arch', '-tarch', dest='kd_teacher_arch', type=str, default='resnet20',
                        help='knowledge distiller teacher architecture(default: resnet20)')
    parser.add_argument('--kd_teacher_checkpoint', '-tpath', dest='kd_teacher_checkpoint', type=str, default='',
                        help='knowledge distiller teacher checkpoint(default: '')')
    parser.add_argument('--kd_temperature', '-t', dest='kd_temperature', type=float, default=20,
                        help='knowledge distiller temperature(default: 20)')
    parser.add_argument('--kd_alpha', '-alpha', dest='kd_alpha', type=float, default=0.9,
                        help='knowledge distiller alpha(default: 0.9)')

    parser.add_argument('--json', type=str, default='',
                        help='json configuration file path(default: '')')

    args = parser.parse_args()
    
    if args.json != '':
        json_path = os.path.join(args.json)
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        params = Params_json(json_path)
        for key in params.dict:
            args.__dict__[key] = params.dict[key]
        

    # debug用
    # args.workers = 0


    distiller_exp = DistillerExp(
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
        suffix_usr=args.suffix_usr,
        log_path=args.log_path,
        test_only=args.test_only,
        milestones=args.milestones,

        kd_teacher_arch=args.kd_teacher_arch,
        kd_teacher_checkpoint=args.kd_teacher_checkpoint,
        kd_temperature=args.kd_temperature,
        kd_alpha=args.kd_alpha,

        visdom = args.visdom, # 使用visdom可视化训练过程
        vis_env=args.vis_env,
        vis_legend=args.vis_legend,
        vis_interval=args.vis_interval,
    )
    distiller_exp.run()
    print("end")

