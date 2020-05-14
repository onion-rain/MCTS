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

import models
from utils import *
from traintest import *
from prune.filter_pruner import FilterPruner

import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

class SFP(object):

    def __init__(self, **kwargs):

        self.config = Configuration()
        self.config.update_config(kwargs) # 解析参数更新默认配置
        assert self.config.check_config() == 0
        # sys.stdout = Logger(self.config.log_path)
        print("| ------------------ Initializing SFP ------------------- |")
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
        self.pruned_ratio = 0
        vis_clear = True

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
        
        # step3: criterion and optimizer
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        
        self.criterion = torch.nn.CrossEntropyLoss()

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self.optimizer,
            milestones=[self.config.max_epoch*0.5, self.config.max_epoch*0.75], 
            gamma=0.1,
            last_epoch=self.start_epoch-1, # 我的训练epoch从1开始，而pytorch要通过当前epoch是否等于0判断是不是resume
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

        # step6: trainer
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
        
        # filter pruner
        self.pruner = FilterPruner(
            model=self.model,
            device=self.device,
            arch=self.config.arch,
            prune_percent=[self.config.prune_percent],
            # target_cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256],
            p=self.config.lp_norm,
        )
        

    def run(self):

        print("")
        start_time = datetime.datetime.now()
        name = (self.config.dataset + "_" + self.config.arch + self.suffix)
        print_flops_params(model=self.model)

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

            # prune
            if epoch == self.config.max_epoch-1:
                # 最后hard prune


                print("\nsimple pruning1...")
                self.simple_pruned_model, self.cfg, self.pruned_ratio = self.pruner.simple_prune(self.model)
                if self.valuator is not None:
                    self.valuator.test(self.simple_pruned_model, epoch=epoch+0.5)

                # print("\nsimple pruning2...")
                # self.simple_pruned_model, self.cfg, self.pruned_ratio = self.pruner.simple_prune(self.model)
                # if self.valuator is not None:
                #     self.valuator.test(self.simple_pruned_model, epoch=epoch+0.5)


                print("\npruning")
                self.best_acc1 = 0
                self.model, self.cfg, self.pruned_ratio = self.pruner.prune(self.model)
                if self.valuator is not None:
                    self.valuator.test(self.model, epoch=epoch+0.5)
            elif epoch%self.config.sfp_intervals == self.config.sfp_intervals-1:
                # 中途soft prune
                print("\nsimple pruning...")
                self.model, self.cfg, self.pruned_ratio = self.pruner.simple_prune(self.model, in_place=True)
                if self.valuator is not None:
                    self.valuator.test(self.model, epoch=epoch+0.5)
                print()

            if self.valuator is not None:
                is_best = self.valuator.top1_acc.avg > self.best_acc1
                self.best_acc1 = max(self.valuator.top1_acc.avg, self.best_acc1)
            else:
                is_best = self.trainer.top1_acc.avg > self.best_acc1
                self.best_acc1 = max(self.top1_acc.avg, self.best_acc1)
                
            # print_bar(start_time, self.config.arch, self.config.dataset,self.best_acc1)
            print_bar_name(start_time, name, self.best_acc1)
            print("")
            
            # save checkpoint
            save_dict = {
                'arch': self.config.arch,
                'ratio': self.pruned_ratio,
                'epoch': epoch,
                'best_acc1': self.best_acc1,
            }
            if self.config.save_object == 'None':
                continue
            elif self.config.save_object == 'state_dict':
                file_name = name + '_state_dict'
                if len(self.config.gpu_idx_list) > 1:
                    state_dict = self.model.module.state_dict()
                else: state_dict = self.model.state_dict()
                save_dict['model_state_dict'] = state_dict
            # FIXME 由于未知原因保存的model无法torch.load加载
            elif self.config.save_object == 'model':
                file_name = name + '_model'
                if len(self.config.gpu_idx_list) > 1:
                    model = self.model.module
                else: model = self.model
                save_dict['model'] = model
            if self.cfg is not None and self.cfg != 0:
                save_dict['cfg'] = self.cfg
            if epoch != self.config.max_epoch-1:
                save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint_path = save_checkpoint(save_dict, is_best=is_best, file_root='checkpoints/', file_name=file_name)

        print("{}{}".format("best_acc1: ", self.best_acc1))
        print('{}{}'.format('==> pruned model save path: ', checkpoint_path))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='network trainer')

    add_trainer_arg_parser(parser)

    add_visdom_arg_parser(parser)
                        
    parser.add_argument('--prune_percent', type=float, default=0.2, metavar='PERCENT', 
                        help='percentage of weight to prune(default: 0.2)')
    parser.add_argument('--lp_norm', '--lp', dest='lp_norm', type=int, default=2, metavar='P', 
                        help='the order of norm(default: 2)')
    parser.add_argument('--sfp_intervals', '--sfp', type=int, default=3, metavar='N', 
                        help='soft filter prune interval(default: 3)')

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


    sfp = SFP(
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
        save_object=args.save_object,

        visdom = args.visdom, # 使用visdom可视化训练过程
        vis_env=args.vis_env,
        vis_legend=args.vis_legend,
        vis_interval=args.vis_interval,

        # pruner
        prune_percent=args.prune_percent,
        lp_norm=args.lp_norm,
        sfp_intervals=args.sfp_intervals,
    )
    sfp.run()
    print("end")



