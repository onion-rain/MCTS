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
    TODO 由于trainer类大改，本类某些函数可能个已过期
    导入的模型必须有BN层，
    并且事先进行稀疏训练，
    并且全连接层前要将左右特征图池化为1x1，即最终卷积输出的通道数即为全连接层输入通道数
    """
    def __init__(self, **kwargs):

        self.config = Configuration()
        self.config.update_config(kwargs) # 解析参数更新默认配置
        assert self.config.check_config() == 0
        sys.stdout = Logger(self.config.log_path)
        print("| ----------------- Initializing Slimmer ----------------- |")
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
        _, self.val_dataloader, self.num_classes = dataloader_init(self.config, val_num=50)
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
                
        # step6: valuator
        self.vis = None
        val_config_dic = {
            'arch': self.model,
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
        cfg = self.pruner.slim()
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



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='network slimmer')
    
    add_trainer_arg_parser(parser)

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
        log_path=args.log_path,
        test_only=args.test_only,

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