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
from prune.filter_pruner import FilterPruner

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"


class Pruner(object):
    """
    """
    def __init__(self, **kwargs):

        self.config = Configuration()
        self.config.update_config(kwargs) # 解析参数更新默认配置
        assert self.config.check_config() == 0
        # sys.stdout = Logger(self.config.log_path)
        print("| ----------------- Initializing Pruner ----------------- |")
        print('{:<30}  {:<8}'.format('==> num_workers: ', self.config.num_workers))
        print('{:<30}  {:<8}'.format('==> batch_size: ', self.config.batch_size))

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

        self.vis = None

        # step5: pruner
        self.pruner = FilterPruner(
            model=self.model,
            device=self.device,
            arch=self.config.arch,
            # prune_percent=[self.config.prune_percent],
            target_cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256],
            p=self.config.lp_norm,
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
        print("| -----------------simple pruning model ------------------ |")
        self.pruned_model, self.cfg, _ = self.pruner.simple_prune(self.model)
        self.valuator.test(self.pruned_model)
        print_flops_params(self.pruned_model, self.config.dataset)

        print("")
        print("| -------------------- pruning model -------------------- |")
        self.pruned_model, self.cfg, self.pruned_ratio = self.pruner.prune()
        self.valuator.test(self.pruned_model)
        print_flops_params(self.pruned_model, self.config.dataset)

        self.best_acc1 = self.valuator.top1_acc.avg
        print("{}{}".format("best_acc1: ", self.best_acc1))
        # save pruned model
        name = ('filter_pruned' + str(self.config.prune_percent) 
                + '_' + self.config.dataset 
                + "_" + self.config.arch
                + self.suffix)
        if len(self.config.gpu_idx_list) > 1:
            state_dict = self.pruned_model.module.state_dict()
        else: state_dict = self.pruned_model.state_dict()
        save_dict = {
            'arch': self.config.arch,
            'ratio': self.pruned_ratio,
            'model_state_dict': state_dict,
            'best_acc1': self.best_acc1,
        }
        if self.cfg is not None:
            save_dict['cfg'] = self.cfg
        checkpoint_path = save_checkpoint(save_dict, file_root='checkpoints/', file_name=name)
        print('{}  {}'.format('==> pruned model save path: ', checkpoint_path))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='network pruner')
    
    add_trainer_arg_parser(parser)
    parser.add_argument('--prune_percent', type=float, default=0.5, 
                        help='percentage of weight to prune(default: 0.5)')
    parser.add_argument('--lp_norm', '-lp', dest='lp_norm', type=int, default=2, 
                        help='the order of norm(default: 2)')
                        

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
        lp_norm=args.lp_norm
    )
    pruner.run()
    print("end")
