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

# python meta_search_exp.py --workers 20 --arch resnet50_pruningnet --dataset imagenet --gpu 2 --resume checkpoints/meta_prune/imagenet_resnet50_meta_myo_best.pth.tar --flops 1500 --population 100 --select-num 30 --mutation-num 30 --crossover-num 30 --log logs/flops1500.txt --usr-suffix _flops1500 --flops-arch resnet50_prunednet

class MetaSearcher(object):

    def __init__(self, **kwargs):

        self.config = Configuration()
        self.config.update_config(kwargs) # 解析参数更新默认配置
        assert self.config.check_config() == 0

        print("| ----------------- Initializing meta searcher ----------------- |")
        sys.stdout = Logger(self.config.log_path)
        print('{:<30}  {:<8}'.format('==> num_workers: ', self.config.num_workers))
        print('{:<30}  {:<8}'.format('==> batch_size: ', self.config.batch_size))

        # 更新一些默认标志
        self.max_iter = self.config.max_epoch
        self.start_iter = 0
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
        flops_model = models.__dict__[self.config.flops_arch](num_classes=self.num_classes).to(self.device)
        print('{:<30}  {:<8}'.format('==> flops_arch: ', self.config.flops_arch))
        
        self.criterion = torch.nn.CrossEntropyLoss()
        # self.criterion_smooth = CrossEntropyLabelSmooth(self.num_classes, 0.1)

        # TODO resume search
        self.candidates = []
        checked_genes_tuple = {}
        tested_genes_tuple = {}
        if self.config.research_resume_path != '':
            search_checkpoint = torch.load(config.research_resume_path, map_location=device)
            assert search_checkpoint['model'] == self.config.arch
            self.start_iter = search_checkpoint['iter']+1
            print('{:<30}  {:<8}'.format('==> checkpoint searched iteration: ', search_checkpoint['iter']))
            vis_clear = False
            self.candidates = search_checkpoint['candidates']
            best_acc1_error = search_checkpoint['best_acc1_error']
            checked_genes_tuple = search_checkpoint['checked_genes_tuple']
            tested_genes_tuple = search_checkpoint['tested_genes_tuple']

        # visdom
        self.vis, self.vis_interval = visdom_init(self.config, self.suffix, vis_clear)

        # step6: searcher
        self.searcher = PrunednetSearcher(
            self.model, 
            self.train_dataloader, 
            self.val_dataloader,
            self.criterion, 
            self.device, 
            self.vis, 
            # hyper-parameters
            self.config.max_flops, 
            self.config.population, 
            self.config.select_num, 
            self.config.mutation_num, 
            self.config.crossover_num, 
            self.config.mutation_prob, 
            # resume
            checked_genes_tuple,
            tested_genes_tuple,
            flops_model=flops_model,
        )
            
        print()
        print('{:<30}  {:<8}'.format('==> max_flops: ', self.config.max_flops))
        print('{:<30}  {:<8}'.format('==> population: ', self.config.population))
        print('{:<30}  {:<8}'.format('==> select_num: ', self.config.select_num))
        print('{:<30}  {:<8}'.format('==> mutation_num: ', self.config.mutation_num))
        print('{:<30}  {:<8}'.format('==> crossover_num: ', self.config.crossover_num))
        print('{:<30}  {:<8}'.format('==> mutation_prob: ', self.config.mutation_prob))
        print('{:<30}  {:<8}'.format('==> search resume from: ', self.config.research_resume_path))
            

    def run(self):
        print("")
        start_time = datetime.datetime.now()
        for iter in range(self.start_iter, self.max_iter):
            print_bar(start_time, self.config.arch, self.config.dataset, epoch=iter)
            self.candidates, checked_genes_tuple, tested_genes_tuple = self.searcher.search(iter, self.candidates)

            # save checkpoint
            name = ('MetaPruneSearch_' + self.config.arch + self.suffix)
            save_dict = {
                'model': self.config.arch,
                'iter': iter,
                'candidates': self.candidates,
                'best_acc1_error': self.candidates[0][-1],
                'checked_genes_tuple': checked_genes_tuple,
                'tested_genes_tuple': tested_genes_tuple,
            }
            save_checkpoint(save_dict, is_best=False, epoch=None, file_root='checkpoints/meta_prune/', file_name=name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='meta Prunednet search')
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH', default='resnet50_pruningnet',
                        choices=models.ALL_MODEL_NAMES,
                        help='model architecture: ' +
                        ' | '.join(name for name in models.ALL_MODEL_NAMES) +
                        ' (default: resnet50_pruningnet)')
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
                        metavar='PATH', help='path to latest train checkpoint (default: '')')
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

    parser.add_argument('--search-resume', dest='search_resume_path', type=str, default='',
                        metavar='PATH', help='path to latest checkpoint (default: '')')
    parser.add_argument('--flops', dest='max_flops', type=float, default=0, 
                        metavar='Flops', help='The maximum amount of computation that can be tolerated, 0 means no limit(default: 0)')
    parser.add_argument('--population', dest='population', type=int, default=100, 
                        metavar='N', help='candidates number(default: 100)')
    parser.add_argument('--select-num', dest='select_num', type=int, default=30, 
                        metavar='N', help='after one iteration, the rest candidates number(default: 30)')
    parser.add_argument('--mutation-num', dest='mutation_num', type=int, default=30, 
                        metavar='N', help='The number of mutations in the candidates(default: 30)')
    parser.add_argument('--crossover-num', dest='crossover_num', type=int, default=30, 
                        metavar='N', help='The number of crossover in the candidates(default: 30)')
    parser.add_argument('--mutation-prob', dest='mutation_prob', type=float, default=0.1, 
                        metavar='prob', help='mutation probability(default: 0.1)')
    parser.add_argument('--flops-arch', type=str, metavar='ARCH', default='resnet50_prunednet',
                        choices=models.ALL_MODEL_NAMES,
                        help='test flops model architecture: ' +
                        ' | '.join(name for name in models.ALL_MODEL_NAMES) +
                        ' (default: resnet50_prunednet)')
                        
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
        log_path=args.log_path,

        visdom = args.visdom, # 使用visdom可视化训练过程
        vis_env=args.vis_env,
        vis_legend=args.vis_legend,
        vis_interval=args.vis_interval,

        search_resume_path=args.search_resume_path, 
        # hyper-parameters
        max_flops=args.max_flops,
        population=args.population,
        select_num=args.select_num,
        mutation_num=args.mutation_num,
        crossover_num=args.crossover_num,
        mutation_prob=args.mutation_prob,
        flops_arch=args.flops_arch,
    )
    MetaSearcher.run()
    print("end")

