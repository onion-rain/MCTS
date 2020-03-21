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

from tester import Tester
from config import Configuration
import models
from utils import *

import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
# fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh

class Trainer(object):
    """
    TODO 待大量测试
    可通过传入config_dic来配置Tester，这种情况下不会在初始化过程中print相关数据
    例：
        train_config_dic = {
            'model': self.model,
            'dataloader': self.train_dataloader,
            'device': self.device,
            'vis': self.vis,
            'vis_interval': self.config.vis_interval,
            'seed': self.config.random_seed,
            'criterion': self.criterion,
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
        }
        self.trainer = Trainer(train_config_dic)
    也可通过**kwargs配置Trainer
    """
    def __init__(self, config_dic=None, **kwargs):

        if config_dic is None:
            self.init_from_kwargs(**kwargs)
        else:
            self.init_from_config(config_dic)
        
    
    def print_bar(self, start_time, arch, dataset):
        """calculate duration time"""
        interval = datetime.datetime.now() - start_time
        print("--------  model: {model}  --  dataset: {dataset}  --  duration: {dh:2}h:{dm:02d}.{ds:02d}  --------".
            format(
                model=arch,
                dataset=dataset,
                dh=interval.seconds//3600,
                dm=interval.seconds%3600//60,
                ds=interval.seconds%60,
            )
        )

    def run(self):

        print("")
        start_time = datetime.datetime.now()
        name = (self.config.dataset + "_" + self.config.arch + self.suffix)
        print_flops_params(model=self.model)

        # initial test
        if self.valuator is not None:
            self.valuator.test(self.model, epoch=self.start_epoch-1)
        self.print_bar(start_time, self.config.arch, self.config.dataset)
        print("")
        for epoch in range(self.start_epoch, self.config.max_epoch):
            # train & valuate
            self.train(epoch=epoch)
            if self.valuator is not None:
                self.valuator.test(self.model, epoch=epoch)
            self.print_bar(start_time, self.config.arch, self.config.dataset)
            print("")
            
            # save checkpoint
            if self.valuator is not None:
                is_best = self.valuator.top1_acc.avg > self.best_acc1
                self.best_acc1 = max(self.valuator.top1_acc.avg, self.best_acc1)
            else:
                is_best = self.top1_acc.avg > self.best_acc1
                self.best_acc1 = max(self.top1_acc.avg, self.best_acc1)
            if len(self.config.gpu_idx_list) > 1:
                state_dict = self.model.module.state_dict()
            else: state_dict = self.model.state_dict()
            save_dict = {
                'model': self.config.arch,
                'epoch': epoch,
                'model_state_dict': state_dict,
                'best_acc1': self.best_acc1,
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            if self.cfg is not None:
                save_dict['cfg'] = self.cfg
            save_checkpoint(save_dict, is_best=is_best, epoch=None, file_root='checkpoints/', file_name=name)
        print("{}{}".format("best_acc1: ", self.best_acc1))

    def train(self, model=None, epoch=None, train_dataloader=None, criterion=None,
                optimizer=None, lr_scheduler=None, vis=None, vis_interval=None):
        """
        在指定数据集上训练指定模型, model和dataset在创建Trainer类时通过修改self.config确定
        args:
            epoch：仅用于显示当前epoch
        """
        if model is not None:
            self.model = model
        if train_dataloader is not None:
            self.train_dataloader = train_dataloader
        if criterion is not None:
            self.criterion = criterion
        if optimizer is not None:
            self.optimizer = optimizer
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler
        if vis is not None:
            self.vis = vis
        if vis_interval is not None:
            self.vis_interval = vis_interval
        
        self.model.train() # 训练模式

        # meters
        self.loss_meter = AverageMeter()
        self.top1_acc = AverageMeter()
        self.top5_acc = AverageMeter()
        self.batch_time = AverageMeter()
        self.dataload_time = AverageMeter()
        if self.vis is not None:
            self.loss_vis = AverageMeter()
            self.top1_vis = AverageMeter()

        end_time = time.time()
        # print("training...")
        # pbar = tqdm(
        #     enumerate(self.train_dataloader), 
        #     total=len(self.train_dataset)/self.config.batch_size,
        # )
        # for batch_index, (input, target) in pbar:
        for batch_index, (input, target) in enumerate(self.train_dataloader):
            # measure data loading time
            self.dataload_time.update(time.time() - end_time)

            # compute output
            input, target = input.to(self.device), target.to(self.device)
            output = self.model(input)
            loss = self.criterion(output, target)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()

            if hasattr(self, 'sr'):
                self.updateBN()
                
            self.optimizer.step()

            # meters update
            self.loss_meter.update(loss.item(), input.size(0))
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            self.top1_acc.update(prec1.data.cpu(), input.size(0))
            self.top5_acc.update(prec5.data.cpu(), input.size(0))

            # measure elapsed time
            self.batch_time.update(time.time() - end_time)
            end_time = time.time()

            # print log
            done = (batch_index+1) * self.train_dataloader.batch_size
            percentage = 100. * (batch_index+1) / len(self.train_dataloader)
            # pbar.set_description(
            print("\r"
                "Train: {epoch:3} "
                "[{done:7}/{total_len:7} ({percentage:3.0f}%)] "
                "loss: {loss_meter:.3f} | "
                "top1: {top1:3.3f}% | "
                # "top5: {top5:3.3f} | "
                "load_time: {time_percent:2.0f}% | "
                "lr   : {lr:0.1e} ".format(
                    epoch=0 if epoch == None else epoch,
                    done=done,
                    total_len=len(self.train_dataloader.dataset),
                    percentage=percentage,
                    loss_meter=self.loss_meter.avg,
                    top1=self.top1_acc.avg,
                    # top5=self.top5_acc.avg,
                    time_percent=self.dataload_time.avg/self.batch_time.avg*100,
                    lr=self.optimizer.param_groups[0]['lr'],
                ), end=""
            )

            # visualize
            if self.vis is not None:
                self.loss_vis.update(loss.item(), input.size(0))
                self.top1_vis.update(prec1.data.cpu(), input.size(0))

                if (batch_index % self.vis_interval == self.vis_interval-1):
                    vis_x = epoch+percentage/100
                    self.vis.plot('train_loss', self.loss_vis.avg, x=vis_x)
                    self.vis.plot('train_top1', self.top1_vis.avg, x=vis_x)
                    self.loss_vis.reset()
                    self.top1_vis.reset()

        print("")

        # visualize
        if self.vis is not None:
            self.vis.log(
                "epoch: {epoch},  lr: {lr}, <br>\
                train_loss: {train_loss}, <br>\
                train_top1: {train_top1}, <br>"
                .format(
                    lr=self.optimizer.param_groups[0]['lr'],
                    epoch=epoch, 
                    train_loss=self.loss_meter.avg,
                    train_top1=self.top1_acc.avg,
                )
            )
        
        # update learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch=epoch)
        
        return self.model
        

    # additional subgradient descent on the sparsity-induced penalty term
    def updateBN(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                # torch.sign(module.weight.data)是对sparsity-induced penalty(g(γ))求导的结果
                module.weight.grad.data.add_(self.config.sr_lambda * torch.sign(module.weight.data))  # L1


    def init_from_kwargs(self, **kwargs):

        print("| ----------------- Initializing Trainer ----------------- |")

        self.config = Configuration()
        self.config.update_config(kwargs) # 解析参数更新默认配置
        if self.config.check_config(): raise # 检测路径、设备是否存在
        print('{:<30}  {:<8}'.format('==> num_workers: ', self.config.num_workers))
        print('{:<30}  {:<8}'.format('==> batch_size: ', self.config.batch_size))
        print('{:<30}  {:<8}'.format('==> max_epoch: ', self.config.max_epoch))
        print('{:<30}  {:<8}'.format('==> lr_scheduler milestones: ', str([self.config.max_epoch*0.5, self.config.max_epoch*0.75])))

        if self.config.sr:
            self.sr = self.config.sr

        self.suffix = get_suffix(self.config)
        print('{:<30}  {:<8}'.format('==> suffix: ', self.suffix))

        # 更新一些默认标志
        self.start_epoch = 0
        self.best_acc1 = 0
        self.checkpoint = None
        vis_clear = True

        # device
        if len(self.config.gpu_idx_list) > 0:
            self.device = torch.device('cuda:{}'.format(min(self.config.gpu_idx_list))) # 起始gpu序号
            print('{:<30}  {:<8}'.format('==> chosen GPU index: ', self.config.gpu_idx))
        else:
            self.device = torch.device('cpu')
            print('{:<30}  {:<8}'.format('==> device: ', 'CPU'))

        # Random Seed 
        # (其实pytorch只保证在同版本并且没有多线程的情况下相同的seed可以得到相同的结果，而加载数据一般没有不用多线程的，这就有点尴尬了)
        if self.config.deterministic:
            if self.config.num_workers > 1:
                print("ERROR: Setting --deterministic requires setting --workers to 0 or 1")
            random.seed(0)
            torch.manual_seed(0)
            np.random.seed(self.config.random_seed)
            torch.backends.cudnn.deterministic = True
        else: 
            torch.backends.cudnn.benchmark = True # 让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速

        # step1: data
        self.train_dataloader, self.val_dataloader, self.num_classes = get_dataloader(self.config)

        # step2: model
        print('{:<30}  {:<8}'.format('==> creating arch: ', self.config.arch))
        self.cfg = None
        checkpoint = None
        if self.config.resume_path != '': # 断点续练hhh
            checkpoint = torch.load(self.config.resume_path, map_location=self.device)
            print('{:<30}  {:<8}'.format('==> resuming from: ', self.config.resume_path))
            if self.config.refine: # 根据cfg加载剪枝后的模型结构
                self.cfg=checkpoint['cfg']
                print(self.cfg)
        if self.cfg is not None:
            self.model = models.__dict__[self.config.arch](cfg=cfg, num_classes=self.num_classes)
        else:
            self.model = models.__dict__[self.config.arch](num_classes=self.num_classes)
        self.model.to(self.device) # 模型转移到设备上
        if len(self.config.gpu_idx_list) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config.gpu_idx_list)
            # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:65535', rank=0, world_size=1)
            # self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)
        if checkpoint is not None:
            if 'epoch' in checkpoint.keys():
                self.start_epoch = checkpoint['epoch'] + 1 # 保存的是已经训练完的epoch，因此start_epoch要+1
                print("{:<30}  {:<8}".format('==> checkpoint trained epoch: ', checkpoint['epoch']))
                if checkpoint['epoch'] > -1:
                    vis_clear = False # 不清空visdom已有visdom env里的内容
            if 'best_acc1' in checkpoint.keys():
                self.best_acc1 = checkpoint['best_acc1']
                print("{:<30}  {:<8}".format('==> checkpoint best acc1: ', checkpoint['best_acc1']))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        # exit(0)
        
        # step3: criterion and optimizer
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        if checkpoint is not None:
            if 'optimizer_state_dict' in checkpoint.keys():
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.criterion = torch.nn.CrossEntropyLoss()

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self.optimizer,
            milestones=[self.config.max_epoch*0.5, self.config.max_epoch*0.75], 
            gamma=0.1,
            last_epoch=self.start_epoch-1, # 我的训练epoch从1开始，而pytorch要通过当前epoch是否等于0判断是不是resume
        )
        
        # step5: visdom
        self.vis = None
        if self.config.visdom:
            if self.config.vis_env == '':
                self.config.vis_env = self.config.dataset + '_' + self.config.arch + self.suffix
            if self.config.vis_legend == '':
                self.config.vis_legend = self.config.arch + self.suffix
            self.vis = Visualizer(self.config.vis_env, self.config.vis_legend, clear=vis_clear) 
            self.vis_interval = self.config.vis_interval

        # step6: valuator
        self.valuator = None
        if self.config.valuate is True:
            val_config_dic = {
                'model': self.model,
                'dataloader': self.val_dataloader,
                'device': self.device,
                'vis': self.vis,
                'seed': self.config.random_seed,
                'criterion': self.criterion,
            }
            self.valuator = Tester(val_config_dic)


    def init_from_config(self, config):
        """这种构造方法一般就是外部调用self.train(),需要啥从train()形参里传入"""
        # visdom
        self.vis = config['vis']
        self.vis_interval = config['vis_interval']

        # device
        self.device = config['device']

        # Random Seed
        random.seed(config['seed'])
        torch.manual_seed(config['seed'])

        # step1: data
        self.train_dataloader = config['dataloader']

        # step2: model
        self.model = config['model']

        # step3: criterion
        self.criterion = config['criterion']

        # step4: optimizer
        self.optimizer = config['optimizer']

        # step4: lr_scheduler
        self.lr_scheduler = config['lr_scheduler']






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='network trainer')
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH', default='vgg19_bn_cifar',
                        choices=models.ALL_MODEL_NAMES,
                        help='model architecture: ' +
                        ' | '.join(name for name in models.ALL_MODEL_NAMES) +
                        ' (default: resnet18)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--workers', type=int, default=10, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 150)')
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
    parser.add_argument('--valuate', action='store_true',
                        help='valuate each training epoch')
    parser.add_argument('--resume', dest='resume_path', type=str, default='',
                        metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--refine', action='store_true',
                        help='refine from pruned model, use construction to build the model')
    parser.add_argument('--usr-suffix', type=str, default='',
                        help='usr_suffix(default:"", means nothing')

    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                        help='train with channel sparsity regularization')
    parser.add_argument('--srl', dest='sr_lambda', type=float, default=1e-4,
                        help='scale sparse rate (default: 1e-4), suggest 1e-4 for vgg, 1e-5 for resnet/densenet')

    parser.add_argument('--visdom', dest='visdom', action='store_true',
                        help='visualize the training process using visdom')
    parser.add_argument('--vis-env', type=str, default='', metavar='ENV',
                        help='visdom environment (default: "", which means env is automatically set to args.dataset + "_" + args.arch)')
    parser.add_argument('--vis-legend', type=str, default='', metavar='LEGEND',
                        help='refine from pruned model (default: "", which means env is automatically set to args.arch)')
    parser.add_argument('--vis-interval', type=int, default=50, metavar='N',
                        help='visdom plot interval batchs (default: 50)')
    # parser.add_argument('--nclear', dest='nclear', action='store_true',
    #                     help='if set this true, the wisdom env will not be cleared before training.')
    args = parser.parse_args()

    # debug用
    # args.workers = 0


    trainer = Trainer(
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

        sr=args.sr,
        sr_lambda=args.sr_lambda,

        visdom = args.visdom, # 使用visdom可视化训练过程
        vis_env=args.vis_env,
        vis_legend=args.vis_legend,
        vis_interval=args.vis_interval,
    )
    trainer.run()
    print("end")

