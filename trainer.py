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

from config import Configuration
import models
from utils import accuracy, print_model_parameters, AverageMeter, get_path

import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)

class Trainer(object):
    """
    可通过传入config类来配置Trainer，这种情况下若要会用visdom必须传入vis类
    也可通过**kwargs配置Trainer
    """
    def __init__(self, config=None, vis=None, **kwargs):
        print("| --------------- Initializing Trainer --------------- |")
        if config == None:
            self.config = Configuration()
            self.config.update_config(kwargs) # 解析参数更新默认配置
            if self.config.check_config(): raise # 检测路径、设备是否存在
            if self.config.use_visdom:
                self.vis = Visualizer(self.config.env, self.config.legend) # 初始化visdom
        else: 
            self.config = config
            self.vis = vis

        if len(self.config.gpu_idx_list) > 0:
            self.device = torch.device('cuda:{}'.format(min(self.config.gpu_idx_list))) # 起始gpu序号
            print('{:<30}  {:<8}'.format('==> chosen GPU index: ', self.config.gpu_idx))
        else:
            self.device = torch.device('cpu')
            print('{:<30}  {:<8}'.format('==> device: ', 'CPU'))

        # Random Seed
        if self.config.random_seed is None:
            self.config.random_seed = random.randint(1, 10000)
        random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

        # step1: data
        print('{:<30}  {:<8}'.format('==> Preparing dataset: ', self.config.dataset))
        if self.config.dataset.startswith("cifar"): # --------------cifar dataset------------------
            transform = tv.transforms.Compose([
                tv.transforms.RandomCrop(32, padding=4),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], 
                    std=[0.5, 0.5, 0.5],
                ) # 标准化的过程为(input-mean)/std
            ])
            if self.config.dataset is "cifar10": # -----------------cifar10 dataset----------------
                self.train_dataset = tv.datasets.CIFAR10(
                    root=self.config.dataset_root, 
                    train=True, 
                    download=False,
                    transform=transform,
                )
                self.num_classes = 10
            elif self.config.dataset is "cifar100": # --------------cifar100 dataset----------------
                self.train_dataset = tv.datasets.CIFAR100(
                    root=self.config.dataset_root, 
                    train=True, 
                    download=False,
                    transform=transform,
                )
                self.num_classes = 100
            else: 
                print("Dataset undefined")
                exit()

        elif self.config.dataset is "imagenet": # ----------------imagenet dataset------------------
            transform = tv.transforms.Compose([
                tv.transforms.RandomResizedCrop(224),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ) # 标准化的过程为(input-mean)/std
            ])
            self.train_dataset = tv.datasets.ImageFolder(
                self.config.dataset_root+'imagenet/img_train/', 
                transform=transform
            )
            self.num_classes = 1000
        else: 
            print("Dataset undefined")
            exit()


        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last = self.config.dataloader_droplast,
        )

        # step2: model
        print('{:<30}  {:<8}'.format('==> creating model: ', self.config.model))
        print('{:<30}  {:<8}'.format('==> loading model: ', self.config.load_model_path if self.config.load_model_path != None else 'None'))
        self.model = models.__dict__[self.config.model](num_classes=self.num_classes) # 从models中获取名为config.model的model
        if len(self.config.gpu_idx_list) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config.gpu_idx_list)
        self.model.to(self.device) # 模型转移到设备上
        if self.config.load_model_path: # 加载目标模型参数
            # self.model.load_state_dict(torch.load(self.config.load_model_path, map_location=self.device))
            checkpoint = torch.load(self.config.load_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("{:<30}  {:<8}".format('==> model epoch: ', checkpoint['epoch']))
            print("{:<30}  {:<8}".format('==> model best acc1: ', checkpoint['best_acc1']))
            # config_state_dict = checkpoint['config_state_dict']
            
        # exit(0)
        
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
            milestones=self.config.lr_scheduler_milestones, 
            gamma=0.1,
            last_epoch=-1,
        )
        
        # step4: meters
        self.loss_meter = AverageMeter()
        self.top1_acc = AverageMeter()
        self.top5_acc = AverageMeter()
        self.batch_time = AverageMeter()
        self.dataload_time = AverageMeter()
        
        if self.config.use_visdom:
            self.loss_vis = AverageMeter()
            self.top1_vis = AverageMeter()


    def run(self):
        for epoch in range(1, self.config.max_epoch):
            self.train(epoch)

        # 训练结束保存最终模型
        if self.config.save_model_path != None:
            if len(self.config.gpu_idx_list) > 1:
                torch.save(self.model.module.state_dict(), self.config.save_model_path)
            else: torch.save(self.model.state_dict(), self.config.save_model_path)

    def train(self, model=None, epoch=None):
        """
        在指定数据集上训练指定模型, 数据集在创建Trainer类时通过修改self.config确定
        args:
            model: 要训练的模型，若不为none则self.model更新为model，若为none则训练self.model
            epoch：仅用于显示当前epoch
        """
        if model is not None:
            self.model = model
        self.model.train() # 训练模式
        self.loss_meter.reset()
        self.top1_acc.reset()
        self.top5_acc.reset()
        self.batch_time.reset()
        self.dataload_time.reset()
        if self.config.use_visdom:
            self.loss_vis.reset()
            self.top1_vis.reset()

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

            if self.config.slim:
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
            done = (batch_index+1) * self.config.batch_size
            percentage = 100. * (batch_index+1) / len(self.train_dataloader)
            # pbar.set_description(
            print("\r"
                "Train: {epoch:3} "
                "[{done:7}/{total_len:7} ({percentage:3.0f}%)] "
                "loss: {loss_meter:.3f} | "
                "top1: {top1:3.3f}% | "
                # "top5: {top5:3.3f} | "
                "load_time: {time_percent:2.0f}% | "
                "lr   : {lr:0.1e}".format(
                    epoch=0 if epoch == None else epoch,
                    done=done,
                    total_len=len(self.train_dataset),
                    percentage=percentage,
                    loss_meter=self.loss_meter.avg,
                    top1=self.top1_acc.avg,
                    # top5=self.top5_acc.avg,
                    time_percent=self.dataload_time.avg/self.batch_time.avg*100,
                    lr=self.optimizer.param_groups[0]['lr'],
                ), end=""
            )

            # visualize
            if self.config.use_visdom:
                self.loss_vis.update(loss.item(), input.size(0))
                self.top1_vis.update(prec1.data.cpu(), input.size(0))

                if (batch_index % self.config.plot_interval == self.config.plot_interval-1):
                    vis_x = epoch-1+percentage/100
                    self.vis.plot('train_loss', self.loss_vis.avg, x=vis_x)
                    self.vis.plot('train_top1', self.top1_vis.avg, x=vis_x)
                    self.loss_vis.reset()
                    self.top1_vis.reset()

        print("")

        # visualize
        if self.config.use_visdom:
            self.vis.log(
                "epoch: {epoch},  lr: {lr}, <br>\
                train_loss: {train_loss}, <br>\
                train_top1: {train_top1}, <br>"
                .format(
                    lr=self.config.lr,
                    epoch=epoch, 
                    train_loss=self.loss_meter.avg,
                    train_top1=self.top1_acc.avg,
                )
            )
        
        # update learning rate
        self.lr_scheduler.step()

    # additional subgradient descent on the sparsity-induced penalty term
    def updateBN(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                # torch.sign(module.weight.data)是约束稀疏那项求导的结果
                module.weight.grad.data.add_(self.config.slim_lambda * torch.sign(module.weight.data))  # L1


if __name__ == "__main__":
    trainer = Trainer(
        max_epoch=150,
        batch_size=100,
        lr=1e-2,
        lr_scheduler_milestones=[50, 110],
        weight_decay=1e-4,
        momentum=0.9,
        model='test',
        dataset="imagenet",
        gpu_idx = "0", # choose gpu
        random_seed=2,
        # num_workers = 5, # 使用多进程加载数据
    )
    print(trainer.model)
    exit(0)
    # trainer.run()
    print("end")