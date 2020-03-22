# import ssl
# #全局取消证书验证
# ssl._create_default_https_context = ssl._create_unverified_context

import torch
from tqdm import tqdm
from torch.nn import functional as F
# import torchvision as tv
import numpy as np
import time
import os
import random
import datetime
import argparse

from train.trainer import Trainer
from utils import *

__all__ = ['SlimmerTrainer']

class SlimmerTrainer(Trainer):
    """
    TODO 待测试
    bn稀疏训练
    """
    def __init__(self, model, dataloader, criterion, optimizer, device, 
                 sr_lambda, vis=None, vis_interval=20, lr_scheduler=None):

        super(SlimmerTrainer, self).__init__(model, dataloader, criterion, optimizer, 
                                             device, vis, vis_interval, lr_scheduler)
        
        self.sr_lambda = sr_lambda
        

    def train(self, model=None, epoch=None, train_dataloader=None, criterion=None,
                optimizer=None, lr_scheduler=None, vis=None, vis_interval=None):
        """注意：如要更新model必须更新optimizer和lr_scheduler"""
        
        if epoch is None:
            epoch = 0
        if model is not None:
            assert optimizer is not None
            assert lr_scheduler is not None
            self.model = model
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        if train_dataloader is not None:
            self.train_dataloader = train_dataloader
        if criterion is not None:
            self.criterion = criterion
        if vis is not None:
            self.vis = vis
        if vis_interval is not None:
            self.vis_interval = vis_interval
        
        self.model.train() # 训练模式
        self.init_meters()

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
                    epoch=epoch,
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




