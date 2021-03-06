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

from traintest.trainer import Trainer
from utils import *

__all__ = ['SlimmerTrainer']

class SlimmerTrainer(Trainer):
    """
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
        self.update_attr(epoch, model, optimizer, train_dataloader, criterion, vis, vis_interval)
        
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
            self.upadate_meters(output, target, loss)

            # measure elapsed time
            self.batch_time.update(time.time() - end_time)
            end_time = time.time()

            # print log
            done = (batch_index+1) * self.train_dataloader.batch_size
            percentage = 100. * (batch_index+1) / len(self.train_dataloader)
            self.print_log(epoch, done, percentage)
            self.visualize_plot(epoch, batch_index, percentage)

        print("")
        self.visualize_log(epoch)
        
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




