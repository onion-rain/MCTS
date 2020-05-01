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

from utils import *
from traintest.trainer import Trainer

__all__ = ['kd_loss', 'DistillerTrainer']

# fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh

def kd_loss(alpha, temperature):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    """
    def loss_kd_in(outputs, labels, teacher_outputs):
        softtarget_loss = F.kl_div(
            F.log_softmax(outputs/temperature, dim=1), 
            F.softmax(teacher_outputs/temperature, dim=1)
        ) * temperature * temperature
        hardtarget_loss = F.cross_entropy(outputs, labels)
        KD_loss = softtarget_loss*(alpha) + hardtarget_loss*(1. - alpha)
        return KD_loss

    return loss_kd_in

class DistillerTrainer(Trainer):

    def __init__(self, model, teacher_model, dataloader, criterion, optimizer, device, 
                 vis=None, vis_interval=20, lr_scheduler=None):
                 
        super(DistillerTrainer, self).__init__(model, dataloader, criterion, optimizer, 
                                             device, vis, vis_interval, lr_scheduler)
        self.teacher_model = teacher_model

    def train(self, model=None, epoch=None, train_dataloader=None, criterion=None,
                optimizer=None, lr_scheduler=None, vis=None, vis_interval=None):
        """注意：如要更新model必须更新optimizer和lr_scheduler"""
        self.update_attr(epoch, model, optimizer, train_dataloader, criterion, vis, vis_interval)
        
        self.model.train() # 训练模式
        self.init_meters()

        end_time = time.time()

        for batch_index, (input, target) in enumerate(self.train_dataloader):
            # measure data loading time
            self.dataload_time.update(time.time() - end_time)

            # compute output
            input, target = input.to(self.device), target.to(self.device)
            output = self.model(input)
            teacher_output = self.teacher_model(input)
            loss = self.criterion(output, target, teacher_output)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
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


        
