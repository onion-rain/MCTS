import torch
import numpy as np
import random
import time

from traintest.trainer import Trainer
from utils import *

__all__ = ['BinaryTrainer']

class BinaryTrainer(Trainer):
    """
    Binarynet专用
    """
    def __init__(self, model, dataloader, criterion, optimizer, device, 
                 vis=None, vis_interval=20, lr_scheduler=None):

        super(BinaryTrainer, self).__init__(model, dataloader, criterion, optimizer, 
                                             device, vis, vis_interval, lr_scheduler)

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

            for param in list(self.model.parameters()):
                if hasattr(param, 'org'):
                    param.data.copy_(param.org)

            self.optimizer.step() # 反向传播传的是全精度gradient

            for param in list(self.model.parameters()):
                if hasattr(param, 'org'):
                    param.org.copy_(param.data.clamp_(-1, 1))

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