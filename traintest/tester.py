import torch
from utils.visualize import Visualizer
from tqdm import tqdm
import torchvision as tv
import numpy as np
import time
import random
import argparse

from utils import *

__all__ = ['Tester']


class Tester(object):
    """
    tester基类，原则上所有trainer都是继承该类
    """
    def __init__(self, dataloader=None, device=None, criterion=None, vis=None):

        # visdom
        self.vis = vis

        # device
        self.device = device

        # step1: data
        self.test_dataloader = dataloader

        # step3: criterion
        self.criterion = criterion

    def init_meters(self):
        self.loss_meter = AverageMeter() # 计算所有数的平均值和标准差，这里用来统计一个epoch中的平均值
        self.top1_acc = AverageMeter()
        self.top5_acc = AverageMeter()
        self.batch_time = AverageMeter()
        self.dataload_time = AverageMeter()


    def test(self, model, epoch=-1, test_dataloader=None, criterion=None, device=None, vis=None):
        """
        args:
            model(torch.nn.Module): 
            epoch(int): (default=-1)
            test_dataloader: (default=None)
            criterion: (default=None)
            device: (default=None)
            vis: (default=None)
        return: self.loss_meter, self.top1_acc, self.top5_acc
        """
        self.model = model
        assert self.model is not None
        if vis is not None:
            self.vis = vis
        if device is not None:
            self.device = device
        if criterion is not None:
            self.criterion = criterion
        if test_dataloader is not None:
            self.test_dataloader = test_dataloader
        
        self.model.eval() # 验证模式
        self.init_meters()

        end_time = time.time()
        # print("testing...")
        with torch.no_grad():
            for batch_index, (input, target) in enumerate(self.test_dataloader):
                # measure data loading time
                self.dataload_time.update(time.time() - end_time)

                # compute output
                input, target = input.to(self.device), target.to(self.device)
                output = self.model(input)
                loss = self.criterion(output, target)

                # meters update and visualize
                self.loss_meter.update(loss.item(), input.size(0))
                prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
                self.top1_acc.update(prec1.data.cpu(), input.size(0))
                # self.self.top5_acc.update(prec5.data.cpu(), input.size(0))

                # measure elapsed time
                self.batch_time.update(time.time() - end_time)
                end_time = time.time()

                done = (batch_index+1) * self.test_dataloader.batch_size
                percentage = 100. * (batch_index+1) / len(self.test_dataloader)
                time_str = time.strftime('%H:%M:%S')
                print("\r"
                "Test: {epoch:4} "
                "[{done:7}/{total_len:7} ({percentage:3.0f}%)] "
                "loss: {loss_meter:.3f} | "
                "top1: {top1:3.3f}% | "
                # "top5: {top5:3.3f} | "
                "load_time: {time_percent:2.0f}% | "
                "UTC+8: {time_str} ".format(
                    epoch=epoch,
                    done=done,
                    total_len=len(self.test_dataloader.dataset),
                    percentage=percentage,
                    loss_meter=self.loss_meter.avg if self.loss_meter.avg<999.999 else 999.999,
                    top1=self.top1_acc.avg,
                    # top5=self.top5_acc.avg,
                    time_percent=self.dataload_time.avg/self.batch_time.avg*100,
                    time_str=time_str
                ), end=""
            )
        print("")
        
        # visualize
        if self.vis is not None:
            self.vis.plot('test_loss', self.loss_meter.avg, x=epoch)
            self.vis.plot('test_top1', self.top1_acc.avg, x=epoch)

        return self.loss_meter, self.top1_acc, self.top5_acc
    
