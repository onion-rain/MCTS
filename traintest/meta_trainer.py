import torch
import numpy as np
import random
import time

from traintest.trainer import Trainer
from utils import *

__all__ = ['PruningnetTrainer', 'PrunednetTrainer']

class PruningnetTrainer(Trainer):
    """
    TODO 待大量测试
    随机生成各层剪枝比例，metaprune专用
    """
    def __init__(self, model, dataloader, criterion, optimizer, device, 
                 vis=None, vis_interval=20, lr_scheduler=None):

        super(PruningnetTrainer, self).__init__(model, dataloader, criterion, optimizer, 
                                             device, vis, vis_interval, lr_scheduler)

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
        
        if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel): # 多gpu训练
            channel_scales = self.model.module.channel_scales
            stage_repeat = self.model.module.stage_repeat
            gene_length = self.model.module.gene_length
            oc_gene_length = self.model.module.oc_gene_length
        else:
            channel_scales = self.model.channel_scales
            stage_repeat = self.model.stage_repeat
            gene_length = self.model.gene_length
            oc_gene_length = self.model.oc_gene_length

        end_time = time.time()
        for batch_index, (input, target) in enumerate(self.train_dataloader):
            # measure data loading time
            self.dataload_time.update(time.time() - end_time)

            # 随机生成网络结构
            gene = np.random.randint(low=0, high=len(channel_scales), size=gene_length).tolist()
            gene[oc_gene_length-1] = -1 # 最后一个stage输出通道数不变

            # compute output
            input, target = input.to(self.device), target.to(self.device)
            output = self.model(input, gene=gene)
            loss = self.criterion(output, target)

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
            # pbar.set_description(
            print("\r"
                "Train: {epoch:3} "
                "[{done:7}/{total_len:7} ({percentage:3.0f}%)] "
                "loss: {loss_meter:7} | "
                "top1: {top1:6}% | "
                # "top5: {top5:6} | "
                "load_time: {time_percent:3.0f}% | "
                "lr   : {lr:0.1e} ".format(
                    epoch=epoch,
                    done=done,
                    total_len=len(self.train_dataloader.dataset),
                    percentage=percentage,
                    loss_meter=self.loss_meter.avg if self.loss_meter.avg<999.999 else 999.999, 
                    top1=self.top1_acc.avg,
                    # top5=top5_acc.avg,
                    time_percent=self.dataload_time.avg/self.batch_time.avg*100,
                    lr=self.optimizer.param_groups[0]['lr'],
                ), end=""
            )

            # visualize
            if self.vis is not None:
                if (batch_index % self.vis_interval == self.vis_interval-1):
                    vis_x = epoch+percentage/100
                    self.vis.plot('train_loss', self.loss_vis.avg, x=vis_x)
                    self.vis.plot('train_top1', self.top1_vis.avg, x=vis_x)
                    self.loss_vis.reset()
                    self.top1_vis.reset()

        print("")

        # visualize
        if vis is not None:
            vis.log(
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



class PrunednetTrainer(Trainer):
    """
    好吧其实就是Trainer
    """
    def __init__(self, model, dataloader, criterion, optimizer, device, 
                 vis=None, vis_interval=20, lr_scheduler=None):

        super(PrunednetTrainer, self).__init__(model, dataloader, criterion, optimizer, 
                                             device, vis, vis_interval, lr_scheduler)
