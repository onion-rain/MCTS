import torch
import numpy as np
import random
import time

from utils import *

__all__ = ['pruningnet_train', 'PruningnetTrainer']

def pruningnet_train(model, epoch=None, train_dataloader=None, criterion=None, device=None,
            optimizer=None, lr_scheduler=None, vis=None, vis_interval=None):

    model.train() # 训练模式

    # meters
    loss_meter = AverageMeter()
    top1_acc = AverageMeter()
    top5_acc = AverageMeter()
    batch_time = AverageMeter()
    dataload_time = AverageMeter()
    if vis is not None:
        loss_vis = AverageMeter()
        top1_vis = AverageMeter()
    
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel): # 多gpu训练
        channel_scales = model.module.channel_scales
        stage_repeat = model.module.stage_repeat
    else:
        channel_scales = model.channel_scales
        stage_repeat = model.stage_repeat

    end_time = time.time()
    for batch_index, (input, target) in enumerate(train_dataloader):
        # measure data loading time
        dataload_time.update(time.time() - end_time)

        # compute output
        input, target = input.to(device), target.to(device)
        mid_scale_ids = np.random.randint(low=0, high=len(channel_scales), size=sum(stage_repeat)).tolist()
        output_scale_ids = np.random.randint(low=0, high=len(channel_scales), size=sum(stage_repeat[:-1])+1).tolist()
        output_scale_ids += [-1,]*(stage_repeat[-1] + 1) # 最后一个stage输出channel不变
        output = model(input, output_scale_ids, mid_scale_ids)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # meters update
        loss_meter.update(loss.item(), input.size(0))
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        top1_acc.update(prec1.data.cpu(), input.size(0))
        top5_acc.update(prec5.data.cpu(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # print log
        done = (batch_index+1) * train_dataloader.batch_size
        percentage = 100. * (batch_index+1) / len(train_dataloader)
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
                total_len=len(train_dataloader.dataset),
                percentage=percentage,
                loss_meter=loss_meter.avg,
                top1=top1_acc.avg,
                # top5=top5_acc.avg,
                time_percent=dataload_time.avg/batch_time.avg*100,
                lr=optimizer.param_groups[0]['lr'],
            ), end=""
        )

        # visualize
        if vis is not None:
            loss_vis.update(loss.item(), input.size(0))
            top1_vis.update(prec1.data.cpu(), input.size(0))

            if (batch_index % vis_interval == vis_interval-1):
                vis_x = epoch+percentage/100
                vis.plot('train_loss', loss_vis.avg, x=vis_x)
                vis.plot('train_top1', top1_vis.avg, x=vis_x)
                loss_vis.reset()
                top1_vis.reset()

    print("")

    # visualize
    if vis is not None:
        vis.log(
            "epoch: {epoch},  lr: {lr}, <br>\
            train_loss: {train_loss}, <br>\
            train_top1: {train_top1}, <br>"
            .format(
                lr=optimizer.param_groups[0]['lr'],
                epoch=epoch, 
                train_loss=loss_meter.avg,
                train_top1=top1_acc.avg,
            )
        )
    
    # update learning rate
    if lr_scheduler is not None:
        lr_scheduler.step(epoch=epoch)
    
    return model


class PruningnetTrainer(object):
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

        # visdom
        self.vis = config_dic['vis']
        self.vis_interval = config_dic['vis_interval']

        # device
        self.device = config_dic['device']

        # Random Seed
        random.seed(config_dic['seed'])
        torch.manual_seed(config_dic['seed'])

        # step1: data
        self.train_dataloader = config_dic['dataloader']

        # step2: model
        self.model = config_dic['model']

        # step3: criterion
        self.criterion = config_dic['criterion']

        # step4: optimizer
        self.optimizer = config_dic['optimizer']

        # step4: lr_scheduler
        self.lr_scheduler = config_dic['lr_scheduler']
        
    
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


    def train(self, model=None, epoch=None, train_dataloader=None, criterion=None,
                optimizer=None, lr_scheduler=None, vis=None, vis_interval=None):

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

        pruningnet_train(
            self.model, 
            epoch=epoch, 
            train_dataloader=self.train_dataloader, 
            criterion=self.criterion, 
            device=self.device, 
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler, 
            vis=self.vis, 
            vis_interval=self.vis_interval,
        )

