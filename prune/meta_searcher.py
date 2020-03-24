import torch
import numpy as np
import random
import time

from traintest.trainer import Trainer
from utils import *

__all__ = ['PruningnetTrainer']

class PrunednetSelecter(object):
    """
    """
    def __init__(self, model, train_dataloader, val_dataloader, criterion, device, max_flops):

        self.model = model
        self.train_dataloader - train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.device = device
        self.max_flops = max_flops

        self.tested_genes = {}
        get_model_inform()
        self.gene_length = len(self.stage_repeat) + sum(self.stage_repeat)

    def get_model_inform(self):
        if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel): # 多gpu训练
            self.channel_scales = self.model.module.channel_scales
            self.stage_repeat = self.model.module.stage_repeat
        else:
            self.channel_scales = self.model.channel_scales
            self.stage_repeat = self.model.stage_repeat

    def init_meters(self):
        self.loss_meter = AverageMeter() # 计算所有数的平均值和标准差，这里用来统计一个epoch中的平均值
        self.top1_acc = AverageMeter()
        self.top5_acc = AverageMeter()
        self.batch_time = AverageMeter()
        self.dataload_time = AverageMeter()

    def test(self, epoch=0, scale_ids):
        self.init_meters()
            
        output_scale_ids = []
        output_scale_ids += [scale_ids[0]]
        for i in range(len(stage_repeat)-1):
            output_scale_ids += [scale_ids[i+1]] * self.stage_repeat[i]
        output_scale_ids += [-1] * stage_repeat[-1]
        mid_scale_ids = scale_ids[len(stage_repeat):]
        
        # 首先以训练模式 recalibrate batchnorm
        self.model.train() # 训练模式
        end_time = time.time()
        print("recalibrating batchnorm")
        for batch_index, (input, target) in enumerate(self.train_dataloader):
            if batch_index >= 100:
                break
            # measure data loading time
            self.dataload_time.update(time.time() - end_time)

            # 仅进行数次正向传播更新bn层参数即可
            input, target = input.to(self.device), target.to(self.device)
            output = self.model(input, output_scale_ids, mid_scale_ids)

            # measure elapsed time
            self.batch_time.update(time.time() - end_time)
            end_time = time.time()
        
        # 验证精度
        self.model.eval()
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

        return self.loss_meter, self.top1_acc, self.top5_acc
    
    def random_genes(self, num):
        """随机生成候选基因"""
        genes = []
        while len(genes) < num:
            gene = np.random.randint(low=0, high=len(channel_scales), size=self.gene_length).tolist()
            gene_tuple = tuple(gene[:-1])
            flops = getflops(self.model, gene) # TODO
            if gene in self.tested_genes or flops > self.max_flops:
                continue
            genes.append(gene)
            self.tested_genes[gene_tuple] = -1
        return genes

    def mutant_genes(self):
        """突变基因"""
        genes = []
        return genes

    def crossover_genes(self):
        """交叉基因"""
        genes = []
        return genes

    def natural_selection(self, candidates, select_num):
        sorted_candidates = sorted(candidates, key=lambda can: can[-1])
        return sorted_candidates[:select_num]

    def search(self):
        candidates = self.random_genes()

