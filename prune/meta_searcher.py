import torch
import numpy as np
import random
import time
import copy as cp

from traintest.trainer import Trainer
from utils import *

__all__ = ['PrunednetSearcher']

class PrunednetSearcher(object):
    """
    """
    def __init__(self, model, train_dataloader, val_dataloader, criterion, device, vis, max_flops=800,
                    population=6, select_num=2, mutation_num=2, crossover_num=2, mutation_prob=0.1,
                    checked_genes_tuple={}, tested_genes_tuple={}, flops_model=None):

        # 一些超参数
        self.max_flops = max_flops
        self.population = population
        self.select_num = select_num
        self.mutation_num = mutation_num
        self.crossover_num = crossover_num
        self.mutation_prob = mutation_prob

        # 一些变量的初始化
        self.candidates = [] # 每个元素的最后一位存储精度信息(top1 error)
        self.survival = []

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.device = device
        self.vis = vis
        self.flops_model = flops_model

        self.checked_genes_tuple = checked_genes_tuple # keys不包含最后一维精度位，键值存储top1 error
        self.tested_genes_tuple = tested_genes_tuple
        self.get_model_inform()
        if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.gene_length = model.module.gene_length + 1 # 最后为精度位（top1 error）
            self.oc_gene_length = model.module.oc_gene_length
        else: 
            self.gene_length = model.gene_length + 1 # 最后为精度位（top1 error）
            self.oc_gene_length = model.oc_gene_length

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

    def test_gene(self, gene, epoch=0):
        # 测试每个候选基因的精度，保存到最后一位
        self.init_meters()
        
        # 首先以训练模式 recalibrate batchnorm
        self.model.train() # 训练模式
        end_time = time.time()
        print("recalibrating batchnorm...", end="", flush=True)
        for batch_index, (input, target) in enumerate(self.train_dataloader):
            if batch_index >= 100:
                break
            # measure data loading time
            self.dataload_time.update(time.time() - end_time)

            # 仅进行数次正向传播更新bn层参数即可
            input, target = input.to(self.device), target.to(self.device)
            output = self.model(input, gene=gene)

            # measure elapsed time
            self.batch_time.update(time.time() - end_time)
            end_time = time.time()
        
        # 验证精度
        self.model.eval()
        end_time = time.time()
        # print("testing...")
        with torch.no_grad():
            for batch_index, (input, target) in enumerate(self.val_dataloader):
                # measure data loading time
                self.dataload_time.update(time.time() - end_time)

                # compute output
                input, target = input.to(self.device), target.to(self.device)
                output = self.model(input, gene=gene)
                loss = self.criterion(output, target)

                # meters update and visualize
                self.loss_meter.update(loss.item(), input.size(0))
                prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
                self.top1_acc.update(prec1.data.cpu(), input.size(0))
                # self.self.top5_acc.update(prec5.data.cpu(), input.size(0))

                # measure elapsed time
                self.batch_time.update(time.time() - end_time)
                end_time = time.time()

                done = (batch_index+1) * self.val_dataloader.batch_size
                percentage = 100. * (batch_index+1) / len(self.val_dataloader)
                print("\r"
                    "Test: {epoch:4} "
                    "[{done:7}/{total_len:7} ({percentage:3.0f}%)] "
                    "loss: {loss_meter:7} | "
                    "top1 error: {top1:6}% | "
                    "load_time: {time_percent:3.0f}% | "
                    "UTC+8: {time_str} ".format(
                        epoch=epoch,
                        done=done,
                        total_len=len(self.val_dataloader.dataset),
                        percentage=percentage,
                        loss_meter=self.loss_meter.avg if self.loss_meter.avg<999.999 else 999.999,
                        top1=self.top1_acc.err_avg,
                        time_percent=self.dataload_time.avg/self.batch_time.avg*100,
                        time_str=time.strftime('%H:%M:%S')
                    ), end=""
                )
        gene[-1] = 100 - self.top1_acc.avg
        print("")

    def test_candidates(self, candidates):
        idx = 0
        for gene in candidates:
            gene_tuple = tuple(gene[:-1])
            if gene_tuple not in self.tested_genes_tuple.keys():
                self.test_gene(gene, epoch=idx)
                self.tested_genes_tuple[gene_tuple] = gene[-1]
            else:
                print(
                    "Test: {epoch:4} | "
                    "{Tested:^36} | "
                    "top1 error: {top1:6}% | "
                    "load_time: {time_percent:3.0f}% | "
                    "UTC+8: {time_str} ".format(
                        epoch=idx,
                        Tested='tested gene',
                        top1=self.tested_genes_tuple[gene_tuple],
                        time_percent=0,
                        time_str=time.strftime('%H:%M:%S')
                    )
                )
            idx += 1
        # print([acc[-1] for acc in candidates]) # 打印本次测试得到的精度列表

    def check_gene(self, gene):
        """
        最后一个stage输出通道不压缩，
        top1 error初始化为100
        检查是否已被检测过
        返回0表示需要被丢弃，返回1表示可用"""
        gene[self.oc_gene_length-1] = -1 # 最后一个stage输出通道数不变
        gene_tuple = tuple(gene[:-1])
        if gene_tuple in self.checked_genes_tuple.keys():
            return 0
        gene[-1] = 100.0 # top1 error初始化为100
        flops = -1
        if self.max_flops > 0:
            flops_model = self.flops_model.__class__(stage_repeat=self.model.stage_repeat, num_classes=self.model.num_classes, gene=gene).to(self.device)
            flops = get_model_flops(flops_model, 'imagenet', pr=False)
            if flops > self.max_flops:
                return 0
        self.checked_genes_tuple[gene_tuple] = -1 # 标记已检测
        return flops

    def get_random_genes(self, num, pr=False):
        """随机生成候选基因(最后一位非精度)"""
        genes = []
        while len(genes) < num:
            gene = np.random.randint(low=0, high=len(self.channel_scales), size=self.gene_length).tolist()
            flops = self.check_gene(gene)
            if flops:
                genes.append(gene)
                if pr ==True:
                    print("\r{prefix:30}"
                        "[{done:3}/{total:3} ({percentage:3.0f}%)] "
                        "flops: {flops:.3f} ".format(
                            prefix="generating random genes: ",
                            done=len(genes),
                            total=num,
                            percentage=len(genes)/num*100,
                            flops=flops,
                        ), end="" if len(genes) < num else '\n', flush=True
                    )
        return genes

    def get_mutant_genes(self, candidates, num, prob, max_iter=10, pr=False):
        """从candidates中产生突变基因"""
        genes = []
        iter = 0
        while len(genes)<num and iter<max_iter:
            mutant_gene_ids = np.random.choice(len(candidates), num)
            mutant_genes = [candidates[id] for id in mutant_gene_ids]
            mutant_layer_ids = np.random.choice(np.arange(0, 2), (num, self.gene_length), p=(1-prob, prob)) # 1表示突变0表示不突变
            mutant_distance = np.random.choice(np.arange(1, len(self.channel_scales)), (num, self.gene_length)) * mutant_layer_ids
            mutanted_genes = ((mutant_genes + mutant_distance) % len(self.channel_scales)).astype(int)
            iter += 1
            for gene in mutanted_genes:
                flops = self.check_gene(gene)
                if flops:
                    genes.append(gene.tolist())
                    if pr ==True:
                        print("\r{prefix:30}"
                            "[{done:3}/{total:3} ({percentage:3.0f}%)] "
                            "flops: {flops:.3f} ".format(
                                prefix="generating mutant genes: ",
                                done=len(genes),
                                total=num,
                                percentage=len(genes)/num*100,
                                flops=flops,
                            ), end="" if len(genes)<num and iter<max_iter else '\n', flush=True
                        )
                    if len(genes) == num:
                        break
        return genes

    def get_crossover_genes(self, candidates, num, max_iter=10, pr=False):
        """从candidates产生交叉基因"""
        genes = []
        iter = 0
        while len(genes)<num and iter<max_iter:
            gene0_id, gene1_id = np.random.choice(len(candidates), 2, replace=False)
            gene0 = candidates[gene0_id]
            gene1 = candidates[gene1_id]
            mask = np.random.choice(np.arange(0, 2), self.gene_length) # 0表示用0，1表示用1
            crossed_gene = (gene0*(1-mask) + gene1*mask).astype(int)
            flops = self.check_gene(crossed_gene)
            if flops:
                genes.append(crossed_gene.tolist())
                iter = 0
                if pr ==True:
                    print("\r{prefix:30}"
                        "[{done:3}/{total:3} ({percentage:3.0f}%)] "
                        "flops: {flops:.3f} ".format(
                            prefix="generating crossover genes: ",
                            done=len(genes),
                            total=num,
                            percentage=len(genes)/num*100,
                            flops=flops,
                        ), end="" if len(genes)<num and iter<max_iter else '\n', flush=True
                    )
            else: iter += 1
        return genes

    def natural_selection(self, candidates, select_num):
        self.test_candidates(candidates)
        sorted_candidates = sorted(candidates, key=lambda x: x[-1])
        print([acc[-1] for acc in sorted_candidates[:select_num]], flush=True) # 打印本次测试得到的精度列表
        return sorted_candidates[:select_num]

    def search(self, iter, candidates):
        """执行一次搜索，返回剩余candidates"""

        # print(' ----------------------------------  iter = {:>2}  ---------------------------------- '.format(iter))
        if iter == 0:
            candidates = self.get_random_genes(self.population, pr=True)
            candidates = self.natural_selection(candidates, self.select_num)
            return candidates, self.checked_genes_tuple, self.tested_genes_tuple
        else:
            mutant = self.get_mutant_genes(candidates, self.mutation_num, self.mutation_prob, pr=True)
            crossover = self.get_crossover_genes(candidates, self.crossover_num, pr=True)
            rand = self.get_random_genes(self.population - len(mutant) - len(crossover) - len(candidates), pr=True)
            candidates.extend(mutant)
            candidates.extend(crossover)
            candidates.extend(rand)
            # print("{:<30}  {:<8}".format('==> total candidates num: ', len(candidates)))
            candidates = self.natural_selection(candidates, self.select_num)
            return candidates, self.checked_genes_tuple, self.tested_genes_tuple
