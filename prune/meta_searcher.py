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
    def __init__(self, model, train_dataloader, val_dataloader, criterion, device, vis, max_flops):

        # 一些默认超参数值
        self.population = 15
        self.mutation_num = 5
        self.crossover_num = 5
        self.select_num = 10
        self.max_iter = 50
        self.mut_prob = 0.1

        # 一些变量的初始化
        self.candidates = [] # 每个元素的最后一位存储精度信息
        self.survival = []

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.device = device
        self.vis = vis
        self.max_flops = max_flops

        self.flopped_genes_tuple = {}
        self.tested_genes_tuple = {}
        self.get_model_inform()
        self.gene_length = len(self.stage_repeat)+1 + sum(self.stage_repeat)

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
        # print("recalibrating batchnorm")
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
                    total_len=len(self.val_dataloader.dataset),
                    percentage=percentage,
                    loss_meter=self.loss_meter.avg if self.loss_meter.avg<999.999 else 999.999,
                    top1=self.top1_acc.avg,
                    # top5=self.top5_acc.avg,
                    time_percent=self.dataload_time.avg/self.batch_time.avg*100,
                    time_str=time_str
                ), end=""
            )
        gene[-1] = 100 - self.top1_acc.avg.item()
        print("")

    def test_candidates(self, candidates, iter):
        print(' ---------------------------  iter = {}  ---------------------------------- '.format(iter))
        idx = 0
        for gene in candidates:
            gene_tuple = tuple(gene[:-1])
            assert gene_tuple not in self.tested_genes_tuple.keys()
            self.test_gene(gene, epoch=idx)
            self.tested_genes_tuple[gene_tuple] = gene[-1]
            idx += 1
        print([acc[-1] for acc in candidates]) # 打印本次测试得到的精度列表

    def get_random_genes(self, num):
        """随机生成候选基因(最后一位非精度)"""
        genes = []
        while len(genes) < num:
            gene = np.random.randint(low=0, high=len(self.channel_scales), size=self.gene_length).tolist()
            gene[len(self.stage_repeat)] = -1 # 最后一个stage输出通道数不变
            gene_tuple = tuple(gene[:-1])



            # test_model = self.model.__class__(self.model.stage_repeat, gene=gene).to(self.device)
            # flops = get_model_flops(test_model, 'imagenet', pr=False)
            # if gene_tuple in self.flopped_genes_tuple.keys() or flops > self.max_flops:
            #     continue



            genes.append(gene)
            self.flopped_genes_tuple[gene_tuple] = -1
        # self.candidates += genes # 纳入候选列表
        return genes

    def get_mutant_genes(self, candidates, num, mut_prob):
        """突变基因"""
        genes = []
        mutant_gene_ids = np.random.choice(len(candidates), num)
        mutant_genes = [candidates[id] for id in mutant_gene_ids]
        mutant_layer_ids = np.random.choice(np.arange(0, 2), (num, self.gene_length), p=(1-mut_prob, mut_prob)) # 1表示突变0表示不突变
        mutant_distance = np.random.choice(np.arange(1, len(self.channel_scales)), (num, self.gene_length)) * mutant_layer_ids
        mutanted_genes = (mutant_genes + mutant_distance) % len(self.channel_scales)
        return mutanted_genes.tolist()

    def get_crossover_genes(self):
        """交叉基因"""
        genes = []
        return genes

    def natural_selection(self, candidates, select_num):
        sorted_candidates = sorted(candidates, key=lambda x: x[-1])
        return sorted_candidates[:select_num]

    def search(self):
        print("preparing candidates...")
        candidates = self.get_random_genes(self.population)


        self.test_candidates(candidates, iter=0)
        candidates = self.natural_selection(candidates, self.select_num)
        # print(candidates)
        exit(0)
        for iter in range(self.max_iter):
            print("preparing candidates...")
            mutant = self.get_mutant_genes(candidates, self.mutation_num)
            crossover = self.get_crossover_genes(self.crossover_num)
            rand = self.get_random_genes(self.population - len(mutant) - len(crossover) - len(candidates))
            candidates = []
            candidates.append(mutant)
            candidates.append(crossover)
            candidates.append(rand)
            self.test_candidates(candidates, iter=iter)
            self.natural_selection(candidates, 10)

            


            

