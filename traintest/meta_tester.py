import torch
import numpy as np
import random
import time

from traintest.tester import Tester
from utils import *

__all__ = ['PruningnetTester', 'PrunednetTester']

class PruningnetTester(Tester):
    """
    随机生成各层剪枝比例，metaprune专用
    """
    def __init__(self, dataloader=None, device=None, criterion=None, vis=None):

        super(PruningnetTester, self).__init__(dataloader, device, criterion, vis)

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
        self.update_attr(model, epoch, test_dataloader, criterion, device, vis)
        
        self.model.eval() # 验证模式
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
        # print("testing...")
        with torch.no_grad():
            for batch_index, (input, target) in enumerate(self.test_dataloader):
                # measure data loading time
                self.dataload_time.update(time.time() - end_time)

                # 随机生成网络结构
                gene = np.random.randint(low=0, high=len(channel_scales), size=gene_length).tolist()
                gene[oc_gene_length-1] = -1 # 最后一个stage输出通道数不变

                # compute output
                input, target = input.to(self.device), target.to(self.device)
                output = self.model(input, gene)
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
                    "loss: {loss_meter:7} | "
                    "top1: {top1:6}% | "
                    # "top5: {top5:6} | "
                    "load_time: {time_percent:3.0f}% | "
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
    


class PrunednetTester(Tester):
    """
    随机生成各层剪枝比例，metaprune专用
    """
    def __init__(self, dataloader=None, device=None, criterion=None, vis=None):

        super(PrunednetTester, self).__init__(dataloader, device, criterion, vis)
