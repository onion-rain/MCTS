import os
import torch
import math
import time
import numpy as np
import torch.nn.functional as F
from ptflops import get_model_complexity_info
import shutil
import datetime


__all__ = ['print_bar', 'write_log', 'print_model_parameters', 'print_nonzeros', 
           'accuracy', 'get_path', 'CrossEntropyLabelSmooth', 'AverageMeter', 
           'print_flops_params', 'save_checkpoint',]

def print_bar(start_time, arch, dataset):
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

def write_log(filename, content):
    """
    Arguments:
        filename (str)
        content (str)
    """
    with open(filename, 'a') as f:
        content += "\n"
        f.write(content)


def print_model_parameters(model, with_values=False):
    """
    打印网络各层参数的name、shape、type
    """
    print(f"{'Param name':30} {'Shape':30} {'Type':15}")
    print('-'*70)
    for name, param in model.named_parameters():
        print(f'{name:25} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            print(param)


def print_nonzeros(model):
    """
    打印网络各层参数非零元素比例、剪枝比例、shape
    以及网络剩余参数数目、剪枝数、参数总数、压缩率(原参数量/现参数量)、剪枝比
    """
    nonzero = total = 0
    for name, param in model.named_parameters():
        if 'mask' in name:
            continue
        array = param.data.cpu().numpy() # tensor转为array便于numpy处理
        nonzero_count = np.count_nonzero(array)
        total_count = np.prod(array.shape) # 连乘shape获得总数
        nonzero += nonzero_count
        total += total_count
        print(f'{name:25} | nonzeros = {nonzero_count:7} / {total_count:7} ({100 * nonzero_count / total_count:6.2f}%) | pruned = {total_count - nonzero_count :7} | shape = {array.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:5.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True) # 取指定维度(第二维度)上的最大值(或最大几个) pred.shape[batch_size, maxk]
    pred = pred.t() # 转置 pred.shape[maxk, batch_size]
    correct = pred.eq(target.view(1, -1).expand_as(pred)) # correct.shape[maxk, batch_size], correct.dtype=bool

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_path(model_name="model"):
    '''
    获取当前时间前缀
    '''
    prefix = "./checkpoints/" + model_name + '_'
    name = time.strftime(prefix + '20%y%m%d_%H.%M.%S.pth')
    return name

# label smooth
class CrossEntropyLabelSmooth(torch.nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = torch.nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
def print_flops_params(model, dataset='cifar'):
    """
    打印网络flops (GMac)
    打印网络参数量params (M)
    """
    if dataset.startswith("cifar"):
        flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=False)
    elif dataset == "imagenet":
        flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    else:
        print("不支持数据集: {}".format(dataset)) 
        raise NotImplementedError
    print('{:<30}  {:<8}'.format('==> Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('==> Number of parameters: ', params))
    # print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    
def save_checkpoint(state, is_best=False, epoch=None, file_root='checkpoints/', file_name='model'):
    """
    args:
        state: model.state_dict()
        is_best(bool): 是则单独保存到model_best.pth.tar，覆盖至前的
        epoch(int): 若为None则覆盖之前的checkpoint，否则分别保存每一次的checkpoint
        file_root(str): checkpoint文件保存目录
        file_name(str): file_name
    return:
        (str)返回文件保存path：file_root + file_name + '_checkpoint.pth.tar'
    """
    if epoch is not None:
        file_root = file_root + "epoch{}_".format(str(epoch))
    torch.save(state, file_root + file_name + '_checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(file_root+file_name+'_checkpoint.pth.tar', file_root + file_name + '_best.pth.tar')
    return (file_root + file_name + '_checkpoint.pth.tar')