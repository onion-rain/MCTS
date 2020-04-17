import os
import sys
import torch
import math
import time
import numpy as np
import torch.nn.functional as F
from ptflops import get_model_complexity_info
import shutil
import datetime
from torch.autograd import Variable


__all__ = ['print_bar', 'write_log', 'print_model_parameters', 'print_nonzeros', 
           'accuracy', 'get_path', 'CrossEntropyLabelSmooth', 'AverageMeter', 
           'print_flops_params', 'save_checkpoint', 'get_model_flops', 'Logger']

def print_bar(start_time, arch, dataset, best_top1=0, epoch=None):
    """calculate duration time"""
    if epoch is not None:
        print("--  {:^3}  ".format(epoch), end='')
    interval = datetime.datetime.now() - start_time
    print("--------  {model}  --  {dataset}  --  best_top1: {best_top1:.3f}  --  duration: {dh:2}h:{dm:02d}.{ds:02d}  --------".
        format(
            model=arch,
            dataset=dataset,
            best_top1=best_top1,
            dh=interval.seconds//3600 + interval.days*24,
            dm=interval.seconds%3600//60,
            ds=interval.seconds%60,
        ), flush=True,
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
    """Computes and stores the average and current value
    均值默认保留3(round)位小数"""
    def __init__(self, round=3):
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
        self.avg = round(float(self.sum / self.count), 3)
        self.err_avg = round(100-self.avg, 3)

def flops_to_string(flops, units=None, precision=2):
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GMac'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MMac'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'

def params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10 ** 6 > 0:
            return str(round(params_num / 10 ** 6, 2)) + ' M'
        elif params_num // 10 ** 3:
            return str(round(params_num / 10 ** 3, 2)) + ' k'
        else:
            return str(params_num)
    else:
        if units == 'M':
            return str(round(params_num / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(params_num / 10.**3, precision)) + ' ' + units
        else:
            return str(params_num)

def print_flops_params(model, dataset='cifar', print_per_layer_stat=False):
    """
    打印网络flops (GMac)
    打印网络参数量params (M)
    """
    if dataset.startswith("cifar"):
        flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings=False, print_per_layer_stat=print_per_layer_stat)
    elif dataset == "imagenet":
        flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=False, print_per_layer_stat=print_per_layer_stat)
    else:
        raise NotImplementedError("不支持数据集: {}".format(dataset))
    flops = flops_to_string(flops)
    params = params_to_string(params)
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


def get_model_flops(one_shot_model, dataset='cifar', pr=False):
    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    multiply_adds = False
    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)


    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)

    foo(one_shot_model)
    device = next(one_shot_model.parameters()).device
    if dataset.startswith("cifar"):
        input = Variable(torch.rand(3,32, 32).unsqueeze(0), requires_grad = True).to(device)
    elif dataset == 'imagenet':
        input = Variable(torch.rand(3,224,224).unsqueeze(0), requires_grad = True).to(device)
    else: 
        raise NotImplementedError("不支持数据集: {}".format(dataset))
    out = one_shot_model(input)
    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    M_flops = total_flops / 1e6
    if pr == True:
        print('{:<30}  {:.2f}M'.format('==> Number of FLOPs: ', M_flops))

    return M_flops

class Logger(object):
    def __init__(self, filepath="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass
