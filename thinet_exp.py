
from utils.visualize import Visualizer
from tqdm import tqdm
from torch.nn import functional as F
import torchvision as tv
import time
import os
import random
import numpy as np
import copy
import argparse
import datetime

from tester import Tester
from config import Configuration
import models
from utils import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"



import torch
import numpy as np
import math

import models

# # 提取隐藏层features
# class FeatureExtractor:
#     features = None
 
#     def __init__(self, model, layer_num):
#         self.hook = model[layer_num].register_forward_hook(self.hook_fn)
 
#     def hook_fn(self, module, input, output):
#         self.features = output.cpu()
 
#     def remove(self):
#         self.hook.remove()

# def get_hidden_output_feature(model, idx, x):
#     """return model第idx层的前向传播输出特征图"""
#     feature_extractor = FeatureExtractor(model, idx) # 注册钩子
#     out = model(x)
#     feature_extractor.remove() # 销毁钩子
#     return feature_extractor.features # 第idx层输出的特征


def print_bar(str, channel_num, start_time):
    """calculate duration time"""
    interval = datetime.datetime.now() - start_time
    print("-------- pruned: {str}  --  channel num: {channel_num}  --  duration: {dh:2}h:{dm:02d}.{ds:02d}  --------".
        format(
            str=str,
            channel_num=channel_num,
            dh=interval.seconds//3600,
            dm=interval.seconds%3600//60,
            ds=interval.seconds%60,
        )
    )

def get_tuples(model):
    """
    Code from https://github.com/synxlin/nn-compression.
    获得计算第i层以及i+1层输入特征图的前向传播函数
    
    return:
        list of tuple, [(module_name, module, next_bn, next_module, fn_input_feature, fn_next_input_feature), ...]
    """
    # 提取各层
    features = model.features
    if isinstance(features, torch.nn.DataParallel):
        features = features.module
    classifier = model.classifier

    module_name_dict = dict()
    for n, m in model.named_modules():
        module_name_dict[m] = n

    conv_indices = []
    conv_modules = []
    conv_names = []
    bn_modules = []
    for i, m in enumerate(features):
        if isinstance(m, torch.nn.modules.conv._ConvNd):
            conv_indices.append(i)
            conv_modules.append(m)
            conv_names.append(module_name_dict[m])
            bn_modules.append(None)
        elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            if bn_modules[-1] is None:
                bn_modules[-1] = m # 其实还有种隐患，不过应该没有哪个模型一个conv后面跟两个bn吧hhh
    
    # 获得第idx个卷积层输入特征图
    def get_fn_conv_input_feature(idx):
        def fn(x):
            if idx == 0:
                return x
            else:
            #     return get_hidden_output_feature(features, conv_indices[idx]-1, x)
                for layer in range(conv_indices[idx]):
                    x = features[layer](x)
                return x
        return fn

    # 获得第idx+1个卷积层输入特征图
    def get_fn_next_input_feature(idx):
        def fn(x):
            if idx+1 < len(conv_indices):
                for layer in range(conv_indices[idx]+1, conv_indices[idx+1]):
                    x = features[layer](x)
            else: # 下层为fc
                for layer in range(conv_indices[-1]+1, len(features)):
                    x = features[layer](x)
                x = model.avgpool(x)
                x = torch.flatten(x, 1)
            return x
        return fn

    modules = []
    module_names = []
    fn_input_feature = []
    fn_next_input_feature = []

    for i in range(len(conv_indices)):
        # modules.append(conv_modules[i])
        # module_names.append(conv_names[i])
        fn_input_feature.append(get_fn_conv_input_feature(i))
        fn_next_input_feature.append(get_fn_next_input_feature(i))

    conv_modules.append(classifier) # 图省事直接append到conv_modules里面

    tuples = []
    for i in range(len(conv_names)):
        tuples.append((conv_names[i], conv_modules[i], bn_modules[i], conv_modules[i+1],
                                fn_input_feature[i], fn_next_input_feature[i]))
    # for i in range(len(conv_names)):
    #     tuples.append((conv_names[-2], conv_modules[-2], bn_modules[-2], conv_modules[-1],
    #                             fn_input_feature[-2], fn_next_input_feature[-2]))

    return tuples

def channel_select(sparsity, output_feature, fn_next_input_feature, next_module, method='greedy', p=2):
    """next(_conv)_output_feature到next2_input_feature之间算是一种恒定的变换，
    因此这里不比较i+2层卷积层的输入，转而比较i+1层卷积层的输出"""
    original_channel_num = output_feature.size(1)
    purned_channel_num = int(math.floor(original_channel_num * sparsity)) # 向下取整

    if method == 'greedy':
        indices_pruned = []
        while len(indices_pruned) < purned_channel_num:
            min_diff = 1e10
            min_idx = 0
            for idx in range(original_channel_num):
                if idx in indices_pruned:
                    continue
                indices_try = indices_pruned + [idx]
                output_feature_try = torch.zeros_like(output_feature)
                output_feature_try[:, indices_try, ...] = output_feature[:, indices_try, ...]
                next_output_feature_try = next_module(fn_next_input_feature(output_feature_try))
                next_output_feature_try_norm = next_output_feature_try.norm(p)
                if next_output_feature_try_norm < min_diff:
                    min_diff = next_output_feature_try_norm
                    min_idx = idx
            indices_pruned.append(min_idx)
    elif method == 'lasso':
        raise NotImplementedError
    elif method == 'random':
        indices_pruned = random.sample(range(original_channel_num), purned_channel_num)
    else:
        raise NotImplementedError

    return indices_pruned

def module_surgery(module, next_bn, next_module, indices_pruned, device):
    """根据indices_pruned实现filter的删除与权重的recover"""
    # operate module
    if isinstance(module, torch.nn.modules.conv._ConvNd):
        indices_stayed = list(set(range(module.out_channels)) - set(indices_pruned))
        num_channels_stayed = len(indices_stayed)
        module.out_channels = num_channels_stayed
    else:
        raise NotImplementedError
    # operate module weight
    new_weight = module.weight[indices_stayed, ...].clone()
    del module.weight
    module.weight = torch.nn.Parameter(new_weight)
    # operate module bias
    if module.bias is not None:
        new_bias = module.bias[indices_stayed, ...].clone()
        del module.bias
        module.bias = torch.nn.Parameter(new_bias)
    

    if next_bn is not None:
        # operate batch_norm
        if isinstance(next_bn, torch.nn.modules.batchnorm._BatchNorm):
            next_bn.num_features = num_channels_stayed
        else:
            raise NotImplementedError
        # operate batch_norm weight
        new_weight = next_bn.weight.data[indices_stayed].clone()
        new_bias = next_bn.bias.data[indices_stayed].clone()
        new_running_mean = next_bn.running_mean[indices_stayed].clone()
        new_running_var = next_bn.running_var[indices_stayed].clone()
        del next_bn.weight, next_bn.bias, next_bn.running_mean, next_bn.running_var
        next_bn.weight = torch.nn.Parameter(new_weight)
        next_bn.bias = torch.nn.Parameter(new_bias)
        next_bn.running_mean = new_running_mean
        next_bn.running_var = new_running_var


    # operate next_module
    if isinstance(next_module, torch.nn.modules.conv._ConvNd):
        next_module.in_channels = num_channels_stayed
    elif isinstance(next_module, torch.nn.modules.linear.Linear):
        print("fc")
    else:
        raise NotImplementedError
    # operate next_module weight
    new_weight = next_module.weight[:, indices_stayed, ...].clone()
    del next_module.weight
    next_module.weight = torch.nn.Parameter(new_weight)

def weight_reconstruction(next_module, next_input_feature, next_output_feature, device=None):
    if next_module.bias is not None:
        bias_size = [1] * next_output_feature.dim()
        bias_size[1] = -1
        next_output_feature -= next_module.bias.view(bias_size)
    if isinstance(next_module, torch.nn.modules.conv._ConvNd):
        unfold = torch.nn.Unfold(kernel_size=next_module.kernel_size,
                                dilation=next_module.dilation,
                                padding=next_module.padding,
                                stride=next_module.stride)
        unfold = unfold.to(device)
        unfold.eval()
        next_input_feature = unfold(next_input_feature)
        next_input_feature = next_input_feature.transpose(1, 2)
        num_fields = next_input_feature.size(0) * next_input_feature.size(1)
        next_input_feature = next_input_feature.reshape(num_fields, -1)
        next_output_feature = next_output_feature.view(next_output_feature.size(0), next_output_feature.size(1), -1)
        next_output_feature = next_output_feature.transpose(1, 2).reshape(num_fields, -1)

    param, _ = torch.lstsq(next_output_feature.data, next_input_feature.data) # 计算最小二乘的解
    param = param[0:next_input_feature.size(1), :].clone().t().contiguous().view(next_output_feature.size(1), -1)
    if isinstance(next_module, torch.nn.modules.conv._ConvNd):
        param = param.view(next_module.out_channels, next_module.in_channels, *next_module.kernel_size)
    del next_module.weight
    next_module.weight = torch.nn.Parameter(param)


def prune(model, sparsity, dataloader, device, method, p):

    start_time = datetime.datetime.now()

    input_iter = iter(dataloader)
    tuples = get_tuples(model)

    print_bar("get_tuples", 0, start_time)

    for (module_name, module, next_bn, next_module, fn_input_feature, fn_next_input_feature) in tuples:
        # 此处module和next_module均为conv module
        input, _ = input_iter.__next__()
        input = input.to(device)
        input_feature = fn_input_feature(input)
        input_feature = input_feature.to(device)

        output_feature = module(input_feature) # 之后我们要对这玩意下刀，去掉几个channel
        next_input_feature = fn_next_input_feature(output_feature)
        next_output_feature = next_module(next_input_feature)

        # sparsity = get_param_sparsity(module_name)
        indices_pruned = channel_select(sparsity, output_feature, fn_next_input_feature, next_module, method)
        module_surgery(module, next_bn, next_module, indices_pruned, device)

        # 通道剪枝后更新特征图
        output_feature = module(input_feature)
        next_input_feature = fn_next_input_feature(output_feature)

        # weight_reconstruction(next_module, next_input_feature, next_output_feature, device)

        print_bar(module_name, module.out_channels, start_time)


class Pruner(object):
    """
    导入的模型必须有BN层，
    并且事先进行稀疏训练，
    并且全连接层前要将左右特征图池化为1x1，即最终卷积输出的通道数即为全连接层输入通道数
    """
    def __init__(self, **kwargs):

        print("| ----------------- Initializing Pruner ----------------- |")

        self.config = Configuration()
        self.config.update_config(kwargs) # 解析参数更新默认配置
        if self.config.check_config(): raise # 检测路径、设备是否存在
        print('{:<30}  {:<8}'.format('==> num_workers: ', self.config.num_workers))

        self.suffix = get_suffix(self.config)
        print('{:<30}  {:<8}'.format('==> suffix: ', self.suffix))
        
        # 更新一些默认标志
        self.best_acc1 = 0
        self.checkpoint = None

        # device
        if len(self.config.gpu_idx_list) > 0:
            self.device = torch.device('cuda:{}'.format(min(self.config.gpu_idx_list))) # 起始gpu序号
            print('{:<30}  {:<8}'.format('==> chosen GPU idx: ', self.config.gpu_idx))
        else:
            self.device = torch.device('cpu')
            print('{:<30}  {:<8}'.format('==> device: ', 'CPU'))

        # step1: data
        _, self.val_dataloader, self.num_classes = get_dataloader(self.config)

        # step2: model
        print('{:<30}  {:<8}'.format('==> creating arch: ', self.config.arch))
        cfg = None
        if self.config.resume_path != '': # 断点续练hhh
            checkpoint = torch.load(self.config.resume_path, map_location=self.device)
            print('{:<30}  {:<8}'.format('==> resuming from: ', self.config.resume_path))
            if self.config.refine: # 根据cfg加载剪枝后的模型结构
                cfg=checkpoint['cfg']
                print(cfg)
        else: 
            print("你剪枝不加载模型剪锤子??")
            exit(0)
        self.model = models.__dict__[self.config.arch](cfg=cfg, num_classes=self.num_classes) # 从models中获取名为config.model的model
        if len(self.config.gpu_idx_list) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config.gpu_idx_list)
        self.model.to(self.device) # 模型转移到设备上
        # if checkpoint is not None:
        self.best_acc1 = checkpoint['best_acc1']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("{:<30}  {:<8}".format('==> checkpoint trained epoch: ', checkpoint['epoch']))
        print("{:<30}  {:<8}".format('==> checkpoint best acc1: ', checkpoint['best_acc1']))

        self.vis = None

        # step6: valuator
        val_config_dic = {
            'model': self.model,
            'dataloader': self.val_dataloader,
            'device': self.device,
            'vis': self.vis,
            'seed': self.config.random_seed
        }
        self.valuator = Tester(val_config_dic)


    def run(self):

        # print("")
        # print("| -------------------- original model -------------------- |")
        # self.valuator.test(self.model)
        # print_flops_params(self.model, self.config.dataset)
        # # print_model_parameters(self.valuator.model)


        print("")
        print("| -------------------- pruning model -------------------- |")
        prune(self.model, self.config.prune_percent, self.val_dataloader, self.device, 'greedy', self.config.lp_norm)
        self.valuator.test(self.model)
        print_flops_params(self.model, self.config.dataset)

        # # save pruned model
        # name = ('weight_pruned' + str(self.config.prune_percent) 
        #         + '_' + self.config.dataset 
        #         + "_" + self.config.arch
        #         + self.suffix)
        # if len(self.config.gpu_idx_list) > 1:
        #     state_dict = self.pruned_model.module.state_dict()
        # else: state_dict = self.pruned_model.state_dict()
        # path = save_checkpoint({
        #     # 'cfg': cfg,
        #     'ratio': self.prune_ratio,
        #     'model_state_dict': state_dict,
        #     'best_acc1': self.valuator.top1_acc.avg,
        # }, file_root='checkpoints/weight_pruned/', file_name=name)
        # print('{:<30}  {}'.format('==> pruned model save path: ', path))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='network pruner')
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH', default='vgg19_bn_cifar',
                        choices=models.ALL_MODEL_NAMES,
                        help='model architecture: ' +
                        ' | '.join(name for name in models.ALL_MODEL_NAMES) +
                        ' (default: resnet18)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--workers', type=int, default=10, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--gpu', type=str, default='0', metavar='gpu_idx',
                        help='training GPU idx(default:"0",which means use GPU0')
    parser.add_argument('--resume', dest='resume_path', type=str, default='',
                        metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--refine', action='store_true',
                        help='refine from pruned model, use construction to build the model')
    parser.add_argument('--prune-percent', type=float, default=0.5, 
                        help='percentage of weight to prune(default: 0.5)')
    parser.add_argument('--lp-norm', '-lp', dest='lp_norm', type=int, default=2, 
                        help='the order of norm(default: 2)')


    args = parser.parse_args()


    pruner = Pruner(
        arch=args.arch,
        dataset=args.dataset,
        num_workers = args.workers, # 使用多进程加载数据
        gpu_idx = args.gpu, # choose gpu
        resume_path=args.resume_path,
        refine=args.refine,

        prune_percent=args.prune_percent,
        lp_norm=args.lp_norm
    )
    pruner.run()
    print("end")
