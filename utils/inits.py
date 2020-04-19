import torch
import time
import torchvision as tv
import numpy as np
import copy as cp
import random

import models
from utils.visualize import Visualizer

__all__ = ['suffix_init',
           'device_init',
           'seed_init',
           'dataset_init', 'dataset_div_init', 'dataloader_init', 'dataloader_div_init',
           'model_init', 'distribute_model_init',
           'visdom_init',]


# .d8888. db    db d88888b d88888b d888888b db    db         d888888b d8b   db d888888b d888888b 
# 88'  YP 88    88 88'     88'       `88'   `8b  d8'           `88'   888o  88   `88'   `~~88~~' 
# `8bo.   88    88 88ooo   88ooo      88     `8bd8'             88    88V8o 88    88       88    
#   `Y8b. 88    88 88~~~   88~~~      88     .dPYb.             88    88 V8o88    88       88    
# db   8D 88b  d88 88      88        .88.   .8P  Y8.           .88.   88  V888   .88.      88    
# `8888Y' ~Y8888P' YP      YP      Y888888P YP    YP C88888D Y888888P VP   V8P Y888888P    YP    
def suffix_init(config, suffix_usr=''):
    """
    获取后缀字符串，用在checkpoin、visdom_envirionment、visdom_legend等的命名
    args:
        config(Configuration)
            config.visdom
        suffix_usr(str): 用户可以添加自己的suffix
    """
    suffix = ''
    if config.sr is True:
        suffix += '_sr'
    if config.refine is True:
        suffix += '_refine'
    if config.sfp_intervals is not None:
        suffix += '_sfp'
    if config.max_flops != 0:
        suffix += '_flops{}'.format(config.max_flops)
    # if config.binarynet is True:
    #     suffix += '_binary'
    if config.quantize_type != '':
        suffix += '_{}'.format(config.quantize_type)
    if config.quantize_type.endswith("dorefa"):
        suffix +='_a{a}w{w}g{g}'.format(a=config.a_bits, w=config.w_bits, g=config.g_bits)
    suffix += suffix_usr
    suffix += config.suffix_usr
    print('{:<30}  {:<8}'.format('==> suffix: ', suffix))
    return suffix




# ------------------------------- DEVICE INIT-------------------------------
# d8888b. d88888b db    db d888888b  .o88b. d88888b   d888888b d8b   db d888888b d888888b 
# 88  `8D 88'     88    88   `88'   d8P  Y8 88'         `88'   888o  88   `88'   `~~88~~' 
# 88   88 88ooooo Y8    8P    88    8P      88ooooo      88    88V8o 88    88       88    
# 88   88 88~~~~~ `8b  d8'    88    8b      88~~~~~      88    88 V8o88    88       88    
# 88  .8D 88.      `8bd8'    .88.   Y8b  d8 88.         .88.   88  V888   .88.      88    
# Y8888D' Y88888P    YP    Y888888P  `Y88P' Y88888P   Y888888P VP   V8P Y888888P    YP    
def device_init(config):
    if len(config.gpu_idx_list) > 0:
        device = torch.device('cuda:{}'.format(min(config.gpu_idx_list))) # 起始gpu序号
        print('{:<30}  {:<8}'.format('==> chosen GPU index: ', config.gpu_idx))
    else:
        device = torch.device('cpu')
        print('{:<30}  {:<8}'.format('==> device: ', 'CPU'))
    return device





# ------------------------------- SEED INIT-------------------------------
# .d8888. d88888b d88888b d8888b.   d888888b d8b   db d888888b d888888b 
# 88'  YP 88'     88'     88  `8D     `88'   888o  88   `88'   `~~88~~' 
# `8bo.   88ooooo 88ooooo 88   88      88    88V8o 88    88       88    
#   `Y8b. 88~~~~~ 88~~~~~ 88   88      88    88 V8o88    88       88    
# db   8D 88.     88.     88  .8D     .88.   88  V888   .88.      88    
# `8888Y' Y88888P Y88888P Y8888D'   Y888888P VP   V8P Y888888P    YP    
def seed_init(config):
    """
    其实pytorch只保证在同版本并且没有多线程的情况下相同的seed可以得到相同的结果，
    而加载数据一般没有不用多线程的，这就有点尴尬了
    """
    if config.deterministic:
        if config.num_workers > 1:
            raise Exception("ERROR: Setting --deterministic requires setting --workers to 0 or 1")
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else: # 让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
        torch.backends.cudnn.benchmark = True




# ------------------------------- DATASET INIT-------------------------------
# d8888b.  .d8b.  d888888b  .d8b.  .d8888. d88888b d888888b   d888888b d8b   db d888888b d888888b 
# 88  `8D d8' `8b `~~88~~' d8' `8b 88'  YP 88'     `~~88~~'     `88'   888o  88   `88'   `~~88~~' 
# 88   88 88ooo88    88    88ooo88 `8bo.   88ooooo    88         88    88V8o 88    88       88    
# 88   88 88~~~88    88    88~~~88   `Y8b. 88~~~~~    88         88    88 V8o88    88       88    
# 88  .8D 88   88    88    88   88 db   8D 88.        88        .88.   88  V888   .88.      88    
# Y8888D' YP   YP    YP    YP   YP `8888Y' Y88888P    YP      Y888888P VP   V8P Y888888P    YP    

def get_cifar_train_transform():
    return tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(
            # mean=[0.5, 0.5, 0.5], 
            # std =[0.5, 0.5, 0.5],
            mean=[0.4914, 0.4822, 0.4465], 
            std=[0.2023, 0.1994, 0.2010],
        ) # 标准化的过程为(input-mean)/std
    ])

def get_cifar_val_transform():
    return tv.transforms.Compose([
        # tv.transforms.CenterCrop(32),
        tv.transforms.ToTensor(), 
        tv.transforms.Normalize(
            # mean=[0.5, 0.5, 0.5], 
            # std =[0.5, 0.5, 0.5],
            mean=[0.4914, 0.4822, 0.4465], 
            std=[0.2023, 0.1994, 0.2010],
        ) # 标准化的过程为(input-mean)/std
    ])

def get_imagenet_train_transform():
    return tv.transforms.Compose([
        tv.transforms.RandomResizedCrop(224),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ) # 标准化的过程为(input-mean)/std
    ])

def get_imagenet_val_transform():
    return tv.transforms.Compose([
        # tv.transforms.RandomResizedCrop(224),
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ) # 标准化的过程为(input-mean)/std
    ])

def dataset_init(config):
    """
    args:
        config(Configuration): 
            需要的配置信息：
            config.dataset, 
            config.dataset_root, 
            config.batch_size, 
            config.num_workers, 
            config.droplast
    return:
        train_dataloader, val_dataloader, num_classes
    """
    print('{:<30}  {:<8}'.format('==> Preparing dataset: ', config.dataset))
    if config.dataset.startswith("cifar"): # --------------cifar dataset------------------
        train_transform = get_cifar_train_transform()
        val_transform = get_cifar_val_transform()
        if config.dataset == 'cifar10': # -----------------cifar10 dataset----------------
            train_dataset = tv.datasets.CIFAR10(
                root=config.dataset_root, 
                train=True, 
                download=False,
                transform=train_transform,)
            val_dataset = tv.datasets.CIFAR10(
                root=config.dataset_root,
                train=False,
                download=False,
                transform=val_transform,)
            num_classes = 10
        elif config.dataset == 'cifar100': # --------------cifar100 dataset----------------
            train_dataset = tv.datasets.CIFAR100(
                root=config.dataset_root, 
                train=True, 
                download=False,
                transform=train_transform,)
            val_dataset = tv.datasets.CIFAR100(
                root=config.dataset_root,
                train=False,
                download=False,
                transform=val_transform,)
            num_classes = 100
        else: 
            raise NotImplementedError("不支持数据集: {}".format(config.dataset))
    elif config.dataset == 'imagenet': # ----------------imagenet dataset------------------
        train_transform = get_imagenet_train_transform()
        val_transform = get_imagenet_val_transform()
        train_dataset = tv.datasets.ImageFolder(
            config.dataset_root+'imagenet/img_train/', 
            transform=train_transform,)
        val_dataset = tv.datasets.ImageFolder(
            config.dataset_root+'imagenet/img_val/', 
            transform=val_transform,)
        num_classes = 1000
    else: 
        raise NotImplementedError("不支持数据集: {}".format(config.dataset))
    return train_dataset, val_dataset, num_classes

def dataset_div_init(config, val_num=50):
    """
    暂时仅支持imagenet
    将dataset拆分成一个train_dataset,一个val_dataset
    val_num为拆分出的验证集每个分类图片个数
    args:
        config(Configuration): 
            需要的配置信息：
            config.dataset, 
            config.dataset_root, 
            config.batch_size, 
            config.num_workers, 
            config.droplast
    return:
        train_dataloader, val_dataloader, num_classes
    """
    if config.dataset == 'imagenet': # ----------------imagenet dataset------------------
        train_transform = get_imagenet_train_transform()
        val_transform = get_imagenet_val_transform()
        dataset = tv.datasets.ImageFolder(
            config.dataset_root+'imagenet/img_train/', 
            transform=train_transform)
        num_classes = 1000
    else: 
        raise NotImplementedError("不支持数据集: {}".format(config.dataset))
    
    from utils import dataset_div
    train_dataset, val_dataset = dataset_div(dataset, val_num)
    val_dataset.transform = val_transform

    return train_dataset, val_dataset, num_classes

def dataloader_init(config):
    train_dataset, val_dataset, num_classes = dataset_init(config)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last = config.droplast,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_dataloader, val_dataloader, num_classes

def dataloader_div_init(config, val_num=50):
    """
    暂时仅支持imagenet
    将dataset拆分成一个train_dataset,一个val_dataset
    val_num为拆分出的验证集每个分类图片个数
    """
    train_dataset, val_dataset, num_classes = dataset_div_init(config, val_num)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last = config.droplast,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_dataloader, val_dataloader, num_classes




# ------------------------------- MODEL INIT-------------------------------
# .88b  d88.  .d88b.  d8888b. d88888b db        d888888b d8b   db d888888b d888888b 
# 88'YbdP`88 .8P  Y8. 88  `8D 88'     88          `88'   888o  88   `88'   `~~88~~' 
# 88  88  88 88    88 88   88 88ooooo 88           88    88V8o 88    88       88    
# 88  88  88 88    88 88   88 88~~~~~ 88           88    88 V8o88    88       88    
# 88  88  88 `8b  d8' 88  .8D 88.     88booo.     .88.   88  V888   .88.      88    
# YP  YP  YP  `Y88P'  Y8888D' Y88888P Y88888P   Y888888P VP   V8P Y888888P    YP    
                                                                                  
def model_init(config, device, num_classes):
    """模型训练model通用初始化"""
    print('{:<30}  {:<8}'.format('==> creating arch: ', config.arch))
    model = None
    cfg = None
    checkpoint = None
    if config.resume_path != '': # 断点续练hhh
        checkpoint = torch.load(config.resume_path, map_location=device)
        assert checkpoint['arch'] == config.arch
        print('{:<30}  {:<8}'.format('==> resuming from: ', config.resume_path))
        if config.refine: # 根据cfg加载剪枝后的模型结构
            cfg=checkpoint['cfg']
            print(cfg)
    model = models.__dict__[config.arch]
    if config.arch.endswith('quantized'):
        model = model(type=config.quantize_type, a_bits=config.a_bits, w_bits=config.w_bits, g_bits=config.g_bits, num_classes=num_classes)
    elif cfg is not None:
        model = model(cfg=cfg, num_classes=num_classes)
    else:
        model = model(num_classes=num_classes)
    # print(model)
    model.to(device)
    if len(config.gpu_idx_list) > 1: # 多gpu
        model = torch.nn.DataParallel(model, device_ids=config.gpu_idx_list)
    if checkpoint is not None: # resume
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, cfg, checkpoint

def distribute_model_init(config, device, num_classes):
    """模型训练model通用初始化
    此种模式下只能通过修改CUDA_VISIBLE_DEVICES来选择GPU"""
    print('{:<30}  {:<8}'.format('==> creating arch: ', config.arch))
    model = None
    cfg = None
    checkpoint = None
    if config.resume_path != '': # 断点续练hhh
        checkpoint = torch.load(config.resume_path, map_location=device)
        print('{:<30}  {:<8}'.format('==> resuming from: ', config.resume_path))
        if config.refine: # 根据cfg加载剪枝后的模型结构
            cfg=checkpoint['cfg']
            print(cfg)
    model = models.__dict__[config.arch]
    if config.arch.endswith('dorefanet'):
        model = model(a_bits=config.a_bits, w_bits=config.w_bits, g_bits=config.g_bits, num_classes=num_classes)
    elif cfg is not None:
        model = model(cfg=cfg, num_classes=num_classes)
    else:
        model = model(num_classes=num_classes)
    # print(model)
    model.to(device)
    torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:65535', rank=0, world_size=1)
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, cfg, checkpoint





# ------------------------------- VISDOM INIT -------------------------------
# db    db d888888b .d8888. d8888b.  .d88b.  .88b  d88.   d888888b d8b   db d888888b d888888b 
# 88    88   `88'   88'  YP 88  `8D .8P  Y8. 88'YbdP`88     `88'   888o  88   `88'   `~~88~~' 
# Y8    8P    88    `8bo.   88   88 88    88 88  88  88      88    88V8o 88    88       88    
# `8b  d8'    88      `Y8b. 88   88 88    88 88  88  88      88    88 V8o88    88       88    
#  `8bd8'    .88.   db   8D 88  .8D `8b  d8' 88  88  88     .88.   88  V888   .88.      88    
#    YP    Y888888P `8888Y' Y8888D'  `Y88P'  YP  YP  YP   Y888888P VP   V8P Y888888P    YP    
                                                                                            
def visdom_init(config, suffix='', vis_clear=True):
    vis = None
    vis_interval = None
    if config.visdom:
        if config.vis_env == '':
            config.vis_env = config.dataset + '_' + config.arch + suffix
        if config.vis_legend == '':
            config.vis_legend = config.arch + suffix
        vis = Visualizer(config.vis_env, config.vis_legend, clear=vis_clear) 
        vis_interval = config.vis_interval
    return vis, vis_interval
