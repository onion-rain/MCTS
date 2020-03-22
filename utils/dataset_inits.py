import torch
import time
import torchvision as tv
import numpy as np
import copy as cp

__all__ = ['dataset_init', 'dataset_div_init', 'dataloader_init', 'dataloader_div_init',
           'seed_init',]


def get_cifar_train_transform():
    return tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(
            mean=[0.5, 0.5, 0.5], 
            std=[0.5, 0.5, 0.5],
        ) # 标准化的过程为(input-mean)/std
    ])

def get_cifar_val_transform():
    return tv.transforms.Compose([
        tv.transforms.ToTensor(), 
        tv.transforms.Normalize(
            mean=(0.5, 0.5, 0.5), 
            std=(0.5, 0.5, 0.5)
        ) # 标准化的过程为(input-mean)/std
    ])

def get_imagenet_train_transform():
    return tv.transforms.Compose([
        tv.transforms.RandomResizedCrop(224),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
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
            std=[0.229, 0.224, 0.225],
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
            print("Dataset undefined")
            raise NotImplementedError
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
        print("Dataset undefined")
        raise NotImplementedError
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
        print("Dataset undefined")
        raise NotImplementedError
    
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


def seed_init(config):
    """
    其实pytorch只保证在同版本并且没有多线程的情况下相同的seed可以得到相同的结果，
    而加载数据一般没有不用多线程的，这就有点尴尬了
    """
    if config.deterministic:
        if config.num_workers > 1:
            print("ERROR: Setting --deterministic requires setting --workers to 0 or 1")
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(self.config.random_seed)
        torch.backends.cudnn.deterministic = True
    else: # 让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
        torch.backends.cudnn.benchmark = True



