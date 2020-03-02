from itertools import chain
import visdom
import torch
import time
import torchvision as tv
import numpy as np


__all__ = ['get_dataloader', 'gram_matrix', 'img2tensor', 'normalize_batch']


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_dataloader(config):
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
        train_transform = tv.transforms.Compose([
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                mean=[0.5, 0.5, 0.5], 
                std=[0.5, 0.5, 0.5],
            ) # 标准化的过程为(input-mean)/std
        ])
        val_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(), 
            tv.transforms.Normalize(
                mean=(0.5, 0.5, 0.5), 
                std=(0.5, 0.5, 0.5)
            ) # 标准化的过程为(input-mean)/std
        ])
        if config.dataset == 'cifar10': # -----------------cifar10 dataset----------------
            train_dataset = tv.datasets.CIFAR10(
                root=config.dataset_root, 
                train=True, 
                download=False,
                transform=train_transform,
            )
            val_dataset = tv.datasets.CIFAR10(
                root=config.dataset_root,
                train=False,
                download=False,
                transform=val_transform,
            )
            num_classes = 10
        elif config.dataset == 'cifar100': # --------------cifar100 dataset----------------
            train_dataset = tv.datasets.CIFAR100(
                root=config.dataset_root, 
                train=True, 
                download=False,
                transform=train_transform,
            )
            val_dataset = tv.datasets.CIFAR100(
                root=config.dataset_root,
                train=False,
                download=False,
                transform=val_transform,
            )
            num_classes = 100
        else: 
            print("Dataset undefined")
            exit(0)

    elif config.dataset == 'imagenet': # ----------------imagenet dataset------------------
        train_transform = tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ) # 标准化的过程为(input-mean)/std
        ])
        val_transform = tv.transforms.Compose([
            # tv.transforms.RandomResizedCrop(224),
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ) # 标准化的过程为(input-mean)/std
        ])
        train_dataset = tv.datasets.ImageFolder(
            config.dataset_root+'imagenet/img_train/', 
            transform=train_transform
        )
        val_dataset = tv.datasets.ImageFolder(
            config.dataset_root+'imagenet/img_val/', 
            transform=val_transform
        )
        num_classes = 1000
    else: 
        print("Dataset undefined")
        exit(0)


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


def gram_matrix(y):
    """
    Input shape: b,c,h,w\n
    Output shape: b,h*w,c
    """
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = torch.bmm(features, features_t) / (ch * h * w)
    return gram

def img2tensor(path):
    """
    load style image，\n
    Return： tensor shape 1*c*h*w, normalized
    """
    style_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    style_image = tv.datasets.folder.default_loader(path)
    style_tensor = style_transform(style_image)
    return style_tensor.unsqueeze(0) # 在第零维增加一个维度


def normalize_batch(batch):
    """
    Input: b,ch,h,w  0~255\n
    Output: b,ch,h,w  -2~2
    """
    mean = batch.detach().new(IMAGENET_MEAN).view(1, -1, 1, 1)
    std = batch.detach().new(IMAGENET_STD).view(1, -1, 1, 1)
    mean = (mean.expand_as(batch.detach()))
    std = (std.expand_as(batch.detach()))
    return (batch / 255.0 - mean) / std
