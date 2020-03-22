from itertools import chain
import visdom
import torch
import time
import torchvision as tv
import numpy as np
import copy as cp


__all__ = ['dataset_div', 'gram_matrix', 'img2tensor', 'normalize_batch']


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def dataset_div(dataset, val_num=50):
    """
    暂时仅支持imagenet
    将dataset拆分成一个train_dataset,一个val_dataset
    val_num为拆分出的验证集每个分类图片个数
    """
    train_dataset = cp.deepcopy(dataset)
    val_dataset = cp.deepcopy(dataset)

    train_imgs = []
    val_imgs = []
    count = [0,]*1000
    for data in dataset.imgs:
        if count[data[1]] < val_num:
            val_imgs.append(data)
            count[data[1]] += 1
        else:
            train_imgs.append(data)

    train_dataset.imgs = train_imgs
    train_dataset.samples = train_imgs
    val_dataset.imgs = val_imgs
    val_dataset.samples = val_imgs
    return train_dataset, val_dataset

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
