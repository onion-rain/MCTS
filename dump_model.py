# 该文件通常用来加载.pth.tar文件以供debug

import torch

from utils import *

import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)

import models

def dump_model(model_name=None,
                cfg=None,
                checkpoint_path=None,
                num_classes=10,
                save_path=None,
                dataset='cifar10'):
    """
    从.pth.tar中取出state.dict并连同模型保存为.pth
    """
    checkpoint = torch.load(checkpoint_path)
    checkpoint['epoch'] = 0
    torch.save(checkpoint, checkpoint_path)
    return

if __name__ == "__main__":
    dump_model(
        # model_name='resnet110_cs', 
        # cfg=None,
        checkpoint_path='checkpoints/sfp/cifar10_vgg16_bn_cifar_sfp_prune0.4_state_dict_best.pth.tar',
        # num_classes=10,
        # save_path='VGG19BN_slimmed0.7_10.0.pth'
        # dataset="cifar10"
    )
    print()