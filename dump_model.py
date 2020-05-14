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
    return

if __name__ == "__main__":
    checkpoint_simple = torch.load('checkpoints/cifar10_test_simple_prune0.5_state_dict_best.pth.tar')
    print(checkpoint_simple['model_state_dict'])
    checkpoint_filter = torch.load('checkpoints/cifar10_test_filter_prune0.5_state_dict_best.pth.tar')
    print(checkpoint_filter['model_state_dict'])
    # dump_model(
    #     model_name='resnet110_cs', 
    #     cfg=None,
    #     checkpoint_path='checkpoints/cifar10_alexnet_cifar_model_checkpoint.pth.tar',
    #     # num_classes=10,
    #     # save_path='VGG19BN_slimmed0.7_10.0.pth'
    #     dataset="cifar10"
    # )
    print()