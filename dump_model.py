# 该文件通常用来加载.pth.tar文件以供debug

import torch
import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)

import models

def dump_model(model_name=None,
                model_structure=None,
                checkpoint_path=None,
                num_classes=10,
                save_path=None):
    """
    从.pth.tar中取出state.dict并连同模型保存为.pth
    """
    checkpoint = torch.load(checkpoint_path)
    if model_structure is None:
        model_structure = checkpoint['structure']
        print(model_structure)
    model = models.__dict__[model_name](structure=model_structure, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if save_path is not None:
        torch.save(model, save_path)

        # exit(0)

if __name__ == "__main__":
    dump_model(
        model_name='vgg19_bn_cifar', 
        model_structure=None,
        checkpoint_path='checkpoints/cifar10_vgg19_bn_cifar_sr_refine_best.pth.tar',
        # num_classes=10,
        # save_path='VGG19BN_slimmed0.7_10.0.pth'
    )