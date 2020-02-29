# import ssl
# #全局取消证书验证
# ssl._create_default_https_context = ssl._create_unverified_context

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
    model = models.__dict__[model_name](structure=model_structure, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    torch.save(model, save_path)

        # exit(0)

if __name__ == "__main__":
    dump_model(
        model_name='vgg_cfg', 
        model_structure=None,
        checkpoint_path='slimmed_checkpoints/slimmed_ratio0.7_cifar10_vgg_cfg_checkpoint.pth.tar',
        num_classes=10,
        save_path='VGG19BN_slimmed0.7_10.0.pth'
    )