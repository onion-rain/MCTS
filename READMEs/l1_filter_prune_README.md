# pytorch-l1_filter_prune

论文地址：[Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) (ICLR 2017)


usage: ```trainer.py [-h] [--arch ARCH] [--dataset DATASET] [--workers N]
                  [--batch-size N] [--epochs N] [--lr LR] [--weight-decay W]
                  [--gpu GPU] [--deterministic] [--momentum M] [--valuate]
                  [--resume PATH] [--refine] [--sparsity-regularization]
                  [--srl SR_LAMBDA] [--visdom] [--vis-env ENV]
                  [--vis-legend LEGEND] [--vis-interval N]```

usage: ```pruner.py [-h] [--arch ARCH] [--dataset DATASET] [--workers N]
                 [--gpu gpu_idx] [--resume PATH] [--refine]```

## 实验（基于CIFAR10数据集）：

### vgg16_bn_cifar

training: ```python trainer.py --arch vgg16_bn_cifar --epochs 150 --gpu 4 --valuate --visdom```

pruning: ```python pruner.py --arch vgg16_bn_cifar --gpu 4 --resume checkpoints/cifar10_vgg16_bn_cifar_best.pth.tar```

fine-tune: ```python trainer.py --arch vgg16_bn_cifar --epochs 20 --gpu 4--valuate --refine --resume checkpoints/pruned_cifar10_vgg16_bn_cifar_checkpoint.pth.tar --visdom```

|  vgg19_bn_cifar  | Baseline | pruned (ratio=0.37) | Fine-tuned (10epochs) |
| :--------------: | :------: | :-----------------: | :-------------------: |
| Top1 Accuracy(%) |  93.48   |        49.47        |        93.46          |
|  Parameters(M)   |  14.73   |        5.26         |         5.26          |
|   FLOPs(GMac)    |   0.31   |        0.21         |         0.21          |


|  Pruned Ratio |                                 architecture                                        |
| :-----------: | :---------------------------------------------------------------------------------: |
|       0       | [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512] |
|    0.37121    | [32, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256] |


## 训练过程（From Scratch）：

### test_loss(交叉熵):

![test_loss](readme_imgs/test_loss.jpg)

### test_top1:

![test_top1](readme_imgs/test_top1.jpg)

## fine-tuning：

### test_loss(交叉熵):

![test_loss](readme_imgs/finetune_test_loss.jpg)

### test_top1:

![test_top1](readme_imgs/finetune_test_top1.jpg)