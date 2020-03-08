# pytorch-weighit_prune

论文地址：[Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626v3) (NIPS 2015)

usage: ```trainer.py [-h] [--arch ARCH] [--dataset DATASET] [--workers N]
                  [--batch-size N] [--epochs N] [--lr LR] [--weight-decay W]
                  [--gpu GPU] [--deterministic] [--momentum M] [--valuate]
                  [--resume PATH] [--refine] [--sparsity-regularization]
                  [--srl SR_LAMBDA] [--visdom] [--vis-env ENV]
                  [--vis-legend LEGEND] [--vis-interval N]```

usage: ```pruner.py [-h] [--arch ARCH] [--dataset DATASET] [--workers N]
                 [--gpu gpu_idx] [--resume PATH] [--refine]
                 [--prune-percent PRUNE_PERCENT]```

## 实验（基于CIFAR10数据集）：

### vgg16_bn_cifar

training: ```python trainer.py --arch vgg16_bn_cifar --epochs 150 --gpu 4 --valuate --visdom```

pruning: ```python pruner.py --arch vgg16_bn_cifar --gpu 4 --resume checkpoints/cifar10_vgg16_bn_cifar_best.pth.tar --prune 0.5```

<!-- fine-tune: ```python trainer.py --arch vgg16_bn_cifar --epochs 20 --gpu 4 --valuate --resume checkpoints/weight_pruned0.5_cifar10_vgg16_bn_cifar_checkpoint.pth.tar --visdom``` -->

|          ratio           | Baseline |   0.3    |   0.4    |   0.5    |   0.6    |   0.7    | 0.75     |   0.8    |   0.9    |
| :----------------------: | :------: | :------: | :------: | :------: | :------: | :------: | -------- | :------: | :------: |
| top1 (without fine-tune) |   92.3   |  93.48   |  93.44   |  93.37   |  93.30   |  92.18   | 83.25    |  40.66   |  10.00   |
|        threshold         |    0     | 1.57 e-3 | 2.40 e-3 | 3.46 e-3 | 4.86 e-3 | 6.95 e-3 | 8.47 e-3 | 1.06 e-2 | 2.07 e-2 |

### resnet20_cs

training: ```python trainer.py --arch resnet20_cs --epochs 100 --gpu 5 --valuate --visdom```

pruning: ```python pruner.py --arch resnet20_cs --gpu 5 --resume checkpoints/cifar10_resnet20_cs_best.pth.tar --prune 0.5```

|          ratio           | Baseline |   0.3    |   0.4    |   0.5    |   0.6    |   0.7    | 0.75     |   0.8    |   0.9    |
| :----------------------: | :------: | :------: | :------: | :------: | :------: | :------: | -------- | :------: | :------: |
| top1 (without fine-tune) |   92.3   |  92.11   |  91.83   |  91.08   |  88.43   |  74.15   | 63.43    |  37.69   |  10.02   |
|        threshold         |    0     | 1.49 e-2 | 2.20 e-2 | 3.02 e-2 | 3.99 e-2 | 5.21 e-2 | 5.96 e-2 | 6.88 e-2 | 9.76 e-2 |

### resnet56_cs

training: ```python trainer.py --arch resnet56_cs --epochs 100 --gpu 6 --valuate --visdom```

pruning: ```python pruner.py --arch resnet56_cs --gpu 6 --resume checkpoints/cifar10_resnet56_cs_best.pth.tar --prune 0.5```

|          ratio           | Baseline |   0.3    |   0.4    |   0.5    |   0.6    |   0.7    |   0.8    |   0.85   |   0.9    |
| :----------------------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| top1 (without fine-tune) |  93.33   |  93.27   |  93.27   |  93.22   |  92.80   |  90.65   |  81.84   |  68.49   |  34.39   |
|        threshold         |    0     | 8.01 e-3 | 1.61 e-2 | 3.02 e-2 | 2.15 e-2 | 2.86 e-2 | 3.91 e-2 | 4.69 e-2 | 5.86 e-2 |