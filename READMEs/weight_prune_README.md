# pytorch-weighit_prune

论文地址：[Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626v3) (NIPS 2015)

## CIFAR10：

### vgg16_bn_cifar

training(最新见[baseline](baseline_README.md)): ```python trainer_exp.py --json experiments/baseline/cifar10_vgg16_bn.json --gpu 2 --visdom```

prune: ```python pruner_exp.py --json experiments/prune/cifar10_weight_prune_vgg16_bn.json --gpu 7 --prune_percent 0.5```

<!-- fine-tune: ```python trainer.py --arch vgg16_bn_cifar --epochs 20 --gpu 4 --valuate --resume checkpoints/weight_pruned0.5_cifar10_vgg16_bn_cifar_checkpoint.pth.tar --visdom``` -->

|          ratio           | Baseline |   0.3    |   0.4    |   0.5    |   0.6    |   0.7    |   0.75   |   0.8    |   0.9    |
| :----------------------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| top1 (without fine-tune) |  93.84   |  93.83   |  93.87   |  93.90   |  93.47   |  89.04   |  71.95   |  48.60   |  10.00   |
|      conv threshold      |    0     | 1.44e-03 | 2.21e-03 | 3.19e-03 | 4.53e-03 | 6.53e-03 | 8.02e-03 | 1.02e-02 | 2.00e-02 |
|       fc threshold       |    0     | 6.76e-02 | 8.14e-02 | 9.41e-02 | 1.09e-01 | 1.26e-01 | 1.35e-01 | 1.46e-01 | 1.78e-01 |

### resnet20

training(最新见[baseline](baseline_README.md)): ```python trainer_exp.py --json experiments/baseline/cifar10_resnet20.json --gpu 2 --visdom```

pruning: ```python pruner_exp.py --json experiments/prune/cifar10_weight_prune_resnet20.json --gpu 7 --prune_percent 0.5```

|          ratio           | Baseline |   0.3    |   0.4    |   0.5    |   0.6    |   0.7    |   0.75    |   0.8    |   0.9    |
| :----------------------: | :------: | :------: | :------: | :------: | :------: | :------: | :-------: | :------: | :------: |
| top1 (without fine-tune) |   92.1   |  91.24   |  90.24   |  87.85   |  78.47   |  53.53   |   44.08   |  19.53   |  11.11   |
|      conv threshold      |    0     | 2.98e-02 | 4.11e-02 | 5.38e-02 | 6.81e-02 | 8.55e-02 | 9.61se-02 | 1.09e-01 | 1.46e-01 |
|       fc threshold       |    0     | 2.79e-01 | 3.51e-01 | 4.29e-01 | 4.96e-01 | 5.71e-01 | 6.16e-01  | 6.70e-01 | 8.53e-01 |

### resnet56

training(最新见[baseline](baseline_README.md)): ```python trainer_exp.py --json experiments/baseline/cifar10_resnet56.json --gpu 2 --visdom```

pruning: ```python pruner_exp.py --json experiments/prune/cifar10_weight_prune_resnet56.json --gpu 7 --prune_percent 0.5```

|          ratio           | Baseline |   0.3    |   0.4    |   0.5    |   0.6    |   0.7    |   0.75   |   0.8    |   0.9    |
| :----------------------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| top1 (without fine-tune) |  92.86   |  92.75   |  92.35   |  92.29   |  90.92   |   86.6   |  76.22   |  65.96   |  18.78   |
|      conv threshold      |    0     | 1.57e-02 | 2.26e-02 | 3.04e-02 | 3.96e-02 | 5.07e-02 | 6.23e-02 | 6.58e-02 | 9.05e-02 |
|       fc threshold       |    0     | 1.80e-01 | 2.55e-01 | 3.23e-01 | 4.04e-01 | 5.25e-01 | 6.14e-01 | 6.48e-01 | 8.37e-01 |

## Imagenet:

### MobileNet v2

training（暂时还没训练出那么高精度的baseline。。直接用的torchvision里的预训练模型）

pruning:```python pruner_exp.py --json experiments/prune/imagenet_weight_prune_tv_mobilenet_v2.json --gpu 7 --prune_percent 0.1```

|          ratio           | Baseline |   0.05   |   0.1    |   0.15   |   0.2    |   0.3    |   0.4    |   0.5    |   0.6    |   0.7    |   0.75   |   0.8    |   0.9    |
| :----------------------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| top1 (without fine-tune) |  71.88   |   71.87  |  71.66   |  71.28   |  70.05   |  59.92   |  14.06   |  0.666   |   0.10   |  0.076   |   0.1    |   0.12   |   0.1    |
|      conv threshold      |    0     | 2.77e-03 | 5.58e-03 | 8.44e-03 | 1.13e-02 | 1.73e-02 | 2.37e-02 | 3.08e-02 | 3.88e-02 | 4.86e-02 | 5.45e-02 | 6.16e-02 | 8.28e-02 |
|       fc threshold       |    0     | 3.77e-03 | 7.57e-03 | 1.14e-02 | 1.52e-02 | 2.30e-02 | 3.11e-02 | 3.96e-02 | 4.88e-02 | 5.93e-02 | 6.52e-02 | 7.18e-02 | 8.96e-02 |

### ResNet-18

training（暂时还没训练出那么高精度的baseline。。直接用的torchvision里的预训练模型）

pruning:```python pruner_exp.py --json experiments/prune/imagenet_weight_prune_tv_resnet18.json --gpu 7 --prune_percent 0.3```

|          ratio           | Baseline |   0.05   |   0.1    |   0.15   |   0.2    |   0.3    |   0.4    |   0.5    |   0.6    |   0.7    |   0.75   |   0.8    |   0.9    |
| :----------------------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| top1 (without fine-tune) |  69.76   |   69.75  |  69.70   |  69.64   |  69.51   |  69.09   |  67.89   |  65.30   |   57.53  |  39.78   |  23.97   |   9.25   |   0.28   |
|      conv threshold      |    0     | 1.01e-03 | 2.04e-03 | 3.07e-03 | 4.13e-03 | 6.31e-03 | 8.64e-03 | 1.12e-02 | 1.42e-02 | 1.77e-02 | 2.00e-02 | 2.26e-02 | 3.08e-02 |
|       fc threshold       |    0     | 3.81e-03 | 7.60e-03 | 1.14e-02 | 1.53e-02 | 2.32e-02 | 3.13e-02 | 4.00e-02 | 4.95e-02 | 6.07e-02 | 6.74e-02 | 7.56e-02 | 1.02e-01 |

### ResNet-34

training（暂时还没训练出那么高精度的baseline。。直接用的torchvision里的预训练模型）

pruning:```python pruner_exp.py --json experiments/prune/imagenet_weight_prune_tv_resnet34.json --gpu 7 --prune_percent 0.3```

|          ratio           | Baseline |   0.3    |   0.4    |   0.5    |   0.6    |   0.7    |   0.75   |   0.8    |   0.9    |
| :----------------------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| top1 (without fine-tune) |  73.31   |  72.98   |  72.39   |  70.22   |  64.95   |  46.26   |  26.60   |   7.53   |   0.15   |
|      conv threshold      |    0     | 5.04e-03 | 6.90e-03 | 8.94e-03 | 1.13e-02 | 1.41e-02 | 1.58e-02 | 1.78e-02 | 2.40e-02 |
|       fc threshold       |    0     | 2.05e-02 | 2.78e-02 | 3.55e-02 | 4.41e-02 | 5.42e-02 | 6.04e-02 | 6.77e-02 | 9.22e-02 |
