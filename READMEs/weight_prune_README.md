# pytorch-weighit_prune

论文地址：[Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626v3) (NIPS 2015)

## CIFAR10：

### vgg16_bn_cifar

training(最新见[baseline](baseline_README.md)): ```python trainer_exp.py --arch vgg16_bn_cifar --epochs 150 --gpu 4 --valuate --visdom```

prune: ```python weight_pruner_exp.py --json experiments/prune/cifar10_weight_prune_vgg16_bn.json --gpu 7 --prune_percent 0.5```

<!-- fine-tune: ```python trainer.py --arch vgg16_bn_cifar --epochs 20 --gpu 4 --valuate --resume checkpoints/weight_pruned0.5_cifar10_vgg16_bn_cifar_checkpoint.pth.tar --visdom``` -->

|          ratio           | Baseline |   0.3    |   0.4    |   0.5    |   0.6    |   0.7    |   0.75   |   0.8    |   0.9    |
| :----------------------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| top1 (without fine-tune) |  93.84   |  93.83   |  93.87   |  93.90   |  93.47   |  89.04   |  71.95   |  48.60   |  10.00   |
|      conv threshold      |    0     | 1.44e-03 | 2.21e-03 | 3.19e-03 | 4.53e-03 | 6.53e-03 | 8.02e-03 | 1.02e-02 | 2.00e-02 |
|       fc threshold       |    0     | 6.76e-02 | 8.14e-02 | 9.41e-02 | 1.09e-01 | 1.26e-01 | 1.35e-01 | 1.46e-01 | 1.78e-01 |

### resnet20

training(最新见[baseline](baseline_README.md)): ```python trainer_exp.py --json experiments/baseline/cifar10_resnet20.json --gpu 2 --visdom```

pruning: ```python pruner.py --arch resnet20_cs --gpu 5 --resume checkpoints/cifar10_resnet20_cs_best.pth.tar --gpu 7 --prune 0.5```

|          ratio           | Baseline |   0.3    |   0.4    |   0.5    |   0.6    |   0.7    |   0.75    |   0.8    |   0.9    |
| :----------------------: | :------: | :------: | :------: | :------: | :------: | :------: | :-------: | :------: | :------: |
| top1 (without fine-tune) |   92.1   |  91.24   |  90.24   |  87.85   |  78.47   |  53.53   |   44.08   |  19.53   |  11.11   |
|      conv threshold      |    0     | 2.98e-02 | 4.11e-02 | 5.38e-02 | 6.81e-02 | 8.55e-02 | 9.61se-02 | 1.09e-01 | 1.46e-01 |
|       fc threshold       |    0     | 2.79e-01 | 3.51e-01 | 4.29e-01 | 4.96e-01 | 5.71e-01 | 6.16e-01  | 6.70e-01 | 8.53e-01 |

### resnet56

training(最新见[baseline](baseline_README.md)): ```python trainer_exp.py --json experiments/baseline/cifar10_resnet56.json --gpu 2 --visdom```

pruning: ```python pruner.py --arch resnet56_cs --gpu 6 --resume checkpoints/cifar10_resnet56_cs_best.pth.tar --gpu 7 --prune 0.5```

|          ratio           | Baseline |   0.3    |   0.4    |   0.5    |   0.6    |   0.7    |   0.75   |   0.8    |   0.9    |
| :----------------------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| top1 (without fine-tune) |  92.86   |  92.75   |  92.35   |  92.29   |  90.92   |   86.6   |  76.22   |  65.96   |  18.78   |
|      conv threshold      |    0     | 1.57e-02 | 2.26e-02 | 3.04e-02 | 3.96e-02 | 5.07e-02 | 6.23e-02 | 6.58e-02 | 9.05e-02 |
|       fc threshold       |    0     | 1.80e-01 | 2.55e-01 | 3.23e-01 | 4.04e-01 | 5.25e-01 | 6.14e-01 | 6.48e-01 | 8.37e-01 |
