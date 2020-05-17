# pytorch-MetaPrune

论文地址：[MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning](https://arxiv.org/abs/1903.10258) (ICCV2019)

参考代码：https://github.com/liuzechun/MetaPruning

注：gene最后一位为top1 error

## 实验（基于Imagenet数据集）：

### resnet50

pruningnet training: ```python meta_trainer_exp.py --arch resnet50_pruningnet --dataset imagenet --batch-size 100 --epochs 32 --gpu 3 --valuate --visdom```

prunednet search: ```python meta_search_exp.py --workers 20 --arch resnet50_pruningnet --dataset imagenet --gpu 2 --resume checkpoints/meta_prune/imagenet_resnet50_pruningnet_best.pth.tar --flops 1500 --population 100 --select-num 30 --mutation-num 30 --crossover-num 30 --log logs/flops1500.txt --flops-arch resnet50_prunednet```

prunednet retrain from scratch: ```python meta_trainer_exp.py --arch resnet50_prunednet --dataset imagenet --search-resume checkpoints/meta_prune/MetaPruneSearch_resnet50_pruningnet_flops0_checkpoint.pth.tar --epochs 60 --gpu 0 --valuate --visdom --log-path logs/resnet50_prunednet_candidate0_flops0.txt --candidate 0```

| Flops limit(MMac) | Baseline |                           No limit                           |  1900  |  1500  |
| :---------------: | :------: | :----: | :----: | :----: |
| Top1(without fine-tuning)(%) |    |                            57.782‬                            |                            57.75‬                             |                            57.494‬                            |
| Top1(train from scratch)(%) |  | 71.852 | 71.212 | 70.61 |
|   Parameters(M)   |                            25.56                             |                            18.57                             |                            14.87                             |                            13.95                             |
| Flops(GMac) | 4.11 | 2.56 | 1.85 | 1.49 |
|    Gene    |  [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, ???]  | [27, 28, 29, 23, -1, 20, 22, 17, 19, 16, 30, 9, 28, 12, 20, 14, 22, 25, 27, 24, 25, 42.218] | [20, 13, 16, 16, -1, 21, 9, 23, 21, 21, 14, 26, 19, 17, 27, 19, 15, 11, 17, 22, 25, 42.25] | [20, 10, 16, 17, -1, 9, 3, 7, 23, 10, 19, 12, 16, 20, 17, 13, 22, 11, 18, 19, 24, 42.506] |

#### baseline:

train: ```python meta_trainer_exp.py --json experiments/baseline/imagenet_resnet50_prunednet.json --gpu 2```（checkpoint保存gene全-1）

或：```python trainer_exp.py --json experiments/baseline/imagenet_resnet50_prunednet.json --gpu 2```（checkpoint不保存gene）

gene: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, ???]

flops: 4.11 GMac

params: 25.56 M

top1 acc:

checkpoint: 

#### flops0(no flops limit):

gene: [27, 28, 29, 23, -1, 20, 22, 17, 19, 16, 30, 9, 28, 12, 20, 14, 22, 25, 27, 24, 25, 42.218]

flops: 2.56 GMac

params: 18.57 M

top1 acc: 71.852 %

pruningnet checkpoint: checkpoints/meta_prune/imagenet_resnet50_pruningnet_best.pth.tar

search checkpoint: checkpoints/meta_prune/

MetaPruneSearch_resnet50_pruningnet_flops0_checkpoint.pth.tar

retrained checkpoint: checkpoints/meta_prune/imagenet_resnet50_prunednet_flops0_best.pth.tar

#### flops1900:

gene: [20, 13, 16, 16, -1, 21, 9, 23, 21, 21, 14, 26, 19, 17, 27, 19, 15, 11, 17, 22, 25, 42.25]

flops: 1.85 GMac

params: 14.87 M

top1 acc: 71.212 %

pruningnet checkpoint: checkpoints/meta_prune/imagenet_resnet50_pruningnet_best.pth.tar
search checkpoint: checkpoints/meta_prune/

MetaPruneSearch_resnet50_pruningnet_flops1900_checkpoint.pth.tar

retrained checkpoint: checkpoints/meta_prune/imagenet_resnet50_prunednet_flops1900_best.pth.tar

#### flops1500:

gene: [20, 10, 16, 17, -1, 9, 3, 7, 23, 10, 19, 12, 16, 20, 17, 13, 22, 11, 18, 19, 24, 42.506]

flops: 1.49 GMac

params: 13.95 M

top1 acc: 70.61 %

pruningnet checkpoint: checkpoints/meta_prune/imagenet_resnet50_pruningnet_best.pth.tar
search checkpoint: checkpoints/meta_prune/

MetaPruneSearch_resnet50_pruningnet_flops1500_checkpoint.pth.tar

retrained checkpoint: checkpoints/meta_prune/imagenet_resnet50_prunednet_flops1500_best.pth.tar

### mobilenetv2

注：baseline结构如下，非torchvision的mobilenetv2结构\
[0,   44, 1, 2], # conv3x3\
[1,   22, 1, 1], # bottleneck\
[6,   33, 2, 2], # bottleneck\
[6,   44, 3, 2], # bottleneck\
[6,   88, 4, 2], # bottleneck\
[6,  132, 3, 1], # bottleneck\
[6,  224, 3, 2], # bottleneck\
[6,  448, 1, 1], # bottleneck\
[0, 1280, 1, 1], # conv1x1\

pruningnet training: ```python meta_trainer_exp.py --arch mobilenetv2_pruningnet --dataset imagenet --batch-size 200 --epochs 64 --gpu 0 --lr 0.25 --weight-decay 0 --valuate --visdom```

prunednet search: ```python meta_search_exp.py --workers 20 --arch mobilenetv2_pruningnet --dataset imagenet --gpu 1 --resume checkpoints/meta_prune/imagenet_mobilenetv2_pruningnet_best.pth.tar --flops 85 --population 100 --select-num 30 --mutation-num 30 --crossover-num 30 --flops -arch mobilenetv2_prunednet --epochs 20```

prunednet retrain from scratch: ```python meta_trainer_exp.py --arch mobilenetv2_prunednet --dataset imagenet --epochs 80 --lr 0.5 --batch_size 200 --valuate --gpu 2 --candidate 0 --search-resume checkpoints/meta_prune/MetaPruneSearch_mobilenetv2_pruningnet_flops300.0_checkpoint.pth.tar --suffix _flops300 --visdom```

| Flops limit(MMac) | Baseline |                           No limit                           |  300  |  141  |  125  |  106  |  85  |  44  |
| :---------------: | :------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Top1(without fine-tuning)(%) |    |                            46.652                            |                             46.4‬                             |                            44.808‬                            |                            44.082                            |                            43.308                            |                            41.54                             |                             33.3                             |
| Top1(train from scratch)(%) |  |                                                              |                                                              |                            45.28                             |                             45.1                             |                             44.2                             |                            43.178                            |                            38.404                            |
|   Parameters(M)   |                             6.55                             |                                                           |                             4.26                             |                             2.74                             |                             2.59                             |                             2.4                              |                             2.21                             |                             1.65                             |
| Flops(MMac) |                            679.9                            |                                                              |                            292.67                            |                            145.69                            |                            128.65                            |                            107.53                            |                            87.41                             |                            46.47                             |
|    Gene    | [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, ???] | [24, 5, 29, 29, 28, 17, 22, 22, -1, 29, 25, 14, 22, 25, 14, 21, 15, 25, 5, 24, 24, 15, 25, 16, 22, 25, 53.348] | [15, 7, 15, 27, 17, 16, 20, 26, -1, 27, 14, 25, 10, 5, 21, 25, 7, 28, 10, 21, 8, 11, 16, 24, 13, 26, 53.6] | [14, 3, 6, 14, 12, 8, 16, 12, -1, 16, 6, 4, 21, 6, 14, 15, 10, 7, 11, 11, 18, 1, 20, 11, 10, 15, 55.192] | [7, 16, 7, 10, 11, 4, 17, 10, -1, 13, 4, 5, 14, 14, 9, 18, 10, 16, 17, 8, 8, 6, 17, 10, 9, 14, 55.918] | [11, 3, 4, 7, 7, 11, 15, 11, -1, 14, 7, 4, 13, 13, 10, 11, 11, 16, 13, 7, 3, 5, 9, 10, 9, 10, 56.692] | [7, 4, 5, 7, 6, 3, 9, 11, -1, 19, 4, 9, 11, 9, 9, 6, 10, 3, 6, 3, 7, 5, 17, 8, 8, 11, 58.46] | [2, 4, 4, 6, 6, 1, 4, 2, -1, 5, 4, 1, 8, 3, 0, 9, 2, 5, 5, 5, 2, 9, 9, 13, 3, 3, 66.7] |

#### baseline:

train: ```python meta_trainer_exp.py --json experiments/baseline/imagenet_mobilenetv2_prunednet.json --gpu 5 --visdom```（checkpoint保存gene全-1）

或：```python trainer_exp.py --json experiments/baseline/imagenet_mobilenetv2_prunednet.json --gpu 5 --visdom```（checkpoint不保存gene）

gene: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, ???]

flops: 320.24 MMac

params: 6.55 M

top1 acc:

checkpoint: 

#### flops0(no flops limit):

gene: [24, 5, 29, 29, 28, 17, 22, 22, -1, 29, 25, 14, 22, 25, 14, 21, 15, 25, 5, 24, 24, 15, 25, 16, 22, 25, 53.348]

flops: 109.16 MMac

params: 2.4 M   

top1 acc: 45.456 

pruningnet checkpoint: checkpoints/meta_prune/imagenet_mobilenetv2_pruningnet_best.pth.tar

search checkpoint: checkpoints/meta_prune/MetaPruneSearch_mobilenetv2_pruningnet_flops300.0_checkpoint.pth.tar

retrained checkpoint: checkpoints/meta_prune/imagenet_mobilenetv2_prunednet_best.pth.tar

#### flops300:

gene: [15, 7, 15, 27, 17, 16, 20, 26, -1, 27, 14, 25, 10, 5, 21, 25, 7, 28, 10, 21, 8, 11, 16, 24, 13, 26, 53.6]

flops: 109.16 GMac

params: 2.59 M

top1 acc: 45.456 

pruningnet checkpoint: checkpoints/meta_prune/imagenet_mobilenetv2_pruningnet_best.pth.tar

search checkpoint: checkpoints/meta_prune/MetaPruneSearch_mobilenetv2_pruningnet_flops300.0_checkpoint.pth.tar

retrained checkpoint: checkpoints/meta_prune/imagenet_mobilenetv2_prunednet_best.pth.tar

#### flops141:

random_range_dividend = 1.2

gene: [14, 3, 6, 14, 12, 8, 16, 12, -1, 16, 6, 4, 21, 6, 14, 15, 10, 7, 11, 11, 18, 1, 20, 11, 10, 15, 55.192]

flops: 145.69 MMac

params: 2.74 M

top1 acc: 45.28

pruningnet checkpoint: checkpoints/meta_prune/imagenet_mobilenetv2_pruningnet_best.pth.tar

search checkpoint: checkpoints/meta_prune/
MetaPruneSearch_mobilenetv2_pruningnet_flops141.0_checkpoint.pth.tar

retrained checkpoint: checkpoints/meta_prune/imagenet_mobilenetv2_prunednet_flops141_best.pth.tar

#### flops125:

random_range_dividend = 1.2

gene: [7, 16, 7, 10, 11, 4, 17, 10, -1, 13, 4, 5, 14, 14, 9, 18, 10, 16, 17, 8, 8, 6, 17, 10, 9, 14, 55.918]

flops: 128.65 MMac

params: 2.59 M

top1 acc: 45.1

pruningnet checkpoint: checkpoints/meta_prune/imagenet_mobilenetv2_pruningnet_best.pth.tar

search checkpoint: checkpoints/meta_prune/MetaPruneSearch_mobilenetv2_pruningnet_flops125.0_checkpoint.pth.tar

retrained checkpoint: checkpoints/meta_prune/imagenet_mobilenetv2_prunednet_flops125_best.pth.tar

#### flops106:

random_range_dividend = 1.5

gene: [11, 3, 4, 7, 7, 11, 15, 11, -1, 14, 7, 4, 13, 13, 10, 11, 11, 16, 13, 7, 3, 5, 9, 10, 9, 10, 56.692]

flops: 107.53 MMac

params: 2.4 M

top1 acc: 44.2

pruningnet checkpoint: checkpoints/meta_prune/imagenet_mobilenetv2_pruningnet_best.pth.tar

search checkpoint: checkpoints/meta_prune/MetaPruneSearch_mobilenetv2_pruningnet_flops106.0_checkpoint.pth.tar

retrained checkpoint: checkpoints/meta_prune/imagenet_mobilenetv2_prunednet_flops106_best.pth.tar

#### flops85:

random_range_dividend = 1.5

gene: [7, 4, 5, 7, 6, 3, 9, 11, -1, 19, 4, 9, 11, 9, 9, 6, 10, 3, 6, 3, 7, 5, 17, 8, 8, 11, 58.46]

flops: 87.41 MMac

params: 2.21 M

top1 acc: 43.178  

pruningnet checkpoint: checkpoints/meta_prune/imagenet_mobilenetv2_pruningnet_best.pth.tar

search checkpoint: checkpoints/meta_prune/MetaPruneSearch_mobilenetv2_pruningnet_flops106.0_checkpoint.pth.tar 

retrained checkpoint: checkpoints/meta_prune/imagenet_mobilenetv2_prunednet_flops85_best.pth.tar

#### flops44:

random_range_dividend = 2.2

gene: [2, 4, 4, 6, 6, 1, 4, 2, -1, 5, 4, 1, 8, 3, 0, 9, 2, 5, 5, 5, 2, 9, 9, 13, 3, 3, 66.7]

flops: 46.47 MMac

params: 1.65 M

top1 acc: 38.404 

pruningnet checkpoint: checkpoints/meta_prune/imagenet_mobilenetv2_pruningnet_best.pth.tar

search checkpoint: checkpoints/meta_prune/MetaPruneSearch_mobilenetv2_pruningnet_flops106.0_checkpoint.pth.tar

retrained checkpoint: checkpoints/meta_prune/imagenet_mobilenetv2_prunednet_flops44_best.pth.tar