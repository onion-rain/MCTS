# pytorch-Thinet

论文地址：[ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression](https://arxiv.org/abs/1707.06342) (ICLR 2017)

## 实验（基于CIFAR10数据集）：

pruning: ```python pruner_exp.py--json experiments/prune/cifar10_channel_prune_vgg16_bn.json --prune_percent 0.3 --gpu 5```

|          ratio           | Baseline |  0.3   |  0.4   |  0.5  |  0.6  |  0.7  |   0.8   |
| :----------------------: | :------: | :----: | :----: | :---: | :---: | :---: | :-----: |
| top1 (without fine-tune) |  93.84   | 60.37  | 79.87  | 71.23 | 53.9  | 38.25 |  24.04  |
|          Params          |  14.73M  | 7.25M  | 5.34M  | 3.69M | 2.37M | 1.34M | 601.04k |
|       Flops(MMac)        |  314.43  | 155.77 | 114.97 | 79.36 | 51.78 | 29.58 |  13.37  |
