# pytorch-Thinet

论文地址：[ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression](https://arxiv.org/abs/1707.06342) (ICLR 2017)

## 实验（基于CIFAR10数据集）：

python pruner_exp.py --arch vgg16_bn_cifar --json experiments/prune/cifar10_channel_prune_vgg16_bn.json --prune_percent 0.3 --gpu 5