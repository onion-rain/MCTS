# pytorch-Thinet

论文地址：[ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression](https://arxiv.org/abs/1707.06342) (ICLR 2017)

## 实验（基于CIFAR10数据集）：

python channel_pruner_exp.py --arch vgg16_bn_cifar --resume checkpoints/baseline/cifar10_vgg16_bn_cifar_best.pth.tar --prune_percent 0.3 --lp_norm 2 --gpu 5