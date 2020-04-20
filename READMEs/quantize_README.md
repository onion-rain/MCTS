# pytorch-quantize

论文地址：[XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279v4)

论文地址：[DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](http://arxiv.org/abs/1606.06160)

论文地址：[Ternary Weight Networks](http://arxiv.org/abs/1605.04711)

参考代码：https://github.com/jiecaoyu/XNOR-Net-PyTorch

# CIFAR10

## network in network(nin)

### baseline

`python trainer_exp.py --arch nin --dataset cifar10 --lr 0.1 --valuate --epochs 100 --deterministic --workers 1 --gpu 3 --visdom`

==> Computational complexity:   149.66 MMac

==> Number of parameters:       674.91 k

Train:  99 [  50000/  50000 (100%)] loss:   0.029 | top1:  99.61% | load_time:   9% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.246 | top1:  92.37% | load_time:  67% | UTC+8: 15:46:52

--------  cifar10_nin  --  best_top1: 92.480  --  duration:  0h:23.38  --------

best_acc1: 92.48

checkpoint_path: checkpoints/cifar10_nin_checkpoint.pth.tar

### xnornet

xnornet把bn层放到conv前面效果会更好，注释里为bn前置的实验结果

`python trainer_exp.py --arch nin_q --dataset cifar10 --lr 0.1 --valuate --epochs 100 --deterministic --workers 1 --gpu 0 --visdom --quantize xnor`

==> Computational complexity:   17.11 MMac

==> Number of parameters:       674.91 k

Train:  99 [  50000/  50000 (100%)] loss:   0.622 | top1: 78.614% | load_time:   3% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.673 | top1:  76.93% | load_time:  50% | UTC+8: 18:40:09

--------  cifar10_nin_q_xnor  --  best_top1: 77.260  --  duration:  0h:29.21  --------

best_acc1: 77.26

checkpoint_path: checkpoints/cifar10_nin_q_xnor_checkpoint.pth.tar

<!-- ==> Computational complexity:   17.26 MMac

==> Number of parameters:       674.91 k

Train:  99 [  50000/  50000 (100%)] loss:   0.522 | top1: 82.016% | load_time:   3% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:    0.56 | top1:   80.7% | load_time:  53% | UTC+8: 16:23:02

--------  cifar10_nin_q_xnor  --  best_top1: 81.220  --  duration:  0h:29.42  --------

best_acc1: 81.22

checkpoint_path: checkpoints/cifar10_nin_q_xnor_checkpoint.pth.tar -->

### ternary

`python trainer_exp.py --arch nin_q --dataset cifar10 --lr 0.1 --valuate --epochs 100 --deterministic --workers 1 --gpu 1 --visdom --quantize ternary`

==> Computational complexity:   17.11 MMac

==> Number of parameters:       674.91 k

Train:  99 [  50000/  50000 (100%)] loss:   0.072 | top1: 98.286% | load_time:   4% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.269 | top1:  91.22% | load_time:  41% | UTC+8: 18:43:30

--------  cifar10_nin_q_ternary  --  best_top1: 91.690  --  duration:  0h:32.31  --------

best_acc1: 91.69

checkpoint_path: checkpoints/cifar10_nin_q_ternary_checkpoint.pth.tar

<!-- ==> Computational complexity:   17.26 MMac

==> Number of parameters:       674.91 k

Train:  99 [  50000/  50000 (100%)] loss:   0.079 | top1: 98.002% | load_time:   3% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.291 | top1:  90.83% | load_time:  39% | UTC+8: 16:26:16

--------  cifar10_nin_q_ternary  --  best_top1: 91.090  --  duration:  0h:33.01  --------

best_acc1: 91.09

checkpoint_path: checkpoints/cifar10_nin_q_ternary_checkpoint.pth.tar -->

`python trainer_exp.py --arch nin_q --dataset cifar10 --lr 0.1 --valuate --epochs 100 --deterministic --workers 1 --gpu 3 --visdom --quantize ternary --suffix _a1`

注：ternary默认全精度activation，如需二值化activation需修改源码(quantize/ternarynet.py)

==> Computational complexity:   17.11 MMac

==> Number of parameters:       674.91 k

Train:  99 [  50000/  50000 (100%)] loss:   0.533 | top1: 81.688% | load_time:   2% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.609 | top1:  79.18% | load_time:  39% | UTC+8: 19:21:40

--------  cifar10_nin_q_ternary_a1  --  best_top1: 79.840  --  duration:  0h:37.15  --------

best_acc1: 79.84

checkpoint_path: checkpoints/cifar10_nin_q_ternary_a1_checkpoint.pth.tar

### dorefa

`python trainer_exp.py --arch nin_q --dataset cifar10 --lr 0.1 --valuate --epochs 50 --deterministic --workers 1 --gpu 2 --visdom --quantize dorefa --a_bits 1 --w_bits 1`

注：a1w1训练不稳定，训练周期过多会导致精度下降到10%且无法通过继续训练提升

==> Computational complexity:   17.11 MMac

==> Number of parameters:       674.91 k

Train:  49 [  50000/  50000 (100%)] loss:   0.953 | top1: 66.588% | load_time:   2% | lr   : 1.0e-03

Test:   49 [  10000/  10000 (100%)] loss:   0.965 | top1:  65.95% | load_time:  24% | UTC+8: 19:51:43

--------  cifar10_nin_q_dorefa_a1w1g32  --  best_top1: 67.040  --  duration:  0h:20.41  --------

best_acc1: 67.04

checkpoint_path: checkpoints/cifar10_nin_q_dorefa_a1w1g32_checkpoint.pth.tar

<!-- ==> Computational complexity:   17.26 MMac

==> Number of parameters:       674.91 k

Train:  99 [  50000/  50000 (100%)] loss:   0.409 | top1: 86.194% | load_time:   2% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.506 | top1:  82.56% | load_time:  28% | UTC+8: 16:39:28

--------  cifar10_nin_q_dorefa_a1w1g32  --  best_top1: 83.890  --  duration:  0h:44.38  --------

best_acc1: 83.89

checkpoint_path: checkpoints/cifar10_nin_q_dorefa_a1w1g32_checkpoint.pth.tar -->

`python trainer_exp.py --arch nin_q --dataset cifar10 --lr 0.1 --valuate --epochs 100 --deterministic --workers 1 --gpu 3 --visdom --quantize dorefa --a_bits 2 --w_bits 2`

==> Computational complexity:   17.11 MMac

==> Number of parameters:       674.91 k

Train:  99 [  50000/  50000 (100%)] loss:   0.368 | top1:  87.42% | load_time:   1% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.696 | top1:   77.4% | load_time:   3% | UTC+8: 18:42:41

--------  cifar10_nin_q_dorefa_a2w2g32  --  best_top1: 81.850  --  duration:  0h:44.19  --------

best_acc1: 81.85

checkpoint_path: checkpoints/cifar10_nin_q_dorefa_a2w2g32_checkpoint.pth.tar

<!-- ==> Computational complexity:   17.26 MMac

==> Number of parameters:       674.91 k

Train:  99 [  50000/  50000 (100%)] loss:   0.498 | top1: 83.026% | load_time:   2% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.609 | top1:  79.59% | load_time:  22% | UTC+8: 16:35:47

--------  cifar10_nin_q_dorefa_a2w2g32  --  best_top1: 80.760  --  duration:  0h:40.31  --------

best_acc1: 80.76

checkpoint_path: checkpoints/cifar10_nin_q_dorefa_a2w2g32_checkpoint.pth.tar -->

`python trainer_exp.py --arch nin_q --dataset cifar10 --lr 0.1 --valuate --epochs 100 --deterministic --workers 1 --gpu 0 --visdom --quantize dorefa --a_bits 4 --w_bits 4`

==> Computational complexity:   17.11 MMac

==> Number of parameters:       674.91 k

Train:  99 [  50000/  50000 (100%)] loss:   0.089 | top1: 97.474% | load_time:   2% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.314 | top1:  90.01% | load_time:  22% | UTC+8: 18:05:14

--------  cifar10_nin_q_dorefa_a4w4g32  --  best_top1: 90.290  --  duration:  0h:41.08  --------

best_acc1: 90.29

checkpoint_path: checkpoints/cifar10_nin_q_dorefa_a4w4g32_checkpoint.pth.tar

<!-- ==> Computational complexity:   17.26 MMac

==> Number of parameters:       674.91 k

Train:  99 [  50000/  50000 (100%)] loss:   0.287 | top1: 90.312% | load_time:   2% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.399 | top1:  86.62% | load_time:  22% | UTC+8: 17:08:25

--------  cifar10_nin_q_dorefa_a4w4g32  --  best_top1: 87.060  --  duration:  0h:40.11  --------

best_acc1: 87.06

checkpoint_path: checkpoints/cifar10_nin_q_dorefa_a4w4g32_checkpoint.pth.tar -->

`python trainer_exp.py --arch nin_q --dataset cifar10 --lr 0.1 --valuate --epochs 100 --deterministic --workers 1 --gpu 1 --visdom --quantize dorefa --a_bits 8 --w_bits 8`

==> Computational complexity:   17.11 MMac

==> Number of parameters:       674.91 k

Train:  99 [  50000/  50000 (100%)] loss:   0.065 | top1:  98.34% | load_time:   2% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.304 | top1:  90.56% | load_time:  28% | UTC+8: 18:05:46

--------  cifar10_nin_q_dorefa_a8w8g32  --  best_top1: 90.820  --  duration:  0h:41.44  --------

best_acc1: 90.82

checkpoint_path: checkpoints/cifar10_nin_q_dorefa_a8w8g32_checkpoint.pth.tar

<!-- ==> Computational complexity:   17.26 MMac

==> Number of parameters:       674.91 k

Train:  99 [  50000/  50000 (100%)] loss:    0.27 | top1: 91.022% | load_time:   2% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.387 | top1:  87.11% | load_time:  30% | UTC+8: 17:08:46

--------  cifar10_nin_q_dorefa_a8w8g32  --  best_top1: 87.450  --  duration:  0h:40.19  --------

best_acc1: 87.45

checkpoint_path: checkpoints/cifar10_nin_q_dorefa_a8w8g32_checkpoint.pth.tar -->

`python trainer_exp.py --arch nin_q --dataset cifar10 --lr 0.1 --valuate --epochs 100 --deterministic --workers 1 --gpu 3 --visdom --quantize dorefa --a_bits 16 --w_bits 16`

==> Computational complexity:   17.11 MMac

==> Number of parameters:       674.91 k

Train:  99 [  50000/  50000 (100%)] loss:   0.064 | top1: 98.458% | load_time:   2% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.301 | top1:  91.01% | load_time:  28% | UTC+8: 18:05:04

--------  cifar10_nin_q_dorefa_a16w16g32  --  best_top1: 91.050  --  duration:  0h:41.13  --------

best_acc1: 91.05

checkpoint_path: checkpoints/cifar10_nin_q_dorefa_a16w16g32_checkpoint.pth.tar

<!-- ==> Computational complexity:   17.26 MMac

==> Number of parameters:       674.91 k

Train:  99 [  50000/  50000 (100%)] loss:   0.263 | top1: 91.258% | load_time:   2% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.383 | top1:  87.43% | load_time:  24% | UTC+8: 17:17:59

--------  cifar10_nin_q_dorefa_a16w16g32  --  best_top1: 87.560  --  duration:  0h:40.09  --------

best_acc1: 87.56

checkpoint_path: checkpoints/cifar10_nin_q_dorefa_a16w16g32_checkpoint.pth.tar -->

| binarize method(bits) | Computational complexity | Number of parameters | best_acc1 |
| :-------------------: | :----------------------: | :------------------: | :-------: |
|       baseline        |       149.66 MMac        |       674.91 k       |   92.48   |
|      xnor(a1w1)       |        17.11 MMac        |       674.91 k       |   77.26   |
|    ternary(a32w2)     |        17.11 MMac        |       674.91 k       |   91.69   |
|     ternary(a1w2)     |        17.11 MMac        |       674.91 k       |   79.84   |
|     dorefa(a1w1)      |        17.11 MMac        |       674.91 k       |   67.04   |
|     dorefa(a2w2)      |        17.11 MMac        |       674.91 k       |   81.85   |
|     dorefa(a4w4)      |        17.11 MMac        |       674.91 k       |   90.29   |
|     dorefa(a8w8)      |        17.11 MMac        |       674.91 k       |   90.82   |
|    dorefa(a16w16)     |        17.11 MMac        |       674.91 k       |   91.05   |

<!-- | binarize method | Computational complexity | Number of parameters | best_acc1 |
| :-------------: | :----------------------: | :------------------: | :-------: |
|    baseline     |       149.66 MMac        |       674.91 k       |   92.48   |
|   xnor(a1w1)    |        17.26 MMac        |       674.91 k       |   81.22   |
| ternary(a32w2)  |        17.26 MMac        |       674.91 k       |   91.09   |
|   dorefa_a1w1   |        17.26 MMac        |       674.91 k       |   83.89   |
|   dorefa_a2w2   |        17.26 MMac        |       674.91 k       |   80.76   |
|   dorefa_a4w4   |        17.26 MMac        |       674.91 k       |   87.06   |
|   dorefa_a8w8   |        17.26 MMac        |       674.91 k       |   87.45   |
|  dorefa_a16w16  |        17.26 MMac        |       674.91 k       |   87.56   | -->

 注：由于没有专用的硬件平台和算法库，此处的量化均为先量化后反量化来模拟，故模型大小和flops不变

![test_top1](imgs/quantize/nin_q_test_top1.jpg)

![train_top1](imgs/quantize/nin_q_train_top1.jpg)

## resnet20

### baseline

`python trainer_exp.py --arch resnet20 --dataset cifar10 --lr 0.1 --valuate --epochs 100 --deterministic --workers 1 --gpu 3 --visdom`

==> Computational complexity:   159.95 MMac

==> Number of parameters:       1.11 M

Train:  99 [  50000/  50000 (100%)] loss:   0.007 | top1: 99.884% | load_time:   3% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:    0.28 | top1:  93.38% | load_time:  56% | UTC+8: 14:40:53

--------  cifar10_resnet20  --  best_top1: 93.450  --  duration:  0h:26.55  --------

best_acc1: 93.45

checkpoint_path: checkpoints/cifar10_resnet20_checkpoint.pth.tar

### xnornet

`python trainer_exp.py --arch resnet20_q --dataset cifar10 --lr 0.1 --valuate --epochs 100 --deterministic --workers 1 --gpu 1 --visdom --quantize xnor`

==> Computational complexity:   1.35 MMac

==> Number of parameters:       1.11 M

Train:  99 [  50000/  50000 (100%)] loss:   0.539 | top1: 81.232% | load_time:   2% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.577 | top1:  80.17% | load_time:  39% | UTC+8: 12:00:23

--------  cifar10_resnet20_q_xnor  --  best_top1: 80.600  --  duration:  0h:39.06  --------

best_acc1: 80.6

checkpoint_path: checkpoints/cifar10_resnet20_q_xnor_checkpoint.pth.tar

### ternary

`python trainer_exp.py --arch resnet20_q --dataset cifar10 --lr 0.1 --valuate --epochs 100 --deterministic --workers 1 --gpu 1 --visdom --quantize ternary`

==> Computational complexity:   1.35 MMac

==> Number of parameters:       1.11 M

Train:  99 [  50000/  50000 (100%)] loss:   0.027 | top1: 99.238% | load_time:   2% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.325 | top1:  92.07% | load_time:  22% | UTC+8: 14:49:57

--------  cifar10_resnet20_q_ternary  --  best_top1: 92.240  --  duration:  0h:48.08  --------

best_acc1: 92.24

checkpoint_path: checkpoints/cifar10_resnet20_q_ternary_checkpoint.pth.tar

`python trainer_exp.py --arch resnet20_q --dataset cifar10 --lr 0.1 --valuate --epochs 100 --deterministic --workers 1 --gpu 1 --visdom --quantize ternary --suffix _a1`

注：ternary默认全精度activation，如需二值化activation需修改源码(quantize/ternarynet.py)

==> Computational complexity:   1.35 MMac

==> Number of parameters:       1.11 M

Train:  99 [  50000/  50000 (100%)] loss:   0.484 | top1: 82.986% | load_time:   2% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.556 | top1:  80.89% | load_time:  12% | UTC+8: 19:43:32

--------  cifar10_resnet20_q_ternary_a1  --  best_top1: 81.920  --  duration:  0h:53.08  --------

best_acc1: 81.92

checkpoint_path: checkpoints/cifar10_resnet20_q_ternary_a1_checkpoint.pth.tar

### dorefa

questions:

在全精度weight上更新在lr=0.1时训练集精度正常上升，测试集精度极低且不稳定，训练一段时间lr自动调整为0.01后测试集精度骤升到接近训练集水平

若在量化后的weight上更新则没有此问题，但最终精度往往没有前者快

而且lr在某一值（0.1）训练epoch过多还会使训练集、测试集精度由上升转变为下降，甚至下降到随机初始化精度(cifar10 10%)，并且此时再改变lr无法提升精度

量化位数过小(如bits=2)lr=0.01，0.001测试集精度仍很低且不稳定，lr=0.1则直接训练集无法收敛

`python trainer_exp.py --arch resnet20_q --dataset cifar10 --lr 0.1 --valuate --epochs 100 --deterministic --workers 1 --gpu 2 --visdom --quantize dorefa --a_bits 1 --w_bits 1`

==> Computational complexity:   1.35 MMac

==> Number of parameters:       1.11 M

Train:  99 [  50000/  50000 (100%)] loss:   1.043 | top1: 63.288% | load_time:   2% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   1.712 | top1:  48.86% | load_time:   5% | UTC+8: 11:03:36

--------  cifar10_resnet20_q_dorefa_a1w1g32  --  best_top1: 61.370  --  duration:  1h:00.09  --------

best_acc1: 61.37

checkpoint_path: checkpoints/cifar10_resnet20_q_dorefa_a1w1g32_checkpoint.pth.tar

`python trainer_exp.py --arch resnet20_q --dataset cifar10 --lr 0.01 --valuate --epochs 100 --deterministic --workers 1 --gpu 2 --visdom --quantize dorefa --a_bits 2 --w_bits 2`

==> Computational complexity:   1.35 MMac

==> Number of parameters:       1.11 M

Train:  99 [  50000/  50000 (100%)] loss:   0.537 | top1:   81.1% | load_time:   2% | lr   : 1.0e-04

Test:   99 [  10000/  10000 (100%)] loss:   0.893 | top1:  71.77% | load_time:   5% | UTC+8: 13:57:07

--------  cifar10_resnet20_q_dorefa_a2w2g32  --  best_top1: 77.120  --  duration:  1h:00.20  --------

best_acc1: 77.12

checkpoint_path: checkpoints/cifar10_resnet20_q_dorefa_a2w2g32_checkpoint.pth.tar

`python trainer_exp.py --arch resnet20_q --dataset cifar10 --lr 0.1 --valuate --epochs 100 --deterministic --workers 1 --gpu 0 --visdom --quantize dorefa --a_bits 4 --w_bits 4`

==> Computational complexity:   1.35 MMac

==> Number of parameters:       1.11 M

Train:  99 [  50000/  50000 (100%)] loss:   0.079 | top1: 97.294% | load_time:   1% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.371 | top1:  90.32% | load_time:   5% | UTC+8: 12:06:07

--------  cifar10_resnet20_q_dorefa_a4w4g32  --  best_top1: 91.010  --  duration:  0h:59.51  --------

best_acc1: 91.01

checkpoint_path: checkpoints/cifar10_resnet20_q_dorefa_a4w4g32_checkpoint.pth.tar

`python trainer_exp.py --arch resnet20_q --dataset cifar10 --lr 0.1 --valuate --epochs 100 --deterministic --workers 1 --gpu 1 --visdom --quantize dorefa --a_bits 8 --w_bits 8`

==> Computational complexity:   1.35 MMac

==> Number of parameters:       1.11 M

Train:  99 [  50000/  50000 (100%)] loss:    0.04 | top1: 98.734% | load_time:   1% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.339 | top1:  92.06% | load_time:   5% | UTC+8: 13:30:20

--------  cifar10_resnet20_q_dorefa_a8w8g32  --  best_top1: 92.200  --  duration:  1h:03.24  --------

best_acc1: 92.2

checkpoint_path: checkpoints/cifar10_resnet20_q_dorefa_a8w8g32_checkpoint.pth.tar

`python trainer_exp.py --arch resnet20_q --dataset cifar10 --lr 0.1 --valuate --epochs 100 --deterministic --workers 1 --gpu 0 --visdom --quantize dorefa --a_bits 16 --w_bits 16`

==> Computational complexity:   1.35 MMac

==> Number of parameters:       1.11 M

Train:  99 [  50000/  50000 (100%)] loss:    0.04 | top1: 98.678% | load_time:   1% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.345 | top1:   91.7% | load_time:   5% | UTC+8: 13:30:25

--------  cifar10_resnet20_q_dorefa_a16w16g32  --  best_top1: 91.970  --  duration:  1h:03.15  --------

best_acc1: 91.97

checkpoint_path: checkpoints/cifar10_resnet20_q_dorefa_a16w16g32_checkpoint.pth.tar

| binarize method(bits) | Computational complexity | Number of parameters | best_acc1 |
| :-------------------: | :----------------------: | :------------------: | :-------: |
|       baseline        |       159.95 MMac        |        1.11 M        |   93.45   |
|      xnor(a1w1)       |        1.35 MMac         |        1.11 M        |   80.6    |
|    ternary(a32w2)     |        1.35 MMac         |        1.11 M        |   92.24   |
|     ternary(a1w2)     |        1.35 MMac         |        1.11 M        |   81.92   |
|     dorefa(a1w1)      |        1.35 MMac         |        1.11 M        |   61.37   |
|     dorefa(a2w2)      |        1.35 MMac         |        1.11 M        |   77.12   |
|     dorefa(a4w4)      |        1.35 MMac         |        1.11 M        |   91.01   |
|     dorefa(a8w8)      |        1.35 MMac         |        1.11 M        |   92.2    |
|    dorefa(a16w16)     |        1.35 MMac         |        1.11 M        |   91.97   |

 注：由于没有专用的硬件平台和算法库，此处的量化均为先量化后反量化来模拟，故模型大小和flops不变

![test_top1](imgs/quantize/resnet_q_test_top1.jpg)

![train_top1](imgs/quantize/resnet_q_train_top1.jpg)