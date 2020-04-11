# pytorch-xnornet

论文地址：[XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279v4)

参考代码：https://github.com/jiecaoyu/XNOR-Net-PyTorch

## network-in-network baseline:

python trainer_exp.py --arch nin --dataset cifar10 --lr 0.01 --gpu 1 --valuate --epochs 200 --deterministic --workers 1

cfg = [192, 160, 96, 192, 192, 192, 192, 192]
groups = [1,]*9

==> Computational complexity:   0.23 GMac
==> Number of parameters:       969.82 k
...
Train: 199 [  50000/  50000 (100%)] loss:   0.007 | top1: 99.956% | load_time:   0% | lr   : 1.0e-04
Test:  199 [  10000/  10000 (100%)] loss:   0.325 | top1:  91.16% | load_time:  67% | UTC+8: 16:52:09
--------  model: nin  --  dataset: cifar10  --  duration:  0h:49.12  --------

best_acc1: 91.45
checkpoint_path: checkpoints/cifar10_nin_checkpoint.pth.tar
end

## network-in-network_groupconv baseline:

python trainer_exp.py --arch nin_gc --dataset cifar10 --lr 0.01 --gpu 1 --valuate --epochs 200 --deterministic --workers 1

cfg = [256, 256, 256, 512, 512, 512, 1024, 1024]
groups = [1, 2, 2, 16, 4, 4, 32, 8, 1]

==> Computational complexity:   0.2 GMac
==> Number of parameters:       722.46 k

Train: 199 [  50000/  50000 (100%)] loss:   0.012 | top1: 99.936% | load_time:   0% | lr   : 1.0e-04
Test:  199 [  10000/  10000 (100%)] loss:   0.328 | top1:  90.29% | load_time:  47% | UTC+8: 16:10:25
--------  model: nin_gc  --  dataset: cifar10  --  duration:  1h:09.14  --------

best_acc1: 90.55
checkpoint_path: checkpoints/cifar10_nin_gc_checkpoint.pth.tar
end

## network-in-network xnornet:

python trainer_exp.py --arch nin_xnornet --dataset cifar10 --lr 0.01 --valuate --epochs 200 --deterministic --workers 1 --gpu 0

cfg = [192, 160, 96, 192, 192, 192, 192, 192]
groups = [1,]*9

==> Computational complexity:   0.02 GMac
==> Number of parameters:       969.82 k
...
Train: 199 [  50000/  50000 (100%)] loss:   0.653 | top1: 77.198% | load_time:   0% | lr   : 1.0e-04
Test:  199 [  10000/  10000 (100%)] loss:   0.697 | top1:  76.08% | load_time:  55% | UTC+8: 13:14:48
--------  model: nin_xnornet  --  dataset: cifar10  --  duration:  2h:09.33  --------

best_acc1: 76.67
checkpoint_path: checkpoints/cifar10_nin_xnornet_checkpoint.pth.tar
end

## network-in-network_groupconv xnornet:

python trainer_exp.py --arch nin_gc_xnornet --dataset cifar10 --lr 0.01 --valuate --epochs 200 --deterministic --workers 1 --gpu 0

==> Computational complexity:   0.02 GMac
==> Number of parameters:       720.93 k

TODO