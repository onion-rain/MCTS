# pytorch-DoReFa-Net

论文地址：[DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](http://arxiv.org/abs/1606.06160)

`python trainer_exp.py --arch nin_dorefanet --dataset cifar10 --lr 0.1 --valuate --epochs 20 --deterministic --workers 1 --gpu 1 --a_bits 4 --w_bits 4 --visdom`

==> Computational complexity:   0.02 GMac

==> Number of parameters:       674.91 k

Train:  19 [  50000/  50000 (100%)] loss:   0.382 | top1: 87.188% | load_time:   2% | lr   : 1.0e-03

Test:   19 [  10000/  10000 (100%)] loss:   0.447 | top1:  85.17% | load_time:  26% | UTC+8: 21:51:34

--------  nin_dorefanet  --  cifar10  --  best_top1: 85.170  --  duration:  0h:08.14  --------

best_acc1: 85.17

checkpoint_path: checkpoints/cifar10_nin_dorefanet_a4w4g32_checkpoint.pth.tar

`python trainer_exp.py --arch nin_dorefanet --dataset cifar10 --lr 0.1 --valuate --epochs 20 --deterministic --workers 1 --gpu 1 --a_bits 16 --w_bits 16 --visdom`

==> Computational complexity:   0.02 GMac

==> Number of parameters:       674.91 k

Train:  19 [  50000/  50000 (100%)] loss:   0.357 | top1: 88.058% | load_time:   2% | lr   : 1.0e-03

Test:   19 [  10000/  10000 (100%)] loss:   0.435 | top1:  85.24% | load_time:  35% | UTC+8: 22:06:46

--------  nin_dorefanet  --  cifar10  --  best_top1: 85.240  --  duration:  0h:07.47  --------

best_acc1: 85.24

checkpoint_path: checkpoints/cifar10_nin_dorefanet_a16w16g32_checkpoint.pth.tar

`python trainer_exp.py --arch nin_dorefanet --dataset cifar10 --lr 0.1 --valuate --epochs 20 --deterministic --workers 1 --gpu 1 --a_bits 32 --w_bits 32 --visdom`

==> Computational complexity:   0.02 GMac

==> Number of parameters:       674.91 k

Train:  19 [  50000/  50000 (100%)] loss:   0.229 | top1: 92.626% | load_time:  12% | lr   : 1.0e-03

Test:   19 [  10000/  10000 (100%)] loss:   0.339 | top1:  88.58% | load_time:  65% | UTC+8: 22:16:42

--------  nin_dorefanet  --  cifar10  --  best_top1: 88.620  --  duration:  0h:04.29  --------

best_acc1: 88.62

checkpoint_path: checkpoints/cifar10_nin_dorefanet_a32w32g32_checkpoint.pth.tar