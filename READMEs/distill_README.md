# distill

## nin to resnet20

python distiller_exp.py --json experiments/distill/cifar10_resnet20-nin_distill.json --gpu 1

==> Computational complexity:   149.66 MMac

==> Number of parameters:       674.91 k

Train:  99 [  50000/  50000 (100%)] loss:   2.291 | top1: 91.734% | load_time:   5% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.584 | top1:  88.77% | load_time:  65% | UTC+8: 22:07:03

--------  cifar10_nin_kd_resnet20  --  best_top1: 88.850  --  duration:  0h:22.03  --------

best_acc1: 88.85

## vgg13_bn_cifar to alexnet

python distiller_exp.py --json experiments/distill/cifar10_vgg13_bn-alexnet_distill.json --gpu 3 --visdom

==> Computational complexity:   1.34 MMac

==> Number of parameters:       129.25 k

Train:  99 [  50000/  50000 (100%)] loss:   0.402 | top1: 79.166% | load_time:   4% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.956 | top1:  77.81% | load_time:  83% | UTC+8: 09:00:26

--------  cifar10_alexnet_cifar_kd_vgg13_bn_cifar  --  best_top1: 78.07  --  duration:  0h:24.50  --------

best_acc1: 78.07