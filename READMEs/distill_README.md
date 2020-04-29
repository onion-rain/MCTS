# distill

python distiller_exp.py --json experiments/distill/cifar10_resnet20-nin_distill.json --gpu 1

==> Computational complexity:   149.66 MMac

==> Number of parameters:       674.91 k

Train:  99 [  50000/  50000 (100%)] loss:   2.291 | top1: 91.734% | load_time:   5% | lr   : 1.0e-03

Test:   99 [  10000/  10000 (100%)] loss:   0.584 | top1:  88.77% | load_time:  65% | UTC+8: 22:07:03

--------  cifar10_nin_kd_resnet20  --  best_top1: 88.850  --  duration:  0h:22.03  --------

best_acc1: 88.85