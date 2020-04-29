# BASELINE

## CIFAR10

### resnet

#### resnet20

python trainer_exp.py --json experiments/baseline/cifar10_resnet20.json --gpu 0

==> Computational complexity:   159.95 MMac

==> Number of parameters:       1.11 M

Train: 199 [  50000/  50000 (100%)] loss:   0.003 | top1: 99.972% | load_time:   3% | lr   : 1.0e-03

Test:  199 [  10000/  10000 (100%)] loss:   0.265 | top1:  93.86% | load_time:  65% | UTC+8: 21:35:35

--------  cifar10_resnet20_forward  --  best_top1: 94.040  --  duration:  0h:55.04  --------

best_acc1: 94.04

#### resnet32

python trainer_exp.py --json experiments/baseline/cifar10_resnet32.json --gpu 0

==> Computational complexity:   273.77 MMac

==> Number of parameters:       1.89 M

Train: 199 [  50000/  50000 (100%)] loss:   0.002 | top1: 99.982% | load_time:   2% | lr   : 1.0e-03

Test:  199 [  10000/  10000 (100%)] loss:    0.24 | top1:   94.4% | load_time:  39% | UTC+8: 23:16:11

--------  cifar10_resnet32  --  best_top1: 94.480  --  duration:  1h:21.15  --------

best_acc1: 94.48

#### resnet44

python trainer_exp.py --json experiments/baseline/cifar10_resnet44.json --gpu 2

==> Computational complexity:   387.59 MMac

==> Number of parameters:       2.66 M

Train: 199 [  50000/  50000 (100%)] loss:   0.002 | top1: 99.986% | load_time:   2% | lr   : 1.0e-03

Test:  199 [  10000/  10000 (100%)] loss:   0.247 | top1:  94.38% | load_time:  12% | UTC+8: 23:43:00

--------  cifar10_resnet44  --  best_top1: 94.490  --  duration:  1h:48.00  --------

best_acc1: 94.49

#### resnet56

python trainer_exp.py --json experiments/baseline/cifar10_resnet56.json --gpu 1

==> Computational complexity:   89.4 MMac

==> Number of parameters:       590.43 k

#### resnet56

python trainer_exp.py --json experiments/baseline/cifar10_resnet110.json --gpu 2

==> Computational complexity:   171.68 MMac

==> Number of parameters:       1.15 M