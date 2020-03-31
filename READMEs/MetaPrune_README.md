# pytorch-MetaPrune

论文地址：[MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning](https://arxiv.org/abs/1903.10258) (ICCV2019)

参考代码：https://github.com/liuzechun/MetaPruning

## 实验（基于Imagenet数据集）：

### resnet50

pruningnet training: ```python meta_trainer_exp.py --arch resnet50_pruningnet --dataset imagenet --batch-size 100 --epochs 32 --gpu 3 --valuate --visdom```

prunednet search: ```python meta_search_exp.py --workers 20 --arch resnet50_pruningnet --dataset imagenet --gpu 2 --resume checkpoints/meta_prune/imagenet_resnet50_pruningnet_best.pth.tar --flops 1500 --population 100 --select-num 30 --mutation-num 30 --crossover-num 30 --log logs/flops1500.txt --flops-arch resnet50_prunednet```

prunednet retrain from scratch: ```python meta_trainer_exp.py --arch resnet50_prunednet --dataset imagenet --search-resume checkpoints/meta_prune/MetaPruneSearch_resnet50_pruningnet_flops0_checkpoint.pth.tar --epochs 60 --gpu 0 --valuate --visdom --log-path logs/resnet50_prunednet_candidate0_flops0.txt --candidate 0```

baseline:
gene: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, ???]
flops: 4.08 GMac
params: 25.56 M
top1 acc:
checkpoint: 

flops0(no flops limit):
gene: [27, 28, 29, 23, -1, 20, 22, 17, 19, 16, 30, 9, 28, 12, 20, 14, 22, 25, 27, 24, 25, 42.218]
flops: 2.56 GMac
params: 18.57 M
top1 acc: 71.852 %
pruningnet checkpoint: checkpoints/meta_prune/imagenet_resnet50_pruningnet_best.pth.tar
search checkpoint: checkpoints/meta_prune/MetaPruneSearch_resnet50_pruningnet_flops0_checkpoint.pth.tar
retrained checkpoint: checkpoints/meta_prune/imagenet_resnet50_prunednet_flops0_best.pth.tar

flops1900:
gene: [20, 13, 16, 16, -1, 21, 9, 23, 21, 21, 14, 26, 19, 17, 27, 19, 15, 11, 17, 22, 25, 42.25]
flops: 1.85 GMac
params: 14.87 M
top1 acc: 
pruningnet checkpoint: checkpoints/meta_prune/imagenet_resnet50_pruningnet_best.pth.tar
search checkpoint: checkpoints/meta_prune/MetaPruneSearch_resnet50_pruningnet_flops1900_checkpoint.pth.tar
retrained checkpoint: checkpoints/meta_prune/imagenet_resnet50_prunednet_flops1900_best.pth.tar # TODO

flops1500:
gene: [20, 10, 16, 17, -1, 9, 3, 7, 23, 10, 19, 12, 16, 20, 17, 13, 22, 11, 18, 19, 24, 42.506]
flops: 1.49 GMac
params: 13.95 M
top1 acc: 
pruningnet checkpoint: checkpoints/meta_prune/imagenet_resnet50_pruningnet_best.pth.tar
search checkpoint: checkpoints/meta_prune/MetaPruneSearch_resnet50_pruningnet_flops1500_checkpoint.pth.tar
retrained checkpoint: checkpoints/meta_prune/imagenet_resnet50_prunednet_flops1500_best.pth.tar # TODO



### mobilenetv2

pruningnet training: ```python meta_trainer_exp.py --arch mobilenetv2_pruningnet --dataset imagenet --batch-size 200 --epochs 64 --gpu 0 --lr 0.25 --weight-decay 0 --valuate --visdom```

prunednet search: ```python meta_search_exp.py --workers 20 --arch mobilenetv2_pruningnet --dataset imagenet --gpu 1 --resume checkpoints/meta_prune/imagenet_mobilenetv2_pruningnet_best.pth.tar --flops 2000 --population 100 --select-num 30 --mutation-num 30 --crossover-num 30 --log logs/flops2000.txt --flops-arch mobilenetv2_prunednet```

prunednet retrain from scratch: ```python meta_trainer_exp.py -a mobilenetv2_prunednet --dataset imagenet --epochs 80 --lr 0.5 --batch-size 500 --valuate --gpu 2 --visdom```

baseline:
gene: 
flops: 0.68 GMac
params: 6.55 M
top1 acc:
checkpoint: 

flops0(no flops limit):
gene: 
flops: 
params: 
top1 acc: 
pruningnet checkpoint: 
search checkpoint: 
retrained checkpoint: 

flops300(no flops limit):
gene: 
flops: 
params: 
top1 acc: 
pruningnet checkpoint: 
search checkpoint: 
retrained checkpoint: 

flops141(no flops limit):
gene: 
flops: 
params: 
top1 acc: 
pruningnet checkpoint: 
search checkpoint: 
retrained checkpoint: 

flops125(no flops limit):
gene: 
flops: 
params: 
top1 acc: 
pruningnet checkpoint: 
search checkpoint: 
retrained checkpoint: 

flops106(no flops limit):
gene: 
flops: 
params: 
top1 acc: 
pruningnet checkpoint: 
search checkpoint: 
retrained checkpoint: 

flops85(no flops limit):
gene: 
flops: 
params: 
top1 acc: 
pruningnet checkpoint: 
search checkpoint: 
retrained checkpoint: 

flops44(no flops limit):
gene: 
flops: 
params: 
top1 acc: 
pruningnet checkpoint: 
search checkpoint: 
retrained checkpoint: 