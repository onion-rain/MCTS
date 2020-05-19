# MCTS(Model Compression Toolset)

![MCTS](READMEs/imgs/MCTS.pdf)

[baseline](READMEs/baseline_README.md)

## prune

[weight prune(all)](READMEs/weight_prune_README.md)

[l1 filter prune(vgg)](READMEs/l1_filter_prune_README.md)

[Thinet(vgg)](READMEs/Thinet_README.md)

[network slimming(vgg, resnet_cs)](READMEs/slimming_README.md)

[SFP(soft filter prune)(vgg)](READMEs/SFP_README.md)

[MetaPurne(resnet, mobilenetv2)](READMEs/MetaPrune_README.md)

## quantize

[binarynet(resnet_cifar)](READMEs/quantize_README.md)

[XNOR-Net(nin)](READMEs/quantize_README.md)

[Dorefa-Net(nin, resnet_cifar)](READMEs/quantize_README.md)

## distill

[distilling-knowledge](READMEs/distill_README.md)

## TODO：

TODO 没法保存完整模型：或者说是保存的完整模型无法加载，判断应该是保存的问题。

TODO 中途resume断点续练无法完全复现：完全没头绪，不就两个东西一个model state dict一个optimizer state dict吗。

TODO 打算把第一个bug解决了直接保存完整模型，弃用cfg重构模型那套方案。

TODO imagenet训练mobilenetv2精度好低。。resnet也比tv里的pth低近3个百分点：超参调一遍训练好久。。。爷吐了。

TODO 舍弃config类直接传args，历史遗留问题。。重构代价有点大，以后有时间再搞