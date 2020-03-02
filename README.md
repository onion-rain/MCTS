# pytorch-slimming

论文地址：[Learning Efficient Convolutional Networks Through Network Slimming](https://arxiv.org/abs/1708.06519v1) (ICCV2017)

代码参考：https://github.com/foolwood/pytorch-slimming

| CIFAR10-CGG19BN  | Baseline | Trained with sparsity (lambda=1e-4) | slimmed (ratio=0.7) | Fine-tuned (10epochs) |
| :--------------: | :------: | :---------------------------------: | :-----------------: | :-------------------: |
| Top1 Accuracy(%) |  93.26   |                94.02                |        10.00        |         92.64         |
|  Parameters(M)   |  20.04   |                20.04                |        2.45         |         2.45          |
|   FLOPs(GMac)    |   0.4    |                 0.4                 |        0.21         |         0.21          |

|             Pruned Ratio             |     0      |     0.1     |     0.2     |    0.3    |    0.4     |    0.5     |    0.6     |    0.7     |
| :----------------------------------: | :--------: | :---------: | :---------: | :-------: | :--------: | :--------: | :--------: | :--------: |
| Top1 Accuracy (%) without Fine-tuned |   93.26    |    93.93    |    93.96    |   93.95   |   94.01    |   94.03    |   94.05    |   93.01    |
|      Parameters(M)/ FLOPs(GMac)      | 20.04/ 0.4 | 15.93/ 0.35 | 12.36/ 0.31 | 9.3/ 0.28 | 6.81/ 0.25 | 4.61/ 0.24 | 3.25/ 0.23 | 2.45/ 0.21 |

| Slimmed Ratio |                         architecture                         |
| :-----------: | :----------------------------------------------------------: |
|       0       | [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512] |
|      0.1      | [62, 64, 'M', 128, 128, 'M', 256, 256, 253, 248, 'M', 444, 420, 421, 460, 'M', 468, 477, 461, 407] |
|      0.2      | [61, 64, 'M', 128, 128, 'M', 256, 256, 251, 242, 'M', 376, 325, 346, 403, 'M', 419, 433, 414, 301] |
|      0.3      | [59, 64, 'M', 128, 128, 'M', 256, 256, 249, 234, 'M', 289, 239, 265, 349, 'M', 372, 387, 358, 219] |
|      0.4      | [56, 64, 'M', 128, 128, 'M', 256, 256, 249, 227, 'M', 224, 144, 183, 297, 'M', 316, 344, 302, 128] |
|      0.5      | [54, 64, 'M', 128, 128, 'M', 256, 256, 249, 226, 'M', 202, 118, 141, 238, 'M', 188, 214, 198, 91] |
|      0.6      | [51, 64, 'M', 128, 128, 'M', 256, 256, 249, 224, 'M', 190, 95, 115, 136, 'M', 71, 63, 84, 91] |
|      0.7      | [49, 64, 'M', 128, 128, 'M', 256, 256, 249, 208, 'M', 116, 38, 18, 8, 'M', 14, 9, 16, 94] |