import torch
from utils.visualize import Visualizer
from tqdm import tqdm
import torchvision as tv
import numpy as np
import time
import random
import argparse

import models
from utils import *
from traintest import *

class TesterExp(object):
    """
    TODO 由于trainer类大改，本类某些函数可能个已过期
    """
    def __init__(self, **kwargs):
        
        self.config = Configuration()
        self.config.update_config(kwargs) # 解析参数更新默认配置
        assert self.config.check_config() == 0
        sys.stdout = Logger(self.config.log_path)
        print("| ----------------- Initializing Tester ------------------ |")
        print('{:<30}  {:<8}'.format('==> num_workers: ', self.config.num_workers))

        # suffix
        self.suffix = suffix_init(self.config)
        # device
        self.device = device_init(self.config)
        # Random Seed 
        seed_init(self.config)
        # data
        self.train_dataloader, self.test_dataloader, self.num_classes = dataloader_div_init(self.config, val_num=50)
        # model
        self.model, self.cfg, checkpoint = model_init(self.config, self.device, self.num_classes)

        # step3: criterion
        self.criterion = torch.nn.CrossEntropyLoss()

        # visdom
        self.vis = None
        if self.config.visdom:
            self.vis, _ = visdom_init(self.config, self.suffix, vis_clear=True)

        self.valuator = Tester(
            dataloader=self.val_dataloader,
            device=self.device,
            criterion=self.criterion,
            vis=self.vis,
        )
        
    def run(self):
        print_model_param_flops(model=self.model, input_res=32, device=self.device)
        print_model_param_nums(model=self.model)
        print_flops_params(model=self.model)
        self.tester.test(self.model, epoch=self.start_epoch)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='network tester')
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH', default='vgg19_bn_cifar',
                        choices=models.ALL_MODEL_NAMES,
                        help='model architecture: ' +
                        ' | '.join(name for name in models.ALL_MODEL_NAMES) +
                        ' (default: resnet18)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--workers', type=int, default=10, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 150)')
    parser.add_argument('--learning-rate', '-lr', dest='lr', type=float, default=1e-1, 
                        metavar='LR', help='initial learning rate (default: 1e-1)')
    parser.add_argument('--weight-decay', '-wd', dest='weight_decay', type=float,
                        default=1e-4, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='training GPU index(default:"0",which means use GPU0')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--valuate', action='store_true',
                        help='valuate each training epoch')
    parser.add_argument('--resume-path', '-rp', dest='resume_path', type=str, default='',
                        metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--refine', action='store_true',
                        help='refine from pruned model, use construction to build the model')
    parser.add_argument('--log-path', type=str, default='logs/log.txt',
                        help='default: logs/log.txt')
    args = parser.parse_args()


    if args.resume_path != '':
        tester = Tester(
            arch=args.arch,
            dataset=args.dataset,
            num_workers = args.workers, # 使用多进程加载数据
            batch_size=args.batch_size,
            max_epoch=args.epochs,
            lr=args.lr,
            gpu_idx = args.gpu, # choose gpu
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            valuate=args.valuate,
            resume_path=args.resume_path,
            refine=args.refine,
            log_path=args.log_path,
        )
    else:
        tester = Tester(
            batch_size=1000,
            arch='vgg19_bn_cifar',
            dataset="cifar10",
            gpu_idx = "4", # choose gpu
            resume_path='checkpoints/cifar10_vgg19_bn_cifar_sr_refine_best.pth.tar',
            refine=True,
            # num_workers = 10, # 使用多进程加载数据
        )
    tester.run()
    print("end")