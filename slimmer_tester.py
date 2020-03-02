# import ssl
# #全局取消证书验证
# ssl._create_default_https_context = ssl._create_unverified_context

import torch
import datetime
import copy
import argparse

import models
from slimmer import Slimmer
from tester import Tester
from trainer import Trainer
from config import Configuration
from utils.visualize import Visualizer
from utils import print_flops_params, print_nonzeros, print_model_parameters, save_checkpoint

class Slimmer_tester(object):

    def __init__(self, **kwargs):

        self.start_time = datetime.datetime.now()
        config = Configuration()
        config.update_config(kwargs) # 解析参数更新默认配置
        if config.check_config(): raise # 检测路径、设备是否存在
        vis = None
        if config.visdom:
            vis = Visualizer(config.env, config.legend) # 初始化visdom

        self.slimmer = Slimmer(config=config)
        self.tester = Tester(config=config, vis=vis)
        self.retrainer = Trainer(config=config, vis=vis)

    def run(self):

        print("")
        print("| -------------------- original model -------------------- |")
        self.tester.test(self.slimmer.model)
        print_flops_params(self.tester.model, self.tester.config.dataset)
        # print_model_parameters(self.tester.model)

        # print("")
        # print("| ----------------- simple slimming model ---------------- |")
        # self.slimmer.simple_slim()
        # self.tester.test(self.slimmer.simple_slimmed_model)
        # print_flops_params(self.tester.model, self.tester.config.dataset)
        # # print_nonzeros(self.tester.model)

        # print("")
        # print("| -------------------- original model -------------------- |")
        # self.tester.test(self.slimmer.model)
        # print_flops_params(self.tester.model, self.tester.config.dataset)

        print("")
        print("| -------------------- slimming model -------------------- |")
        structure = self.slimmer.slim()
        self.tester.test(self.slimmer.slimmed_model)
        print_flops_params(self.tester.model, self.tester.config.dataset)

        # save slimmed model
        name = ('slimmed_ratio' 
                + str(self.slimmer.config.slim_percent) 
                + '_' 
                + self.slimmer.config.dataset 
                + "_" + self.slimmer.config.arch)
        if len(self.slimmer.config.gpu_idx_list) > 1:
            state_dict = self.slimmer.slimmed_model.module.state_dict()
        else: state_dict = self.slimmer.slimmed_model.state_dict()
        path = save_checkpoint({
            'structure': structure,
            'ratio': self.slimmer.slim_ratio,
            'model_state_dict': state_dict,
            'best_acc1': self.tester.top1_acc.avg,
        }, file_root='slimmed_checkpoints/', file_name=name)
        print('{:<30}  {}'.format('==> save path: ', path))

        print("")
        print("| ---------------------- finetuning ---------------------- |")
        best_acc1 = 0
        name = ('finetuned_ratio' 
                + str(self.slimmer.config.slim_percent) 
                + '_' 
                + self.retrainer.config.dataset 
                + "_" 
                + self.retrainer.config.arch)
        self.retrainer.model = self.slimmer.slimmed_model
        for epoch in range(1, self.retrainer.config.max_epoch+1):
            # train & valuate
            self.retrainer.train(epoch)
            self.tester.test(self.retrainer.model, epoch)
            print("")

            # save checkpoint
            is_best = self.tester.top1_acc.avg > best_acc1
            best_acc1 = max(self.tester.top1_acc.avg, best_acc1)
            if len(self.retrainer.config.gpu_idx_list) > 1:
                state_dict = self.retrainer.model.module.state_dict()
            else: state_dict = self.retrainer.model.state_dict()
            path = save_checkpoint({
                'epoch': epoch,
                'structure': structure,
                'ratio': self.slimmer.slim_ratio,
                'model_state_dict': state_dict,
                'best_acc1': self.tester.top1_acc.avg,
                'optimizer_state_dict': self.retrainer.optimizer.state_dict(),
            }, is_best=is_best, file_root='slimmed_checkpoints/', file_name=name)
        print('{:<30}  {}'.format('==> best acc1: ', best_acc1))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='network slimming')
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH', default='vgg_cfg',
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
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 150)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', type=float, default=1e-1, metavar='LR',
                        help='initial learning rate (default: 1e-1)')
    parser.add_argument('--weight-decay', '-wd', dest='weight_decay', type=float,
                        default=1e-4, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='training GPU index(default:"0",which means use GPU0')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--valuate', action='store_true',
                        help='valuate each training epoch')

    parser.add_argument('--resume', type=str, default='', metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--refine', type=str, default='', metavar='PATH',
                        help='refine from pruned model')
    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                        help='train with channel sparsity regularization')
    parser.add_argument('--sr-lambda', dest='sr_lambda', type=float, default=1e-4,
                        help='scale sparse rate (default: 1e-4)')
    args = parser.parse_args()



    slimmer_tester = Slimmer_tester(
        arch=args.arch,
        dataset=args.dataset,
        gpu_idx = args.cuda, # choose gpu
        random_seed=args.seed,
        load_model_path="checkpoints/cifar10_vgg_cfg_best.pth.tar",
        num_workers = 6, # 使用多进程加载数据
        slim_percent=0.7,
        # retrain
        max_epoch=10,
        batch_size=100,
        lr=1e-1,
        # lr_scheduler_milestones=[4, 7],
        weight_decay=1e-4,
        momentum=0.9,
    )
    slimmer_tester.run()
    print("end")

