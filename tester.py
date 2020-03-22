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

__all__ = ['test', 'Tester']

def test(model, epoch=-1, test_dataloader=None, criterion=None, device=None, vis=None):
    
    model.eval() # 验证模式
    
    # meters
    loss_meter = AverageMeter() # 计算所有数的平均值和标准差，这里用来统计一个epoch中的平均值
    top1_acc = AverageMeter()
    top5_acc = AverageMeter()
    batch_time = AverageMeter()
    dataload_time = AverageMeter()

    end_time = time.time()
    # print("testing...")
    with torch.no_grad():
        for batch_index, (input, target) in enumerate(test_dataloader):
            # measure data loading time
            dataload_time.update(time.time() - end_time)

            # compute output
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = criterion(output, target)

            # meters update and visualize
            loss_meter.update(loss.item(), input.size(0))
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            top1_acc.update(prec1.data.cpu(), input.size(0))
            # self.top5_acc.update(prec5.data.cpu(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            done = (batch_index+1) * test_dataloader.batch_size
            percentage = 100. * (batch_index+1) / len(test_dataloader)
            time_str = time.strftime('%H:%M:%S')
            print("\r"
            "Test: {epoch:4} "
            "[{done:7}/{total_len:7} ({percentage:3.0f}%)] "
            "loss: {loss_meter:.3f} | "
            "top1: {top1:3.3f}% | "
            # "top5: {top5:3.3f} | "
            "load_time: {time_percent:2.0f}% | "
            "UTC+8: {time_str} ".format(
                epoch=epoch,
                done=done,
                total_len=len(test_dataloader.dataset),
                percentage=percentage,
                loss_meter=loss_meter.avg if loss_meter.avg<999.999 else 999.999,
                top1=top1_acc.avg,
                # top5=top5_acc.avg,
                time_percent=dataload_time.avg/batch_time.avg*100,
                time_str=time_str
            ), end=""
        )
    print("")
    
    # visualize
    if vis is not None:
        vis.plot('test_loss', loss_meter.avg, x=epoch)
        vis.plot('test_top1', top1_acc.avg, x=epoch)

    return loss_meter, top1_acc, top5_acc


class Tester(object):
    """
    可通过传入config_dic来配置Tester，这种情况下不会在初始化过程中print相关数据
    例：
        val_config_dic = {
            'model': self.model,
            'dataloader': self.val_dataloader,
            'device': self.device,
            'vis': self.vis,
            'seed': self.config.random_seed,
            'criterion': self.criterion,
        }
        self.valuator = Tester(val_config_dic)
    也可通过**kwargs配置Tester
    """
    def __init__(self, config_dic=None, **kwargs):

        if config_dic is None:
            self.init_from_kwargs(**kwargs)
        else:
            self.init_from_config(config_dic)
        
    def run(self):
        print_model_param_flops(model=self.model, input_res=32, device=self.device)
        print_model_param_nums(model=self.model)
        print_flops_params(model=self.model)
        self.test()

    def test(self, model=None, epoch=-1):
        """
        测试指定模型在指定数据集上的表现, 数据集在创建Tester类时通过修改self.config确定
        args:
            model: 要测试的模型，若不为none则self.model更新为model，若为none则测试self.model
            epoch：仅用于显示当前epoch
        """
        if model is not None:
            self.model = model

        self.loss_meter, self.top1_acc, self.top5_acc = test(
            self.model, 
            epoch=epoch, 
            test_dataloader=self.test_dataloader, 
            criterion=self.criterion, 
            device=self.device, 
            vis=self.vis
        )
    

    def init_from_kwargs(self, **kwargs):
        
        print("| ----------------- Initializing Tester ------------------ |")
        
        self.config = Configuration()
        self.config.update_config(kwargs) # 解析参数更新默认配置
        if self.config.check_config(): raise # 检测路径、设备是否存在
        print('{:<30}  {:<8}'.format('==> num_workers: ', self.config.num_workers))

        # visdom
        self.vis = None
        if self.config.visdom:
            self.vis = Visualizer(self.config.vis_env, self.config.vis_legend) # 初始化visdom

        # device
        if len(self.config.gpu_idx_list) > 0:
            self.device = torch.device('cuda:{}'.format(min(self.config.gpu_idx_list))) # 起始gpu序号
            print('{:<30}  {:<8}'.format('==> chosen GPU index: ', self.config.gpu_idx))
        else:
            self.device = torch.device('cpu')
            print('{:<30}  {:<8}'.format('==> device: ', 'CPU'))

        # Random Seed
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(self.config.random_seed)
        torch.backends.cudnn.deterministic = True

        # step1: data
        _, self.test_dataloader, self.num_classes = get_dataloader(self.config)

        # step2: model
        print('{:<30}  {:<8}'.format('==> creating arch: ', self.config.arch))
        self.cfg = None
        checkpoint = None
        if self.config.resume_path != '': # 断点续练hhh
            checkpoint = torch.load(self.config.resume_path, map_location=self.device)
            print('{:<30}  {:<8}'.format('==> resuming from: ', self.config.resume_path))
            if self.config.refine: # 根据cfg加载剪枝后的模型结构
                self.cfg=checkpoint['cfg']
                print(self.cfg)
        else: 
            print("你test不加载模型测锤子??")
            exit(0)
        if self.cfg is not None:
            self.model = models.__dict__[self.config.arch](cfg=cfg, num_classes=self.num_classes)
        else:
            self.model = models.__dict__[self.config.arch](num_classes=self.num_classes)
        if len(self.config.gpu_idx_list) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config.gpu_idx_list)
        self.model.to(self.device) # 模型转移到设备上
        # if checkpoint is not None:
        self.best_acc1 = checkpoint['best_acc1']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("{:<30}  {:<8}".format('==> checkpoint trained epoch: ', checkpoint['epoch']))
        print("{:<30}  {:<8}".format('==> checkpoint best acc1: ', checkpoint['best_acc1']))

        # step3: criterion
        self.criterion = torch.nn.CrossEntropyLoss()


    def init_from_config(self, config):
    
        # visdom
        self.vis = config['vis']

        # device
        self.device = config['device']

        # Random Seed
        random.seed(config['seed'])
        torch.manual_seed(config['seed'])

        # step1: data
        self.test_dataloader = config['dataloader']

        # step2: model
        self.model = config['model']

        # step3: criterion
        self.criterion = config['criterion']


    

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