import torch
from utils.visualize import Visualizer
from tqdm import tqdm
import torchvision as tv
import time
import random

from config import Configuration
import models
from utils import accuracy, print_model_parameters, AverageMeter, print_flops_params, get_dataloader

class Tester(object):
    """
    可通过传入config_dic来配置Tester，这种情况下不会在初始化过程中print相关数据
    也可通过**kwargs配置Tester
    """
    def __init__(self, config_dic=None, **kwargs):

        if config_dic is None:
            self.init_from_kwargs(**kwargs)
        else:
            self.init_from_config(config_dic)
        
    def run(self):
        print_flops_params(model=self.model)
        self.test()

    def test(self, model=None, epoch=None):
        """
        测试指定模型在指定数据集上的表现, 数据集在创建Tester类时通过修改self.config确定
        args:
            model: 要测试的模型，若不为none则self.model更新为model，若为none则测试self.model
            epoch：仅用于显示当前epoch
        """
        if model is not None:
            self.model = model
        self.model.eval() # 验证模式
        self.loss_meter.reset()
        self.top1_acc.reset()
        self.top5_acc.reset()
        self.batch_time.reset()
        self.dataload_time.reset()

        end_time = time.time()
        # print("testing...")
        with torch.no_grad():
            for batch_index, (input, target) in enumerate(self.test_dataloader):
                # measure data loading time
                self.dataload_time.update(time.time() - end_time)

                # compute output
                input, target = input.to(self.device), target.to(self.device)
                output = self.model(input)
                loss = self.criterion(output, target)

                # meters update and visualize
                self.loss_meter.update(loss.item(), input.size(0))
                prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
                self.top1_acc.update(prec1.data.cpu(), input.size(0))
                # self.top5_acc.update(prec5.data.cpu(), input.size(0))

                # measure elapsed time
                self.batch_time.update(time.time() - end_time)
                end_time = time.time()

                done = (batch_index+1) * self.test_dataloader.batch_size
                percentage = 100. * (batch_index+1) / len(self.test_dataloader)
                time_str = time.strftime('%H:%M:%S')
                print("\r"
                "Test: {epoch:4} "
                "[{done:7}/{total_len:7} ({percentage:3.0f}%)] "
                "loss: {loss_meter:.3f} | "
                "top1: {top1:3.3f}% | "
                # "top5: {top5:3.3f} | "
                "load_time: {time_percent:2.0f}% | "
                "UTC+8: {time_str} ".format(
                    epoch=0 if epoch == None else epoch,
                    done=done,
                    total_len=len(self.test_dataloader.dataset),
                    percentage=percentage,
                    loss_meter=self.loss_meter.avg,
                    top1=self.top1_acc.avg,
                    # top5=self.top5_acc.avg,
                    time_percent=self.dataload_time.avg/self.batch_time.avg*100,
                    time_str=time_str
                ), end=""
            )
        print("")
        
        # visualize
        if self.vis is not None:
            self.vis.plot('test_loss', self.loss_meter.avg, x=epoch)
            self.vis.plot('test_top1', self.top1_acc.avg, x=epoch)

    def init_from_kwargs(self, **kwargs):
        
        print("| ----------------- Initializing Tester ------------------ |")
        
        self.config = Configuration()
        self.config.update_config(kwargs) # 解析参数更新默认配置
        if self.config.check_config(): raise # 检测路径、设备是否存在

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
        if self.config.random_seed is None:
            self.config.random_seed = random.randint(1, 10000)
        random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

        # step1: data
        _, self.test_dataloader, self.num_classes = get_dataloader(self.config)

        # step2: model
        print('{:<30}  {:<8}'.format('==> creating model: ', self.config.arch))
        print('{:<30}  {:<8}'.format('==> loading model: ', self.config.load_model_path if self.config.load_model_path != None else 'None'))
        self.model = models.__dict__[self.config.arch](num_classes=self.num_classes) # 从models中获取名为config.arch的model
        if len(self.config.gpu_idx_list) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config.gpu_idx_list)
        self.model.to(self.device) # 模型转移到设备上
        if self.config.load_model_path: # 加载目标模型参数
            # self.model.load_state_dict(torch.load(self.config.load_model_path, map_location=self.device))
            checkpoint = torch.load(self.config.load_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("{:<30}  {:<8}".format('==> model epoch: ', checkpoint['epoch']))
            print("{:<30}  {:<8}".format('==> model best acc1: ', checkpoint['best_acc1']))
            # print(self.model)
            # print_model_parameters(self.model)
            # print_flops_params(self.model, self.config.dataset)

        # step3: criterion
        self.criterion = torch.nn.CrossEntropyLoss()

        #step4: meters
        self.loss_meter = AverageMeter() # 计算所有数的平均值和标准差，这里用来统计一个epoch中的平均值
        self.top1_acc = AverageMeter()
        self.top5_acc = AverageMeter()
        self.batch_time = AverageMeter()
        self.dataload_time = AverageMeter()

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
        self.criterion = torch.nn.CrossEntropyLoss()

        #step4: meters
        self.loss_meter = AverageMeter() # 计算所有数的平均值和标准差，这里用来统计一个epoch中的平均值
        self.top1_acc = AverageMeter()
        self.top5_acc = AverageMeter()
        self.batch_time = AverageMeter()
        self.dataload_time = AverageMeter()

    

if __name__ == "__main__":
    tester = Tester(
        batch_size=200,
        model='vgg_cfg',
        dataset="cifar10",
        gpu_idx = "4", # choose gpu
        load_model_path='checkpoints/with_sparsity/cifar10_vgg_cfg_best.pth.tar',
        random_seed=1,
        # visdom = True, # 使用visdom可视化训练过程
        # env='test_test',
        # print_config=True,
        # print_device=True,
        num_workers = 10, # 使用多进程加载数据
    )
    tester.run()
    print("end")