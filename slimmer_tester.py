# import ssl
# #全局取消证书验证
# ssl._create_default_https_context = ssl._create_unverified_context

import torch
import datetime
import copy

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
        if config.use_visdom:
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
                + "_" + self.slimmer.config.model)
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
                + self.retrainer.config.model)
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
    slimmer_tester = Slimmer_tester(
        model='vgg_cfg',
        dataset="cifar10",
        gpu_idx = "4", # choose gpu
        random_seed=2,
        load_model_path="checkpoints/with_sparsity/cifar10_vgg_cfg_best.pth.tar",
        num_workers = 6, # 使用多进程加载数据
        slim_percent=0.7,
        # retrain
        max_epoch=10,
        batch_size=100,
        lr=1e-1,
        lr_scheduler_milestones=[4, 7],
        weight_decay=1e-4,
        momentum=0.9,
        sparsity=False,
        use_visdom = True, # 使用visdom可视化训练过程
        env='cifar10_vgg_cfg_retrain',
        legend='vgg_cfg_sparsity_retrain',
        plot_interval=50,
    )
    slimmer_tester.run()
    print("end")

