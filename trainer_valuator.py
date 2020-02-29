# import ssl
# #全局取消证书验证
# ssl._create_default_https_context = ssl._create_unverified_context

import torch
import datetime

from trainer import Trainer
from tester import Tester
from config import Configuration
from utils.visualize import Visualizer
from utils import print_flops_params, print_nonzeros, print_model_parameters, save_checkpoint

class Trainer_valuator(object):

    def __init__(self, **kwargs):

        self.start_time = datetime.datetime.now()
        config = Configuration()
        config.update_config(kwargs) # 解析参数更新默认配置
        if config.check_config(): raise # 检测路径、设备是否存在
        vis = None
        if config.use_visdom:
            vis = Visualizer(config.env, config.legend) # 初始化visdom

        self.trainer = Trainer(config=config, vis=vis)
        self.valuator = Tester(config=config, vis=vis)
    
    def print_bar(self):
        """calculate duration time"""
        interval = datetime.datetime.now() - self.start_time
        print("--------  model: {model}  --  dataset: {dataset}  --  duration: {dh:2}h:{dm:02d}.{ds:02d}  --------".
            format(
                model=self.trainer.config.model,
                dataset=self.trainer.config.dataset,
                dh=interval.seconds//3600,
                dm=interval.seconds%3600//60,
                ds=interval.seconds%60,
            )
        )

    def run(self):
        best_acc1 = 0
        name = (self.trainer.config.dataset + "_" + self.trainer.config.model)

        # initial test
        print_flops_params(model=self.valuator.model)
        self.valuator.test(epoch=0)
        self.print_bar()
        print("")
        for epoch in range(1, self.trainer.config.max_epoch+1):
            # train & valuate
            self.trainer.train(epoch=epoch)
            self.valuator.test(model=self.trainer.model, epoch=epoch)
            self.print_bar()
            print("")
            
            # save checkpoint
            is_best = self.valuator.top1_acc.avg > best_acc1
            best_acc1 = max(self.valuator.top1_acc.avg, best_acc1)
            if len(self.trainer.config.gpu_idx_list) > 1:
                state_dict = self.trainer.model.module.state_dict()
            else: state_dict = self.trainer.model.state_dict()
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'best_acc1': best_acc1,
                'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            }, is_best=is_best, epoch=None, file_root='checkpoints/with_sparsity/', file_name=name)


# train vgg with sparsity
if __name__ == "__main__":
    trainer_valuator = Trainer_valuator(
        max_epoch=150,
        batch_size=100,
        lr=1e-1,
        lr_scheduler_milestones=[81, 122],
        model='vgg_cfg',
        env='cifar10_vgg_cfg_sparsity',
        legend='vgg_cfg_sparsity',
        use_visdom = True, # 使用visdom可视化训练过程
        plot_interval=50,
        sparsity=True,
        sparsity_lambda=1e-4,
        gpu_idx = "6", # choose gpu
        dataset="cifar10",
        weight_decay=1e-4,
        momentum=0.9,
        random_seed=1,
        num_workers = 10, # 使用多进程加载数据
    )
    trainer_valuator.run()
    print("end")

# # train vgg without sparsity
# if __name__ == "__main__":
#     trainer_valuator = Trainer_valuator(
#         max_epoch=150,
#         batch_size=100,
#         lr=1e-1,
#         lr_scheduler_milestones=[81, 122],
#         model='vgg_cfg',
#         env='cifar10_vgg_cfg',
#         legend='vgg_cfg',
#         use_visdom = True, # 使用visdom可视化训练过程
#         plot_interval=50,
#         sparsity=False,
#         gpu_idx = "5", # choose gpu
#         dataset="cifar10",
#         weight_decay=1e-4,
#         momentum=0.9,
#         random_seed=1,
#         num_workers = 10, # 使用多进程加载数据
#     )
#     trainer_valuator.run()
#     print("end")

