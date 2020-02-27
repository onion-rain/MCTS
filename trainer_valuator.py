# import ssl
# #全局取消证书验证
# ssl._create_default_https_context = ssl._create_unverified_context

import torch
import datetime

from trainer import Trainer
from tester import Tester
from config import Configuration
from utils.visualize import Visualizer

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
        # initial test
        self.valuator.print_flops_params()
        self.valuator.test(epoch=0)
        self.print_bar()
        for epoch in range(1, self.trainer.config.max_epoch+1):
            # train & valuate
            self.trainer.train(epoch)
            self.valuator.model = self.trainer.model
            self.valuator.test(epoch)
            self.print_bar()

            # save checkpoint
            if ((epoch%(self.trainer.config.max_epoch//4) == 0) 
                and (epoch != self.trainer.config.max_epoch)):
                checkpoint_path = "checkpoints/" + self.trainer.config.dataset + "_" + self.trainer.config.model + "_epoch{epoch}_acc{acc:.2f}.pth".format(epoch=epoch, acc=self.valuator.top1_acc.avg)
                if len(self.trainer.config.gpu_idx_list) > 1:
                    torch.save(self.trainer.model.module.state_dict(), checkpoint_path)
                else: torch.save(self.trainer.model.state_dict(), checkpoint_path)
        
        # save last model
        if self.trainer.config.save_model_path is None:
            save_model_path = "checkpoints/" + self.trainer.config.dataset + "_" + self.trainer.config.model + "_epoch{epoch}_acc{acc:.2f}.pth".format(epoch=self.trainer.config.max_epoch, acc=self.valuator.top1_acc.avg)
        if len(self.trainer.config.gpu_idx_list) > 1:
            torch.save(self.trainer.model.module.state_dict(), save_model_path)
        else: torch.save(self.trainer.model.state_dict(), save_model_path)


# vgg retrain
if __name__ == "__main__":
    trainer_valuator = Trainer_valuator(
        max_epoch=40,
        batch_size=100,
        lr=1e-2,
        lr_scheduler_milestones=[20],
        model='vgg_cfg',
        gpu_idx = "4", # choose gpu
        dataset="cifar10",
        slim_lambda=1e-4,
        weight_decay=1e-4,
        momentum=0.9,
        random_seed=2,
        num_workers = 10, # 使用多进程加载数据
        load_model_path="slimmed_checkpoints/cifar10_vgg_cfg_acc57.48.pth"
    )
    trainer_valuator.run()
    print("end")


# # vgg
# if __name__ == "__main__":
#     trainer_valuator = Trainer_valuator(
#         max_epoch=150,
#         batch_size=100,
#         lr=1e-2,
#         lr_scheduler_milestones=[81, 122],
#         model='vgg_cfg',
#         env='slim_vgg_cfg',
#         legend='sparsity_vgg_cfg',
#         slim=True,
#         gpu_idx = "4", # choose gpu
#         dataset="cifar10",
#         slim_lambda=1e-4,
#         weight_decay=1e-4,
#         momentum=0.9,
#         random_seed=2,
#         num_workers = 10, # 使用多进程加载数据
#         use_visdom = True, # 使用visdom可视化训练过程
#         plot_interval=50,
#     )
#     trainer_valuator.run()
#     print("end")


# # nin
# if __name__ == "__main__":
#     trainer_valuator = Trainer_valuator(
#         max_epoch=164,
#         batch_size=100,
#         lr=1e-2,
#         lr_scheduler_milestones=[81, 122],
#         model='nin',
#         env='slim_nin',
#         legend='sparsity_nin',
#         slim=True,
#         gpu_idx = "6", # choose gpu
#         dataset="cifar10",
#         slim_lambda=1e-4,
#         weight_decay=1e-4,
#         momentum=0.9,
#         random_seed=2,
#         num_workers = 5, # 使用多进程加载数据
#         use_visdom = True, # 使用visdom可视化训练过程
#         plot_interval=50,
#     )
#     trainer_valuator.run()
#     print("end")

# # nin_gc
# if __name__ == "__main__":
#     trainer_valuator = Trainer_valuator(
#         max_epoch=164,
#         batch_size=100,
#         lr=1e-2,
#         lr_scheduler_milestones=[81, 122],
#         model='nin_gc',
#         env='slim_nin_gc',
#         legend='sparsity_nin_gc',
#         slim=True,
#         gpu_idx = "6", # choose gpu
#         dataset="cifar10",
#         slim_lambda=1e-4,
#         weight_decay=1e-4,
#         momentum=0.9,
#         random_seed=2,
#         num_workers = 5, # 使用多进程加载数据
#         use_visdom = True, # 使用visdom可视化训练过程
#         plot_interval=50,
#     )
#     trainer_valuator.run()
#     print("end")

