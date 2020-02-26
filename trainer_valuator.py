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
        print("--------  model: {model}  --  dataset: {dataset}  --  duration: {dh:2}h:{dm:02d}.{ds:02d}  --------".format(
            model=self.trainer.config.model,
            dataset=self.trainer.config.dataset,
            dh=interval.seconds//3600,
            dm=interval.seconds%3600//60,
            ds=interval.seconds%60,
        ))

    def run(self):
        # initial test
        self.valuator.test(epoch=0)
        self.print_bar()
        for epoch in range(1, self.trainer.config.max_epoch):
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
        checkpoint_path = "checkpoints/" + self.trainer.config.dataset + "_" + self.trainer.config.model + "_epoch{epoch}_acc{acc:.2f}.pth".format(epoch=self.trainer.config.max_epoch, acc=self.valuator.top1_acc.avg)
        if len(self.trainer.config.gpu_idx_list) > 1:
            torch.save(self.trainer.model.module.state_dict(), checkpoint_path)
        else: torch.save(self.trainer.model.state_dict(), checkpoint_path)



# if __name__ == "__main__":
#     trainer_valuator = Trainer_valuator(
#         max_epoch=200,
#         batch_size=100,
#         lr=1e-2,
#         lr_scheduler_milestones=[50, 170],
#         weight_decay=1e-4,
#         momentum=0.9,
#         # plot_interval=5,
#         model='alexnet_cifar',
#         dataset="cifar10",
#         gpu_idx = "0", # choose gpu
#         # save_model_path="checkpoints/resnet34.pth",
#         # load_model_path='checkpoints/cifar100_resnet18_epoch99_acc92.83.pth', 
#         random_seed=2,
#         # print_config=True,
#         # print_device=True,
#         num_workers = 5, # 使用多进程加载数据
#         # use_visdom = True, # 使用visdom可视化训练过程
#     )
#     trainer_valuator.run()
#     print("end")

# # test
# if __name__ == "__main__":
#     trainer_valuator = Trainer_valuator(
#         max_epoch=90,
#         batch_size=50,
#         lr=1e-1,
#         lr_scheduler_milestones=[31, 61],
#         model='test',
#         dataset="cifar10",
#         gpu_idx = "7", # choose gpu
#         weight_decay=1e-4,
#         momentum=0.9,
#         random_seed=2,
#         # num_workers = 20, # 使用多进程加载数据
#         use_visdom = True, # 使用visdom可视化训练过程
#         plot_interval=50,
#         env='test_test',
#     )
#     trainer_valuator.run()
#     print("end")

# ---------------------------imagenet------------------------------------------------------

# # resnet
# if __name__ == "__main__":
#     trainer_valuator = Trainer_valuator(
#         max_epoch=50,
#         batch_size=200,
#         lr=1e-1,
#         lr_scheduler_milestones=[20, 30, 40],
#         model='resnext50_32x4d',
#         dataset="imagenet",
#         gpu_idx = "2, 3", # choose gpu
#         weight_decay=1e-4,
#         momentum=0.9,
#         random_seed=2,
#         num_workers = 10, # 使用多进程加载数据
#         use_visdom = True, # 使用visdom可视化训练过程
#         plot_interval=200,
#         env='imagenet_classification_resnext50_32x4d',
#     )
#     trainer_valuator.run()
#     print("end")
    

# # vgg
# if __name__ == "__main__":
#     trainer_valuator = Trainer_valuator(
#         max_epoch=100,
#         batch_size=200,
#         lr=1e-2,
#         lr_scheduler_milestones=[50],
#         model='vgg11_cifar',
#         dataset="imagenet",
#         gpu_idx = "0, 1", # choose gpu
#         weight_decay=1e-4,
#         momentum=0.9,
#         random_seed=2,
#         num_workers = 10, # 使用多进程加载数据
#         use_visdom = True, # 使用visdom可视化训练过程
#         plot_interval=200,
#         env='imagenet_classification',
#     )
#     trainer_valuator.run()
#     print("end")


# ---------------------------cifar------------------------------------------------------

# # shufflenet
# if __name__ == "__main__":
#     trainer_valuator = Trainer_valuator(
#         max_epoch=200,
#         batch_size=100,
#         lr=1e-2,
#         lr_scheduler_milestones=[100, 150],
#         weight_decay=1e-4,
#         momentum=0.9,
#         model='shufflenet_v2_x2_0',
#         dataset="cifar10",
#         gpu_idx = "1", # choose gpu
#         random_seed=2,
#         num_workers = 5, # 使用多进程加载数据
#     )
#     trainer_valuator.run()
#     print("end")


# # mobilenet
# if __name__ == "__main__":
#     trainer_valuator = Trainer_valuator(
#         max_epoch=200,
#         batch_size=100,
#         lr=1e-2,
#         lr_scheduler_milestones=[100, 150],
#         weight_decay=1e-4,
#         momentum=0.9,
#         model='mobilenet_v2',
#         dataset="cifar10",
#         gpu_idx = "0", # choose gpu
#         random_seed=2,
#         num_workers = 5, # 使用多进程加载数据
#     )
#     trainer_valuator.run()
#     print("end")


# # densenet
# if __name__ == "__main__":
#     trainer_valuator = Trainer_valuator(
#         max_epoch=80,
#         batch_size=100,
#         lr=1e-2,
#         lr_scheduler_milestones=[40, 60],
#         weight_decay=1e-4,
#         momentum=0.9,
#         model='densenet_cifar2',
#         dataset="cifar100",
#         gpu_idx = "3", # choose gpu
#         random_seed=2,
#         num_workers = 5, # 使用多进程加载数据
#     )
#     trainer_valuator.run()
#     print("end")


# # resnext
# if __name__ == "__main__":
#     trainer_valuator = Trainer_valuator(
#         max_epoch=80,
#         batch_size=100,
#         lr=1e-2,
#         lr_scheduler_milestones=[40, 60],
#         weight_decay=1e-4,
#         momentum=0.9,
#         model='resnext29_8x16d',
#         dataset="cifar10",
#         gpu_idx = "1, 2", # choose gpu
#         random_seed=2,
#         num_workers = 5, # 使用多进程加载数据
#     )
#     trainer_valuator.run()
#     print("end")


# # resnet
# if __name__ == "__main__":
#     trainer_valuator = Trainer_valuator(
#         max_epoch=200,
#         batch_size=100,
#         lr=1e-1,
#         lr_scheduler_milestones=[100, 150],
#         weight_decay=1e-4,
#         momentum=0.9,
#         model='resnet20',
#         dataset="cifar10",
#         gpu_idx = "7", # choose gpu
#         random_seed=2,
#         num_workers = 5, # 使用多进程加载数据
#     )
#     trainer_valuator.run()
# #     print("end")


# vgg
if __name__ == "__main__":
    trainer_valuator = Trainer_valuator(
        max_epoch=164,
        batch_size=100,
        lr=1e-2,
        lr_scheduler_milestones=[81, 122],
        model='vgg19_cifar',
        env='cifar10_vgg19_xxx',
        dataset="cifar10",
        gpu_idx = "7", # choose gpu
        weight_decay=1e-4,
        momentum=0.9,
        random_seed=2,
        num_workers = 5, # 使用多进程加载数据
        use_visdom = True, # 使用visdom可视化训练过程
        plot_interval=50,
    )
    trainer_valuator.run()
    print("end")


# # alexnet
# if __name__ == "__main__":
#     trainer_valuator = Trainer_valuator(
#         max_epoch=150,
#         batch_size=100,
#         lr=1e-2,
#         lr_scheduler_milestones=[50, 110],
#         weight_decay=1e-4,
#         momentum=0.9,
#         model='alexnet_cifar',
#         dataset="cifar100",
#         gpu_idx = "7", # choose gpu
#         random_seed=2,
#         num_workers = 5, # 使用多进程加载数据
#     )
#     trainer_valuator.run()
#     print("end")