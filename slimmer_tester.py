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
from utils import print_flops_params, print_nonzeros, print_model_parameters

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
        cfg = []

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
        cfg = self.slimmer.slim()
        self.tester.test(self.slimmer.slimmed_model)
        print_flops_params(self.tester.model, self.tester.config.dataset)

        # print("")
        # print("| ----------------- saving slimmed model ----------------- |")
        # save_model_path = ("slimmed_checkpoints/" 
        #                     + self.slimmer.config.dataset 
        #                     + "_" 
        #                     + self.slimmer.config.model 
        #                     + "_ratio{:.2f}".format(self.slimmer.slim_ratio)
        #                     + "_acc{:.2f}".format(self.tester.top1_acc.avg)
        #                     + "_" + str(cfg) + ".pth")
        # print(save_model_path)
        # torch.save(self.slimmer.slimmed_model.state_dict(), save_model_path)

        # print("")
        # print("| ------------------- retraining model ------------------- |")
        # for epoch in range(1, self.retrainer.config.max_epoch+1):
        #     # train & valuate
        #     self.retrainer.train(self.slimmer.slimmed_model, epoch)
        #     self.tester.test(self.retrainer.model, epoch)
        #     print("")

        # print("")
        # print("| -----------------saving retrained model ---------------- |")
        # if self.retrainer.config.save_model_path is None:
        #     save_model_path = ("checkpoints/" + self.retrainer.config.dataset 
        #                         + "_" + self.retrainer.config.model 
        #                         + "_epoch{epoch}_acc{acc:.2f}.pth".format(
        #                             epoch=self.retrainer.config.max_epoch, 
        #                             acc=self.valuator.top1_acc.avg
        #                         ))
        # else: save_model_path = self.retrainer.config.save_model_path
        # if len(self.retrainer.config.gpu_idx_list) > 1:
        #     torch.save(self.retrainer.model.module.state_dict(), save_model_path)
        # else: torch.save(self.retrainer.model.state_dict(), save_model_path)


if __name__ == "__main__":
    slimmer_tester = Slimmer_tester(
        model='vgg',
        dataset="cifar10",
        gpu_idx = "5", # choose gpu
        random_seed=2,
        load_model_path="checkpoints/github_slim/cifar10_vgg_best.pth.tar",
        num_workers = 5, # 使用多进程加载数据
        slim_percent=0.3,
        # save_model_path="slimmed_checkpoints/xxx.pth"
        # retrain
        max_epoch=30,
        batch_size=100,
        lr=1e-2,
        lr_scheduler_milestones=[10, 20],
        weight_decay=1e-4,
        momentum=0.9,
        slim=False,
        use_visdom = True, # 使用visdom可视化训练过程
        env='slim_vgg_cfg_retrain',
        legend='sparsity_vgg_cfg_retrain',
        plot_interval=50,
    )
    slimmer_tester.run()
    print("end")

# if __name__ == "__main__":
#     slimmer_tester = Slimmer_tester(
#         model='nin',
#         dataset="cifar10",
#         gpu_idx = "5", # choose gpu
#         random_seed=2,
#         load_model_path="checkpoints/cifar10_nin_epoch123_acc90.83.pth",
#         num_workers = 5, # 使用多进程加载数据
#         slim_percent=0.1,
#     )
#     slimmer_tester.run()
#     print("end")

