# import ssl
# #全局取消证书验证
# ssl._create_default_https_context = ssl._create_unverified_context

import torch
import datetime
import copy

from slimmer import Slimmer
from tester import Tester
from config import Configuration
from utils import print_flops_params, print_nonzeros, print_model_parameters

class Slimmer_tester(object):

    def __init__(self, **kwargs):

        self.start_time = datetime.datetime.now()
        config = Configuration()
        config.update_config(kwargs) # 解析参数更新默认配置
        if config.check_config(): raise # 检测路径、设备是否存在

        self.slimmer = Slimmer(config=config)
        self.tester = Tester(config=config)

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

        # save slimmed model
        if self.slimmer.config.save_model_path is None:
            self.slimmer.config.save_model_path = ("slimmed_checkpoints/" 
                                                + self.slimmer.config.dataset 
                                                + "_" 
                                                + self.slimmer.config.model 
                                                + "_ratio{:.2f}".format(self.slimmer.slim_ratio)
                                                + "_acc{:.2f}".format(self.tester.top1_acc.avg) 
                                                + "_" + str(cfg) + ".pth")
        torch.save(self.slimmer.model.state_dict(), self.slimmer.config.save_model_path)

if __name__ == "__main__":
    slimmer_tester = Slimmer_tester(
        model='vgg_cfg',
        dataset="cifar10",
        gpu_idx = "5", # choose gpu
        random_seed=2,
        load_model_path="checkpoints/cifar10_vgg_slim_epoch150_acc91.86.pth",
        num_workers = 5, # 使用多进程加载数据
        slim_percent=0.8,
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

