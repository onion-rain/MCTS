import warnings
import torch
from pathlib import Path

__all__ = ['Configuration']

class Configuration(object):
    '''
    使用范例：
        import models
        from config import Configuration

        config = Configuration()\n
        lr = config.lr\n
        model = getattr(models, config.arch)\n
        dataset = DogCat_dataset(config.train_data_root)\n
    '''

    arch = 'resnet34' # 要训练的网络结构，名字必须与models/__init__.py中的名字一致
    gpu_idx = "" # choose gpu
    max_epoch = 100
    lr = 1e-1 # initial learning rate
    lr_decay = 0.2 # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
    momentum = 0.9
    valuate = False # 每训练一个epoch进行一次valuate
    print_config = False
    print_device = False
    resume_path = '' # 断点续练hhh
    refine = False # 是否根据structure加载剪枝后的模型结构
    deterministic = False # 结论确定，若为True则在相同pytorch版本和相同随机种子相同workers情况下结果可复现
    random_seed = 0
    usr_suffix = ''
    log_path = 'logs/log.txt'

    # test专用
    load_model_path = None # 加载预训练参数的路径
    
    # dataloader config
    dataset = "cifar10" # Dataset can only be cifar10 or cifar100 or imagenet
    dataset_root = '/home/xueruini/onion_rain/pytorch/dataset/' # 训练集存放路径
    batch_size = 100
    num_workers = 1 # 默认便于调试
    droplast = False

    # visdom
    visdom = False
    vis_env = 'main' # visdom 环境
    vis_legend = None # visdom 图例名称，为None从取env第一个"_"之后的字符串作为legend
    vis_interval = 20 # visdom plot info every N batch

    # slimming
    sr = False
    sr_lambda = 1e-4
    slim_percent = 0.7

    # prune
    prune_percent = 0.5
    lp_norm=2
    prune_object = 'all'
    sfp_intervals = None

    # thinet
    method = 'greedy'

    # meta prune
    max_flops = 0
    population = 100
    select_num = 30
    mutation_num = 30
    crossover_num = 30
    mutation_prob = 0.1
    research_resume_path = ''


    def update_config(self, kwargs):
        '''
        根据字典kwargs 更新 config 参数\n
        config = Configuration()\n
        new_config = {'lr':0.1,'use_gpu':False}\n
        config.update_config(new_config)\n
        config.lr == 0.1\n
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                # 警告还是报错，取决于你个人的喜好
                warnings.warn("\nWarning: opt has not attribut %s" %k)
            setattr(self, k, v)
        
        # 打印配置信息
        if self.print_config == True:
            print('user config:')
            for k, v in self.__class__.__dict__.items():
                if not k.startswith('__'):
                    print(k, "\t\t", getattr(self, k))
    
    def sting2list(self, x):
        if x == "":
            return []
        else: return list(map(int, x.split(",")))

    def check_config(self):
        '''
        自动检测config配置是否合理
        '''
        # self.multi_gpus = False
        self.gpu_idx_list = self.sting2list(self.gpu_idx)
        # if len(self.gpu_idx_list) > 1:
        #     self.multi_gpus = True
        errors = 0

        if torch.cuda.is_available():
            if self.print_device == True:
                print("usable GPU devices: ")
                for n in range(torch.cuda.device_count()): # 打印出所有可用gpu
                    print("device {n}: {name}".format(n=n, name=torch.cuda.get_device_name(n)))
            
            if len(self.gpu_idx_list) == 0: # 有gpu却没用
                print("config warning: You can use the gpu for higher performance")
            elif max(self.gpu_idx_list) >= torch.cuda.device_count():
                print("invalid device index: {}, config.gpu_inx should no more than {}"
                    .format(self.gpu_idx, torch.cuda.device_count()-1))
                errors += 1
        else:
            if not len(self.gpu_idx_list) == 0: # 没 gpu 还想用
                print("config error: torch.cuda.is_available() is False. \
                    If you are running on a CPU-only machine, please change config.use_gpu to False.")
                errors += 1
        
        if not (self.dataset_root == None or self.dataset_root == ''):
            if not Path(self.dataset_root).exists():
                print("config error: No such file or directory: '{}'".format(self.dataset_root))
                errors += 1

        if not (self.load_model_path == None or self.load_model_path == ''):
            if not Path(self.load_model_path).exists():
                print("config error: No such file or directory: '{}'".format(self.load_model_path))
                errors += 1

        if not (self.resume_path == None or self.resume_path == ''):
            if not Path(self.resume_path).exists():
                print("config error: No such file or directory: '{}'".format(self.resume_path))
                errors += 1
        
        if errors != 0: print("config errors : {}".format(errors))
        return errors

# config = Configuration()

if __name__ == "__main__":
    config = Configuration()
    config.test_config()
    print("end")