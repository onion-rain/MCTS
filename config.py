import warnings
import torch
from pathlib import Path

class Configuration(object):
    '''
    使用范例：
    import models
    from config import Configuration

    config = Configuration()\n
    lr = config.lr\n
    model = getattr(models, config.model)\n
    dataset = DogCat_dataset(config.train_data_root)\n
    '''
    use_visdom = False
    env = 'cifar10_vgg16' # visdom 环境
    legend = None # visdom 图例名称，为None从取env第一个"_"之后的字符串作为legend
    model = 'resnet34' # 要训练的模型，名字必须与models/__init__.py中的名字一致
    gpu_idx = "" # choose gpu
    num_workers = 0 # 默认不使用多进程加载数据，便于调试
    plot_interval = 20 # visdom plot info every N batch
    max_epoch = 100
    batch_size = 100
    lr = 1e-2 # initial learning rate
    lr_decay = 0.2 # when val_loss increase, lr = lr*lr_decay
    random_seed = None
    print_config = False
    print_device = False
    weight_decay = 5e-4
    momentum = 0.9
    lr_scheduler_milestones = [100, 150]
    dataloader_droplast = False
    dataset = "cifar10" # Dataset can only be cifar10 or cifar100
    
    slim = False
    slim_lambda = 1e-4
    slim_percent = 0.1

    # # windows
    # dataset_root = 'E:\competition\Python\PyTorch\dataset/' # 训练集存放路径
    # load_model_path = None # 加载预训练的模型的路径，为None代表不加载
    # save_model_path = None

    # gpu_server
    dataset_root = '/home/xueruini/onion_rain/pytorch/dataset/' # 训练集存放路径
    load_model_path = None # 加载预训练的模型的路径，为None代表不加载
    save_model_path = None

    log_path = "log.txt"
    sensitivity = 2 # sensitivity灵敏度，sensitivity*std得到prune的threshold
    percentile = 95 # 剪枝比例
    bits = 5 # 量化比特数

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
        self.multi_gpus = False
        self.gpu_idx_list = self.sting2list(self.gpu_idx)
        if len(self.gpu_idx_list) > 1:
            self.multi_gpus = True
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
        
        if not self.dataset_root == None:
            if not Path(self.dataset_root).exists():
                print("config error: No such file or directory: '{}'".format(self.dataset_root))
                errors += 1

        if not self.load_model_path == None:
            if not Path(self.load_model_path).exists():
                print("config error: No such file or directory: '{}'".format(self.load_model_path))
                errors += 1
        
        if errors != 0: print("config errors : {}".format(errors))
        return errors

# config = Configuration()

if __name__ == "__main__":
    config = Configuration()
    config.test_config()
    print("end")