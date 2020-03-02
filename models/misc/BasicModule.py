import torch
import time

class BasicModule(torch.nn.Module):
    '''
    封装了nn.Module,主要是提供了save和load两个方法
    '''
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))# 默认名字

    def load(self, path, device="cuda"):
        '''
        可加载指定路径的模型
        '''
        self.load_state_dict(torch.load(path, map_location=device))

    def save(self, name=None):
        '''
        保存模型参数，默认使用“模型名字+时间”作为文件名
        '''
        if name is None:
            prefix = "./checkpoints/" + self.model_name + '_'
            name = time.strftime(prefix + '20%y%m%d_%H.%M.%S.pth')
        torch.save(self.state_dict(), name)
        return name


class Flat(torch.nn.Module):
    '''
    把输入reshape成（batch_size,dim_length）
    '''
    def __init__(self):
        super(Flat, self).__init__()
        #self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)

if __name__ == "__main__":
    BasicModule = BasicModule()
    BasicModule.model_name = "BasicModule"
    BasicModule.save()
    print(time.strftime('20%y%m%d_%H.%M.%S.pth'))