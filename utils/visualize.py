import visdom
import time
import numpy as np


__all__ = ['Visualizer']


class Visualizer(object):
    '''
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    Args:
        env(str): 环境名，Default: 'default'
        clear(bool): 是否清空环境，Default: True
    '''
    def __init__(self, env='default', legend=None, clear=True, **kwargs):
    
        self.vis = visdom.Visdom(env=env, **kwargs)
        if(clear == True):
            self.vis.close(env=env)
        
        if legend is None:
            index = env.find("_") + 1
            if index > 0:
                self.legend = env[index:]
            else: self.legend = None
        else: self.legend = legend
        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {} 
        self.log_text = ''
    def reinit(self, env='default', **kwargs):
        '''
        修改visdom的配置
        '''
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        '''
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        '''
        for k, v in d.iteritems():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)

    def plot(self, name, y, x=None, **kwargs):
        '''
        self.plot('loss',1.00)
        '''
        if x == None:
            x = self.index.get(name, 0) # 找键值为name的item，如果不存在返回0
        self.vis.line(
            Y=np.array([y]), X=np.array([x]),
            win=str(name), # 指定pane的名字，不指定的话，visdom将自动分配一个新的pane，同名会覆盖
            opts=dict(
                title=name,
                showlegend=False if self.legend is None else True,
                legend=[self.legend],
            ), # 选项，接受一个字典，常见的option包括title、xlabel、ylabel、width等，主要用于设置pane的显示格式。
            update=None if x == 0 else 'append', # 指定参数update='append'来避免覆盖之前的数值。
            **kwargs
        )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        '''
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)

        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        '''
        self.vis.images(
            img_.cpu().numpy(),
            win=str(name),
            opts=dict(title=name),
            **kwargs
        )

    def img_grid_many(self, d):
        for k, v in d.items():
            self.img_grid(k, v)

    def img_grid(self, name, input_3d):
        """
        convert batch images to grid of images
        i.e. input（36，64，64） ->  6*6 grid，each grid is an image of size 64*64
        """
        self.img(name, tv.utils.make_grid(
            input_3d.cpu()[0].unsqueeze(1).clamp(max=1, min=0)))

    def log(self, info, win='log_text'):
        '''
        self.log({'loss':1,'lr':0.0001})
        '''

        self.log_text += (
            '[{time}] {info} <br>'.format(
                time=time.strftime('20%y%m%d_%H.%M.%S'),
                info=info
            )
        ) 
        self.vis.text(self.log_text, win)   

    def __getattr__(self, name):
        return getattr(self.vis, name)

# test
if __name__ == "__main__":
    visdom = Visualizer(env="hhh")
    x = 1
    visdom.index["loss"] = x
    y = x**2
    visdom.plot("loss", y=y)
    # visdom.vis.line(
    #     X=x, Y=y,
    #     name=str("hhh"),
    #     opts=dict(title="www")
    # )