import torch
import numpy as np
import math
import datetime
from sklearn.linear_model import Lasso

def pruner_print_bar(str, channel_num, start_time):
    """calculate duration time"""
    interval = datetime.datetime.now() - start_time
    print("-------- layer: {str}  -- remain channel num: {channel_num}  --  duration: {dh:2}h:{dm:02d}.{ds:02d}  --------".
        format(
            str=str,
            channel_num=channel_num,
            dh=interval.seconds//3600,
            dm=interval.seconds%3600//60,
            ds=interval.seconds%60,
        )
    )

def get_tuples(model):
    """
    Code from https://github.com/synxlin/nn-compression.
    获得计算第i层以及i+1层输入特征图的前向传播函数
    
    return:
        list of tuple, [(module_name, module, next_bn, next_module, fn_input_feature, fn_next_input_feature), ...]
    """
    # 提取各层
    features = model.features
    if isinstance(features, torch.nn.DataParallel):
        features = features.module
    classifier = model.classifier

    module_name_dict = dict()
    for n, m in model.named_modules():
        module_name_dict[m] = n

    conv_indices = []
    conv_modules = []
    conv_names = []
    bn_modules = []
    for i, m in enumerate(features):
        if isinstance(m, torch.nn.modules.conv._ConvNd):
            conv_indices.append(i)
            conv_modules.append(m)
            conv_names.append(module_name_dict[m])
            bn_modules.append(None)
        elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            if bn_modules[-1] is None:
                bn_modules[-1] = m # 其实还有种隐患，不过应该没有哪个模型一个conv后面跟两个bn吧hhh
    
    # 获得第idx个卷积层输入特征图
    def get_fn_conv_input_feature(idx):
        def fn(x):
            if idx == 0:
                return x
            else:
            #     return get_hidden_output_feature(features, conv_indices[idx]-1, x)
                for layer in range(conv_indices[idx]):
                    x = features[layer](x)
                return x
        return fn

    # 获得第idx+1个卷积层输入特征图
    def get_fn_next_input_feature(idx):
        def fn(x):
            if idx+1 < len(conv_indices):
                for layer in range(conv_indices[idx]+1, conv_indices[idx+1]):
                    x = features[layer](x)
            else: # 下层为fc
                for layer in range(conv_indices[-1]+1, len(features)):
                    x = features[layer](x)
                x = model.avgpool(x)
                x = torch.flatten(x, 1)
            return x
        return fn

    modules = []
    module_names = []
    fn_input_feature = []
    fn_next_input_feature = []

    for i in range(len(conv_indices)):
        # modules.append(conv_modules[i])
        # module_names.append(conv_names[i])
        fn_input_feature.append(get_fn_conv_input_feature(i))
        fn_next_input_feature.append(get_fn_next_input_feature(i))

    conv_modules.append(classifier) # 图省事直接append到conv_modules里面

    tuples = []
    for i in range(len(conv_names)):
        tuples.append((conv_names[i], conv_modules[i], bn_modules[i], conv_modules[i+1],
                                fn_input_feature[i], fn_next_input_feature[i]))
    # for i in range(len(conv_names)):
    #     tuples.append((conv_names[-2], conv_modules[-2], bn_modules[-2], conv_modules[-1],
    #                             fn_input_feature[-2], fn_next_input_feature[-2]))

    return tuples

def channel_select(sparsity, output_feature, fn_next_input_feature, next_module, method='greedy', p=2):
    """
    output_feature中选一些不重要的channel，使得fn_next_input_feature(output_feature_try)的lp norm最小
    next(_conv)_output_feature到next2_input_feature之间算是一种恒定的变换，
    因此这里不比较i+2层卷积层的输入，转而比较i+1层卷积层的输出
    """
    original_num = output_feature.size(1)
    pruned_num = int(math.floor(original_num * sparsity)) # 向下取整

    if method == 'greedy': # ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression
        indices_pruned = []
        while len(indices_pruned) < pruned_num:
            min_diff = 1e10
            min_idx = 0
            for idx in range(original_num):
                if idx in indices_pruned:
                    continue
                indices_try = indices_pruned + [idx]
                output_feature_try = torch.zeros_like(output_feature)
                output_feature_try[:, indices_try, ...] = output_feature[:, indices_try, ...]
                next_output_feature_try = next_module(fn_next_input_feature(output_feature_try))
                next_output_feature_try_norm = next_output_feature_try.norm(p)
                if next_output_feature_try_norm < min_diff:
                    min_diff = next_output_feature_try_norm
                    min_idx = idx
            indices_pruned.append(min_idx)
    elif method == 'lasso': # Channel Pruning for Accelerating Very Deep Neural Networks
        # FIXME 无法收敛。。。待解决
        next_output_feature = next_module(fn_next_input_feature(output_feature))
        num_el = next_output_feature.numel()
        next_output_feature = next_output_feature.data.view(num_el).cpu()
        next_output_feature_divided = []
        for idx in range(original_num): # 每个channel单独拿出来，其他通道为0
            output_feature_try = torch.zeros_like(output_feature)
            output_feature_try[:, idx, ...] = output_feature[:, idx, ...]
            next_output_feature_try = next_module(fn_next_input_feature(output_feature_try))
            next_output_feature_divided.append(next_output_feature_try.data.view(num_el, 1))
        next_output_feature_divided = torch.cat(next_output_feature_divided, dim=1).cpu()



        # import matplotlib.pyplot as plt  # 可视化绘制
        # X = next_output_feature_divided[:, 1:2]
        # y = next_output_feature
        # model = Lasso(alpha=0.000001, warm_start=True, selection='random', tol=4000)
        # model.fit(X, y)
        # predicted = model.predict(X)
        # # 绘制散点图 参数：x横轴 y纵轴
        # plt.scatter(X, y, marker='x')
        # plt.plot(X, predicted, c='r')
        # # 绘制x轴和y轴坐标
        # plt.xlabel("next_output_feature_divided[:, 0:1]")
        # plt.ylabel("next_output_feature")
        # # 显示图形
        # plt.savefig('Lasso1.png')



        # first, try to find a alpha that provides enough pruned channels
        alpha_try = 5e-5
        pruned_num_try = 0
        solver = Lasso(alpha=alpha_try, warm_start=True, selection='random')
        while pruned_num_try < pruned_num:
            alpha_try *= 2
            solver.alpha = alpha_try
            solver.fit(next_output_feature_divided, next_output_feature)
            pruned_num_try = sum(solver.coef_ == 0)
            print("lasso_alpha = {}, pruned_num_try = {}".format(alpha_try, pruned_num_try))

        # then, narrow down alpha to get more close to the desired number of pruned channels
        alpha_min = 0
        alpha_max = alpha_try
        pruned_num_tolerate_coeff = 1.1 # 死区
        pruned_num_max = int(pruned_num * pruned_num_tolerate_coeff)
        while True:
            alpha = (alpha_min + alpha_max) / 2
            solver.alpha = alpha
            solver.fit(next_output_feature_divided, next_output_feature)
            pruned_num_try = sum(solver.coef_ == 0)
            if pruned_num_try > pruned_num_max:
                alpha_max = alpha
            elif pruned_num_try < pruned_num:
                alpha_min = alpha
            else:
                print("lasso_alpha = {}".format(alpha))
                break

        # finally, convert lasso coeff to indices
        indices_pruned = solver.coef_.nonzero()[0].tolist()
        
    elif method == 'random':
        indices_pruned = random.sample(range(original_num), pruned_num)
    else:
        raise NotImplementedError

    return indices_pruned

def module_surgery(module, next_bn, next_module, indices_pruned, device):
    """根据indices_pruned实现filter的删除与权重的recover"""
    # operate module
    if isinstance(module, torch.nn.modules.conv._ConvNd):
        indices_stayed = list(set(range(module.out_channels)) - set(indices_pruned))
        num_channels_stayed = len(indices_stayed)
        module.out_channels = num_channels_stayed
    else:
        raise NotImplementedError
    # operate module weight
    new_weight = module.weight[indices_stayed, ...].clone()
    del module.weight
    module.weight = torch.nn.Parameter(new_weight)
    # operate module bias
    if module.bias is not None:
        new_bias = module.bias[indices_stayed, ...].clone()
        del module.bias
        module.bias = torch.nn.Parameter(new_bias)
    

    if next_bn is not None:
        # operate batch_norm
        if isinstance(next_bn, torch.nn.modules.batchnorm._BatchNorm):
            next_bn.num_features = num_channels_stayed
        else:
            raise NotImplementedError
        # operate batch_norm weight
        new_weight = next_bn.weight.data[indices_stayed].clone()
        new_bias = next_bn.bias.data[indices_stayed].clone()
        new_running_mean = next_bn.running_mean[indices_stayed].clone()
        new_running_var = next_bn.running_var[indices_stayed].clone()
        del next_bn.weight, next_bn.bias, next_bn.running_mean, next_bn.running_var
        next_bn.weight = torch.nn.Parameter(new_weight)
        next_bn.bias = torch.nn.Parameter(new_bias)
        next_bn.running_mean = new_running_mean
        next_bn.running_var = new_running_var


    # operate next_module
    if isinstance(next_module, torch.nn.modules.conv._ConvNd):
        next_module.in_channels = num_channels_stayed
    elif isinstance(next_module, torch.nn.modules.linear.Linear):
        next_module.in_features = num_channels_stayed
    else:
        raise NotImplementedError
    # operate next_module weight
    new_weight = next_module.weight[:, indices_stayed, ...].clone()
    del next_module.weight
    next_module.weight = torch.nn.Parameter(new_weight)

def weight_reconstruction(next_module, next_input_feature, next_output_feature, device=None):
    """
    通过最小二乘寻找最合适next_module的权重，
    使得去掉一些通道的next_input_feature经过next_module运算与原输出差距最小，
    next_output_feature为原输出，即目标输出
    """
    if next_module.bias is not None: # 还原bias影响
        bias_size = [1] * next_output_feature.dim()
        bias_size[1] = -1
        next_output_feature -= next_module.bias.view(bias_size)
    if isinstance(next_module, torch.nn.modules.conv._ConvNd):
        unfold = torch.nn.Unfold(kernel_size=next_module.kernel_size,
                                dilation=next_module.dilation,
                                padding=next_module.padding,
                                stride=next_module.stride)
        unfold = unfold.to(device)
        unfold.eval()
        # 卷积层输入
        next_input_feature = unfold(next_input_feature) # [B, C*kh*kw, L]
        next_input_feature = next_input_feature.transpose(1, 2) # 维度1,2颠倒一下才好理解，next_input_feature[B, L, :]表示从输入特种图中提取一个卷积核体积的子特征图(shape=C*kh*kw)，L就是每个输入特征图里这种子特征图有多少个(卷积核能滑多少次)
        num_fields = next_input_feature.size(0) * next_input_feature.size(1) # B * L
        next_input_feature = next_input_feature.reshape(num_fields, -1)
        # 目标输出
        next_output_feature = next_output_feature.view(next_output_feature.size(0), next_output_feature.size(1), -1)
        next_output_feature = next_output_feature.transpose(1, 2).reshape(num_fields, -1)

    # 计算最小二乘的解 Returned tensor XX has shape (\max(m, n) \times k)(max(m,n)×k) . The first nn rows of XX contains the solution.
    param, _ = torch.lstsq(next_output_feature.data.cpu(), next_input_feature.data.cpu()) # The case when m < n is not supported on the GPU.
    param = param.to(device)[:next_input_feature.size(1), :].clone().t().reshape(next_output_feature.size(1), -1)

    if isinstance(next_module, torch.nn.modules.conv._ConvNd):
        param = param.view(next_module.out_channels, next_module.in_channels, *next_module.kernel_size)
    del next_module.weight
    next_module.weight = torch.nn.Parameter(param)

def Thinet_prune(model, sparsity, dataloader, device, method, p):

    start_time = datetime.datetime.now()

    input_iter = iter(dataloader)
    tuples = get_tuples(model)

    pruner_print_bar("got_tuples", None, start_time)

    for (module_name, module, next_bn, next_module, fn_input_feature, fn_next_input_feature) in tuples:
        # 此处module和next_module均为conv module
        input, _ = input_iter.__next__()
        input = input.to(device)
        input_feature = fn_input_feature(input)
        input_feature = input_feature.to(device)

        output_feature = module(input_feature) # 之后我们要对这玩意下刀，去掉几个channel
        next_input_feature = fn_next_input_feature(output_feature)
        next_output_feature = next_module(next_input_feature)

        # sparsity = get_param_sparsity(module_name)
        indices_pruned = channel_select(sparsity, output_feature, fn_next_input_feature, next_module, method)
        module_surgery(module, next_bn, next_module, indices_pruned, device)

        # 通道剪枝后更新特征图
        output_feature = module(input_feature)
        next_input_feature = fn_next_input_feature(output_feature)

        weight_reconstruction(next_module, next_input_feature, next_output_feature, device)

        pruner_print_bar(module_name, module.out_channels, start_time)

    return model, 0, sparsity


class ChannelPruner(object):
    def __init__(self, model, prune_percent, dataloader, device, method, p):
        self.model = model
        self.prune_percent = prune_percent
        self.dataloader = dataloader
        self.device = device
        self.method = method
        self.p = p

    def prune(self):
        return Thinet_prune(self.model, self.prune_percent, self.dataloader, self.device, self.method, self.p)

        

# # 提取隐藏层features
# class FeatureExtractor:
#     features = None

#     def __init__(self, model, layer_num):
#         self.hook = model[layer_num].register_forward_hook(self.hook_fn)

#     def hook_fn(self, module, input, output):
#         self.features = output.cpu()

#     def remove(self):
#         self.hook.remove()

# def get_hidden_output_feature(model, idx, x):
#     """return model第idx层的前向传播输出特征图"""
#     feature_extractor = FeatureExtractor(model, idx) # 注册钩子
#     out = model(x)
#     feature_extractor.remove() # 销毁钩子
#     return feature_extractor.features # 第idx层输出的特征