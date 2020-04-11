import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# pruningnet means use fc to generate conv weights, prunednet means construct model with gene
__all__ = ['ResNet_Pruningnet', 'resnet50_pruningnet', 'resnet101_pruningnet', 'resnet152_pruningnet',
           'ResNet_Prunednet',  'resnet50_prunednet',  'resnet101_prunednet',  'resnet152_prunednet',]


def parse_gene(gene, stage_repeat):
    """根据gene生成output_scale_ids与mid_scale_ids"""
    if gene is None:
        # gene = [-1, ]*(len(stage_repeat)+1 + sum(stage_repeat))
        output_scale_ids = [-1,]*(sum(stage_repeat)+1)
        mid_scale_ids = [-1,]*sum(stage_repeat)
    else:
        mid_scale_ids = gene[len(stage_repeat)+1:len(stage_repeat)+1+sum(stage_repeat)]
        output_scale_ids = [gene[0]] # stage 0
        for i in range(len(stage_repeat)):
            output_scale_ids += [gene[i+1]]*stage_repeat[i]
        # output_scale_ids.append(-1) # features输出通道不变
    return output_scale_ids, mid_scale_ids

# ------------------------------ Pruningnet ------------------------------

class first_conv_Pruningnet(nn.Module):
    def __init__(self, in_channels, out_channels, channel_scales, stride=1):
        super(first_conv_Pruningnet, self).__init__()

        self.channel_scales = channel_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.fc11 = nn.Linear(1, 32)
        self.fc12 = nn.Linear(32, self.out_channels * self.in_channels * 7 * 7)
        # self.conv1 = conv1x1(self.in_channels, self.out_channels, stride=stride)
        # self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm1 = nn.ModuleList()
        for scale in channel_scales: # 所有可能的通道数都造一个bn层
            self.norm1.append(nn.BatchNorm2d(int(self.out_channels*scale), affine=False))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x, scale_id):
        
        # self.in_channels为原始通道数，in_channels为pruned通道数
        in_channels = self.in_channels
        out_channels = int(self.out_channels * self.channel_scales[scale_id])
        # block_cfg = torch.FloatTensor([out_channels]).to(x.device)
        block_cfg = torch.FloatTensor([self.channel_scales[scale_id]]).to(x.device)

        # conv1
        conv1_param = F.relu(self.fc11(block_cfg), inplace=True)
        conv1_param = self.fc12(conv1_param).view(self.out_channels, self.in_channels, 7, 7)
        out = F.conv2d(x, conv1_param[:out_channels, :in_channels, :, :], 
                            bias=None, stride=self.stride, padding=1, groups=1)
        out = self.norm1[scale_id](F.relu(out, inplace=True))
        
        out = self.maxpool(out)

        return out

class Bottleneck_Pruningnet(nn.Module):
    def __init__(self, in_channels, out_channels, channel_scales, stride=1):
        super(Bottleneck_Pruningnet, self).__init__()
        expansion = 4

        self.channel_scales = channel_scales
        self.in_channels = in_channels
        self.mid_channels = int(out_channels/expansion)
        self.out_channels = out_channels
        self.stride = stride

        self.fc11 = nn.Linear(3, 32)
        self.fc12 = nn.Linear(32, self.mid_channels * self.in_channels * 1 * 1)
        # self.conv1 = conv1x1(self.in_channels, self.mid_channels, stride=1)
        # self.norm1 = nn.BatchNorm2d(self.mid_channels)
        self.norm1 = nn.ModuleList()
        for scale in channel_scales: # 所有可能的通道数都造一个bn层
            self.norm1.append(nn.BatchNorm2d(int(self.mid_channels*scale), affine=False))

        self.fc21 = nn.Linear(3, 32)
        self.fc22 = nn.Linear(32, self.mid_channels * self.mid_channels * 3 * 3)
        # self.conv2 = conv3x3(self.mid_channels, self.mid_channels, stride=stride)
        # self.norm2 = nn.BatchNorm2d(self.mid_channels)
        self.norm2 = nn.ModuleList()
        for scale in channel_scales: # 所有可能的通道数都造一个bn层
            self.norm2.append(nn.BatchNorm2d(int(self.mid_channels*scale), affine=False))

        self.fc31 = nn.Linear(3, 32)
        self.fc32 = nn.Linear(32, self.out_channels * self.mid_channels * 1 * 1)
        # self.conv3 = conv1x1(self.mid_channels, self.out_channels, stride=1)
        # self.norm3 = nn.BatchNorm2d(self.out_channels)
        self.norm3 = nn.ModuleList()
        for scale in channel_scales: # 所有可能的通道数都造一个bn层
            self.norm3.append(nn.BatchNorm2d(int(self.out_channels*scale), affine=False))

        if stride != 1 or self.in_channels != self.out_channels:
            self.fc_shortcut1 = nn.Linear(3, 32)
            self.fc_shortcut2 = nn.Linear(32, self.out_channels * self.in_channels * 1 * 1)
            # self.shortcut = conv1x1(in_channels, out_channels, stride=stride)
            # self.norm_shortcut = nn.BatchNorm2d(self.out_channels)
            self.norm_shortcut = nn.ModuleList()
            for scale in channel_scales: # 所有可能的通道数都造一个bn层
                self.norm_shortcut.append(nn.BatchNorm2d(int(self.out_channels*scale), affine=False))

    def forward(self, x, scale_ids):

        in_channels  = int(self.in_channels  * self.channel_scales[scale_ids[0]])
        mid_channels = int(self.mid_channels * self.channel_scales[scale_ids[1]])
        out_channels = int(self.out_channels * self.channel_scales[scale_ids[2]])
        block_cfg = torch.FloatTensor([
            self.channel_scales[scale_ids[0]],
            self.channel_scales[scale_ids[1]],
            self.channel_scales[scale_ids[2]],
        ]).to(x.device)

        # shortcut
        shortcut = x
        if self.stride != 1 or self.in_channels != self.out_channels:
            conv_shortcut_param = F.relu(self.fc_shortcut1(block_cfg), inplace=True)
            conv_shortcut_param = self.fc_shortcut2(conv_shortcut_param).view(self.out_channels, self.in_channels, 1, 1)
            shortcut = F.conv2d(shortcut, conv_shortcut_param[:out_channels, :in_channels, :, :], 
                                bias=None, stride=self.stride, padding=0, groups=1)
            shortcut = self.norm_shortcut[scale_ids[2]](shortcut)

        # conv1
        conv1_param = F.relu(self.fc11(block_cfg), inplace=True)
        conv1_param = self.fc12(conv1_param).view(self.mid_channels, self.in_channels, 1, 1)
        residual = F.conv2d(x, conv1_param[:mid_channels, :in_channels, :, :], 
                            bias=None, stride=1, padding=0, groups=1)
        residual = self.norm1[scale_ids[1]](F.relu(residual, inplace=True))

        # conv2
        conv2_param = F.relu(self.fc21(block_cfg), inplace=True)
        conv2_param = self.fc22(conv2_param).view(self.mid_channels, self.mid_channels, 3, 3)
        residual = F.conv2d(residual, conv2_param[:mid_channels, :mid_channels, :, :], 
                            bias=None, stride=self.stride, padding=1, groups=1)
        residual = self.norm2[scale_ids[1]](F.relu(residual, inplace=True))

        # conv3
        conv3_param = F.relu(self.fc31(block_cfg), inplace=True)
        conv3_param = self.fc32(conv3_param).view(self.out_channels, self.mid_channels, 1, 1)
        residual = F.conv2d(residual, conv3_param[:out_channels, :mid_channels, :, :], 
                            bias=None, stride=1, padding=0, groups=1)
        residual = self.norm3[scale_ids[2]](residual)

        out = residual + shortcut
        out = F.relu(out, inplace=True)
        return out

class ResNet_Pruningnet(nn.Module):

    def __init__(self, stage_repeat=[3, 4, 6, 3], num_classes=1000):
        """
        args:
            stage_repeat(list)：每个stage重复的block数
            num_classes(int)：分类数
        """
        super(ResNet_Pruningnet, self).__init__()

        self.num_classes = num_classes
        self.stage_repeat = stage_repeat
        self.gene_length = len(stage_repeat)+1 + sum(stage_repeat)
        self.oc_gene_length = len(stage_repeat)+1

        self.channel_scales = [] # 压缩率摇奖池
        for i in range(31):
            self.channel_scales.append((10 + i * 3)/100)
        # stage_channels = [64, 128, 256, 512, 2048]
        stage_channels = [64, 256, 512, 1024, 2048] # 原始每层stage的输出通道数，与metaprune作者相同
        first_conv = first_conv_Pruningnet
        Bottleneck = Bottleneck_Pruningnet

        self.features = nn.ModuleList()

        # first conv(stage0)
        self.features.append(first_conv(3, stage_channels[0], self.channel_scales, stride=2))

        # stage1
        self.features.append(Bottleneck(stage_channels[0], stage_channels[1], self.channel_scales, stride=1))
        for i in range(1, stage_repeat[0]):
            self.features.append(Bottleneck(stage_channels[1], stage_channels[1], self.channel_scales, stride=1))

        #stage2
        self.features.append(Bottleneck(stage_channels[1], stage_channels[2], self.channel_scales, stride=2))
        for i in range(1, stage_repeat[1]):
            self.features.append(Bottleneck(stage_channels[2], stage_channels[2], self.channel_scales, stride=1))

        #stage3
        self.features.append(Bottleneck(stage_channels[2], stage_channels[3], self.channel_scales, stride=2))
        for i in range(1, stage_repeat[2]):
            self.features.append(Bottleneck(stage_channels[3], stage_channels[3], self.channel_scales, stride=1))

        #stage4
        self.features.append(Bottleneck(stage_channels[3], stage_channels[4], self.channel_scales, stride=2))
        for i in range(1, stage_repeat[3]):
            self.features.append(Bottleneck(stage_channels[4], stage_channels[4], self.channel_scales, stride=1))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 输出尺寸为1*1
        self.classifier = nn.Linear(stage_channels[4], num_classes)

        # for m in self.modules():
        #     if isinstance(m, (nn.Linear, nn.Conv2d)):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x, gene=None):

        output_scale_ids, mid_scale_ids = parse_gene(gene, self.stage_repeat)
        for idx, block in enumerate(self.features):
            if idx == 0:
                x = block(x, output_scale_ids[idx])
            else:
                x = block(x, [output_scale_ids[idx-1], 
                            mid_scale_ids[idx-1], 
                            output_scale_ids[idx]])
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 这些应该是basicblock。。懒得写了
# def resnet18_pruningnet(cfg=None, num_classes=1000):
#     return ResNet_Pruningnet([2, 2, 2, 2], num_classes)


# def resnet34_pruningnet(cfg=None, num_classes=1000):
#     return ResNet_Pruningnet([3, 4, 6, 3], num_classes)


def resnet50_pruningnet(num_classes=1000):
    return ResNet_Pruningnet([3, 4, 6, 3], num_classes)


def resnet101_pruningnet(num_classes=1000):
    return ResNet_Pruningnet([3, 4, 23, 3], num_classes)


def resnet152_pruningnet(num_classes=1000):
    return ResNet_Pruningnet([3, 8, 36, 3], num_classes)











# ------------------------------ Prunednet ------------------------------
# FIXME conv7x7 padding应该为3
def conv7x7(in_channels, out_channels, stride=1):
    """7x7 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, bias=False)

def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class first_conv_Prunednet(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(first_conv_Prunednet, self).__init__()

        self.conv1 = conv7x7(in_channels, out_channels, stride=stride)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(F.relu(out, inplace=True))
        out = self.maxpool(out)
        return out

class Bottleneck_Prunednet(nn.Module):
    
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(Bottleneck_Prunednet, self).__init__()

        self.conv1 = conv1x1(in_channels, mid_channels, stride=1)
        self.norm1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = conv3x3(mid_channels, mid_channels, stride=stride)
        self.norm2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = conv1x1(mid_channels, out_channels, stride=1)
        self.norm3 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = conv1x1(in_channels, out_channels, stride=stride)
            self.norm_shortcut = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # shortcut
        shortcut = x
        if hasattr(self, 'shortcut'):
            shortcut = self.shortcut(shortcut)
            shortcut = self.norm_shortcut(shortcut)

        # conv1
        residual = self.conv1(x)
        residual = self.norm1(F.relu(residual, inplace=True))

        # conv2
        residual = self.conv2(residual)
        residual = self.norm2(F.relu(residual, inplace=True))

        # conv3
        residual = self.conv3(residual)
        residual = self.norm3(residual)

        out = residual + shortcut
        out = F.relu(out, inplace=True)
        return out

class ResNet_Prunednet(nn.Module):

    def __init__(self, stage_repeat=[3, 4, 6, 3], num_classes=1000, gene=None):
        """
        args:
            stage_repeat(list)：每个stage重复的block数
            num_classes(int)：分类数
            gene(list): 存储每层卷积的输入输出通道，用来构造剪之后的网络，前部表示output_scale_ids，后部表示mid_scale_ids
                        若gene为None则构造原始resnet
        """
        super(ResNet_Prunednet, self).__init__()

        self.num_classes = num_classes
        self.stage_repeat = stage_repeat
        self.gene_length = len(stage_repeat)+1 + sum(stage_repeat)
        self.oc_gene_length = len(stage_repeat)+1
        self.gene = gene

        self.channel_scales = [] # 压缩率摇奖池
        for i in range(31):
            self.channel_scales.append((10 + i * 3)/100)
            
        # stage_channels = [64, 128, 256, 512, 2048]
        stage_channels = [64, 256, 512, 1024, 2048] # 原始每层stage的输出通道数，与作者相同
        first_conv = first_conv_Prunednet
        Bottleneck = Bottleneck_Prunednet
        if gene is None:
            gene = [-1, ]*(len(stage_repeat)+1 + sum(stage_repeat))
        output_scale_ids, mid_scale_ids = parse_gene(gene, self.stage_repeat)
        output_channels_o = [stage_channels[0]]
        for i in range(1, len(stage_channels)):
            output_channels_o += [stage_channels[i],]*stage_repeat[i-1]
        mid_channels_o = []
        for i in range(1, len(stage_channels)):
            mid_channels_o += [int(stage_channels[i]/4),]*stage_repeat[i-1]
        output_channels = np.asarray([output_channels_o[i]*self.channel_scales[output_scale_ids[i]] for i in range(len(output_scale_ids))], dtype=int).tolist()
        mid_channels    = np.asarray([mid_channels_o[i]   *self.channel_scales[mid_scale_ids[i]   ] for i in range(len(  mid_scale_ids ))], dtype=int).tolist()

        self.features = nn.ModuleList()

        # first conv(stage0)
        self.features.append(first_conv(3, output_channels[0], stride=2))

        block_num = 1
        for stage in range(len(stage_repeat)):
            if stage == 0:
                self.features.append(Bottleneck(output_channels[block_num-1], mid_channels[block_num-1], output_channels[block_num], stride=1))
                block_num += 1
            else:
                self.features.append(Bottleneck(output_channels[block_num-1], mid_channels[block_num-1], output_channels[block_num], stride=2))
                block_num += 1

            for i in range(1, stage_repeat[stage]):
                self.features.append(Bottleneck(output_channels[block_num-1], mid_channels[block_num-1], output_channels[block_num], stride=1))
                block_num +=1

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 输出尺寸为1*1
        self.classifier = nn.Linear(stage_channels[4], num_classes)

        # for m in self.modules():
        #     if isinstance(m, (nn.Linear, nn.Conv2d)):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):

        for idx, block in enumerate(self.features):
            x = block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        
# 这些应该是basicblock。。懒得写了
# def resnet18_prunednet(cfg=None, num_classes=1000):
#     return ResNet_Prunednet([2, 2, 2, 2], num_classes, gene=cfg)


# def resnet34_prunednet(cfg=None, num_classes=1000):
#     return ResNet_Prunednet([3, 4, 6, 3], num_classes, gene=cfg)


def resnet50_prunednet(gene=None, num_classes=1000):
    return ResNet_Prunednet([3, 4, 6, 3], num_classes, gene=gene)


def resnet101_prunednet(gene=None, num_classes=1000):
    return ResNet_Prunednet([3, 4, 23, 3], num_classes, gene=gene)


def resnet152_prunednet(gene=None, num_classes=1000):
    return ResNet_Prunednet([3, 8, 36, 3], num_classes, gene=gene)

