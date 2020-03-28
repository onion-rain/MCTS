import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# pruningnet means use fc to generate conv weights, prunednet means construct model with gene
__all__ = ['MobileNetV2_Pruningnet', 'mobilenetv2_pruningnet',
           'MobileNetV2_Prunednet',  'mobilenetv2_prunednet',]


def parse_gene(gene, stage_repeat):
    """根据gene生成output_scale_ids与mid_scale_ids"""
    if gene is None:
        # gene = [-1, ]*(len(stage_repeat)+1 + sum(stage_repeat))
        output_scale_ids = [-1,]*sum(stage_repeat)
        mid_scale_ids = [-1,]*(sum(stage_repeat)-2)
    else:
        mid_scale_ids = gene[len(stage_repeat):len(stage_repeat)+sum(stage_repeat)-2]
        output_scale_ids = []
        for i in range(len(stage_repeat)):
            output_scale_ids += [gene[i]]*stage_repeat[i]
    return output_scale_ids, mid_scale_ids

# ------------------------------ Pruningnet ------------------------------

class first_conv_Pruningnet(nn.Module):
    def __init__(self, in_channels, out_channels, channel_scales, stride=2):
        super(first_conv_Pruningnet, self).__init__()

        self.channel_scales = channel_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.fc11 = nn.Linear(1, 32)
        self.fc12 = nn.Linear(32, self.out_channels * self.in_channels * 3 * 3)
        # self.conv1 = conv1x1(in_channels, out_channels, stride=stride)
        # self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm1 = nn.ModuleList()
        for scale in channel_scales: # 所有可能的通道数都造一个bn层
            self.norm1.append(nn.BatchNorm2d(int(self.out_channels*scale), affine=False))

    def forward(self, x, scale_id):
        
        # self.in_channels为原始通道数，in_channels为pruned通道数
        in_channels = self.in_channels
        out_channels = int(self.out_channels * self.channel_scales[scale_id])
        block_cfg = torch.FloatTensor([self.channel_scales[scale_id]]).to(x.device)

        # conv1
        conv1_param = F.relu(self.fc11(block_cfg), inplace=True)
        conv1_param = self.fc12(conv1_param).view(self.out_channels, self.in_channels, 3, 3)
        out = F.conv2d(x, conv1_param[:out_channels, :in_channels, :, :], 
                            bias=None, stride=self.stride, padding=1, groups=1)
        out = F.relu(self.norm1[scale_id](out), inplace=True)

        return out

class last_conv_Pruningnet(nn.Module):
    def __init__(self, in_channels, out_channels, channel_scales, stride=1):
        super(last_conv_Pruningnet, self).__init__()

        self.channel_scales = channel_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.fc11 = nn.Linear(2, 32)
        self.fc12 = nn.Linear(32, self.out_channels * self.in_channels * 1 * 1)
        # self.conv1 = conv1x1(in_channels, out_channels, stride=stride)
        # self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm1 = nn.ModuleList()
        for scale in channel_scales: # 所有可能的通道数都造一个bn层
            self.norm1.append(nn.BatchNorm2d(int(self.out_channels*scale), affine=False))

    def forward(self, x, scale_ids):
        
        # self.in_channels为原始通道数，in_channels为pruned通道数
        in_channels  = int(self.in_channels  * self.channel_scales[scale_ids[0]])
        out_channels = int(self.out_channels * self.channel_scales[scale_ids[1]])
        block_cfg = torch.FloatTensor([
            self.channel_scales[scale_ids[0]],
            self.channel_scales[scale_ids[1]],
        ]).to(x.device)

        # conv1
        conv1_param = F.relu(self.fc11(block_cfg), inplace=True)
        conv1_param = self.fc12(conv1_param).view(self.out_channels, self.in_channels, 1, 1)
        out = F.conv2d(x, conv1_param[:out_channels, :in_channels, :, :], 
                            bias=None, stride=self.stride, padding=1, groups=1)
        out = F.relu(self.norm1[scale_ids[1]](out), inplace=True)

        return out

class Bottleneck_Pruningnet(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, channel_scales, stride=1):
        super(Bottleneck_Pruningnet, self).__init__()
        self.residual_connect = False

        self.channel_scales = channel_scales
        self.in_channels  = in_channels
        self.mid_channels = mid_channels
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

        if stride==1 and in_channels==out_channels:
            self.residual_connect = True

    def forward(self, x, scale_ids):

        in_channels  = int(self.in_channels  * self.channel_scales[scale_ids[0]])
        mid_channels = int(self.mid_channels * self.channel_scales[scale_ids[1]])
        out_channels = int(self.out_channels * self.channel_scales[scale_ids[2]])
        block_cfg = torch.FloatTensor([
            self.channel_scales[scale_ids[0]],
            self.channel_scales[scale_ids[1]],
            self.channel_scales[scale_ids[2]],
        ]).to(x.device)

        # conv1
        conv1_param = F.relu(self.fc11(block_cfg), inplace=True)
        conv1_param = self.fc12(conv1_param).view(self.mid_channels, self.in_channels, 1, 1)
        residual = F.conv2d(x, conv1_param[:mid_channels, :in_channels, :, :], 
                            bias=None, stride=1, padding=0, groups=1)
        residual = F.relu(self.norm1[scale_ids[1]](residual), inplace=True)

        # conv2
        conv2_param = F.relu(self.fc21(block_cfg), inplace=True)
        conv2_param = self.fc22(conv2_param).view(self.mid_channels, self.mid_channels, 3, 3)
        residual = F.conv2d(residual, conv2_param[:mid_channels, :mid_channels, :, :], 
                            bias=None, stride=self.stride, padding=1, groups=1)
        residual = F.relu(self.norm2[scale_ids[1]](residual), inplace=True)

        # conv3
        conv3_param = F.relu(self.fc31(block_cfg), inplace=True)
        conv3_param = self.fc32(conv3_param).view(self.out_channels, self.mid_channels, 1, 1)
        residual = F.conv2d(residual, conv3_param[:out_channels, :mid_channels, :, :], 
                            bias=None, stride=1, padding=0, groups=1)
        residual = self.norm3[scale_ids[2]](residual)


        if self.residual_connect == True:
            return residual + x
        else: return residual

class MobileNetV2_Pruningnet(nn.Module):

    def __init__(self, num_classes=1000):
        """
        args:
            stage_repeat(list)：每个stage重复的block数
            num_classes(int)：分类数
        """
        super(MobileNetV2_Pruningnet, self).__init__()

        self.num_classes = num_classes
        first_conv = first_conv_Pruningnet
        Bottleneck = Bottleneck_Pruningnet
        last_conv  = last_conv_Pruningnet

        self.channel_scales = [] # 压缩率摇奖池
        for i in range(31):
            self.channel_scales.append((10 + i * 3)/100)

        network_structure = [ # 原始网络结构
            # t,  c,  n, s
            [0,   32, 1, 2], # conv3x3
            [1,   16, 1, 1], # bottleneck
            [6,   24, 2, 2], # bottleneck
            [6,   32, 3, 2], # bottleneck
            [6,   64, 4, 2], # bottleneck
            [6,   96, 3, 1], # bottleneck
            [6,  160, 3, 2], # bottleneck
            [6,  320, 1, 1], # bottleneck
            [0, 1280, 1, 1], # conv1x1
        ]
        stage_expansion = []
        stage_channels  = []
        stage_repeat    = []
        stage_strides   = []
        for layer in range(len(network_structure)):
            stage_expansion.append(network_structure[layer][0])
            stage_channels.append(network_structure[layer][1])
            stage_repeat.append(network_structure[layer][2])
            stage_strides.append(network_structure[layer][3])
        self.stage_repeat = stage_repeat
        self.gene_length = len(stage_repeat) + sum(stage_repeat)-2
        self.oc_gene_length = len(stage_repeat)

        gene = [-1, ]*(len(stage_repeat) + sum(stage_repeat)-2)
        output_scale_ids, mid_scale_ids = parse_gene(gene, stage_repeat)
        output_channels_o = [stage_channels[0]]
        for i in range(1, len(stage_channels)):
            output_channels_o += [stage_channels[i],]*stage_repeat[i]
        mid_channels_o = []
        for i in range(1, len(stage_channels)-1):
            mid_channels_o += [int(stage_channels[i]*stage_expansion[i]),]*stage_repeat[i]
        output_channels = np.asarray([output_channels_o[i]*self.channel_scales[output_scale_ids[i]] for i in range(len(output_scale_ids))], dtype=int).tolist()
        mid_channels    = np.asarray([mid_channels_o[i]   *self.channel_scales[mid_scale_ids[i]   ] for i in range(len(  mid_scale_ids ))], dtype=int).tolist()

        self.features = nn.ModuleList()
        # first conv(stage0)
        self.features.append(first_conv(3, output_channels[0], self.channel_scales, stride=stage_strides[0]))
        # bottleneck
        block_num = 1
        for stage in range(1, len(stage_repeat)-1):
            self.features.append(Bottleneck(output_channels[block_num-1], mid_channels[block_num-1], output_channels[block_num], self.channel_scales, stride=stage_strides[stage]))
            block_num += 1
            for i in range(1, stage_repeat[stage]):
                self.features.append(Bottleneck(output_channels[block_num-1], mid_channels[block_num-1], output_channels[block_num], self.channel_scales, stride=1))
                block_num +=1
        # last conv(stage-1)
        self.features.append(last_conv(output_channels[-2], output_channels[-1], self.channel_scales, stride=stage_strides[0]))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 输出尺寸为1*1
        
        self.classifier = nn.Linear(stage_channels[-1], num_classes)


    def forward(self, x, gene=None):

        output_scale_ids, mid_scale_ids = parse_gene(gene, self.stage_repeat)
        for idx, block in enumerate(self.features):
            if idx==0:
                x = block(x, output_scale_ids[idx])
            elif idx==len(self.features)-1:
                x = block(x, [output_scale_ids[idx-1], 
                              output_scale_ids[idx]])
            else:
                x = block(x, [output_scale_ids[idx-1], 
                              mid_scale_ids[idx-1], 
                              output_scale_ids[idx]])
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def mobilenetv2_pruningnet(num_classes=1000):
    return MobileNetV2_Pruningnet(num_classes)










# ------------------------------ Prunednet ------------------------------

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

    def __init__(self, in_channels, out_channels, stride=2):
        super(first_conv_Prunednet, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.norm1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.norm1(out), inplace=True)
        return out

class last_conv_Prunednet(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(last_conv_Prunednet, self).__init__()

        self.conv1 = conv1x1(in_channels, out_channels, stride=stride)
        self.norm1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.norm1(out), inplace=True)
        return out

class Bottleneck_Prunednet(nn.Module):
    
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(Bottleneck_Prunednet, self).__init__()
        self.residual_connect = False

        self.conv1 = conv1x1(in_channels, mid_channels, stride=1)
        self.norm1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = conv3x3(mid_channels, mid_channels, stride=stride)
        self.norm2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = conv1x1(mid_channels, out_channels, stride=1)
        self.norm3 = nn.BatchNorm2d(out_channels)

        if stride==1 and in_channels==out_channels:
            self.residual_connect = True

    def forward(self, x):
        # conv1
        residual = self.conv1(x)
        residual = F.relu(self.norm1(residual), inplace=True)

        # conv2
        residual = self.conv2(residual)
        residual = F.relu(self.norm2(residual), inplace=True)

        # conv3
        residual = self.conv3(residual)
        residual = self.norm3(residual)

        if self.residual_connect == True:
            return residual + x
        else: return residual

class MobileNetV2_Prunednet(nn.Module):

    def __init__(self, num_classes=1000, gene=None):
        """
        args:
            num_classes(int)：分类数
            gene(list): 存储每层卷积的输入输出通道，用来构造剪之后的网络，前部表示output_scale_ids，后部表示mid_scale_ids
                        若gene为None则构造原始resnet
        """
        super(MobileNetV2_Prunednet, self).__init__()

        self.num_classes = num_classes
        self.gene = gene
        first_conv = first_conv_Prunednet
        Bottleneck = Bottleneck_Prunednet
        last_conv  = last_conv_Prunednet

        self.channel_scales = [] # 压缩率摇奖池
        for i in range(31):
            self.channel_scales.append((10 + i * 3)/100)

        network_structure = [ # 原始网络结构
            # t,  c,  n, s
            [0,   32, 1, 2], # conv3x3
            [1,   16, 1, 1], # bottleneck
            [6,   24, 2, 2], # bottleneck
            [6,   32, 3, 2], # bottleneck
            [6,   64, 4, 2], # bottleneck
            [6,   96, 3, 1], # bottleneck
            [6,  160, 3, 2], # bottleneck
            [6,  320, 1, 1], # bottleneck
            [0, 1280, 1, 1], # conv1x1
        ]
        stage_expansion = []
        stage_channels  = []
        stage_repeat    = []
        stage_strides   = []
        for layer in range(len(network_structure)):
            stage_expansion.append(network_structure[layer][0])
            stage_channels.append(network_structure[layer][1])
            stage_repeat.append(network_structure[layer][2])
            stage_strides.append(network_structure[layer][3])
        self.stage_repeat = stage_repeat
        self.gene_length = len(stage_repeat) + sum(stage_repeat)-2
        self.oc_gene_length = len(stage_repeat)

        if gene is None:
            gene = [-1, ]*(len(stage_repeat) + sum(stage_repeat)-2)
        output_scale_ids, mid_scale_ids = parse_gene(gene, stage_repeat)
        output_channels_o = [stage_channels[0]]
        for i in range(1, len(stage_channels)):
            output_channels_o += [stage_channels[i],]*stage_repeat[i]
        mid_channels_o = []
        for i in range(1, len(stage_channels)-1):
            mid_channels_o += [int(stage_channels[i]*stage_expansion[i]),]*stage_repeat[i]
        output_channels = np.asarray([output_channels_o[i]*self.channel_scales[output_scale_ids[i]] for i in range(len(output_scale_ids))], dtype=int).tolist()
        mid_channels    = np.asarray([mid_channels_o[i]   *self.channel_scales[mid_scale_ids[i]   ] for i in range(len(  mid_scale_ids ))], dtype=int).tolist()

        self.features = nn.ModuleList()
        # first conv(stage0)
        self.features.append(first_conv(3, output_channels[0], stride=stage_strides[0]))
        # bottleneck
        block_num = 1
        for stage in range(1, len(stage_repeat)-1):
            self.features.append(Bottleneck(output_channels[block_num-1], mid_channels[block_num-1], output_channels[block_num], stride=stage_strides[block_num]))
            block_num += 1
            for i in range(1, stage_repeat[stage]):
                self.features.append(Bottleneck(output_channels[block_num-1], mid_channels[block_num-1], output_channels[block_num], stride=1))
                block_num +=1
        # last conv(stage-1)
        self.features.append(last_conv(output_channels[-2], output_channels[-1], stride=stage_strides[0]))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 输出尺寸为1*1
        
        self.classifier = nn.Linear(stage_channels[-1], num_classes)

    def forward(self, x):

        for idx, block in enumerate(self.features):
            x = block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        

def mobilenetv2_prunednet(gene=None, num_classes=1000):
    return MobileNetV2_Prunednet(num_classes, gene=gene)
