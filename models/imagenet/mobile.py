import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math

import numpy as np

__all__ = ['mobile_pruningnet']

mid_channel_scale = []
for i in range(31):
    mid_channel_scale += [(10 + i * 3)/100]

overall_channel_scale = []
for i in range(31):
    overall_channel_scale += [(10 + i * 3)/100]

stage_out_channel = [44] + [22] + [33] * 2 + [44] * 3 + [88] * 4 + [132] * 3 + [224] * 3 + [448]

class conv2d_3x3(nn.Module):
    def __init__(self, base_inp, base_oup, stride):
        super(conv2d_3x3, self).__init__()

        self.stride = stride
        assert stride in [1, 2]

        self.max_overall_scale = overall_channel_scale[-1]
        self.base_inp = base_inp
        self.base_oup = base_oup

        self.max_oup_channel = int(self.max_overall_scale * self.base_oup)
        self.fc11 = nn.Linear(1, 32)
        self.fc12 = nn.Linear(32, self.max_oup_channel * self.base_inp * 3 * 3)

        self.first_bn = nn.ModuleList()
        for oup_scale in overall_channel_scale:
            oup = int(self.base_oup * oup_scale)
            self.first_bn.append(nn.BatchNorm2d(oup, affine=False))

    def forward(self, x, oup_scale_id):

        oup_scale = overall_channel_scale[oup_scale_id]
        oup = int(self.base_oup * oup_scale)
        scale_tensor = torch.FloatTensor([oup_scale/self.max_overall_scale]).to(x.device)

        fc11_out = F.relu(self.fc11(scale_tensor))
        conv1_weight = self.fc12(fc11_out).view(self.max_oup_channel, self.base_inp, 3, 3)

        out = F.conv2d(x, conv1_weight[:oup, :, :, :], bias=None, stride=self.stride, padding=1)
        out = self.first_bn[oup_scale_id](out)
        out = F.relu6(out)

        return out

class conv2d_1x1(nn.Module):
    def __init__(self, base_inp, base_oup, stride):
        super(conv2d_1x1, self).__init__()

        self.stride = stride
        assert stride in [1, 2]

        self.max_overall_scale = overall_channel_scale[-1]
        self.base_inp = base_inp
        self.base_oup = base_oup

        self.max_inp_channel = int(self.max_overall_scale * self.base_inp)
        self.fc11 = nn.Linear(1, 32)
        self.fc12 = nn.Linear(32, self.base_oup * self.max_inp_channel * 1 * 1)
        #self.conv1_weight = nn.Parameter(torch.randn(base_oup, self.max_inp_channel, 1, 1))

        self.first_bn = nn.ModuleList()
        for inp_scale in overall_channel_scale:
            inp = int(self.base_inp * inp_scale)
            self.first_bn.append(nn.BatchNorm2d(base_oup, affine=False))

    def forward(self, x, inp_scale_id):

        inp_scale = overall_channel_scale[inp_scale_id]

        inp = int(self.base_inp * inp_scale)

        scale_tensor = torch.FloatTensor([inp_scale/self.max_overall_scale]).to(x.device)

        fc11_out = F.relu(self.fc11(scale_tensor))
        conv1_weight = self.fc12(fc11_out).view(self.base_oup, self.max_inp_channel, 1, 1)

        out = F.conv2d(x, conv1_weight[:, :inp, :, :], bias=None, stride=self.stride, padding=0)
        out = self.first_bn[inp_scale_id](out)
        out = F.relu6(out)

        return out

class bottleneck(nn.Module):
    def __init__(self, base_inp, base_oup, stride, expand_ratio=6):
        super(bottleneck, self).__init__()

        self.max_overall_scale = overall_channel_scale[-1]

        max_inp = base_inp
        max_oup = base_oup
        max_mid = max_inp * expand_ratio

        self.max_inp = base_inp
        self.max_oup = base_oup
        self.max_mid = max_mid
        self.stride = stride

        self.fc11 = nn.Linear(3, 64)
        self.fc12 = nn.Linear(64, max_mid * max_inp * 1 * 1)

        self.fc21 = nn.Linear(3, 64)
        self.fc22 = nn.Linear(64, max_mid * 1 * 3 * 3)

        self.fc31 = nn.Linear(3, 64)
        self.fc32 = nn.Linear(64, max_oup * max_mid * 1 * 1)


        self.bn1 = nn.ModuleList()
        for mid_scale in mid_channel_scale:
            mid = int(self.max_mid * mid_scale)
            self.bn1.append(nn.BatchNorm2d(mid, affine=False))

        self.bn2 = nn.ModuleList()
        for mid_scale in mid_channel_scale:
            mid = int(self.max_mid * mid_scale)
            self.bn2.append(nn.BatchNorm2d(mid, affine=False))

        self.bn3 = nn.ModuleList()
        for oup_scale in overall_channel_scale:
            oup = int(max_oup * oup_scale)
            self.bn3.append(nn.BatchNorm2d(oup, affine=False))


    def forward(self, x, mid_scale_id, inp_scale_id, oup_scale_id):

        mid_scale = mid_channel_scale[mid_scale_id]
        inp_scale = overall_channel_scale[inp_scale_id]
        oup_scale = overall_channel_scale[oup_scale_id]

        mid = int(self.max_mid * mid_scale)
        inp = int(self.max_inp * inp_scale)
        oup = int(self.max_oup * oup_scale)

        scale_ratio_tensor = torch.FloatTensor([mid_scale, inp_scale, oup_scale]).to(x.device)

        fc11_out = F.relu(self.fc11(scale_ratio_tensor))
        conv1_weight = self.fc12(fc11_out).view(self.max_mid, self.max_inp, 1, 1)

        fc21_out = F.relu(self.fc21(scale_ratio_tensor))
        conv2_weight = self.fc22(fc21_out).view(self.max_mid, 1, 3, 3)

        fc31_out = F.relu(self.fc31(scale_ratio_tensor))
        conv3_weight = self.fc32(fc31_out).view(self.max_oup, self.max_mid, 1, 1)

        out = F.conv2d(x, conv1_weight[:mid, :inp, :, :], bias=None, stride=1, padding=0, groups=1)
        out = self.bn1[mid_scale_id](out)
        out = F.relu6(out)

        out = F.conv2d(out, conv2_weight[:mid, :, :, :], bias=None, stride=self.stride, padding=1, groups=mid)
        out = self.bn2[mid_scale_id](out)
        out = F.relu6(out)

        out = F.conv2d(out, conv3_weight[:oup, :mid, :, :], bias=None, stride=1, padding=0, groups=1)
        out = self.bn3[oup_scale_id](out)

        if self.max_inp == self.max_oup:
            return (out + x)

        else:
            return out


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


class MobileNetV2(nn.Module):
    def __init__(self, input_size=224, num_classes=1000):
        super(MobileNetV2, self).__init__()
        
        self.channel_scales = overall_channel_scale
        stage_repeat = [1,1,2,3,4,3,3,1,1]
        self.stage_repeat = stage_repeat
        self.gene_length = len(stage_repeat) + sum(stage_repeat)-2
        self.oc_gene_length = len(stage_repeat)

        self.feature = nn.ModuleList()

        for i in range(19):
            if i == 0:
                self.feature.append(conv2d_3x3(3, stage_out_channel[i], 2))
            elif i == 1:
                self.feature.append(bottleneck(stage_out_channel[i-1], stage_out_channel[i], 1, expand_ratio=1))
            elif i == 18:
                self.feature.append(conv2d_1x1(stage_out_channel[i-1], 1280, 1))
            else:
                if stage_out_channel[i-1]!=stage_out_channel[i] and stage_out_channel[i]!=132 and stage_out_channel[i]!=448:
                    self.feature.append(bottleneck(stage_out_channel[i-1], stage_out_channel[i], 2))
                else:
                    self.feature.append(bottleneck(stage_out_channel[i-1], stage_out_channel[i], 1))

        self.pool1 = nn.AvgPool2d(7)
        self.fc = nn.Linear(1280, 1000)

    def forward(self, x, gene=None):
        
        output_scale_ids, mid_scale_ids = parse_gene(gene, [1,1,2,3,4,3,3,1,1])

        for i, block in enumerate(self.feature):
            if i == 0 :
                x = block(x, output_scale_ids[i])
            elif i == 18 :
                x = block(x, output_scale_ids[i-1])
            else :
                x = block(x, mid_scale_ids[i-1], output_scale_ids[i-1], output_scale_ids[i])

        #print(x.shape, flush=True)
        x = self.pool1(x)
        x = x.view(-1, 1280)
        x = self.fc(x)

        return x

def mobile_pruningnet(num_classes=1000):
    return MobileNetV2(num_classes=num_classes)

# if __name__ == "__main__":
#     model = MobileNetV2()
#     print(model)
