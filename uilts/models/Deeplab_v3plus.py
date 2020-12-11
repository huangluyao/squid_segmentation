"""  input size 224×224
Model DeepLabv3_plus : params: 54649004
Model DeepLabv3_plus : size: 208.469406M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv3x3_bn_relu(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Sequential(conv3x3(in_planes, out_planes, stride, groups, dilation),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU(inplace=True))


def fixed_padding(inputs, kernel_size, dilation):
    '''根据卷积核和采样率自动计算padding尺寸'''
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)    # Knew = Kori + (Kori-1)(rate-1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparabelConv2d(nn.Module):
    '''带空洞的深度可分离卷积'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparabelConv2d, self).__init__()

        """先进行分组卷积"""
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, 0, dilation, groups=in_channels, bias=bias)
        """再用1×1的卷积进行处理"""
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                                   padding=0, dilation=1, groups=1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class ASPP(nn.Module):
    """空洞SPPNet"""
    def __init__(self, in_channels, out_channels, os):
        super(ASPP, self).__init__()

        if os == 16:
            dilations = [1, 6, 12, 18]
        elif os == 8:
            dilations = [1, 12, 24, 36]

        self.aspp1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                             dilation=dilations[0], bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU())

        self.aspp2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilations[1],
                                             dilation=dilations[1], bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))

        self.aspp3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilations[2],
                                             dilation=dilations[2], bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))

        self.aspp4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilations[3],
                                             dilation=dilations[3], bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))

        self.gp = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Conv2d(2048, 256, 1, stride=1, padding=0, bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(256*5, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.gp(x)         # [n, c, 1, 1]
        # 线性插值
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        # 进行拼接
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)

        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, reps, stride=1, dilation=1, grow_first=True):
        super(Block, self).__init__()

        # 定义跳跃连接部分
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                      nn.BatchNorm2d(out_channels))
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        # 每一块的第一个卷积块
        if grow_first:
            rep.append(SeparabelConv2d(in_channels, out_channels, 3, stride=1, dilation=dilation))
            rep.append(nn.BatchNorm2d(out_channels))
            rep.append(nn.ReLU(inplace=True))
            # 循环卷积次数
            for i in range(reps - 1):
                rep.append(SeparabelConv2d(out_channels, out_channels, 3, stride=1, dilation=dilation))
                rep.append(nn.BatchNorm2d(out_channels))
                rep.append(nn.ReLU(inplace=True))

        else:
            rep.append(SeparabelConv2d(in_channels, in_channels, 3, stride=1, dilation=dilation))
            rep.append(nn.BatchNorm2d(in_channels))
            rep.append(nn.ReLU(inplace=True))
            # 循环卷积次数
            for i in range(reps - 1):
                rep.append(SeparabelConv2d(in_channels, out_channels, 3, stride=1, dilation=dilation))
                rep.append(nn.BatchNorm2d(out_channels))
                rep.append(nn.ReLU(inplace=True))

        # 最后一个卷积，决定是否下采样
        rep.append(SeparabelConv2d(out_channels, out_channels, 3, stride=stride))

        self.block = nn.Sequential(*rep)

    def forward(self, x):
        x1 = self.block(x)
        if self.skip is not None:
            x = self.skip(x)
        x = x + x1
        x = self.relu(x)
        return x


class Xception(nn.Module):
    """定义Xception网络"""
    def __init__(self, in_channels, os=16):
        super(Xception,self).__init__()

        if os==16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif os == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError

        self.relu = nn.ReLU(inplace=True)
        # Entry flow
        self.conv1_bn_relu = conv3x3_bn_relu(in_channels, 32)
        self.conv2_bn_relu = conv3x3_bn_relu(32, 64)

        self.block1 = Block(64, 128, reps=2, stride=2)
        self.block2 = Block(128, 256, reps=2, stride=2)
        self.block3 = Block(256, 728,reps=2, stride=entry_block3_stride)

        # Middle flow
        mid_block = []
        for i in range(16):
            mid_block.append(Block(728, 728, reps=2, stride=1, dilation=middle_block_dilation,grow_first=False))
        self.mid_flow = nn.Sequential(*mid_block)

        # Exit flow
        self.exitflow1 = Block(728, 1024, reps=2, stride=2, dilation=exit_block_dilations[0], grow_first=False)
        self.exitflow2 = SeparabelConv2d(1024, 1536, 3, dilation=exit_block_dilations[1])
        self.exitflow3 = SeparabelConv2d(1536, 1536, 3, dilation=exit_block_dilations[1])
        self.exitflow4 = SeparabelConv2d(1536, 2048, 3, dilation=exit_block_dilations[1])

        # 初始化网络权重
        self._init_weight()

    def forward(self, x):
        x = self.conv1_bn_relu(x)
        x = self.conv2_bn_relu(x)
        x = self.block1(x)
        x = self.block2(x)
        low_level_feat = x
        x = self.block3(x)

        x = self.mid_flow(x)
        x = self.exitflow1(x)
        x = self.exitflow2(x)
        x = self.exitflow3(x)
        x = self.exitflow4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus(nn.Module):
    def __init__(self, input_channels, num_calsses, os=16):
        super(DeepLabv3_plus, self).__init__()

        """空洞卷积编码器"""
        self.xception_features = Xception(input_channels, os)
        """空洞空间金字塔池化"""
        self.ASPP = ASPP(2048, 256, os=os)

        self.conv1_bn_relu = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU())
        self.conv2_bn_relu = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False),
                                           nn.BatchNorm2d(48),
                                           nn.ReLU())

        self.last_conv = nn.Sequential(conv3x3_bn_relu(304, 256),
                                       conv3x3_bn_relu(256, 256),
                                       nn.Conv2d(256, num_calsses, kernel_size=1, stride=1))

    def forward(self, input):
        x, low_leval_feat = self.xception_features(input)
        x = self.ASPP(x)
        # print('ASPP out_size', x.size())
        x = self.conv1_bn_relu(x)
        # print('size', (int(math.ceil(input.size()[-2]/4)), int(math.ceil(input.size()[-1]/4))))
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2]/4)), int(math.ceil(input.size()[-1]/4))),
                          mode='bilinear',align_corners=True)

        low_leval_feat = self.conv2_bn_relu(low_leval_feat)

        x = torch.cat([low_leval_feat, x], dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[-2:], mode='bilinear', align_corners=True)
        return x


if __name__=="__main__":

    a = torch.rand((1, 3, 224, 224))
    model = DeepLabv3_plus(3, num_calsses=12, os=16)
    model.eval()
    x = model(a)
    print('x.size', x.size())

    # 计算网络模型尺寸大小
    import numpy as np
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    type_size = 4 # float32 占4个字节
    print('Model {} : params: {}'.format(model._get_name(), para))
    print('Model {} : size: {:4f}M'.format(model._get_name(), para*type_size/1024/1024))