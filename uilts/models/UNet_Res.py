""" inpnut size 224×224
Model UNet : params: 60369624
Model UNet : size: 230.291840M
"""

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 残差块作为编码
class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 构建残差块
def make_layers(block, in_channels, out_channels, blocks, stride=1):

    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            conv1x1(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels))

    layers = []
    layers.append(block(in_channels, out_channels, stride, downsample))

    for _ in range(1, blocks):
        layers.append(block(out_channels, out_channels))

    return nn.Sequential(*layers)


# UNet论文中的解码器
class Decoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Decoder, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels))

    def forward(self, e, d):
        d = self.up(d)
        cat = torch.cat([e, d], dim=1)
        out = self.block(cat)
        return out


def final_block(in_channels, out_channels):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                          nn.ReLU(inplace=True),
                          nn.BatchNorm2d(out_channels))
    return block


class UNet(nn.Module):
    def __init__(self,num_classes):
        super(UNet, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encode
        self.encode1 = make_layers(BasicBlock, 3, 64, 2, stride=1)
        self.encode2 = make_layers(BasicBlock, 64, 128, 2, stride=2)
        self.encode3 = make_layers(BasicBlock, 128, 256, 2,  stride=2)
        self.encode4 = make_layers(BasicBlock, 256, 512, 2, stride=2)

        # 编码器最底部
        self.bottleneck = make_layers(BasicBlock, 512, 1024, 2, stride=2)

        # decoder
        self.decode4 = Decoder(1024, 512, 512)
        self.decode3 = Decoder(512, 256, 256)
        self.decode2 = Decoder(256, 128, 128)
        self.decode1 = Decoder(128, 64, 64)

        self.final = final_block(64, num_classes)

    def forward(self, x):
        encode_block1 = self.encode1(x)  # print('encode_block1', encode_block1.size()) torch.Size([2, 128, 416, 416])
        encode_block2 = self.encode2(encode_block1)  # print('encode_block2', encode_block2.size()) torch.Size([2, 256, 208, 208])
        encode_block3 = self.encode3(encode_block2)  # print('encode_block3', encode_block3.size()) torch.Size([2, 512, 104, 104])
        encode_block4 = self.encode4(encode_block3)  # print('encode_block4', encode_block4.size()) torch.Size([2, 1024, 52, 52])

        bottleneck = self.bottleneck(encode_block4)  # print('bottleneck', bottleneck.size()) torch.Size([2, 1024, 26, 26])

        decode_block4 = self.decode4(encode_block4, bottleneck)  # print('decode_block4', decode_block4.size())
        decode_block3 = self.decode3(encode_block3, decode_block4)  # print('decode_block3', decode_block3.size())
        decode_block2 = self.decode2(encode_block2, decode_block3)  # print('decode_block2', decode_block2.size())
        decode_block1 = self.decode1(encode_block1, decode_block2)  # print('decode_block1', decode_block1.size())

        out = self.final(decode_block1)
        return out


if __name__ == "__main__":
    rgb = torch.randn(1, 3, 224, 224)

    model = UNet(3, 12)

    out = model(rgb)
    print(out.shape)

    # 计算网络模型尺寸大小
    import numpy as np
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    type_size = 4 # float32 占4个字节
    print('Model {} : params: {}'.format(model._get_name(), para))
    print('Model {} : size: {:4f}M'.format(model._get_name(), para*type_size/1024/1024))

