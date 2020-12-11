import torch
import torch.nn as nn


# UNet 论文中的编码器
def Encoder(in_channels, out_channels):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
                  nn.ReLU(),
                  nn.BatchNorm2d(out_channels),
                  nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False),
                  nn.ReLU(),
                  nn.BatchNorm2d(out_channels)
                  )
    return block


# UNet论文中的解码器
class Decoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Decoder, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels))

    def forward(self, e, d):
        d = self.up(d)
        # 将encoder得到的尺寸进行裁剪和decoder进行拼接
        diffY = e.size()[2] - d.size()[2]
        diffX = e.size()[3] - d.size()[3]
        e = e[:, :, diffY//2:e.size()[2]-diffY//2, diffX//2: e.size()[3]-diffX//2]
        cat = torch.cat([e, d], dim=1)
        out = self.block(cat)
        return out


def final_block(in_channels, out_channels):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                          nn.ReLU(),
                          nn.BatchNorm2d(out_channels))
    return block


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encode
        self.encode1 = Encoder(in_channels, 64)
        self.encode2 = Encoder(64, 128)
        self.encode3 = Encoder(128, 256)
        self.encode4 = Encoder(256, 512)

        # 编码器最底部
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(1024),
                                        nn.Conv2d(1024, 1024, kernel_size=3, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(1024)
                                        )

        # decoder
        self.decode4 = Decoder(1024, 512, 512)
        self.decode3 = Decoder(512, 256, 256)
        self.decode2 = Decoder(256, 128, 128)
        self.decode1 = Decoder(128, 64, 64)

        self.final = final_block(64, num_classes)

    def forward(self, x):
        encode_block1 = self.encode1(x)
        pool1 = self.pool(encode_block1)
        encode_block2 = self.encode2(pool1); print('encode_block2', encode_block2.size())
        pool2 = self.pool(encode_block2)
        encode_block3 = self.encode3(pool2)
        pool3 = self.pool(encode_block3)
        encode_block4 = self.encode4(pool3)
        pool4 = self.pool(encode_block4)

        bottleneck = self.bottleneck(pool4)

        decode_block4 = self.decode4(encode_block4, bottleneck)
        decode_block3 = self.decode3(encode_block3, decode_block4); print('decode_block3', decode_block3.size())
        decode_block2 = self.decode2(encode_block2, decode_block3)
        decode_block1 = self.decode1(encode_block1, decode_block2)

        out = self.final(decode_block1)
        return out


if __name__ == "__main__":
    rgb = torch.randn(1, 3, 572, 572)

    net = UNet(3, 12)

    out = net(rgb)

    print(out.shape)