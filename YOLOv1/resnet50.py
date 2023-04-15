import torch
from torch.nn import Sequential, Conv2d, MaxPool2d, ReLU, BatchNorm2d
from torch import nn
from torch.utils import model_zoo

model_urls = {'resnet50' : 'https://download.pytorch.org/models/resnet50-19c8e357.pth'}
CLASS_NUM = 4

class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, stride, downsample):
        super(Bottleneck, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.bottleneck = Sequential(
            Conv2d(in_channel, out_channel, kernel_size=1, stride=stride[0], padding=0, bias=False),
            BatchNorm2d(out_channel),
            ReLU(inplace=True),

            Conv2d(out_channel, out_channel, kernel_size=3, stride=stride[1], padding=1, bias=False),
            BatchNorm2d(out_channel),
            ReLU(inplace=True),

            Conv2d(out_channel, out_channel * 4, kernel_size=1, stride=stride[2], padding=0, bias=False),
            BatchNorm2d(out_channel * 4),
        )
        if self.downsample is False:
            self.shortcut = Sequential()
        else:
            self.shortcut = Sequential(
                Conv2d(self.in_channel, self.out_channel * 4, kernel_size=1, stride=stride[0], bias=False),
                BatchNorm2d(self.out_channel * 4)
            )

    def forward(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class output_net(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        super(output_net, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False, dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.downsample = nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_planes != self.expansion * planes or block_type == 'B':
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = self.relu(out)
        return out



















