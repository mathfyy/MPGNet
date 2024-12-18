import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy
from AlgorithmPool.FunctionalMethodPool.DataIO import writeNII


# 几种归一化
def normalization2D(planes, norm='bn'):
    if norm == 'bn':
        # torch.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True,
        # track_running_stats=True, device=None, dtype=None)
        m = nn.BatchNorm2d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm2d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


def normalization3D(planes, norm='bn'):
    if norm == 'bn':
        # torch.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True,
        # track_running_stats=True, device=None, dtype=None)
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class Conv2D(nn.Module):
    def __init__(self, inC, outC, kerSize=3, kerStride=2, kerPad=1, dropoutRate=0.5, norm="bn"):
        super(Conv2D, self).__init__()
        self.layer0 = nn.Sequential()
        self.layer0.add_module("conv", nn.Conv2d(inC, inC, kerSize, stride=1, padding=kerPad, bias=False))
        self.layer0.add_module("norm", normalization2D(inC, norm))
        self.layer0.add_module("relu", nn.LeakyReLU(inplace=True))
        # self.layer0.add(nn.Dropout3d(p=dropoutRate, inplace=True))

        self.layer = nn.Sequential()
        self.layer.add_module("conv", nn.Conv2d(inC, outC, kerSize, stride=kerStride, padding=kerPad, bias=False))
        # torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0,
        # dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.layer.add_module("norm", normalization2D(outC, norm))
        # torch.nn.RReLU(lower=0.125, upper=0.3333333333333333, inplace=False)
        # inplace是否将得到的值计算得到的值覆盖之前的值(减少内存占用)
        self.layer.add_module("relu", nn.LeakyReLU(inplace=True))
        # self.layer.add(nn.Dropout3d(p=dropoutRate, inplace=True))

    def forward(self, x):
        x = self.layer0(x)
        y = self.layer(x)
        return y


class Conv3D(nn.Module):
    def __init__(self, inC, outC, kerSize=(3, 3, 3), kerStride=(2, 2, 2), kerPad=(1, 1, 1), dilate=(1, 1, 1),
                 dropoutRate=0.5,
                 norm="bn"):
        super(Conv3D, self).__init__()
        self.layer0 = nn.Sequential()
        self.layer0.add_module("conv",
                               nn.Conv3d(inC, inC, kerSize, stride=1, padding=kerPad, bias=True, dilation=dilate))
        self.layer0.add_module("norm", normalization3D(inC, norm))
        self.layer0.add_module("relu", nn.LeakyReLU(inplace=True))

        self.layer = nn.Sequential()
        self.layer.add_module("conv",
                              nn.Conv3d(inC, outC, kerSize, stride=kerStride, padding=kerPad, bias=True,
                                        dilation=dilate))
        self.layer.add_module("norm", normalization3D(outC, norm))
        self.layer.add_module("relu", nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.layer0(x)
        y = self.layer(x)
        return y


class deConv2D(nn.Module):
    def __init__(self, inC, outC, kerSize=3, kerStride=1, kerPad=1, dropoutRate=0.5, norm="bn", insert=True,
                 scale_factor=2):
        super(deConv2D, self).__init__()
        self.insert = insert
        self.scale_factor = scale_factor

        self.layer0 = nn.Sequential()
        self.layer0.add_module("conv", nn.Conv2d(inC, inC, kerSize, stride=1, padding=kerPad, bias=False))
        self.layer0.add_module("norm", normalization2D(inC, norm))
        self.layer0.add_module("relu", nn.LeakyReLU(inplace=True))
        # self.layer.add(nn.Dropout3d(p=dropoutRate, inplace=True))

        self.layer = nn.Sequential()
        self.layer.add_module("conv", nn.Conv2d(inC, outC, kerSize, stride=kerStride, padding=kerPad, bias=False))
        self.layer.add_module("norm", normalization2D(outC, norm))
        # self.layer.add(nn.Dropout3d(p=dropoutRate, inplace=True))

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.layer0(x)
        y = self.layer(x)
        if self.insert:
            y = self.relu(F.interpolate(y, scale_factor=self.scale_factor, mode='bicubic'))
        return y


class deConv3D(nn.Module):
    def __init__(self, inC, outC, kerSize=(3, 3, 3), kerStride=1, kerPad=(1, 1, 1), dilate=(1, 1, 1), dropoutRate=0.5,
                 norm="bn",
                 insert=True,
                 scale_factor=(2, 2, 2)):
        super(deConv3D, self).__init__()
        self.insert = insert
        self.scale_factor = scale_factor

        self.layer0 = nn.Sequential()
        self.layer0.add_module("conv",
                               nn.Conv3d(inC, inC, kerSize, stride=1, padding=kerPad, bias=True, dilation=dilate))
        self.layer0.add_module("norm", normalization3D(inC, norm))
        self.layer0.add_module("relu", nn.LeakyReLU(inplace=True))

        self.layer = nn.Sequential()
        self.layer.add_module("conv", nn.Conv3d(inC, outC, kerSize, stride=kerStride, padding=kerPad, bias=True,
                                                dilation=dilate))
        self.layer.add_module("norm", normalization3D(outC, norm))
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.layer0(x)
        y = self.layer(x)
        if self.insert:
            y = self.relu(
                F.interpolate(y, scale_factor=(self.scale_factor[0], self.scale_factor[1], self.scale_factor[2]),
                              mode='trilinear', align_corners=True))
        return y


class DenseNet3D(nn.Module):
    def __init__(self, inC, outC, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1), dilate=(1, 1, 1),
                 norm="bn"):
        super(DenseNet3D, self).__init__()
        self.layer1 = nn.Sequential()
        self.layer1.add_module("norm", normalization3D(inC, norm))
        self.layer1.add_module("relu", nn.LeakyReLU(inplace=True))
        self.layer1.add_module("conv",
                               nn.Conv3d(inC, inC // 4, kerSize, stride=kerStride, padding=kerPad, bias=True,
                                         dilation=dilate))
        # self.layer2 = nn.Sequential()
        # self.layer2.add_module("norm", normalization3D(5 * inC // 4, norm))
        # self.layer2.add_module("relu", nn.LeakyReLU(inplace=True))
        # self.layer2.add_module("conv",
        #                        nn.Conv3d(5 * inC // 4, inC // 4, kerSize, stride=kerStride, padding=kerPad, bias=True,
        #                                  dilation=dilate))
        # self.layer3 = nn.Sequential()
        # self.layer3.add_module("norm", normalization3D(6 * inC // 4, norm))
        # self.layer3.add_module("relu", nn.LeakyReLU(inplace=True))
        # self.layer3.add_module("conv",
        #                        nn.Conv3d(6 * inC // 4, inC // 4, kerSize, stride=kerStride, padding=kerPad, bias=True,
        #                                  dilation=dilate))
        # self.layer4 = nn.Sequential()
        # self.layer4.add_module("norm", normalization3D(7 * inC // 4, norm))
        # self.layer4.add_module("relu", nn.LeakyReLU(inplace=True))
        # self.layer4.add_module("conv",
        #                        nn.Conv3d(7 * inC // 4, inC // 4, kerSize, stride=kerStride, padding=kerPad, bias=True,
        #                                  dilation=dilate))
        self.out = nn.Conv3d(5 * inC // 4, outC, kerSize, stride=kerStride, padding=kerPad, bias=True,
                             dilation=dilate)

    def forward(self, x):
        y = self.layer1(x)
        x2 = torch.cat((x, y), dim=1)
        # y1 = self.layer2(x2)
        # x3 = torch.cat((x2, y1), dim=1)
        # y2 = self.layer3(x3)
        # x4 = torch.cat((x3, y2), dim=1)
        # y3 = self.layer4(x4)
        # z = torch.cat((x4, y3), dim=1)
        return self.out(x2)


class DAResNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(DAResNet3D, self).__init__()
        self.conLayer1 = nn.Sequential()
        self.conLayer1.add_module("conv",
                                  nn.Conv3d(in_channels, 32, 3, stride=1, padding=1, bias=True))
        self.conLayer1.add_module("norm", normalization3D(32, "bn"))
        self.conLayer1.add_module("relu", nn.LeakyReLU(inplace=True))

        self.conLayer2 = nn.Sequential()
        self.conLayer2.add_module("conv",
                                  nn.Conv3d(96, 48, 3, stride=(1, 2, 2), padding=1, bias=True))
        self.conLayer2.add_module("norm", normalization3D(48, "bn"))
        self.conLayer2.add_module("relu", nn.LeakyReLU(inplace=True))

        self.conLayer3 = nn.Sequential()
        self.conLayer3.add_module("conv",
                                  nn.Conv3d(112, 56, 3, stride=(1, 2, 2), padding=1, bias=True))
        self.conLayer3.add_module("norm", normalization3D(56, "bn"))
        self.conLayer3.add_module("relu", nn.LeakyReLU(inplace=True))

        self.conLayer4 = nn.Sequential()
        self.conLayer4.add_module("conv",
                                  nn.Conv3d(120, 60, 3, stride=(1, 2, 2), padding=1, bias=True))
        self.conLayer4.add_module("norm", normalization3D(60, "bn"))
        self.conLayer4.add_module("relu", nn.LeakyReLU(inplace=True))

        self.FC1 = nn.Conv3d(96, 48, kernel_size=1, stride=1)
        self.bn1 = normalization3D(48, "bn")
        self.FC1_ = nn.Conv3d(48, out_channels, kernel_size=1, stride=1)

        self.FC2 = nn.Conv3d(112, 56, kernel_size=1, stride=1)
        self.bn2 = normalization3D(56, "bn")
        self.DC2 = nn.ConvTranspose3d(56, 48, 3, padding=1, stride=1)
        self.bn2d = normalization3D(48, "bn")
        self.FC2_ = nn.Conv3d(48, out_channels, kernel_size=1, stride=1)

        self.FC3 = nn.Conv3d(120, 60, kernel_size=1, stride=1)
        self.bn3 = normalization3D(60, "bn")
        self.DC3 = nn.ConvTranspose3d(60, 56, 1, stride=1)
        self.bn3d = normalization3D(56, "bn")
        self.DC3_ = nn.ConvTranspose3d(56, 48, 3, padding=1, stride=1)
        self.bn3d_ = normalization3D(48, "bn")
        self.FC3_ = nn.Conv3d(48, out_channels, kernel_size=1, stride=1)

        self.FC4 = nn.Conv3d(124, 62, kernel_size=1, stride=1)
        self.bn4 = normalization3D(62, "bn")
        self.DC4 = nn.ConvTranspose3d(62, 60, 3, padding=1, stride=1)
        self.bn4d = normalization3D(60, "bn")
        self.DC4_ = nn.ConvTranspose3d(60, 56, 3, padding=1, stride=1)
        self.bn4d_ = normalization3D(56, "bn")
        self.DC4__ = nn.ConvTranspose3d(56, 48, 3, padding=1, stride=1)
        self.bn4d__ = normalization3D(48, "bn")
        self.FC4_ = nn.Conv3d(48, out_channels, kernel_size=1, stride=1)

        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        input = torch.max_pool3d(input, kernel_size=1, stride=(1, 2, 2))
        conv1 = self.conLayer1(input)
        dense1 = self.DenseNet3D1(conv1)
        pre1 = self.FC1_(self.LeakyReLU(self.bn1(self.FC1(dense1))))
        pre1 = F.interpolate(pre1, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)

        conv2 = self.conLayer2(dense1)
        dense2 = self.DenseNet3D2(conv2)
        pre2 = self.LeakyReLU(self.bn2d(self.DC2(self.LeakyReLU(self.bn2(self.FC2(dense2))))))
        pre2 = F.interpolate(pre2, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        pre2 = self.FC2_(pre2)
        pre2 = F.interpolate(pre2, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)

        conv3 = self.conLayer3(dense2)
        dense3 = self.DenseNet3D3(conv3)
        pre3 = self.LeakyReLU(self.bn3d(self.DC3(self.LeakyReLU(self.bn3(self.FC3(dense3))))))
        pre3 = F.interpolate(pre3, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        pre3 = F.interpolate(self.LeakyReLU(self.bn3d_(self.DC3_(pre3))), scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        pre3 = self.FC3_(pre3)
        pre3 = F.interpolate(pre3, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)

        conv4 = self.conLayer4(dense3)
        dense4 = self.DenseNet3D4(conv4)
        pre4 = self.LeakyReLU(self.bn4d(self.DC4(self.LeakyReLU(self.bn4(self.FC4(dense4))))))
        pre4 = F.interpolate(pre4, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        pre4 = F.interpolate(self.LeakyReLU(self.bn4d_(self.DC4_(pre4))), scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        pre4 = F.interpolate(self.LeakyReLU(self.bn4d__(self.DC4__(pre4))), scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        pre4 = self.FC4_(pre4)
        pre4 = F.interpolate(pre4, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)

        pre = torch.add(torch.add(torch.add(pre1, pre2), pre3), pre4)

        return self.relu(pre), self.relu(pre1), self.relu(pre2), self.relu(pre3), self.relu(pre4)
