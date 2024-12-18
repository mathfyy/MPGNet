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


class Conv3D_0(nn.Module):
    def __init__(self, inC, outC, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1), dilate=(1, 1, 1),
                 dropoutRate=0.5,
                 norm="bn"):
        super(Conv3D_0, self).__init__()
        self.layer0 = nn.Sequential()
        self.layer0.add_module("conv",
                               nn.Conv3d(inC, outC, kerSize, stride=kerStride, padding=kerPad, bias=True,
                                         dilation=dilate))
        self.layer0.add_module("norm", normalization3D(outC, norm))
        self.layer0.add_module("relu", nn.Tanh())

    def forward(self, x):
        x = self.layer0(x)
        return x


class Conv3D(nn.Module):
    def __init__(self, inC, outC, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1), dilate=(1, 1, 1),
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


class Conv3D_e(nn.Module):
    def __init__(self, inC, outC, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1), dilate=(1, 1, 1),
                 dropoutRate=0.5,
                 norm="bn"):
        super(Conv3D_e, self).__init__()
        self.layer0 = nn.Sequential()
        self.layer0.add_module("conv",
                               nn.Conv3d(inC, inC, kerSize, stride=kerStride, padding=kerPad, bias=True,
                                         dilation=dilate))
        self.layer0.add_module("norm", normalization3D(inC, norm))
        self.layer0.add_module("relu", nn.LeakyReLU(inplace=True))

        self.layer = nn.Sequential()
        self.layer.add_module("conv",
                              nn.Conv3d(inC, outC, kerSize, stride=1, padding=kerPad, bias=True,
                                        dilation=dilate))
        self.layer.add_module("norm", normalization3D(outC, norm))
        self.layer.add_module("relu", nn.LeakyReLU(inplace=True))
        self.layer.add_module("conv",
                              nn.Conv3d(inC, outC, kerSize, stride=1, padding=kerPad, bias=True,
                                        dilation=dilate))
        self.layer.add_module("norm", normalization3D(outC, norm))
        self.layer.add_module("relu", nn.LeakyReLU(inplace=True))
        self.layer.add_module("conv",
                              nn.Conv3d(inC, outC, kerSize, stride=1, padding=kerPad, bias=True,
                                        dilation=dilate))
        self.layer.add_module("norm", normalization3D(outC, norm))
        self.layer.add_module("relu", nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.layer0(x)
        y = self.layer(x)
        return y


class Conv3DAdd(nn.Module):
    def __init__(self, inC, outC, kerSize=(3, 3, 3), kerStride=(2, 2, 2), kerPad=(1, 1, 1), dilate=(1, 1, 1),
                 dropoutRate=0.5,
                 norm="bn"):
        super(Conv3DAdd, self).__init__()
        self.layer0 = nn.Sequential()
        self.layer0.add_module("conv",
                               nn.Conv3d(inC, inC, kerSize, stride=1, padding=kerPad, bias=True, dilation=dilate))
        self.layer0.add_module("norm", normalization3D(inC, norm))
        self.layer0.add_module("relu", nn.LeakyReLU(inplace=True))
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


class resConv3D(nn.Module):
    def __init__(self, inC, outC, kerSize=(3, 3, 3), kerStride=(2, 2, 2), kerPad=(1, 1, 1), dilate=(1, 1, 1),
                 dropoutRate=0.5,
                 norm="bn"):
        super(resConv3D, self).__init__()
        self.layer0 = nn.Sequential()
        self.layer0.add_module("conv",
                               nn.Conv3d(inC, outC, kerSize, stride=1, padding=kerPad, bias=True, dilation=dilate))
        self.layer0.add_module("norm", normalization3D(outC, norm))
        self.layer0.add_module("relu", nn.LeakyReLU(inplace=True))

        self.layer = nn.Sequential()
        self.layer.add_module("conv",
                              nn.Conv3d(outC, outC, kerSize, stride=kerStride, padding=kerPad, bias=True,
                                        dilation=dilate))
        self.layer.add_module("norm", normalization3D(outC, norm))
        self.layer.add_module("relu", nn.LeakyReLU(inplace=True))

        self.layer1 = nn.Sequential()
        self.layer1.add_module("conv",
                               nn.Conv3d(inC, outC, kernel_size=1, stride=1, padding=0, bias=True,
                                         dilation=dilate))
        self.layer1.add_module("norm", normalization3D(outC, norm))
        self.layer1.add_module("relu", nn.LeakyReLU(inplace=True))

    def forward(self, x):
        y = self.layer0(x)
        z = self.layer1(x)
        out = torch.add(y, z)
        out = self.layer(out)
        return out


class resConv3DAdd(nn.Module):
    def __init__(self, inC, outC, kerSize=(3, 3, 3), kerStride=(2, 2, 2), kerPad=(1, 1, 1), dilate=(1, 1, 1),
                 dropoutRate=0.5,
                 norm="bn"):
        super(resConv3DAdd, self).__init__()
        self.layer0 = nn.Sequential()
        self.layer0.add_module("conv",
                               nn.Conv3d(inC, inC, kerSize, stride=1, padding=kerPad, bias=True, dilation=dilate))
        self.layer0.add_module("norm", normalization3D(inC, norm))
        self.layer0.add_module("relu", nn.LeakyReLU(inplace=True))
        self.layer0.add_module("conv",
                               nn.Conv3d(inC, outC, kerSize, stride=1, padding=kerPad, bias=True, dilation=dilate))
        self.layer0.add_module("norm", normalization3D(outC, norm))
        self.layer0.add_module("relu", nn.LeakyReLU(inplace=True))

        self.layer = nn.Sequential()
        self.layer.add_module("conv",
                              nn.Conv3d(outC, outC, kerSize, stride=kerStride, padding=kerPad, bias=True,
                                        dilation=dilate))
        self.layer.add_module("norm", normalization3D(outC, norm))
        self.layer.add_module("relu", nn.LeakyReLU(inplace=True))

        self.layer1 = nn.Sequential()
        self.layer1.add_module("conv",
                               nn.Conv3d(inC, outC, kernel_size=1, stride=1, padding=0, bias=True,
                                         dilation=dilate))
        self.layer1.add_module("norm", normalization3D(outC, norm))
        self.layer1.add_module("relu", nn.LeakyReLU(inplace=True))

    def forward(self, x):
        y = self.layer0(x)
        z = self.layer1(x)
        out = torch.add(y, z)
        out = self.layer(out)
        return out


class ResBlock_v1(nn.Module):
    def __init__(self, inC, outC, kerSize=(3, 3, 3), kerStride=(2, 2, 2), kerPad=(1, 1, 1), dilate=(1, 1, 1),
                 dropoutRate=0.5,
                 norm="bn"):
        super(ResBlock_v1, self).__init__()
        with self.name_scope():
            self.layers = nn.Sequential()
            self.layers.add_module("conv",
                                   nn.Conv3d(inC, inC, kerSize, stride=1, padding=kerPad, bias=True, dilation=dilate))
            self.layers.add_module("norm", normalization3D(outC, norm))
            self.layers.add_module("relu", nn.ReLU(inplace=True))
            self.layers.add_module("drop", nn.Dropout(0.5))
            self.layers.add_module("conv",
                                   nn.Conv3d(inC, outC, kerSize, stride=kerStride, padding=kerPad, bias=True,
                                             dilation=dilate))
            self.layers.add_module("norm", normalization3D(outC, norm))
            self.layers.add_module("relu", nn.ReLU(inplace=True))
            self.layers.add_module("drop", nn.Dropout(0.5))

    def hybrid_forward(self, x):
        data = self.layers(x)
        data = torch.cat((x, data), dim=1)
        return data


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
                 scale_factor=(1, 2, 2)):
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


class deConv3D_e(nn.Module):
    def __init__(self, inC, outC, kerSize=(3, 3, 3), kerStride=1, kerPad=(1, 1, 1), dilate=(1, 1, 1), dropoutRate=0.5,
                 norm="bn",
                 insert=True,
                 scale_factor=(1, 2, 2)):
        super(deConv3D_e, self).__init__()
        self.insert = insert
        self.scale_factor = scale_factor

        self.layer0 = nn.Sequential()
        self.layer0.add_module("conv",
                               nn.Conv3d(inC, inC, kerSize, stride=1, padding=kerPad, bias=True, dilation=dilate))
        self.layer0.add_module("norm", normalization3D(inC, norm))
        self.layer0.add_module("relu", nn.LeakyReLU(inplace=True))
        self.layer0.add_module("conv",
                               nn.Conv3d(inC, inC, kerSize, stride=1, padding=kerPad, bias=True, dilation=dilate))
        self.layer0.add_module("norm", normalization3D(inC, norm))
        self.layer0.add_module("relu", nn.LeakyReLU(inplace=True))
        self.layer0.add_module("conv",
                               nn.Conv3d(inC, inC, kerSize, stride=1, padding=kerPad, bias=True, dilation=dilate))
        self.layer0.add_module("norm", normalization3D(inC, norm))
        self.layer0.add_module("relu", nn.LeakyReLU(inplace=True))

        self.layer = nn.Sequential()
        self.layer.add_module("conv", nn.Conv3d(inC, outC, kerSize, stride=kerStride, padding=kerPad, bias=True,
                                                dilation=dilate))
        self.layer.add_module("norm", normalization3D(outC, norm))
        self.layer.add_module("relu", nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.layer0(x)
        y = self.layer(x)
        if self.insert:
            y = F.interpolate(y, scale_factor=(self.scale_factor[0], self.scale_factor[1], self.scale_factor[2]),
                              mode='trilinear', align_corners=True)
        return y


class upSample_ChangeChannel_3D(nn.Module):
    def __init__(self, inC, outC, kerSize=(1, 1, 1), kerStride=1, kerPad=(0, 0, 0), dilate=(1, 1, 1),
                 norm="bn",
                 insert=True,
                 scale_factor=(2, 2, 2)):
        super(upSample_ChangeChannel_3D, self).__init__()
        self.insert = insert
        self.scale_factor = scale_factor

        self.layer = nn.Sequential()
        self.layer.add_module("conv", nn.Conv3d(inC, outC, kerSize, stride=kerStride, padding=kerPad, bias=True,
                                                dilation=dilate))
        self.layer.add_module("norm", normalization3D(outC, norm))
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        y = self.layer(x)
        if self.insert:
            y = self.relu(
                F.interpolate(y, scale_factor=(self.scale_factor[0], self.scale_factor[1], self.scale_factor[2]),
                              mode='trilinear', align_corners=True))
        out = y

        return out


class resdeConv3D(nn.Module):
    def __init__(self, inC, outC, kerSize=(3, 3, 3), kerStride=1, kerPad=(1, 1, 1), dilate=(1, 1, 1), dropoutRate=0.5,
                 norm="bn",
                 insert=True,
                 scale_factor=(2, 2, 2)):
        super(resdeConv3D, self).__init__()
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
        self.layer.add_module("relu", nn.LeakyReLU(inplace=True))

        self.layer1 = nn.Sequential()
        self.layer1.add_module("conv",
                               nn.Conv3d(inC, outC, kernel_size=1, stride=1, padding=0, bias=True,
                                         dilation=dilate))
        self.layer1.add_module("norm", normalization3D(outC, norm))
        self.layer1.add_module("relu", nn.LeakyReLU(inplace=True))

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        y = self.layer0(x)
        y = self.layer(y)
        z = self.layer1(x)
        out = torch.add(y, z)

        if self.insert:
            out = self.relu(
                F.interpolate(out, scale_factor=(self.scale_factor[0], self.scale_factor[1], self.scale_factor[2]),
                              mode='trilinear', align_corners=True))
        return out


class resdeConv3DT(nn.Module):
    def __init__(self, inC, outC, kerSize=(3, 3, 3), kerStride=1, kerPad=(1, 1, 1), dilate=(1, 1, 1), dropoutRate=0.5,
                 norm="bn",
                 insert=True,
                 scale_factor=(2, 2, 2)):
        super(resdeConv3DT, self).__init__()
        self.insert = insert
        self.scale_factor = scale_factor

        self.layer0 = nn.Sequential()
        self.layer0.add_module("conv",
                               nn.ConvTranspose3d(inC, inC, kerSize, stride=1, padding=kerPad, bias=True,
                                                  dilation=dilate))
        self.layer0.add_module("norm", normalization3D(inC, norm))
        self.layer0.add_module("relu", nn.LeakyReLU(inplace=True))

        self.layer = nn.Sequential()
        self.layer.add_module("conv",
                              nn.ConvTranspose3d(inC, outC, kerSize, stride=kerStride, padding=kerPad, bias=True,
                                                 dilation=dilate))
        self.layer.add_module("norm", normalization3D(outC, norm))
        self.layer.add_module("relu", nn.LeakyReLU(inplace=True))

        self.layer1 = nn.Sequential()
        self.layer1.add_module("conv",
                               nn.ConvTranspose3d(inC, outC, kernel_size=1, stride=1, padding=0, bias=True,
                                                  dilation=dilate))
        self.layer1.add_module("norm", normalization3D(outC, norm))
        self.layer1.add_module("relu", nn.LeakyReLU(inplace=True))

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        y = self.layer0(x)
        y = self.layer(y)
        z = self.layer1(x)
        out = torch.add(y, z)

        if self.insert:
            out = self.relu(
                F.interpolate(out, scale_factor=(self.scale_factor[0], self.scale_factor[1], self.scale_factor[2]),
                              mode='trilinear', align_corners=True))
        return out


class resdeConv3DTAdd(nn.Module):
    def __init__(self, inC, outC, kerSize=(3, 3, 3), kerStride=1, kerPad=(1, 1, 1), dilate=(1, 1, 1), dropoutRate=0.5,
                 norm="bn",
                 insert=True,
                 scale_factor=(2, 2, 2)):
        super(resdeConv3DTAdd, self).__init__()
        self.insert = insert
        self.scale_factor = scale_factor

        self.layer0 = nn.Sequential()
        self.layer0.add_module("conv",
                               nn.ConvTranspose3d(inC, inC, kerSize, stride=1, padding=kerPad, bias=True,
                                                  dilation=dilate))
        self.layer0.add_module("norm", normalization3D(inC, norm))
        self.layer0.add_module("relu", nn.LeakyReLU(inplace=True))
        self.layer0.add_module("conv",
                               nn.ConvTranspose3d(inC, inC, kerSize, stride=1, padding=kerPad, bias=True,
                                                  dilation=dilate))
        self.layer0.add_module("norm", normalization3D(inC, norm))
        self.layer0.add_module("relu", nn.LeakyReLU(inplace=True))

        self.layer = nn.Sequential()
        self.layer.add_module("conv",
                              nn.ConvTranspose3d(inC, outC, kerSize, stride=kerStride, padding=kerPad, bias=True,
                                                 dilation=dilate))
        self.layer.add_module("norm", normalization3D(outC, norm))
        self.layer.add_module("relu", nn.LeakyReLU(inplace=True))

        self.layer1 = nn.Sequential()
        self.layer1.add_module("conv",
                               nn.ConvTranspose3d(inC, outC, kernel_size=1, stride=1, padding=0, bias=True,
                                                  dilation=dilate))
        self.layer1.add_module("norm", normalization3D(outC, norm))
        self.layer1.add_module("relu", nn.LeakyReLU(inplace=True))

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        y = self.layer0(x)
        y = self.layer(y)
        z = self.layer1(x)
        out = torch.add(y, z)

        if self.insert:
            out = self.relu(
                F.interpolate(out, scale_factor=(self.scale_factor[0], self.scale_factor[1], self.scale_factor[2]),
                              mode='trilinear', align_corners=True))
        return out


class deConv3DT(nn.Module):
    def __init__(self, inC, outC, kerSize=(3, 3, 3), kerStride=1, kerPad=(0, 0, 0), dilate=(1, 1, 1), dropoutRate=0.5,
                 norm="bn",
                 insert=True,
                 scale_factor=(2, 2, 2)):
        super(deConv3DT, self).__init__()
        self.insert = insert
        self.scale_factor = scale_factor

        self.layer0 = nn.Sequential()
        self.layer0.add_module("conv", nn.ConvTranspose3d(inC, inC, kerSize, stride=1, padding=kerPad, bias=True,
                                                          dilation=dilate))
        self.layer0.add_module("norm", normalization3D(inC, norm))
        self.layer0.add_module("relu", nn.LeakyReLU(inplace=True))

        self.layer = nn.Sequential()
        self.layer.add_module("conv",
                              nn.ConvTranspose3d(inC, outC, kerSize, stride=kerStride, padding=kerPad, bias=True,
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


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ChannelAttention3D(nn.Module):
    def __init__(self, inC=4, outC=4):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((24, 256, 256))
        self.max_pool = nn.AdaptiveMaxPool3d((24, 256, 256))

        self.fc1 = nn.Conv3d(inC, outC, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(outC, inC, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SpatialAttention3D(nn.Module):
    def __init__(self, inC=3, outC=3, kernel_size=3):
        super(SpatialAttention3D, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, outC, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CRFB2D(nn.Module):
    def __init__(self, en_cin, en_cout, de_cin, de_cout, kerSize=3, kerStride=1, kerPad=1, insert=True, scale_factor=2,
                 dropoutRate=0.5,
                 norm="bn"):
        super(CRFB2D, self).__init__()
        self.encoder = Conv2D(en_cin, en_cout, kerSize, kerStride, kerPad)
        self.decoder = deConv2D(de_cin, de_cout, kerSize, 1, kerPad, insert, scale_factor=scale_factor)

    def forward(self, u, k):
        u_ = self.decoder(u)
        k_ = self.encoder(k)
        uin = torch.add(u, k_)
        kin = torch.add(k, u_)
        return kin, uin


class CRFB3D(nn.Module):
    def __init__(self, en_cin, en_cout, de_cin, de_cout, kerSize=3, kerStride=1, kerPad=1, insert=True,
                 scale_factor=2,
                 dropoutRate=0.5,
                 norm="bn"):
        super(CRFB3D, self).__init__()
        self.encoder = Conv3D(en_cin, en_cout, kerSize, kerStride, kerPad)
        self.decoder = deConv3D(de_cin, de_cout, kerSize, 1, kerPad, insert, scale_factor=scale_factor)

    def forward(self, u, k):
        u_ = self.decoder(u)
        k_ = self.encoder(k)
        uin = torch.add(u, k_)
        kin = torch.add(k, u_)
        return kin, uin


class simpleKiunet2D(nn.Module):
    def __init__(self, training, cin=3, cout=4, label=1):
        super(simpleKiunet2D, self).__init__()
        self.training = training
        # Unet相关
        self.UNet_encoder1 = Conv2D(cin, cout, 3, 2, 1)
        self.UNet_encoder2 = Conv2D(cout, 2 * cout, 3, 2, 1)
        self.UNet_encoder3 = Conv2D(2 * cout, 4 * cout, 3, 2, 1)

        self.UNet_decoder3 = deConv2D(4 * cout, 2 * cout, 3, 1, 1)
        self.UNet_decoder2 = deConv2D(2 * cout, cout, 3, 1, 1)
        self.UNet_decoder1 = deConv2D(cout, cin, 3, 1, 1)

        # Kinet相关
        self.KiNet_encoder1 = Conv2D(cout, cin, 3, 1, 1)
        self.KiNet_encoder2 = Conv2D(2 * cout, cout, 3, 1, 1)
        self.KiNet_encoder3 = Conv2D(4 * cout, 2 * cout, 3, 1, 1)

        self.KiNet_decoder3 = deConv2D(2 * cout, 4 * cout, 3, 1, 1, insert=False)
        self.KiNet_decoder2 = deConv2D(cout, 2 * cout, 3, 1, 1, insert=False)
        self.KiNet_decoder1 = deConv2D(cin, cout, 3, 1, 1, insert=False)

        # CRFB
        self.CRFB1 = CRFB2D(cout, cout, cout, cout, kerStride=2, insert=False)
        self.CRFB2 = CRFB2D(2 * cout, 2 * cout, 2 * cout, 2 * cout, kerStride=4, insert=True, scale_factor=4)
        self.CRFB3 = CRFB2D(4 * cout, 4 * cout, 4 * cout, 4 * cout, kerStride=8, insert=True, scale_factor=8)
        self.CRFB4 = CRFB2D(2 * cout, 2 * cout, 2 * cout, 2 * cout, kerStride=4, insert=True, scale_factor=4)
        self.CRFB5 = CRFB2D(cout, cout, cout, cout, kerStride=2, insert=True, scale_factor=2)

        # conv
        self.conv = nn.Conv2d(cin, label, 1, bias=False)
        self.soft = nn.Softmax(dim=1)

    def forward(self, input):
        u_e1 = self.UNet_encoder1(input)
        k_d1 = self.KiNet_decoder1(input)
        kin1, uin1 = self.CRFB1(u_e1, k_d1)

        u_e2 = self.UNet_encoder2(uin1)
        k_d2 = self.KiNet_decoder2(kin1)
        kin2, uin2 = self.CRFB2(u_e2, k_d2)

        u_e3 = self.UNet_encoder3(uin2)
        k_d3 = self.KiNet_decoder3(kin2)
        kin3, uin3 = self.CRFB3(u_e3, k_d3)

        u_d3 = self.UNet_decoder3(uin3)
        k_e3 = self.KiNet_encoder3(kin3)
        kin4, uin4 = self.CRFB4(u_d3, k_e3)

        uin4 = torch.add(uin4, u_e2)
        kin4 = torch.add(kin4, k_d2)

        u_d2 = self.UNet_decoder2(uin4)
        k_e2 = self.KiNet_encoder2(kin4)
        kin5, uin5 = self.CRFB5(u_d2, k_e2)

        uin5 = torch.add(uin5, u_e1)
        kin5 = torch.add(kin5, k_d1)

        uout = self.UNet_decoder1(uin5)
        kout = self.KiNet_encoder1(kin5)

        netout = F.relu(self.conv(torch.add(uout, kout)))
        netout = self.soft(netout)

        return netout


class simpleKiunet3D(nn.Module):
    def __init__(self, training, cin=1, cout=4, label=1):
        super(simpleKiunet3D, self).__init__()
        self.training = training
        # Unet相关
        self.UNet_encoder1 = Conv3D(cin, cout, 3, 2, 1)
        self.UNet_encoder2 = Conv3D(cout, 2 * cout, 3, 2, 1)
        self.UNet_encoder3 = Conv3D(2 * cout, 4 * cout, 3, 2, 1)

        self.UNet_decoder3 = deConv3D(4 * cout, 2 * cout, 3, 1, 1)
        self.UNet_decoder2 = deConv3D(2 * cout, cout, 3, 1, 1)
        self.UNet_decoder1 = deConv3D(cout, cin, 3, 1, 1)

        # Kinet相关
        self.KiNet_encoder1 = Conv3D(cout, cin, 3, 1, 1)
        self.KiNet_encoder2 = Conv3D(2 * cout, cout, 3, 1, 1)
        self.KiNet_encoder3 = Conv3D(4 * cout, 2 * cout, 3, 1, 1)

        self.KiNet_decoder3 = deConv3D(2 * cout, 4 * cout, 3, 1, 1, insert=False)
        self.KiNet_decoder2 = deConv3D(cout, 2 * cout, 3, 1, 1, insert=False)
        self.KiNet_decoder1 = deConv3D(cin, cout, 3, 1, 1, insert=False)

        # CRFB
        self.CRFB1 = CRFB3D(cout, cout, cout, cout, kerStride=2, insert=False)
        self.CRFB2 = CRFB3D(2 * cout, 2 * cout, 2 * cout, 2 * cout, kerStride=4, insert=True, scale_factor=4)
        self.CRFB3 = CRFB3D(4 * cout, 4 * cout, 4 * cout, 4 * cout, kerStride=8, insert=True, scale_factor=8)
        self.CRFB4 = CRFB3D(2 * cout, 2 * cout, 2 * cout, 2 * cout, kerStride=4, insert=True, scale_factor=4)
        self.CRFB5 = CRFB3D(cout, cout, cout, cout, kerStride=2, insert=True, scale_factor=2)

        # conv
        self.conv = nn.Conv3d(cin, label, 1, bias=False)
        self.soft = nn.Softmax(dim=1)

    def forward(self, input):
        u_e1 = self.UNet_encoder1(input)
        k_d1 = self.KiNet_decoder1(input)
        kin1, uin1 = self.CRFB1(u_e1, k_d1)

        u_e2 = self.UNet_encoder2(uin1)
        k_d2 = self.KiNet_decoder2(kin1)
        kin2, uin2 = self.CRFB2(u_e2, k_d2)

        u_e3 = self.UNet_encoder3(uin2)
        k_d3 = self.KiNet_decoder3(kin2)
        kin3, uin3 = self.CRFB3(u_e3, k_d3)

        u_d3 = self.UNet_decoder3(uin3)
        k_e3 = self.KiNet_encoder3(kin3)
        kin4, uin4 = self.CRFB4(u_d3, k_e3)

        uin4 = torch.add(uin4, u_e2)
        kin4 = torch.add(kin4, k_d2)

        u_d2 = self.UNet_decoder2(uin4)
        k_e2 = self.KiNet_encoder2(kin4)
        kin5, uin5 = self.CRFB5(u_d2, k_e2)

        uin5 = torch.add(uin5, u_e1)
        kin5 = torch.add(kin5, k_d1)

        uout = self.UNet_decoder1(uin5)
        kout = self.KiNet_encoder1(kin5)

        netout = F.relu(self.conv(torch.add(uout, kout)))
        netout = self.soft(netout)

        return netout


class smallUNet3D(nn.Module):
    # 仅限（192，256，256）大小输入
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(smallUNet3D, self).__init__()

        features = init_features
        self.encoder1 = Conv3D(in_channels, features)
        self.encoder2 = Conv3D(features, features * 2)
        self.encoder3 = Conv3D(features * 2, features * 4)
        self.encoder4 = Conv3D(features * 4, features * 8)

        self.bottleneck = Conv3D(features * 8, features * 16)
        # self.dense = nn.Linear(6144, 6144)

        self.upconv4 = deConv3D(features * 16, features * 8)
        self.upconv3 = deConv3D(features * 8 * 2, features * 4)
        self.upconv2 = deConv3D(features * 4 * 2, features * 2)
        self.upconv1 = deConv3D(features * 2 * 2, features)

        self.upconv0 = deConv3D(features * 2, out_channels)

        self.conv = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, x):
        # x = torch.nn.functional.normalize(x)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)
        # bottleneck1 = self.dense(bottleneck.view(-1, 6144))
        # bottleneck2 = torch.reshape(bottleneck1, (-1, 16, 6, 8, 8))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec0 = self.upconv0(dec1)
        outputs = self.conv(dec0)
        return self.relu(outputs)


class smallUNet2D(nn.Module):
    # 仅限（256，256）大小输入
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(smallUNet2D, self).__init__()

        features = init_features
        self.encoder1 = Conv2D(in_channels, features)
        self.encoder2 = Conv2D(features, features * 2)
        self.encoder3 = Conv2D(features * 2, features * 4)
        self.encoder4 = Conv2D(features * 4, features * 8)

        self.bottleneck = Conv2D(features * 8, features * 16)
        # self.dense = nn.Linear(6144, 6144)

        self.upconv4 = deConv2D(features * 16, features * 8)
        self.upconv3 = deConv2D(features * 8 * 2, features * 4)
        self.upconv2 = deConv2D(features * 4 * 2, features * 2)
        self.upconv1 = deConv2D(features * 2 * 2, features)

        self.upconv0 = deConv2D(features * 2, out_channels)

        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, x):
        # x = torch.nn.functional.normalize(x)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)
        # bottleneck1 = self.dense(bottleneck.view(-1, 6144))
        # bottleneck2 = torch.reshape(bottleneck1, (-1, 16, 6, 8, 8))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec0 = self.upconv0(dec1)
        outputs = self.conv(dec0)
        return self.relu(outputs)


class PAM_Module_2D(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module_2D, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class PAM_Module_3D(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module_3D, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width, deep = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height * deep).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height * deep)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height * deep)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width, deep)

        out = self.gamma * out + x
        return out


class CAM_Module_2D(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module_2D, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module_3D(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module_3D, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width, deep = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width, deep)

        out = self.gamma * out + x
        return out


class CAM_Module_3D_conv(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module_3D_conv, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.layer = nn.Sequential()
        self.layer.add_module("conv", nn.Conv3d(in_dim, in_dim // 2, 1, stride=1, padding=0, bias=False))
        self.layer.add_module("norm", normalization3D(in_dim // 2, 'bn'))
        self.layer.add_module("relu", nn.RReLU(inplace=True))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        x = self.layer(x)
        m_batchsize, C, height, width, deep = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width, deep)

        out = self.gamma * out + x
        return out


class BasicBlock(nn.Module):
    def __init__(self, inC, outC, kerSize=3, kerStride=1, kerPad=1, dropoutRate=0.5, norm="bn"):
        super(BasicBlock, self).__init__()
        self.layer = nn.Sequential()
        self.layer.add_module("conv", nn.Conv2d(inC, outC, kerSize, stride=kerStride, padding=kerPad, bias=False))
        self.layer.add_module("norm", normalization2D(outC, norm))
        self.layer.add_module("relu", nn.RReLU(inplace=True))

    def forward(self, x):
        y = torch.add(x, self.layer(x))
        return y


class BasicBlock3D(nn.Module):
    def __init__(self, inC, outC, kerSize=3, kerStride=(1, 1, 1), kerPad=1, dropoutRate=0.5, norm="bn"):
        super(BasicBlock3D, self).__init__()
        self.layer = nn.Sequential()
        self.layer.add_module("conv", nn.Conv3d(inC, outC, kerSize, stride=kerStride, padding=kerPad, bias=False))
        self.layer.add_module("norm", normalization3D(outC, norm))
        self.layer.add_module("relu", nn.RReLU(inplace=True))

    def forward(self, x):
        y = torch.add(x, self.layer(x))
        return y


class DAResBlock(nn.Module):
    def __init__(self, inC, outC, kerSize=3, kerStride=1, kerPad=1, dropoutRate=0.5, norm="bn"):
        super(DAResBlock, self).__init__()
        self.layerS = nn.Sequential()
        self.layerS.add_module("conv", nn.Conv2d(inC, outC, kerSize, stride=kerStride, padding=kerPad, bias=False))
        self.layerS.add_module("norm", normalization2D(outC, norm))
        self.layerS.add_module("relu", nn.RReLU(inplace=True))

        self.SMO = PAM_Module_2D(outC)

        self.layerS1 = nn.Sequential()
        self.layerS1.add_module("conv", nn.Conv2d(outC, outC, kerSize, stride=kerStride, padding=kerPad, bias=False))
        self.layerS1.add_module("norm", normalization2D(outC, norm))
        self.layerS1.add_module("relu", nn.RReLU(inplace=True))

        self.layerC = nn.Sequential()
        self.layerC.add_module("conv", nn.Conv2d(inC, outC, kerSize, stride=kerStride, padding=kerPad, bias=False))
        self.layerC.add_module("norm", normalization2D(outC, norm))
        self.layerC.add_module("relu", nn.RReLU(inplace=True))

        self.CMO = CAM_Module_2D(outC)

        self.layerC1 = nn.Sequential()
        self.layerC1.add_module("conv", nn.Conv2d(outC, outC, kerSize, stride=kerStride, padding=kerPad, bias=False))
        self.layerC1.add_module("norm", normalization2D(outC, norm))
        self.layerC1.add_module("relu", nn.RReLU(inplace=True))

        self.layer = nn.Sequential()
        self.layer.add_module("conv", nn.Conv2d(2 * outC, outC, kerSize, stride=kerStride, padding=kerPad, bias=False))
        self.layer.add_module("norm", normalization2D(outC, norm))
        self.layer.add_module("relu", nn.RReLU(inplace=True))

    def forward(self, x):
        sBlock = self.layerS(x)
        sBlock = self.SMO(sBlock)
        sBlock = self.layerS1(sBlock)

        cBlock = self.layerC(x)
        cBlock = self.CMO(cBlock)
        cBlock = self.layerC1(cBlock)

        block = self.layer(torch.cat((sBlock, cBlock), dim=1))

        return block


class DAResBlock3D(nn.Module):
    def __init__(self, inC, outC, kerSize=(3, 3, 3), kerStride=1, kerPad=1, dropoutRate=0.5, norm="bn"):
        super(DAResBlock3D, self).__init__()
        self.layerS = nn.Sequential()
        self.layerS.add_module("conv", nn.Conv3d(inC, inC, kerSize, stride=kerStride, padding=kerPad, bias=False))
        self.layerS.add_module("norm", normalization3D(inC, norm))
        self.layerS.add_module("relu", nn.RReLU(inplace=True))

        # self.SMO = PAM_Module_3D(outC//2)
        self.SMO = SpatialAttention3D(inC, outC // 2)

        self.layerS1 = nn.Sequential()
        self.layerS1.add_module("conv",
                                nn.Conv3d(outC // 2, outC // 2, kerSize, stride=kerStride, padding=kerPad, bias=False))
        self.layerS1.add_module("norm", normalization3D(outC // 2, norm))
        self.layerS1.add_module("relu", nn.RReLU(inplace=True))

        self.layerC = nn.Sequential()
        self.layerC.add_module("conv", nn.Conv3d(inC, inC, kerSize, stride=kerStride, padding=kerPad, bias=False))
        self.layerC.add_module("norm", normalization3D(inC, norm))
        self.layerC.add_module("relu", nn.RReLU(inplace=True))

        self.CMO = CAM_Module_3D(inC)
        # self.CMO = ChannelAttention3D(outC//2, outC//2)

        self.layerC1 = nn.Sequential()
        self.layerC1.add_module("conv",
                                nn.Conv3d(inC, outC // 2, kerSize, stride=kerStride, padding=kerPad, bias=False))
        self.layerC1.add_module("norm", normalization3D(outC // 2, norm))
        self.layerC1.add_module("relu", nn.RReLU(inplace=True))

        self.layer = nn.Sequential()
        self.layer.add_module("conv", nn.Conv3d(outC, outC, kerSize, stride=kerStride, padding=kerPad, bias=False))
        self.layer.add_module("norm", normalization3D(outC, norm))
        self.layer.add_module("relu", nn.RReLU(inplace=True))

    def forward(self, x):
        sBlock = self.layerS(x)
        sBlock = self.SMO(sBlock)
        sBlock = self.layerS1(sBlock)

        cBlock = self.layerC(x)
        cBlock = self.CMO(cBlock)
        cBlock = self.layerC1(cBlock)

        block = self.layer(torch.cat((sBlock, cBlock), dim=1))

        return block


class DAResBlock3D_(nn.Module):
    def __init__(self, inC, outC, kerSize=(3, 3, 3), kerStride=1, kerPad=1, dropoutRate=0.5, norm="bn"):
        super(DAResBlock3D_, self).__init__()
        self.layerS = nn.Sequential()
        self.layerS.add_module("conv", nn.Conv3d(inC, inC, kerSize, stride=kerStride, padding=kerPad, bias=False))
        self.layerS.add_module("norm", normalization3D(inC, norm))
        self.layerS.add_module("relu", nn.RReLU(inplace=True))

        # self.SMO = PAM_Module_3D(inC)
        self.SMO = SpatialAttention3D(inC, inC)

        self.layerS1 = nn.Sequential()
        self.layerS1.add_module("conv", nn.Conv3d(inC, outC, kerSize, stride=kerStride, padding=kerPad, bias=False))
        self.layerS1.add_module("norm", normalization3D(outC, norm))
        self.layerS1.add_module("relu", nn.RReLU(inplace=True))

    def forward(self, x):
        sBlock = self.layerS(x)
        sBlock = self.SMO(sBlock)
        sBlock = self.layerS1(sBlock)
        return sBlock


# base
# ------------------------------------------------------------------------------------------------#

class DaResUNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(DaResUNet2D, self).__init__()

        features = in_channels

        self.basicBlock = BasicBlock(in_channels, features)
        self.daresBlock = DAResBlock(features, features)

        self.encoder1 = Conv2D(in_channels, features)
        self.encoder2 = Conv2D(features, features * 2)
        self.encoder3 = Conv2D(features * 2, features * 4)
        self.encoder4 = Conv2D(features * 4, features * 8)

        self.bottleneck = Conv2D(features * 8, features * 16)

        self.upconv4 = deConv2D(features * 16, features * 8)
        self.upconv3 = deConv2D(features * 8 * 2, features * 4)
        self.upconv2 = deConv2D(features * 4 * 2, features * 2)
        self.upconv1 = deConv2D(features * 2 * 2, features)

        self.deconv = deConv2D(features * 2, out_channels)
        self.conv = Conv2D(features * 2, out_channels, kerStride=1)

    def forward(self, x):
        # DARes部分
        basicOut = self.basicBlock(x)
        daresOut = self.daresBlock(basicOut)

        # UNet部分
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.deconv(dec1)

        outputs = self.conv(torch.cat((daresOut, dec1), dim=1))
        # outputs = self.conv(dec1)

        return torch.sigmoid(outputs)


class DaResUNet3D_(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(DaResUNet3D_, self).__init__()

        features = init_features
        self.basicBlock = BasicBlock3D(in_channels, in_channels)
        self.daresBlock = DAResBlock3D(in_channels, features)

        self.encoder1 = Conv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder2 = Conv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder3 = Conv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder4 = Conv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))

        self.bottleneck = Conv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))

        self.upconv4 = deConv3D(features * 16, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv3 = deConv3D(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv2 = deConv3D(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv1 = deConv3D(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1), scale_factor=(1, 2, 2))

        self.upconv0 = deConv3D(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1), scale_factor=(1, 2, 2))

        self.conv = nn.Conv3d(out_channels + in_channels + features, out_channels, kernel_size=1, stride=1,
                              padding=0, bias=False)
        # self.conv = Conv3D(out_channels + in_channels + features, out_channels, kerStride=1)

        self.relu = nn.Sigmoid()

    def forward(self, input):
        basicOut = self.basicBlock(input)
        daresOut = self.daresBlock(basicOut)

        enc1 = self.encoder1(input)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)

        dec0 = self.upconv0(dec1)
        dec = torch.cat((dec0, input), dim=1)

        output = self.conv(torch.cat((daresOut, dec), dim=1))

        # output = self.conv(dec)

        return self.relu(output)


# ------------------------------------------------------------------------------------------------#
# brainLesion
class STN_Block_sym(nn.Module):
    def __init__(self, cin=1, cout=8):
        super(STN_Block_sym, self).__init__()

        # 空间变换器定位 - 网络
        self.localization = nn.Sequential(
            resConv3DAdd(cin, cout, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1)),
            resConv3DAdd(cout, cout, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1)),
            resConv3DAdd(cout, 2 * cout, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1)),
            resConv3DAdd(2 * cout, 4 * cout, kerSize=(3, 3, 3), kerStride=(2, 2, 2), kerPad=(1, 1, 1)),
            resConv3DAdd(4 * cout, 8 * cout, kerSize=(3, 3, 3), kerStride=(2, 2, 2), kerPad=(1, 1, 1)),
        )
        # 3 * 2 affine矩阵的回归量
        self.fc_loc = nn.Sequential(
            nn.Linear(24576, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 3)
        )

    def forward(self, input):
        xs = self.localization(input)
        xs = xs.view(-1, xs.size()[1] * xs.size()[2] * xs.size()[3] * xs.size()[4])
        torch.flatten(xs, start_dim=1)
        ref = torch.tanh(self.fc_loc(xs)) * 0.5

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # vectorX = torch.tensor([[1, 0, 0],
        #                         [0, 0, 0],
        #                         [0, 0, 0]]).to(device)
        #
        # vectorX1 = torch.tensor([[0, 0, 0],
        #                          [0, 1, 0],
        #                          [0, 0, -1]]).to(device)
        #
        # vectorX2 = torch.tensor([[0, 0, 0],
        #                          [0, 0, -1],
        #                          [0, 1, 0]]).to(device)

        vectorY = torch.tensor([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]]).to(device)

        vectorY1 = torch.tensor([[1, 0, 0],
                                 [0, 0, 0],
                                 [0, 0, 1]]).to(device)

        vectorY2 = torch.tensor([[0, 0, 1],
                                 [0, 0, 0],
                                 [-1, 0, 0]]).to(device)

        vectorZ = torch.tensor([[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 1]]).to(device)

        vectorZ1 = torch.tensor([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]]).to(device)

        vectorZ2 = torch.tensor([[0, -1, 0],
                                 [1, 0, 0],
                                 [0, 0, 0]]).to(device)

        # rotateX = vectorX + torch.mul(vectorX1.view(1, 3, 3),
        #                               torch.cos(ref[:, 0]).view(input.size()[0], 1, 1)) + torch.mul(
        #     vectorX2.view(1, 3, 3), torch.sin(ref[:, 0]).view(input.size()[0], 1, 1))
        rotateY = vectorY + torch.mul(vectorY1.view(1, 3, 3),
                                      torch.cos(ref[:, 1]).view(input.size()[0], 1, 1)) + torch.mul(
            vectorY2.view(1, 3, 3), torch.sin(ref[:, 1]).view(input.size()[0], 1, 1))
        # rotateZ = vectorZ + torch.mul(vectorZ1.view(1, 3, 3),
        #                               torch.cos(ref[:, 2]).view(input.size()[0], 1, 1)) + torch.mul(
        #     vectorZ2.view(1, 3, 3), torch.sin(ref[:, 2]).view(input.size()[0], 1, 1))

        rotateZ = vectorZ + torch.mul(vectorZ1.view(1, 3, 3),
                                      torch.cos(ref[:, 0]).view(input.size()[0], 1, 1)) + torch.mul(
            vectorZ2.view(1, 3, 3), torch.sin(ref[:, 0]).view(input.size()[0], 1, 1))

        # temp = torch.bmm(torch.bmm(rotateX, rotateY), rotateZ).view(input.size()[0], 3, 3)
        temp = torch.bmm(rotateY, rotateZ).view(input.size()[0], 3, 3)
        # temp = rotateZ.view(input.size()[0], 3, 3)

        trans1 = torch.tensor([[1],
                               [0],
                               [0]]).to(device)
        # trans2 = torch.tensor([[0],
        #                        [1],
        #                        [0]]).to(device)
        # trans3 = torch.tensor([[0],
        #                        [0],
        #                        [1]]).to(device)
        # move = torch.mul(trans1.view(1, 3, 1), ref[:, 3].view(input.size()[0], 1, 1)) \
        #        + torch.mul(trans2.view(1, 3, 1), ref[:, 4].view(input.size()[0], 1, 1)) \
        #        + torch.mul(trans3.view(1, 3, 1), ref[:, 5].view(input.size()[0], 1, 1))

        move = torch.mul(trans1.view(1, 3, 1), ref[:, 2].view(input.size()[0], 1, 1))

        theta = torch.cat([temp, move], dim=2)
        grid = F.affine_grid(theta, input.size(), align_corners=True)
        netout = F.grid_sample(input, grid, align_corners=True)

        return theta, netout


class STN_Block_sym_small(nn.Module):
    def __init__(self, cin=1, cout=2):
        super(STN_Block_sym_small, self).__init__()

        # 空间变换器定位 - 网络
        self.localization = nn.Sequential(
            Conv3D_0(cin, cout, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1)),
            Conv3D_0(cout, cout, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1)),
            Conv3D_0(cout, 2 * cout, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1)),
            Conv3D_0(2 * cout, 4 * cout, kerSize=(3, 3, 3), kerStride=(2, 2, 2), kerPad=(1, 1, 1)),
            Conv3D_0(4 * cout, 8 * cout, kerSize=(3, 3, 3), kerStride=(2, 2, 2), kerPad=(1, 1, 1)),
        )
        # 3 * 2 affine矩阵的回归量
        self.fc_loc = nn.Sequential(
            nn.Linear(3 * 4 * 4 * cout * 8, 256),
            nn.ReLU(True),
            nn.Linear(256, 2)
        )

    def forward(self, input):
        input = torch.max_pool3d(input, kernel_size=1, padding=0, stride=(2, 2, 2))
        xs = self.localization(input)
        xs = xs.view(-1, xs.size()[1] * xs.size()[2] * xs.size()[3] * xs.size()[4])
        torch.flatten(xs, start_dim=1)
        ref = torch.tanh(self.fc_loc(xs)) * 0.5

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # vectorX = torch.tensor([[1, 0, 0],
        #                         [0, 0, 0],
        #                         [0, 0, 0]]).to(device)
        #
        # vectorX1 = torch.tensor([[0, 0, 0],
        #                          [0, 1, 0],
        #                          [0, 0, -1]]).to(device)
        #
        # vectorX2 = torch.tensor([[0, 0, 0],
        #                          [0, 0, -1],
        #                          [0, 1, 0]]).to(device)

        vectorY = torch.tensor([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]]).to(device)

        vectorY1 = torch.tensor([[1, 0, 0],
                                 [0, 0, 0],
                                 [0, 0, 1]]).to(device)

        vectorY2 = torch.tensor([[0, 0, 1],
                                 [0, 0, 0],
                                 [-1, 0, 0]]).to(device)

        vectorZ = torch.tensor([[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 1]]).to(device)

        vectorZ1 = torch.tensor([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]]).to(device)

        vectorZ2 = torch.tensor([[0, -1, 0],
                                 [1, 0, 0],
                                 [0, 0, 0]]).to(device)

        # rotateX = vectorX + torch.mul(vectorX1.view(1, 3, 3),
        #                               torch.cos(ref[:, 0]).view(input.size()[0], 1, 1)) + torch.mul(
        #     vectorX2.view(1, 3, 3), torch.sin(ref[:, 0]).view(input.size()[0], 1, 1))
        # rotateY = vectorY + torch.mul(vectorY1.view(1, 3, 3),
        #                               torch.cos(ref[:, 1]).view(input.size()[0], 1, 1)) + torch.mul(
        #     vectorY2.view(1, 3, 3), torch.sin(ref[:, 1]).view(input.size()[0], 1, 1))
        # rotateZ = vectorZ + torch.mul(vectorZ1.view(1, 3, 3),
        #                               torch.cos(ref[:, 2]).view(input.size()[0], 1, 1)) + torch.mul(
        #     vectorZ2.view(1, 3, 3), torch.sin(ref[:, 2]).view(input.size()[0], 1, 1))

        rotateY = vectorY + torch.mul(vectorY1.view(1, 3, 3),
                                      torch.cos(ref[:, 0]).view(input.size()[0], 1, 1)) + torch.mul(
            vectorY2.view(1, 3, 3), torch.sin(ref[:, 0]).view(input.size()[0], 1, 1))
        rotateZ = vectorZ + torch.mul(vectorZ1.view(1, 3, 3),
                                      torch.cos(ref[:, 1]).view(input.size()[0], 1, 1)) + torch.mul(
            vectorZ2.view(1, 3, 3), torch.sin(ref[:, 1]).view(input.size()[0], 1, 1))

        # rotateZ = vectorZ + torch.mul(vectorZ1.view(1, 3, 3),
        #                               torch.cos(ref[:, 0]).view(input.size()[0], 1, 1)) + torch.mul(
        #     vectorZ2.view(1, 3, 3), torch.sin(ref[:, 0]).view(input.size()[0], 1, 1))

        # temp = torch.bmm(torch.bmm(rotateX, rotateY), rotateZ).view(input.size()[0], 3, 3)
        temp = torch.bmm(rotateY, rotateZ).view(input.size()[0], 3, 3)
        # temp = rotateZ.view(input.size()[0], 3, 3)

        trans1 = torch.tensor([[0],
                               [0],
                               [0]]).to(device)

        # trans1 = torch.tensor([[1],
        #                        [0],
        #                        [0]]).to(device)
        # trans2 = torch.tensor([[0],
        #                        [1],
        #                        [0]]).to(device)
        # trans3 = torch.tensor([[0],
        #                        [0],
        #                        [1]]).to(device)
        # move = torch.mul(trans1.view(1, 3, 1), ref[:, 3].view(input.size()[0], 1, 1)) \
        #        + torch.mul(trans2.view(1, 3, 1), ref[:, 4].view(input.size()[0], 1, 1)) \
        #        + torch.mul(trans3.view(1, 3, 1), ref[:, 5].view(input.size()[0], 1, 1))

        move = torch.mul(trans1.view(1, 3, 1), ref[:, 1].view(input.size()[0], 1, 1))

        theta = torch.cat([temp, move], dim=2)
        grid = F.affine_grid(theta, input.size(), align_corners=True)
        netout = F.grid_sample(input, grid, align_corners=True)

        return theta, netout


class STN_Block_sym_ncct(nn.Module):
    def __init__(self, cin=1, cout=2):
        super(STN_Block_sym_ncct, self).__init__()

        # 空间变换器定位 - 网络
        self.localization = nn.Sequential(
            Conv3D_0(cin, cout, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1)),
            Conv3D_0(cout, cout, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1)),
            Conv3D_0(cout, 2 * cout, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1)),
            Conv3D_0(2 * cout, 4 * cout, kerSize=(3, 3, 3), kerStride=(2, 2, 2), kerPad=(1, 1, 1)),
            Conv3D_0(4 * cout, 8 * cout, kerSize=(3, 3, 3), kerStride=(2, 2, 2), kerPad=(1, 1, 1)),
        )
        # 3 * 2 affine矩阵的回归量
        self.fc_loc = nn.Sequential(
            nn.Linear(2 * 2 * 2 * cout * 8, 64),
            nn.Dropout(p=0.25, inplace=True),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, input):
        input = torch.max_pool3d(input, kernel_size=1, padding=0, stride=(4, 4, 4))
        xs = self.localization(input)
        xs = xs.view(-1, xs.size()[1] * xs.size()[2] * xs.size()[3] * xs.size()[4])
        torch.flatten(xs, start_dim=1)
        ref = torch.tanh(self.fc_loc(xs)) * 2

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # vectorX = torch.tensor([[1, 0, 0],
        #                         [0, 0, 0],
        #                         [0, 0, 0]]).to(device)
        #
        # vectorX1 = torch.tensor([[0, 0, 0],
        #                          [0, 1, 0],
        #                          [0, 0, -1]]).to(device)
        #
        # vectorX2 = torch.tensor([[0, 0, 0],
        #                          [0, 0, -1],
        #                          [0, 1, 0]]).to(device)

        # vectorY = torch.tensor([[0, 0, 0],
        #                         [0, 1, 0],
        #                         [0, 0, 0]]).to(device)
        #
        # vectorY1 = torch.tensor([[1, 0, 0],
        #                          [0, 0, 0],
        #                          [0, 0, 1]]).to(device)
        #
        # vectorY2 = torch.tensor([[0, 0, 1],
        #                          [0, 0, 0],
        #                          [-1, 0, 0]]).to(device)

        vectorZ = torch.tensor([[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 1]]).to(device)

        vectorZ1 = torch.tensor([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]]).to(device)

        vectorZ2 = torch.tensor([[0, -1, 0],
                                 [1, 0, 0],
                                 [0, 0, 0]]).to(device)

        # rotateX = vectorX + torch.mul(vectorX1.view(1, 3, 3),
        #                               torch.cos(ref[:, 0]).view(input.size()[0], 1, 1)) + torch.mul(
        #     vectorX2.view(1, 3, 3), torch.sin(ref[:, 0]).view(input.size()[0], 1, 1))
        # rotateY = vectorY + torch.mul(vectorY1.view(1, 3, 3),
        #                               torch.cos(ref[:, 1]).view(input.size()[0], 1, 1)) + torch.mul(
        #     vectorY2.view(1, 3, 3), torch.sin(ref[:, 1]).view(input.size()[0], 1, 1))
        # rotateZ = vectorZ + torch.mul(vectorZ1.view(1, 3, 3),
        #                               torch.cos(ref[:, 2]).view(input.size()[0], 1, 1)) + torch.mul(
        #     vectorZ2.view(1, 3, 3), torch.sin(ref[:, 2]).view(input.size()[0], 1, 1))

        # rotateY = vectorY + torch.mul(vectorY1.view(1, 3, 3),
        #                               torch.cos(ref[:, 0]).view(input.size()[0], 1, 1)) + torch.mul(
        #     vectorY2.view(1, 3, 3), torch.sin(ref[:, 0]).view(input.size()[0], 1, 1))
        # rotateZ = vectorZ + torch.mul(vectorZ1.view(1, 3, 3),
        #                               torch.cos(ref[:, 1]).view(input.size()[0], 1, 1)) + torch.mul(
        #     vectorZ2.view(1, 3, 3), torch.sin(ref[:, 1]).view(input.size()[0], 1, 1))

        rotateZ = vectorZ + torch.mul(vectorZ1.view(1, 3, 3),
                                      torch.cos(ref[:, 0]).view(input.size()[0], 1, 1)) + torch.mul(
            vectorZ2.view(1, 3, 3), torch.sin(ref[:, 0]).view(input.size()[0], 1, 1))

        # temp = torch.bmm(torch.bmm(rotateX, rotateY), rotateZ).view(input.size()[0], 3, 3)
        # temp = torch.bmm(rotateY, rotateZ).view(input.size()[0], 3, 3)
        temp = rotateZ.view(input.size()[0], 3, 3)

        trans1 = torch.tensor([[0.25 / 2],
                               [0],
                               [0]]).to(device)

        # trans1 = torch.tensor([[1],
        #                        [0],
        #                        [0]]).to(device)
        # trans2 = torch.tensor([[0],
        #                        [1],
        #                        [0]]).to(device)
        # trans3 = torch.tensor([[0],
        #                        [0],
        #                        [1]]).to(device)
        # move = torch.mul(trans1.view(1, 3, 1), ref[:, 3].view(input.size()[0], 1, 1)) \
        #        + torch.mul(trans2.view(1, 3, 1), ref[:, 4].view(input.size()[0], 1, 1)) \
        #        + torch.mul(trans3.view(1, 3, 1), ref[:, 5].view(input.size()[0], 1, 1))

        move = torch.mul(trans1.view(1, 3, 1), ref[:, 1].view(input.size()[0], 1, 1))

        theta = torch.cat([temp, move], dim=2)
        grid = F.affine_grid(theta, input.size(), align_corners=True)
        netout = F.grid_sample(input, grid, align_corners=True)

        return theta, netout


class STN_Block_sym_small_bak(nn.Module):
    def __init__(self, cin=1, cout=4):
        super(STN_Block_sym_small_bak, self).__init__()

        # 空间变换器定位 - 网络
        self.localization = nn.Sequential(
            Conv3D_0(cin, cout, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1)),
            Conv3D_0(cout, cout, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1)),
            Conv3D_0(cout, 2 * cout, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1)),
            Conv3D_0(2 * cout, 4 * cout, kerSize=(3, 3, 3), kerStride=(2, 2, 2), kerPad=(1, 1, 1)),
            Conv3D_0(4 * cout, 8 * cout, kerSize=(3, 3, 3), kerStride=(2, 2, 2), kerPad=(1, 1, 1)),
        )
        # 3 * 2 affine矩阵的回归量
        self.fc_loc = nn.Sequential(
            nn.Linear(3 * 8 * 8 * cout * 8, 256),
            nn.ReLU(True),
            nn.Linear(256, 3)
        )

    def forward(self, input):
        xs = self.localization(input)
        xs = xs.view(-1, xs.size()[1] * xs.size()[2] * xs.size()[3] * xs.size()[4])
        torch.flatten(xs, start_dim=1)
        ref = torch.tanh(self.fc_loc(xs)) * 0.5

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # vectorX = torch.tensor([[1, 0, 0],
        #                         [0, 0, 0],
        #                         [0, 0, 0]]).to(device)
        #
        # vectorX1 = torch.tensor([[0, 0, 0],
        #                          [0, 1, 0],
        #                          [0, 0, -1]]).to(device)
        #
        # vectorX2 = torch.tensor([[0, 0, 0],
        #                          [0, 0, -1],
        #                          [0, 1, 0]]).to(device)

        vectorY = torch.tensor([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]]).to(device)

        vectorY1 = torch.tensor([[1, 0, 0],
                                 [0, 0, 0],
                                 [0, 0, 1]]).to(device)

        vectorY2 = torch.tensor([[0, 0, 1],
                                 [0, 0, 0],
                                 [-1, 0, 0]]).to(device)

        vectorZ = torch.tensor([[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 1]]).to(device)

        vectorZ1 = torch.tensor([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]]).to(device)

        vectorZ2 = torch.tensor([[0, -1, 0],
                                 [1, 0, 0],
                                 [0, 0, 0]]).to(device)

        # rotateX = vectorX + torch.mul(vectorX1.view(1, 3, 3),
        #                               torch.cos(ref[:, 0]).view(input.size()[0], 1, 1)) + torch.mul(
        #     vectorX2.view(1, 3, 3), torch.sin(ref[:, 0]).view(input.size()[0], 1, 1))
        rotateY = vectorY + torch.mul(vectorY1.view(1, 3, 3),
                                      torch.cos(ref[:, 1]).view(input.size()[0], 1, 1)) + torch.mul(
            vectorY2.view(1, 3, 3), torch.sin(ref[:, 1]).view(input.size()[0], 1, 1))
        # rotateZ = vectorZ + torch.mul(vectorZ1.view(1, 3, 3),
        #                               torch.cos(ref[:, 2]).view(input.size()[0], 1, 1)) + torch.mul(
        #     vectorZ2.view(1, 3, 3), torch.sin(ref[:, 2]).view(input.size()[0], 1, 1))

        rotateZ = vectorZ + torch.mul(vectorZ1.view(1, 3, 3),
                                      torch.cos(ref[:, 0]).view(input.size()[0], 1, 1)) + torch.mul(
            vectorZ2.view(1, 3, 3), torch.sin(ref[:, 0]).view(input.size()[0], 1, 1))

        # temp = torch.bmm(torch.bmm(rotateX, rotateY), rotateZ).view(input.size()[0], 3, 3)
        temp = torch.bmm(rotateY, rotateZ).view(input.size()[0], 3, 3)
        # temp = rotateZ.view(input.size()[0], 3, 3)

        trans1 = torch.tensor([[1],
                               [0],
                               [0]]).to(device)
        # trans2 = torch.tensor([[0],
        #                        [1],
        #                        [0]]).to(device)
        # trans3 = torch.tensor([[0],
        #                        [0],
        #                        [1]]).to(device)
        # move = torch.mul(trans1.view(1, 3, 1), ref[:, 3].view(input.size()[0], 1, 1)) \
        #        + torch.mul(trans2.view(1, 3, 1), ref[:, 4].view(input.size()[0], 1, 1)) \
        #        + torch.mul(trans3.view(1, 3, 1), ref[:, 5].view(input.size()[0], 1, 1))

        move = torch.mul(trans1.view(1, 3, 1), ref[:, 2].view(input.size()[0], 1, 1))

        theta = torch.cat([temp, move], dim=2)
        grid = F.affine_grid(theta, input.size(), align_corners=True)
        netout = F.grid_sample(input, grid, align_corners=True)

        return theta, netout


class mSTN_uNet_sym_solo(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet_sym_solo, self).__init__()

        features = init_features
        self.encoder1 = resConv3DAdd(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder2 = resConv3DAdd(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder3 = resConv3DAdd(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 2, 2),
                                     kerPad=(1, 1, 1))
        self.encoder4 = resConv3DAdd(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 2, 2),
                                     kerPad=(1, 1, 1))

        self.bottleneck = resConv3DAdd(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 2, 2),
                                       kerPad=(1, 1, 1))
        self.bn = normalization3D(features * 16, "bn")
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        self.upconv4 = resdeConv3DTAdd(features * 16 * 2, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))
        self.upconv3 = resdeConv3DTAdd(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))
        self.upconv2 = resdeConv3DTAdd(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))
        self.upconv1 = resdeConv3DTAdd(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))

        self.upconv0 = resdeConv3DTAdd(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        # enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        # enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        # enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        # enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        # bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        temp = self.global_pool(bottleneck)
        temp = self.bn(temp)
        bottleneck = torch.cat((bottleneck, temp), dim=1)

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)

        dec0 = self.upconv0(dec1)
        dec = torch.cat((dec0, input), dim=1)

        output = self.conv(dec)
        return self.relu(output)


class mSTN_uNet_sym_solo_CP(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet_sym_solo_CP, self).__init__()

        features = init_features
        self.encoder1 = resConv3DAdd(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder2 = resConv3DAdd(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder3 = resConv3DAdd(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 2, 2),
                                     kerPad=(1, 1, 1))
        self.encoder4 = resConv3DAdd(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 2, 2),
                                     kerPad=(1, 1, 1))

        self.bottleneck = resConv3DAdd(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 2, 2),
                                       kerPad=(1, 1, 1))
        # self.bn = normalization3D(features * 16, "bn")
        # self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        self.upconv4 = resdeConv3DTAdd(features * 16, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))
        self.upconv3 = resdeConv3DTAdd(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))
        self.upconv2 = resdeConv3DTAdd(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))
        self.upconv1 = resdeConv3DTAdd(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))

        self.upconv0 = resdeConv3DTAdd(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))

        self.CP4 = CAM_Module_3D(features * 16)
        self.CP3 = CAM_Module_3D(features * 8)
        self.CP2 = CAM_Module_3D(features * 4)
        self.CP1 = CAM_Module_3D(features * 2)
        self.CP0 = CAM_Module_3D(out_channels + in_channels)

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        # enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        # enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        # enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        # enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        # bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        # temp = self.global_pool(bottleneck)
        # temp = self.bn(temp)
        # bottleneck = torch.cat((bottleneck, temp), dim=1)

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)
        dec4 = self.CP4(dec4)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)
        dec3 = self.CP3(dec3)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)
        dec2 = self.CP2(dec2)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)
        dec1 = self.CP1(dec1)

        dec0 = self.upconv0(dec1)
        dec0 = torch.cat((dec0, input), dim=1)
        dec0 = self.CP0(dec0)

        output = self.conv(dec0)
        return self.relu(output)


class mSTN_uNet_sym_solo_CP2(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet_sym_solo_CP2, self).__init__()

        features = init_features
        self.encoder1 = resConv3DAdd(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder2 = resConv3DAdd(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder3 = resConv3DAdd(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 2, 2),
                                     kerPad=(1, 1, 1))
        self.encoder4 = resConv3DAdd(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 2, 2),
                                     kerPad=(1, 1, 1))

        self.bottleneck = resConv3DAdd(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 2, 2),
                                       kerPad=(1, 1, 1))
        self.bn = normalization3D(features * 16, "bn")
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        self.upconv4 = resdeConv3DTAdd(features * 16 * 2, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))
        self.upconv3 = resdeConv3DTAdd(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))
        self.upconv2 = resdeConv3DTAdd(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))
        self.upconv1 = resdeConv3DTAdd(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))

        self.upconv0 = resdeConv3DTAdd(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))

        self.CP4 = CAM_Module_3D(features * 16)
        self.CP3 = CAM_Module_3D(features * 8)
        self.CP2 = CAM_Module_3D(features * 4)
        self.CP1 = CAM_Module_3D(features * 2)
        self.CP0 = CAM_Module_3D(out_channels + in_channels)

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        # enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        # enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        # enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        # enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        # bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        temp = self.global_pool(bottleneck)
        temp = self.bn(temp)
        bottleneck = torch.cat((bottleneck, temp), dim=1)

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)
        dec4 = self.CP4(dec4)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)
        dec3 = self.CP3(dec3)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)
        dec2 = self.CP2(dec2)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)
        dec1 = self.CP1(dec1)

        dec0 = self.upconv0(dec1)
        dec0 = torch.cat((dec0, input), dim=1)
        dec0 = self.CP0(dec0)

        output = self.conv(dec0)
        return self.relu(output)


class symDiffNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(symDiffNet, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))
        self.bn = normalization3D(features * 16, "bn")
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        self.upconv4 = resdeConv3DT(features * 16 * 2, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                    scale_factor=(1, 2, 2))
        self.upconv3 = resdeConv3DT(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                    scale_factor=(1, 2, 2))
        self.upconv2 = resdeConv3DT(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                    scale_factor=(1, 2, 2))
        self.upconv1 = resdeConv3DT(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                    scale_factor=(1, 2, 2))

        self.upconv0 = resdeConv3DT(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                    scale_factor=(1, 2, 2))

        self.CP4 = CAM_Module_3D(features * 16)
        self.CP3 = CAM_Module_3D(features * 8)
        self.CP2 = CAM_Module_3D(features * 4)
        self.CP1 = CAM_Module_3D(features * 2)
        self.CP0 = CAM_Module_3D(out_channels + in_channels)

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        temp = self.global_pool(bottleneck)
        temp = self.bn(temp)
        bottleneck = torch.cat((bottleneck, temp), dim=1)

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)
        dec4 = self.CP4(dec4)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)
        dec3 = self.CP3(dec3)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)
        dec2 = self.CP2(dec2)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)
        dec1 = self.CP1(dec1)

        dec0 = self.upconv0(dec1)
        dec0 = torch.cat((dec0, input), dim=1)
        dec0 = self.CP0(dec0)

        output = self.conv(dec0)
        return self.relu(output)


class symDiffNet_(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(symDiffNet_, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))
        self.bn = normalization3D(features * 16, "bn")
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        self.upconv4 = resdeConv3D(features * 16 * 2, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                   scale_factor=(1, 2, 2))
        self.upconv3 = resdeConv3D(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                   scale_factor=(1, 2, 2))
        self.upconv2 = resdeConv3D(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                   scale_factor=(1, 2, 2))
        self.upconv1 = resdeConv3D(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                   scale_factor=(1, 2, 2))

        self.upconv0 = resdeConv3D(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                   scale_factor=(1, 2, 2))

        self.CP4 = CAM_Module_3D(features * 16)
        self.CP3 = CAM_Module_3D(features * 8)
        self.CP2 = CAM_Module_3D(features * 4)
        self.CP1 = CAM_Module_3D(features * 2)
        self.CP0 = CAM_Module_3D(out_channels + in_channels)

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                              bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        temp = self.global_pool(bottleneck)
        temp = self.bn(temp)
        bottleneck = torch.cat((bottleneck, temp), dim=1)

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)
        dec4 = self.CP4(dec4)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)
        dec3 = self.CP3(dec3)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)
        dec2 = self.CP2(dec2)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)
        dec1 = self.CP1(dec1)

        dec0 = self.upconv0(dec1)
        dec0 = torch.cat((dec0, input), dim=1)
        dec0 = self.CP0(dec0)

        output = self.conv(dec0)
        return self.relu(output)


class symDiffNet__(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(symDiffNet__, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))
        self.bn = normalization3D(features * 16, "bn")
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        self.upconv4 = deConv3DT(features * 16 * 2, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv3 = deConv3DT(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv2 = deConv3DT(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv1 = deConv3DT(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.upconv0 = deConv3DT(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.CP4 = CAM_Module_3D(features * 16)
        self.CP3 = CAM_Module_3D(features * 8)
        self.CP2 = CAM_Module_3D(features * 4)
        self.CP1 = CAM_Module_3D(features * 2)
        self.CP0 = CAM_Module_3D(out_channels + in_channels)

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                              bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        temp = self.global_pool(bottleneck)
        temp = self.bn(temp)
        bottleneck = torch.cat((bottleneck, temp), dim=1)

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)
        dec4 = self.CP4(dec4)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)
        dec3 = self.CP3(dec3)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)
        dec2 = self.CP2(dec2)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)
        dec1 = self.CP1(dec1)

        dec0 = self.upconv0(dec1)
        dec0 = torch.cat((dec0, input), dim=1)
        dec0 = self.CP0(dec0)

        output = self.conv(dec0)
        return self.relu(output)


class symDiffNet__Globepool(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(symDiffNet__Globepool, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))

        self.upconv4 = deConv3DT(features * 16, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv3 = deConv3DT(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv2 = deConv3DT(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv1 = deConv3DT(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.upconv0 = deConv3DT(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.CP4 = CAM_Module_3D(features * 16)
        self.CP3 = CAM_Module_3D(features * 8)
        self.CP2 = CAM_Module_3D(features * 4)
        self.CP1 = CAM_Module_3D(features * 2)
        self.CP0 = CAM_Module_3D(out_channels + in_channels)

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                              bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)
        dec4 = self.CP4(dec4)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)
        dec3 = self.CP3(dec3)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)
        dec2 = self.CP2(dec2)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)
        dec1 = self.CP1(dec1)

        dec0 = self.upconv0(dec1)
        dec0 = torch.cat((dec0, input), dim=1)
        dec0 = self.CP0(dec0)

        output = self.conv(dec0)
        return self.relu(output)


class symDiffNet__Flip_CA(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(symDiffNet__Flip_CA, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))
        self.bn = normalization3D(features * 16, "bn")
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        self.upconv4 = deConv3DT(features * 16 * 2, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv3 = deConv3DT(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv2 = deConv3DT(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv1 = deConv3DT(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.upconv0 = deConv3DT(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                              bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        temp = self.global_pool(bottleneck)
        temp = self.bn(temp)
        bottleneck = torch.cat((bottleneck, temp), dim=1)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)

        dec0 = self.upconv0(dec1)
        dec0 = torch.cat((dec0, input), dim=1)

        output = self.conv(dec0)
        return self.relu(output)


class symDiffNet___CA(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(symDiffNet___CA, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))
        self.bn = normalization3D(features * 16, "bn")
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        self.upconv4 = deConv3DT(features * 16 * 2, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv3 = deConv3DT(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv2 = deConv3DT(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv1 = deConv3DT(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.upconv0 = deConv3DT(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                              bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        temp = self.global_pool(bottleneck)
        temp = self.bn(temp)
        bottleneck = torch.cat((bottleneck, temp), dim=1)

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)

        dec0 = self.upconv0(dec1)
        dec0 = torch.cat((dec0, input), dim=1)

        output = self.conv(dec0)
        return self.relu(output)


class symDiffNet__Globepool_CA(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(symDiffNet__Globepool_CA, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))

        self.upconv4 = deConv3DT(features * 16, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv3 = deConv3DT(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv2 = deConv3DT(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv1 = deConv3DT(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.upconv0 = deConv3DT(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                              bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)

        dec0 = self.upconv0(dec1)
        dec0 = torch.cat((dec0, input), dim=1)

        output = self.conv(dec0)
        return self.relu(output)


class symDiffNet___(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(symDiffNet___, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))
        self.bn = normalization3D(features * 16, "bn")
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        self.upconv4 = deConv3D(features * 16 * 2, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv3 = deConv3D(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv2 = deConv3D(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv1 = deConv3D(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))

        self.upconv0 = deConv3D(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))

        self.CP4 = CAM_Module_3D(features * 16)
        self.CP3 = CAM_Module_3D(features * 8)
        self.CP2 = CAM_Module_3D(features * 4)
        self.CP1 = CAM_Module_3D(features * 2)
        self.CP0 = CAM_Module_3D(out_channels + in_channels)

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                              bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        temp = self.global_pool(bottleneck)
        temp = self.bn(temp)
        bottleneck = torch.cat((bottleneck, temp), dim=1)

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)
        dec4 = self.CP4(dec4)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)
        dec3 = self.CP3(dec3)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)
        dec2 = self.CP2(dec2)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)
        dec1 = self.CP1(dec1)

        dec0 = self.upconv0(dec1)
        dec0 = torch.cat((dec0, input), dim=1)
        dec0 = self.CP0(dec0)

        output = self.conv(dec0)
        return self.relu(output)


class mSTN_uNet_sym_solo_CP_UP(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet_sym_solo_CP_UP, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))
        self.bn = normalization3D(features * 16, "bn")
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        self.upconv4 = deConv3DT(features * 16 * 2, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv3 = deConv3DT(features * 8, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv2 = deConv3DT(features * 4, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv1 = deConv3DT(features * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.upconv0 = deConv3DT(features, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.CP4 = CAM_Module_3D_conv(features * 16)
        self.CP3 = CAM_Module_3D_conv(features * 8)
        self.CP2 = CAM_Module_3D_conv(features * 4)
        self.CP1 = CAM_Module_3D_conv(features * 2)
        self.CP0 = CAM_Module_3D_conv(out_channels + in_channels)

        self.UP4 = upSample_ChangeChannel_3D(features * 16 // 2, out_channels, scale_factor=(1, 16, 16))
        self.UP3 = upSample_ChangeChannel_3D(features * 8 // 2, out_channels, scale_factor=(1, 8, 8))
        self.UP2 = upSample_ChangeChannel_3D(features * 4 // 2, out_channels, scale_factor=(1, 4, 4))
        self.UP1 = upSample_ChangeChannel_3D(features * 2 // 2, out_channels, scale_factor=(1, 2, 2))
        self.UP0 = upSample_ChangeChannel_3D((out_channels + in_channels) // 2, out_channels, scale_factor=(1, 1, 1))

        self.conv = nn.Conv3d(5 * out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        temp = self.global_pool(bottleneck)
        temp = self.bn(temp)
        bottleneck = torch.cat((bottleneck, temp), dim=1)

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)
        dec4 = self.CP4(dec4)

        dec3 = self.upconv3(dec4)
        dec4 = self.UP4(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)
        dec3 = self.CP3(dec3)

        dec2 = self.upconv2(dec3)
        dec3 = self.UP3(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)
        dec2 = self.CP2(dec2)

        dec1 = self.upconv1(dec2)
        dec2 = self.UP2(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)
        dec1 = self.CP1(dec1)

        dec0 = self.upconv0(dec1)
        dec1 = self.UP1(dec1)
        dec0 = torch.cat((dec0, input), dim=1)
        dec0 = self.CP0(dec0)
        dec0 = self.UP0(dec0)

        dec = torch.cat((dec4, dec3, dec2, dec1, dec0), dim=1)
        # dec = torch.cat((dec4, dec3, dec2, dec1, dec0, input), dim=1)

        output = self.conv(dec)
        return self.relu(output)


class mSTN_uNet_sym_solo_CP_UP_ncct(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet_sym_solo_CP_UP_ncct, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))
        self.bn = normalization3D(features * 16, "bn")
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        self.upconv4 = deConv3DT(features * 16 * 2, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv3 = deConv3DT(features * 8, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv2 = deConv3DT(features * 4, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv1 = deConv3DT(features * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.upconv0 = deConv3DT(features, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.CP4 = CAM_Module_3D_conv(features * 16)
        self.CP3 = CAM_Module_3D_conv(features * 8)
        self.CP2 = CAM_Module_3D_conv(features * 4)
        self.CP1 = CAM_Module_3D_conv(features * 2)
        self.CP0 = CAM_Module_3D_conv(out_channels + in_channels)

        self.UP4 = upSample_ChangeChannel_3D(features * 16 // 2, out_channels, scale_factor=(1, 16, 16))
        self.UP3 = upSample_ChangeChannel_3D(features * 8 // 2, out_channels, scale_factor=(1, 8, 8))
        self.UP2 = upSample_ChangeChannel_3D(features * 4 // 2, out_channels, scale_factor=(1, 4, 4))
        self.UP1 = upSample_ChangeChannel_3D(features * 2 // 2, out_channels, scale_factor=(1, 2, 2))
        self.UP0 = upSample_ChangeChannel_3D((out_channels + in_channels) // 2, out_channels, scale_factor=(1, 1, 1))

        self.conv = nn.Conv3d(5 * out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                              bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        temp = self.global_pool(bottleneck)
        temp = self.bn(temp)
        bottleneck = torch.cat((bottleneck, temp), dim=1)

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)
        dec4 = self.CP4(dec4)

        dec3 = self.upconv3(dec4)
        dec4 = self.UP4(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)
        dec3 = self.CP3(dec3)

        dec2 = self.upconv2(dec3)
        dec3 = self.UP3(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)
        dec2 = self.CP2(dec2)

        dec1 = self.upconv1(dec2)
        dec2 = self.UP2(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)
        dec1 = self.CP1(dec1)

        dec0 = self.upconv0(dec1)
        dec1 = self.UP1(dec1)
        dec0 = torch.cat((dec0, input), dim=1)
        dec0 = self.CP0(dec0)
        dec0 = self.UP0(dec0)

        dec = torch.cat((dec4, dec3, dec2, dec1, dec0, input), dim=1)

        output = self.conv(dec)
        return self.relu(output)


class mSTN_uNet_sym_solo_CP_UP_ncct_real(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet_sym_solo_CP_UP_ncct_real, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))
        self.bn = normalization3D(features * 16, "bn")
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        self.upconv4 = deConv3DT(features * 16 * 2, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv3 = deConv3DT(features * 8, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv2 = deConv3DT(features * 4, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv1 = deConv3DT(features * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.upconv0 = deConv3DT(features, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.CP4 = CAM_Module_3D_conv(features * 16)
        self.CP3 = CAM_Module_3D_conv(features * 8)
        self.CP2 = CAM_Module_3D_conv(features * 4)
        self.CP1 = CAM_Module_3D_conv(features * 2)
        self.CP0 = CAM_Module_3D_conv(out_channels + in_channels)

        self.UP4 = upSample_ChangeChannel_3D(features * 16 // 2, out_channels, scale_factor=(1, 16, 16))
        self.UP3 = upSample_ChangeChannel_3D(features * 8 // 2, out_channels, scale_factor=(1, 8, 8))
        self.UP2 = upSample_ChangeChannel_3D(features * 4 // 2, out_channels, scale_factor=(1, 4, 4))
        self.UP1 = upSample_ChangeChannel_3D(features * 2 // 2, out_channels, scale_factor=(1, 2, 2))
        self.UP0 = upSample_ChangeChannel_3D((out_channels + in_channels) // 2, out_channels, scale_factor=(1, 1, 1))

        self.conv = nn.Conv3d(5 * out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                              bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        temp = self.global_pool(bottleneck)
        temp = self.bn(temp)
        bottleneck = torch.cat((bottleneck, temp), dim=1)

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [4])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)
        dec4 = self.CP4(dec4)

        dec3 = self.upconv3(dec4)
        dec4 = self.UP4(dec4)
        enc3FLIP = torch.flip(enc3, [4])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)
        dec3 = self.CP3(dec3)

        dec2 = self.upconv2(dec3)
        dec3 = self.UP3(dec3)
        enc2FLIP = torch.flip(enc2, [4])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)
        dec2 = self.CP2(dec2)

        dec1 = self.upconv1(dec2)
        dec2 = self.UP2(dec2)
        enc1FLIP = torch.flip(enc1, [4])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)
        dec1 = self.CP1(dec1)

        dec0 = self.upconv0(dec1)
        dec1 = self.UP1(dec1)
        dec0 = torch.cat((dec0, input), dim=1)
        dec0 = self.CP0(dec0)
        dec0 = self.UP0(dec0)

        dec = torch.cat((dec4, dec3, dec2, dec1, dec0, input), dim=1)

        output = self.relu(self.conv(dec))
        return output


class mSTN_uNet_sym_solo_CP_UP_gb(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet_sym_solo_CP_UP_gb, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))

        self.upconv4 = deConv3DT(features * 16, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv3 = deConv3DT(features * 8, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv2 = deConv3DT(features * 4, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv1 = deConv3DT(features * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.upconv0 = deConv3DT(features, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.CP4 = CAM_Module_3D_conv(features * 16)
        self.CP3 = CAM_Module_3D_conv(features * 8)
        self.CP2 = CAM_Module_3D_conv(features * 4)
        self.CP1 = CAM_Module_3D_conv(features * 2)
        self.CP0 = CAM_Module_3D_conv(out_channels + in_channels)

        self.UP4 = upSample_ChangeChannel_3D(features * 16 // 2, out_channels, scale_factor=(1, 16, 16))
        self.UP3 = upSample_ChangeChannel_3D(features * 8 // 2, out_channels, scale_factor=(1, 8, 8))
        self.UP2 = upSample_ChangeChannel_3D(features * 4 // 2, out_channels, scale_factor=(1, 4, 4))
        self.UP1 = upSample_ChangeChannel_3D(features * 2 // 2, out_channels, scale_factor=(1, 2, 2))
        self.UP0 = upSample_ChangeChannel_3D((out_channels + in_channels) // 2, out_channels, scale_factor=(1, 1, 1))

        self.conv = nn.Conv3d(5 * out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [4])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)
        dec4 = self.CP4(dec4)

        dec3 = self.upconv3(dec4)
        dec4 = self.UP4(dec4)
        enc3FLIP = torch.flip(enc3, [4])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)
        dec3 = self.CP3(dec3)

        dec2 = self.upconv2(dec3)
        dec3 = self.UP3(dec3)
        enc2FLIP = torch.flip(enc2, [4])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)
        dec2 = self.CP2(dec2)

        dec1 = self.upconv1(dec2)
        dec2 = self.UP2(dec2)
        enc1FLIP = torch.flip(enc1, [4])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)
        dec1 = self.CP1(dec1)

        dec0 = self.upconv0(dec1)
        dec1 = self.UP1(dec1)
        dec0 = torch.cat((dec0, input), dim=1)
        dec0 = self.CP0(dec0)
        dec0 = self.UP0(dec0)

        dec = torch.cat((dec4, dec3, dec2, dec1, dec0), dim=1)

        output = self.conv(dec)
        return self.relu(output)


class mSTN_uNet_sym_solo_CP_UP_FMNL(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet_sym_solo_CP_UP_FMNL, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))
        self.bn = normalization3D(features * 16, "bn")
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        self.upconv4 = deConv3DT(features * 16 * 2, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv3 = deConv3DT(features * 8, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv2 = deConv3DT(features * 4, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv1 = deConv3DT(features * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.upconv0 = deConv3DT(features, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.CP4 = CAM_Module_3D_conv(features * 16)
        self.CP3 = CAM_Module_3D_conv(features * 8)
        self.CP2 = CAM_Module_3D_conv(features * 4)
        self.CP1 = CAM_Module_3D_conv(features * 2)
        self.CP0 = CAM_Module_3D_conv(out_channels + in_channels)

        self.UP4 = upSample_ChangeChannel_3D(features * 16 // 2, out_channels, scale_factor=(1, 16, 16))
        self.UP3 = upSample_ChangeChannel_3D(features * 8 // 2, out_channels, scale_factor=(1, 8, 8))
        self.UP2 = upSample_ChangeChannel_3D(features * 4 // 2, out_channels, scale_factor=(1, 4, 4))
        self.UP1 = upSample_ChangeChannel_3D(features * 2 // 2, out_channels, scale_factor=(1, 2, 2))
        self.UP0 = upSample_ChangeChannel_3D((out_channels + in_channels) // 2, out_channels, scale_factor=(1, 1, 1))

        self.conv = nn.Conv3d(5 * out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        temp = self.global_pool(bottleneck)
        temp = self.bn(temp)
        bottleneck = torch.cat((bottleneck, temp), dim=1)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.CP4(dec4)

        dec3 = self.upconv3(dec4)
        dec4 = self.UP4(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.CP3(dec3)

        dec2 = self.upconv2(dec3)
        dec3 = self.UP3(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.CP2(dec2)

        dec1 = self.upconv1(dec2)
        dec2 = self.UP2(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.CP1(dec1)

        dec0 = self.upconv0(dec1)
        dec1 = self.UP1(dec1)
        dec0 = torch.cat((dec0, input), dim=1)
        dec0 = self.CP0(dec0)
        dec0 = self.UP0(dec0)

        dec = torch.cat((dec4, dec3, dec2, dec1, dec0), dim=1)
        # dec = torch.cat((dec4, dec3, dec2, dec1, dec0, input), dim=1)

        output = self.conv(dec)
        return self.relu(output)


class mSTN_uNet_sym_solo_CP_UP_FMNL_ncct(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet_sym_solo_CP_UP_FMNL_ncct, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))
        self.bn = normalization3D(features * 16, "bn")
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        self.upconv4 = deConv3DT(features * 16 * 2, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv3 = deConv3DT(features * 8, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv2 = deConv3DT(features * 4, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv1 = deConv3DT(features * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.upconv0 = deConv3DT(features, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.CP4 = CAM_Module_3D_conv(features * 16)
        self.CP3 = CAM_Module_3D_conv(features * 8)
        self.CP2 = CAM_Module_3D_conv(features * 4)
        self.CP1 = CAM_Module_3D_conv(features * 2)
        self.CP0 = CAM_Module_3D_conv(out_channels + in_channels)

        self.UP4 = upSample_ChangeChannel_3D(features * 16 // 2, out_channels, scale_factor=(1, 16, 16))
        self.UP3 = upSample_ChangeChannel_3D(features * 8 // 2, out_channels, scale_factor=(1, 8, 8))
        self.UP2 = upSample_ChangeChannel_3D(features * 4 // 2, out_channels, scale_factor=(1, 4, 4))
        self.UP1 = upSample_ChangeChannel_3D(features * 2 // 2, out_channels, scale_factor=(1, 2, 2))
        self.UP0 = upSample_ChangeChannel_3D((out_channels + in_channels) // 2, out_channels, scale_factor=(1, 1, 1))

        self.conv = nn.Conv3d(5 * out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                              bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        temp = self.global_pool(bottleneck)
        temp = self.bn(temp)
        bottleneck = torch.cat((bottleneck, temp), dim=1)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.CP4(dec4)

        dec3 = self.upconv3(dec4)
        dec4 = self.UP4(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.CP3(dec3)

        dec2 = self.upconv2(dec3)
        dec3 = self.UP3(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.CP2(dec2)

        dec1 = self.upconv1(dec2)
        dec2 = self.UP2(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.CP1(dec1)

        dec0 = self.upconv0(dec1)
        dec1 = self.UP1(dec1)
        dec0 = torch.cat((dec0, input), dim=1)
        dec0 = self.CP0(dec0)
        dec0 = self.UP0(dec0)

        dec = torch.cat((dec4, dec3, dec2, dec1, dec0, input), dim=1)

        output = self.conv(dec)
        return self.relu(output)


class mSTN_uNet_sym_solo_CP_UP_exp(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet_sym_solo_CP_UP_exp, self).__init__()

        features = init_features
        self.basicBlock = BasicBlock3D(in_channels, in_channels)
        self.daresBlock = DAResBlock3D_(in_channels, features)

        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))
        self.bn = normalization3D(features * 16, "bn")
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        self.upconv4 = deConv3DT(features * 16 * 2, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv3 = deConv3DT(features * 8, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv2 = deConv3DT(features * 4, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv1 = deConv3DT(features * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.upconv0 = deConv3DT(features, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.CP4 = CAM_Module_3D_conv(features * 16)
        self.CP3 = CAM_Module_3D_conv(features * 8)
        self.CP2 = CAM_Module_3D_conv(features * 4)
        self.CP1 = CAM_Module_3D_conv(features * 2)
        self.CP0 = CAM_Module_3D_conv(out_channels + in_channels)

        self.UP4 = upSample_ChangeChannel_3D(features * 16 // 2, out_channels, scale_factor=(1, 16, 16))
        self.UP3 = upSample_ChangeChannel_3D(features * 8 // 2, out_channels, scale_factor=(1, 8, 8))
        self.UP2 = upSample_ChangeChannel_3D(features * 4 // 2, out_channels, scale_factor=(1, 4, 4))
        self.UP1 = upSample_ChangeChannel_3D(features * 2 // 2, out_channels, scale_factor=(1, 2, 2))
        self.UP0 = upSample_ChangeChannel_3D((out_channels + in_channels) // 2, out_channels, scale_factor=(1, 1, 1))

        self.conv = nn.Conv3d(5 * out_channels + features, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        basicOut = self.basicBlock(input)
        daresOut = self.daresBlock(basicOut)

        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        temp = self.global_pool(bottleneck)
        temp = self.bn(temp)
        bottleneck = torch.cat((bottleneck, temp), dim=1)

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)
        dec4 = self.CP4(dec4)

        dec3 = self.upconv3(dec4)
        dec4 = self.UP4(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)
        dec3 = self.CP3(dec3)

        dec2 = self.upconv2(dec3)
        dec3 = self.UP3(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)
        dec2 = self.CP2(dec2)

        dec1 = self.upconv1(dec2)
        dec2 = self.UP2(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)
        dec1 = self.CP1(dec1)

        dec0 = self.upconv0(dec1)
        dec1 = self.UP1(dec1)
        dec0 = torch.cat((dec0, input), dim=1)
        dec0 = self.CP0(dec0)
        dec0 = self.UP0(dec0)

        dec = torch.cat((dec4, dec3, dec2, dec1, dec0), dim=1)

        output = self.conv(torch.cat((daresOut, dec), dim=1))

        # output = self.conv(dec)

        return self.relu(output)


class mSTN_uNet_sym_solo_CP_UP_EXT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet_sym_solo_CP_UP_EXT, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))
        self.bn = normalization3D(features * 16, "bn")
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        self.upconv4 = deConv3DT(features * 16 * 2, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv3 = deConv3DT(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv2 = deConv3DT(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv1 = deConv3DT(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.upconv0 = deConv3DT(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.CP4 = CAM_Module_3D_conv(features * 16)
        self.CP3 = CAM_Module_3D_conv(features * 8)
        self.CP2 = CAM_Module_3D_conv(features * 4)
        self.CP1 = CAM_Module_3D_conv(features * 2)
        self.CP0 = CAM_Module_3D_conv(out_channels + in_channels)

        self.UP4 = upSample_ChangeChannel_3D(features * 16 // 2, out_channels, scale_factor=(1, 16, 16))
        self.UP3 = upSample_ChangeChannel_3D(features * 8 // 2, out_channels, scale_factor=(1, 8, 8))
        self.UP2 = upSample_ChangeChannel_3D(features * 4 // 2, out_channels, scale_factor=(1, 4, 4))
        self.UP1 = upSample_ChangeChannel_3D(features * 2 // 2, out_channels, scale_factor=(1, 2, 2))
        self.UP0 = upSample_ChangeChannel_3D((out_channels + in_channels) // 2, out_channels, scale_factor=(1, 1, 1))

        self.conv = nn.Conv3d(5 * out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        temp = self.global_pool(bottleneck)
        temp = self.bn(temp)
        bottleneck = torch.cat((bottleneck, temp), dim=1)

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)

        dec3 = self.upconv3(dec4)
        dec4 = self.CP4(dec4)
        dec4 = self.UP4(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)

        dec2 = self.upconv2(dec3)
        dec3 = self.CP3(dec3)
        dec3 = self.UP3(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)

        dec1 = self.upconv1(dec2)
        dec2 = self.CP2(dec2)
        dec2 = self.UP2(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)

        dec0 = self.upconv0(dec1)
        dec1 = self.CP1(dec1)
        dec1 = self.UP1(dec1)
        dec0 = torch.cat((dec0, input), dim=1)
        dec0 = self.CP0(dec0)
        dec0 = self.UP0(dec0)

        dec = torch.cat((dec4, dec3, dec2, dec1, dec0), dim=1)

        output = self.conv(dec)
        return self.relu(output)


class mSTN_uNet_sym_solo_CP_UP_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet_sym_solo_CP_UP_unet, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))

        self.upconv4 = deConv3DT(features * 16, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv3 = deConv3DT(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv2 = deConv3DT(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv1 = deConv3DT(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.upconv0 = deConv3DT(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.conv = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)

        dec0 = self.upconv0(dec1)

        output = self.conv(dec0)
        return self.relu(output)


class mSTN_uNet_sym_solo_CP_UP_DAResUnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet_sym_solo_CP_UP_DAResUnet, self).__init__()

        features = init_features
        self.basicBlock = BasicBlock3D(in_channels, in_channels)
        self.daresBlock = DAResBlock3D(in_channels, features)

        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                  kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))

        self.upconv4 = deConv3DT(features * 16, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv3 = deConv3DT(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv2 = deConv3DT(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv1 = deConv3DT(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.upconv0 = deConv3DT(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.conv = nn.Conv3d(out_channels + features, out_channels, kernel_size=1, stride=1, padding=0,
                              bias=False)

        self.relu = nn.Sigmoid()

    def forward(self, input):
        basicOut = self.basicBlock(input)
        daresOut = self.daresBlock(basicOut)

        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)

        dec0 = self.upconv0(dec1)

        output = self.conv(torch.cat((daresOut, dec0), dim=1))

        return self.relu(output)


class mSTN_uNet_sym_solo_CP_UP_(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet_sym_solo_CP_UP_, self).__init__()

        features = init_features
        self.encoder1 = resConv3DAdd(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder2 = resConv3DAdd(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder3 = resConv3DAdd(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 2, 2),
                                     kerPad=(1, 1, 1))
        self.encoder4 = resConv3DAdd(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 2, 2),
                                     kerPad=(1, 1, 1))

        self.bottleneck = resConv3DAdd(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 2, 2),
                                       kerPad=(1, 1, 1))
        self.bn = normalization3D(features * 16, "bn")
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        self.upconv4 = resdeConv3DTAdd(features * 16 * 2, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))
        self.upconv3 = resdeConv3DTAdd(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))
        self.upconv2 = resdeConv3DTAdd(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))
        self.upconv1 = resdeConv3DTAdd(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))

        self.upconv0 = resdeConv3DTAdd(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))

        self.CP4 = CAM_Module_3D(features * 16)
        self.CP3 = CAM_Module_3D(features * 8)
        self.CP2 = CAM_Module_3D(features * 4)
        self.CP1 = CAM_Module_3D(features * 2)
        self.CP0 = CAM_Module_3D(out_channels + in_channels)

        self.UP4 = upSample_ChangeChannel_3D(features * 16, out_channels, scale_factor=(1, 16, 16))
        self.UP3 = upSample_ChangeChannel_3D(features * 8, out_channels, scale_factor=(1, 8, 8))
        self.UP2 = upSample_ChangeChannel_3D(features * 4, out_channels, scale_factor=(1, 4, 4))
        self.UP1 = upSample_ChangeChannel_3D(features * 2, out_channels, scale_factor=(1, 2, 2))
        self.UP0 = upSample_ChangeChannel_3D(out_channels + in_channels, out_channels, scale_factor=(1, 1, 1))

        self.conv = nn.Conv3d(5 * out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        # enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        # enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        # enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        # enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        # bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        temp = self.global_pool(bottleneck)
        temp = self.bn(temp)
        bottleneck = torch.cat((bottleneck, temp), dim=1)

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)
        dec4 = self.CP4(dec4)
        feature4 = self.UP4(dec4)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)
        dec3 = self.CP3(dec3)
        feature3 = self.UP3(dec3)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)
        dec2 = self.CP2(dec2)
        feature2 = self.UP2(dec2)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)
        dec1 = self.CP1(dec1)
        feature1 = self.UP1(dec1)

        dec0 = self.upconv0(dec1)
        dec0 = torch.cat((dec0, input), dim=1)
        dec0 = self.CP0(dec0)
        feature0 = self.UP0(dec0)

        dec = torch.cat((feature4, feature3, feature2, feature1, feature0), dim=1)

        output = self.conv(dec)
        return self.relu(output)


class mSTN_uNet_sym(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet_sym, self).__init__()

        features = init_features
        self.encoder1 = resConv3DAdd(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder2 = resConv3DAdd(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder3 = resConv3DAdd(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 2, 2),
                                     kerPad=(1, 1, 1))
        self.encoder4 = resConv3DAdd(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 2, 2),
                                     kerPad=(1, 1, 1))

        self.bottleneck = resConv3DAdd(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 2, 2),
                                       kerPad=(1, 1, 1))
        self.bn = normalization3D(features * 16, "bn")
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        self.upconv4 = resdeConv3DTAdd(features * 16 * 2, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))
        self.upconv3 = resdeConv3DTAdd(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))
        self.upconv2 = resdeConv3DTAdd(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))
        self.upconv1 = resdeConv3DTAdd(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))

        self.upconv0 = resdeConv3DTAdd(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                       scale_factor=(1, 2, 2))

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input, theta):
        inputGrid = F.affine_grid(theta, input.size(), align_corners=True)
        inputAlign = F.grid_sample(input, inputGrid, align_corners=True)

        enc1 = self.encoder1(inputAlign)
        # enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        # enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        # enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        # enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        # bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        temp = self.global_pool(bottleneck)
        temp = self.bn(temp)
        bottleneck = torch.cat((bottleneck, temp), dim=1)

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)

        dec0 = self.upconv0(dec1)
        inputAlignFLIP = torch.flip(inputAlign, [3])
        diffInput = torch.abs(torch.sub(inputAlign, inputAlignFLIP))
        dec = torch.cat((dec0, diffInput), dim=1)

        output = self.conv(dec)

        outputGrid = F.affine_grid(-theta, output.size(), align_corners=True)
        outputs = F.grid_sample(output, outputGrid, align_corners=True)

        return self.relu(outputs)


class mSTN_uNet_sym_small(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet_sym_small, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 2, 2),
                                  kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 2, 2),
                                  kerPad=(1, 1, 1))

        # self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 2, 2),
        #                             kerPad=(1, 1, 1))
        # self.bn = normalization3D(features * 16, "bn")
        # self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        # self.upconv4 = resdeConv3DT(features * 16, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
        #                             scale_factor=(1, 2, 2))
        self.upconv3 = resdeConv3DT(features * 8, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                    scale_factor=(1, 2, 2))
        self.upconv2 = resdeConv3DT(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                    scale_factor=(1, 2, 2))
        self.upconv1 = resdeConv3DT(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                    scale_factor=(1, 2, 2))

        self.upconv0 = resdeConv3DT(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                    scale_factor=(1, 2, 2))

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input, theta):
        inputGrid = F.affine_grid(theta, input.size(), align_corners=True)
        inputAlign = F.grid_sample(input, inputGrid, align_corners=True)

        enc1 = self.encoder1(inputAlign)
        # enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        # enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        # enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        bottleneck = self.encoder4(enc3)
        # enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        # bottleneck = self.bottleneck(enc4)
        # bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        # temp = self.global_pool(bottleneck)
        # temp = self.bn(temp)
        # bottleneck = torch.cat((bottleneck, temp), dim=1)

        # dec4 = self.upconv4(bottleneck)
        # enc4FLIP = torch.flip(enc4, [3])
        # enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        # dec4 = torch.cat((dec4, enc4DIFF), dim=1)

        dec3 = self.upconv3(bottleneck)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)

        dec0 = self.upconv0(dec1)
        inputAlignFLIP = torch.flip(inputAlign, [3])
        diffInput = torch.abs(torch.sub(inputAlign, inputAlignFLIP))
        dec = torch.cat((dec0, diffInput), dim=1)

        output = self.conv(dec)

        outputGrid = F.affine_grid(-theta, output.size(), align_corners=True)
        outputs = F.grid_sample(output, outputGrid, align_corners=True)

        return self.relu(outputs)


class STN_Block(nn.Module):
    def __init__(self, cin=1, cout=8):
        super(STN_Block, self).__init__()

        # 空间变换器定位 - 网络
        self.localization = nn.Sequential(
            Conv3D(cin, cout, 5, 2),
            Conv3D(cout, cout, 3, 2),
            Conv3D(cout, 2 * cout, 3, 2),
            Conv3D(2 * cout, 4 * cout, 3, 2),
            Conv3D(4 * cout, 8 * cout, 3, 2),
        )
        # 3 * 2 affine矩阵的回归量
        self.fc_loc = nn.Sequential(
            nn.Linear(24576, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 3)
        )

    def forward(self, input):
        xs = self.localization(input)
        xs = xs.view(-1, xs.size()[1] * xs.size()[2] * xs.size()[3] * xs.size()[4])
        torch.flatten(xs, start_dim=1)
        ref = torch.tanh(self.fc_loc(xs)) * 0.5

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # vectorX = torch.tensor([[1, 0, 0],
        #                         [0, 0, 0],
        #                         [0, 0, 0]]).to(device)
        #
        # vectorX1 = torch.tensor([[0, 0, 0],
        #                          [0, 1, 0],
        #                          [0, 0, -1]]).to(device)
        #
        # vectorX2 = torch.tensor([[0, 0, 0],
        #                          [0, 0, -1],
        #                          [0, 1, 0]]).to(device)

        vectorY = torch.tensor([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]]).to(device)

        vectorY1 = torch.tensor([[1, 0, 0],
                                 [0, 0, 0],
                                 [0, 0, 1]]).to(device)

        vectorY2 = torch.tensor([[0, 0, 1],
                                 [0, 0, 0],
                                 [-1, 0, 0]]).to(device)

        vectorZ = torch.tensor([[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 1]]).to(device)

        vectorZ1 = torch.tensor([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]]).to(device)

        vectorZ2 = torch.tensor([[0, -1, 0],
                                 [1, 0, 0],
                                 [0, 0, 0]]).to(device)

        # rotateX = vectorX + torch.mul(vectorX1.view(1, 3, 3),
        #                               torch.cos(ref[:, 0]).view(input.size()[0], 1, 1)) + torch.mul(
        #     vectorX2.view(1, 3, 3), torch.sin(ref[:, 0]).view(input.size()[0], 1, 1))
        rotateY = vectorY + torch.mul(vectorY1.view(1, 3, 3),
                                      torch.cos(ref[:, 1]).view(input.size()[0], 1, 1)) + torch.mul(
            vectorY2.view(1, 3, 3), torch.sin(ref[:, 1]).view(input.size()[0], 1, 1))
        # rotateZ = vectorZ + torch.mul(vectorZ1.view(1, 3, 3),
        #                               torch.cos(ref[:, 2]).view(input.size()[0], 1, 1)) + torch.mul(
        #     vectorZ2.view(1, 3, 3), torch.sin(ref[:, 2]).view(input.size()[0], 1, 1))

        rotateZ = vectorZ + torch.mul(vectorZ1.view(1, 3, 3),
                                      torch.cos(ref[:, 0]).view(input.size()[0], 1, 1)) + torch.mul(
            vectorZ2.view(1, 3, 3), torch.sin(ref[:, 0]).view(input.size()[0], 1, 1))

        # temp = torch.bmm(torch.bmm(rotateX, rotateY), rotateZ).view(input.size()[0], 3, 3)
        temp = torch.bmm(rotateY, rotateZ).view(input.size()[0], 3, 3)
        # temp = rotateZ.view(input.size()[0], 3, 3)

        trans1 = torch.tensor([[1],
                               [0],
                               [0]]).to(device)
        # trans2 = torch.tensor([[0],
        #                        [1],
        #                        [0]]).to(device)
        # trans3 = torch.tensor([[0],
        #                        [0],
        #                        [1]]).to(device)
        # move = torch.mul(trans1.view(1, 3, 1), ref[:, 3].view(input.size()[0], 1, 1)) \
        #        + torch.mul(trans2.view(1, 3, 1), ref[:, 4].view(input.size()[0], 1, 1)) \
        #        + torch.mul(trans3.view(1, 3, 1), ref[:, 5].view(input.size()[0], 1, 1))

        move = torch.mul(trans1.view(1, 3, 1), ref[:, 2].view(input.size()[0], 1, 1))

        theta = torch.cat([temp, move], dim=2)
        grid = F.affine_grid(theta, input.size(), align_corners=True)
        netout = F.grid_sample(input, grid, align_corners=True)

        return theta, netout


class mSTNUnet3D_t1(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTNUnet3D_t1, self).__init__()

        features = init_features
        self.encoder1 = Conv3D(in_channels, features)
        self.encoder2 = Conv3D(features, features * 2)
        self.encoder3 = Conv3D(features * 2, features * 4)
        self.encoder4 = Conv3D(features * 4, features * 8)

        self.bottleneck = Conv3D(features * 8, features * 16)

        self.upconv4 = deConv3D(features * 16, features * 8)
        self.upconv3 = deConv3D(features * 8 * 2, features * 4)
        self.upconv2 = deConv3D(features * 4 * 2, features * 2)
        self.upconv1 = deConv3D(features * 2 * 2, features)

        self.upconv0 = deConv3D(features * 2, out_channels)

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)

        dec0 = self.upconv0(dec1)
        dec = torch.cat((dec0, input), dim=1)

        output = self.conv(dec)
        return self.relu(output)


class uNetT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(uNetT, self).__init__()

        features = init_features
        self.encoder1 = Conv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = Conv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = Conv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder4 = Conv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))

        self.bottleneck = Conv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))

        self.upconv4 = deConv3DT(features * 16, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv3 = deConv3DT(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv2 = deConv3DT(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))
        self.upconv1 = deConv3DT(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.upconv0 = deConv3DT(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                 scale_factor=(1, 2, 2))

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)

        dec0 = self.upconv0(dec1)
        dec = torch.cat((dec0, input), dim=1)

        output = self.conv(dec)
        return self.relu(output)


class uNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(uNet, self).__init__()

        features = init_features
        self.encoder1 = Conv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = Conv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = Conv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder4 = Conv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))

        self.bottleneck = Conv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))

        self.upconv4 = deConv3D(features * 16, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv3 = deConv3D(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv2 = deConv3D(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv1 = deConv3D(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1), scale_factor=(1, 2, 2))

        self.upconv0 = deConv3D(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1), scale_factor=(1, 2, 2))

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)

        dec0 = self.upconv0(dec1)
        dec = torch.cat((dec0, input), dim=1)

        output = self.conv(dec)
        return self.relu(output)


class mSTN_uNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))

        self.upconv4 = deConv3D(features * 16, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv3 = deConv3D(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv2 = deConv3D(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv1 = deConv3D(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1), scale_factor=(1, 2, 2))

        self.upconv0 = deConv3D(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1), scale_factor=(1, 2, 2))

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)

        dec0 = self.upconv0(dec1)
        dec = torch.cat((dec0, input), dim=1)

        output = self.conv(dec)
        return self.relu(output)


class mSTN_uNet__(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet__, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))
        self.bn = normalization3D(features * 16, "bn")
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        self.upconv4 = deConv3D(features * 16 * 2, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv3 = deConv3D(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv2 = deConv3D(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv1 = deConv3D(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1), scale_factor=(1, 2, 2))

        self.upconv0 = deConv3D(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1), scale_factor=(1, 2, 2))

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        temp = self.global_pool(bottleneck)
        temp = self.bn(temp)
        bottleneck = torch.cat((bottleneck, temp), dim=1)

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)

        dec0 = self.upconv0(dec1)
        dec = torch.cat((dec0, input), dim=1)

        output = self.conv(dec)
        return self.relu(output)


class mSTN_uNet__Globepool(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet__Globepool, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))

        self.upconv4 = deConv3D(features * 16, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv3 = deConv3D(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv2 = deConv3D(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv1 = deConv3D(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1), scale_factor=(1, 2, 2))

        self.upconv0 = deConv3D(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1), scale_factor=(1, 2, 2))

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)

        dec0 = self.upconv0(dec1)
        dec = torch.cat((dec0, input), dim=1)

        output = self.conv(dec)
        return self.relu(output)


class mSTN_uNet_Flip(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet_Flip, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))
        self.bn = normalization3D(features * 16, "bn")
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        self.upconv4 = deConv3D(features * 16 * 2, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv3 = deConv3D(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv2 = deConv3D(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv1 = deConv3D(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1), scale_factor=(1, 2, 2))

        self.upconv0 = deConv3D(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1), scale_factor=(1, 2, 2))

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        temp = self.global_pool(bottleneck)
        temp = self.bn(temp)
        bottleneck = torch.cat((bottleneck, temp), dim=1)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)

        dec0 = self.upconv0(dec1)
        dec = torch.cat((dec0, input), dim=1)

        output = self.conv(dec)
        return self.relu(output)


class mSTN_uNet___(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet___, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))

        self.upconv4 = deConv3D(features * 16, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv3 = deConv3D(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv2 = deConv3D(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv1 = deConv3D(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1), scale_factor=(1, 2, 2))

        self.upconv0 = deConv3D(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1), scale_factor=(1, 2, 2))

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)

        dec0 = self.upconv0(dec1)
        dec = torch.cat((dec0, input), dim=1)

        output = self.conv(dec)
        return self.relu(output)


class mSTN_uNet___loss(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet___loss, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))

        self.upconv4 = deConv3D(features * 16, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv3 = deConv3D(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv2 = deConv3D(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv1 = deConv3D(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1), scale_factor=(1, 2, 2))

        self.upconv0 = deConv3D(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1), scale_factor=(1, 2, 2))

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)

        dec0 = self.upconv0(dec1)
        dec = torch.cat((dec0, input), dim=1)

        output = self.conv(dec)
        return self.relu(output)


class mSTN_uNet__1(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTN_uNet__1, self).__init__()

        features = init_features
        self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
        self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))

        self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
                                    kerPad=(1, 1, 1))

        self.upconv4 = resdeConv3DT(features * 16, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                    scale_factor=(1, 2, 2))
        self.upconv3 = resdeConv3DT(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                    scale_factor=(1, 2, 2))
        self.upconv2 = resdeConv3DT(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                    scale_factor=(1, 2, 2))
        self.upconv1 = resdeConv3DT(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                    scale_factor=(1, 2, 2))

        self.upconv0 = resdeConv3DT(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                    scale_factor=(1, 2, 2))

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc2 = self.encoder2(enc1)
        enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc3 = self.encoder3(enc2)
        enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(1, 2, 2))
        enc4 = self.encoder4(enc3)
        enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))

        bottleneck = self.bottleneck(enc4)
        bottleneck = torch.max_pool3d(bottleneck, kernel_size=3, padding=1, stride=(1, 2, 2))

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)

        dec0 = self.upconv0(dec1)
        dec = torch.cat((dec0, input), dim=1)

        output = self.conv(dec)
        return self.relu(output)


class uNet_rsize(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(uNet_rsize, self).__init__()

        features = init_features
        self.encoder1 = Conv3D_e(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder2 = Conv3D_e(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder3 = Conv3D_e(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder4 = Conv3D_e(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))

        self.bottleneck = Conv3D_e(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 2, 2),
                                   kerPad=(1, 1, 1))
        # self.bn = normalization3D(features * 16, "bn")
        # self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        self.upconv4 = deConv3D_e(features * 16, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                  scale_factor=(1, 2, 2))
        self.upconv3 = deConv3D_e(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                  scale_factor=(1, 2, 2))
        self.upconv2 = deConv3D_e(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                  scale_factor=(1, 2, 2))
        self.upconv1 = deConv3D_e(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                  scale_factor=(1, 2, 2))

        self.upconv0 = deConv3D_e(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                  scale_factor=(1, 2, 2))

        self.conv = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        # temp = self.global_pool(bottleneck)
        # temp = self.bn(temp)
        # bottleneck = torch.cat((bottleneck, temp), dim=1)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)

        dec0 = self.upconv0(dec1)
        # dec = torch.cat((dec0, input), dim=1)

        output = self.conv(dec0)
        return self.relu(output)


class uNet_rsize_ncct(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(uNet_rsize_ncct, self).__init__()

        features = init_features
        self.encoder1 = Conv3D_e(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder2 = Conv3D_e(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder3 = Conv3D_e(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder4 = Conv3D_e(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))

        self.bottleneck = Conv3D_e(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 2, 2),
                                   kerPad=(1, 1, 1))
        # self.bn = normalization3D(features * 16, "bn")
        # self.global_pool = nn.AdaptiveAvgPool3d(output_size=(24, 8, 8))

        self.upconv4 = deConv3D_e(features * 16, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                  scale_factor=(1, 2, 2))
        self.upconv3 = deConv3D_e(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                  scale_factor=(1, 2, 2))
        self.upconv2 = deConv3D_e(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                  scale_factor=(1, 2, 2))
        self.upconv1 = deConv3D_e(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                  scale_factor=(1, 2, 2))

        self.upconv0 = deConv3D_e(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                  scale_factor=(1, 2, 2))

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        # temp = self.global_pool(bottleneck)
        # temp = self.bn(temp)
        # bottleneck = torch.cat((bottleneck, temp), dim=1)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)

        dec0 = self.upconv0(dec1)
        dec = torch.cat((dec0, input), dim=1)

        output = self.conv(dec)
        return self.relu(output)


class UNetA(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(UNetA, self).__init__()

        features = init_features
        self.encoder1 = Conv3DAdd(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder2 = Conv3DAdd(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder3 = Conv3DAdd(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder4 = Conv3DAdd(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))

        self.bottleneck = Conv3DAdd(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 2, 2),
                                    kerPad=(1, 1, 1))

        self.upconv4 = deConv3D(features * 16, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv3 = deConv3D(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv2 = deConv3D(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv1 = deConv3D(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1), scale_factor=(1, 2, 2))

        self.upconv0 = deConv3D(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1), scale_factor=(1, 2, 2))

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)

        dec0 = self.upconv0(dec1)
        dec = torch.cat((dec0, input), dim=1)

        output = self.conv(dec)
        return self.relu(output)


class mSTNUnet3D_rsize(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTNUnet3D_rsize, self).__init__()

        features = init_features
        self.encoder1 = Conv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder2 = Conv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder3 = Conv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))
        self.encoder4 = Conv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))

        self.bottleneck = Conv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 2, 2), kerPad=(1, 1, 1))

        self.upconv4 = deConv3D(features * 16, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv3 = deConv3D(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv2 = deConv3D(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
                                scale_factor=(1, 2, 2))
        self.upconv1 = deConv3D(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1), scale_factor=(1, 2, 2))

        self.upconv0 = deConv3D(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1), scale_factor=(1, 2, 2))

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)

        dec0 = self.upconv0(dec1)
        dec = torch.cat((dec0, input), dim=1)

        output = self.conv(dec)
        return self.relu(output)


class mSTNUnet3D_t2(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTNUnet3D_t2, self).__init__()

        features = init_features
        self.encoder1 = Conv3D(in_channels, features)
        self.encoder2 = Conv3D(features, features * 2)
        self.encoder3 = Conv3D(features * 2, features * 4)
        self.encoder4 = Conv3D(features * 4, features * 8)

        self.bottleneck = Conv3D(features * 8, features * 16)

        self.upconv4 = deConv3D(features * 16, features * 8)
        self.upconv3 = deConv3D(features * 8 * 2, features * 4)
        self.upconv2 = deConv3D(features * 4 * 2, features * 2)
        self.upconv1 = deConv3D(features * 2 * 2, features)

        self.upconv0 = deConv3D(features * 2, out_channels)

        self.conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input, theta):
        inputGrid = F.affine_grid(theta, input.size(), align_corners=True)
        inputAlign = F.grid_sample(input, inputGrid, align_corners=True)

        # x = torch.nn.functional.normalize(x)
        enc1 = self.encoder1(inputAlign)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)

        dec0 = self.upconv0(dec1)
        inputAlignFLIP = torch.flip(inputAlign, [3])
        diffInput = torch.abs(torch.sub(inputAlign, inputAlignFLIP))
        dec = torch.cat((dec0, diffInput), dim=1)

        output = self.conv(dec)

        outputGrid = F.affine_grid(-theta, output.size(), align_corners=True)
        outputs = F.grid_sample(output, outputGrid, align_corners=True)

        return self.relu(outputs)


class STN_Block_2D(nn.Module):
    def __init__(self, cin=1, cout=8):
        super(STN_Block_2D, self).__init__()

        # 空间变换器定位 - 网络
        self.localization = nn.Sequential(
            Conv2D(cin, cout, 3, 2),
            Conv2D(cout, cout, 3, 2),
            Conv2D(cout, 2 * cout, 3, 2),
            Conv2D(2 * cout, 4 * cout, 3, 2),
            Conv2D(4 * cout, 8 * cout, 3, 2),
        )
        # 3 * 2 affine矩阵的回归量
        self.fc_loc = nn.Sequential(
            nn.Linear(4096, 1024),  # nn.Linear(12288, 32),  # nn.Linear(9408, 32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 2)
        )

    def forward(self, input):
        xs = self.localization(input)
        xs = xs.view(-1, xs.size()[1] * xs.size()[2] * xs.size()[3])
        torch.flatten(xs, start_dim=1)
        ref = torch.tanh(self.fc_loc(xs)) * 0.25

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        vectorZ1 = torch.tensor([[1, 0], [0, 1]]).to(device)

        vectorZ2 = torch.tensor([[0, 1], [-1, 0]]).to(device)

        rotateZ = torch.mul(vectorZ1.view(1, 2, 2), torch.cos(ref[:, 0]).view(input.size()[0], 1, 1)) + torch.mul(
            vectorZ2.view(1, 2, 2), torch.sin(ref[:, 0]).view(input.size()[0], 1, 1))

        temp = rotateZ.view(input.size()[0], 2, 2)

        trans1 = torch.tensor([[1],
                               [0]]).to(device)
        move = torch.mul(trans1.view(1, 2, 1), ref[:, 1].view(input.size()[0], 1, 1))

        theta = torch.cat([temp, move], dim=2)
        grid = F.affine_grid(theta, input.size(), align_corners=True)
        netout = F.grid_sample(input, grid, align_corners=True)

        return theta, netout


class mSTNUnet2D_t1(nn.Module):
    # 仅限（256，256）大小输入
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTNUnet2D_t1, self).__init__()

        features = init_features
        self.encoder1 = Conv2D(in_channels, features)
        self.encoder2 = Conv2D(features, features * 2)
        self.encoder3 = Conv2D(features * 2, features * 4)
        self.encoder4 = Conv2D(features * 4, features * 8)

        self.bottleneck = Conv2D(features * 8, features * 16)
        # self.dense = nn.Linear(6144, 6144)

        self.upconv4 = deConv2D(features * 16, features * 8)
        self.upconv3 = deConv2D(features * 8 * 2, features * 4)
        self.upconv2 = deConv2D(features * 4 * 2, features * 2)
        self.upconv1 = deConv2D(features * 2 * 2, features)

        self.upconv0 = deConv2D(features * 2, out_channels)

        self.conv = nn.Conv2d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        enc1 = self.encoder1(input)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)

        dec0 = self.upconv0(dec1)

        dec = torch.cat((dec0, input), dim=1)

        output = self.conv(dec)

        return self.relu(output)


class mSTNUnet2D_t2(nn.Module):
    # 仅限（256，256）大小输入
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTNUnet2D_t2, self).__init__()

        features = init_features
        self.encoder1 = Conv2D(in_channels, features)
        self.encoder2 = Conv2D(features, features * 2)
        self.encoder3 = Conv2D(features * 2, features * 4)
        self.encoder4 = Conv2D(features * 4, features * 8)

        self.bottleneck = Conv2D(features * 8, features * 16)
        # self.dense = nn.Linear(6144, 6144)

        self.upconv4 = deConv2D(features * 16, features * 8)
        self.upconv3 = deConv2D(features * 8 * 2, features * 4)
        self.upconv2 = deConv2D(features * 4 * 2, features * 2)
        self.upconv1 = deConv2D(features * 2 * 2, features)

        self.upconv0 = deConv2D(features * 2, out_channels)

        self.conv = nn.Conv2d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input, theta):
        inputGrid = F.affine_grid(theta, input.size(), align_corners=True)
        inputAlign = F.grid_sample(input, inputGrid, align_corners=True)

        # x = torch.nn.functional.normalize(x)
        enc1 = self.encoder1(inputAlign)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)
        # bottleneck1 = self.dense(bottleneck.view(-1, 6144))
        # bottleneck2 = torch.reshape(bottleneck1, (-1, 16, 6, 8, 8))

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)

        dec0 = self.upconv0(dec1)
        inputAlignFLIP = torch.flip(inputAlign, [3])
        diffInput = torch.abs(torch.sub(inputAlign, inputAlignFLIP))
        dec = torch.cat((dec0, diffInput), dim=1)

        output = self.conv(dec)

        outputGrid = F.affine_grid(-theta, output.size(), align_corners=True)
        outputs = F.grid_sample(output, outputGrid, align_corners=True)

        return self.relu(outputs)


class mSTNUnet2D_Combine(nn.Module):
    # 仅限（256，256）大小输入
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(mSTNUnet2D_Combine, self).__init__()

        features = init_features
        self.rigid = STN_Block_2D(cin=in_channels)
        self.encoder1 = Conv2D(in_channels, features)
        self.encoder2 = Conv2D(features, features * 2)
        self.encoder3 = Conv2D(features * 2, features * 4)
        self.encoder4 = Conv2D(features * 4, features * 8)

        self.bottleneck = Conv2D(features * 8, features * 16)
        # self.dense = nn.Linear(6144, 6144)

        self.upconv4 = deConv2D(features * 16, features * 8)
        self.upconv3 = deConv2D(features * 8 * 2, features * 4)
        self.upconv2 = deConv2D(features * 4 * 2, features * 2)
        self.upconv1 = deConv2D(features * 2 * 2, features)

        self.upconv0 = deConv2D(features * 2, out_channels)

        self.conv = nn.Conv2d(out_channels + in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        theta, inputAlign = self.rigid(input)

        # inputGrid = F.affine_grid(theta, input.size(), align_corners=True)
        # inputAlign = F.grid_sample(input, inputGrid, align_corners=True)

        # x = torch.nn.functional.normalize(x)
        enc1 = self.encoder1(inputAlign)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)
        # bottleneck1 = self.dense(bottleneck.view(-1, 6144))
        # bottleneck2 = torch.reshape(bottleneck1, (-1, 16, 6, 8, 8))

        dec4 = self.upconv4(bottleneck)
        enc4FLIP = torch.flip(enc4, [3])
        enc4DIFF = torch.abs(torch.sub(enc4, enc4FLIP))
        dec4 = torch.cat((dec4, enc4DIFF), dim=1)

        dec3 = self.upconv3(dec4)
        enc3FLIP = torch.flip(enc3, [3])
        enc3DIFF = torch.abs(torch.sub(enc3, enc3FLIP))
        dec3 = torch.cat((dec3, enc3DIFF), dim=1)

        dec2 = self.upconv2(dec3)
        enc2FLIP = torch.flip(enc2, [3])
        enc2DIFF = torch.abs(torch.sub(enc2, enc2FLIP))
        dec2 = torch.cat((dec2, enc2DIFF), dim=1)

        dec1 = self.upconv1(dec2)
        enc1FLIP = torch.flip(enc1, [3])
        enc1DIFF = torch.abs(torch.sub(enc1, enc1FLIP))
        dec1 = torch.cat((dec1, enc1DIFF), dim=1)

        dec0 = self.upconv0(dec1)
        inputAlignFLIP = torch.flip(inputAlign, [3])
        diffInput = torch.abs(torch.sub(inputAlign, inputAlignFLIP))
        dec = torch.cat((dec0, diffInput), dim=1)

        output = self.conv(dec)

        outputGrid = F.affine_grid(-theta, output.size(), align_corners=True)
        outputs = F.grid_sample(output, outputGrid, align_corners=True)

        return self.relu(outputs), inputAlign
