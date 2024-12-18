import torch
import torch.nn as nn
import torch.nn.functional as F


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

        output = self.conv(dec0)
        return self.relu(output)
