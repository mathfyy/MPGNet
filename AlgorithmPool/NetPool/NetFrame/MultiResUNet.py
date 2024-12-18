import torch
import torch.nn as nn
import torch.nn.functional as F


def normalization3D(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class MultiResBlock3D(nn.Module):
    def __init__(self, inC, outC, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1), dilate=(1, 1, 1),
                 norm="bn"):
        super(MultiResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(inC, inC, kernel_size=kerSize, stride=kerStride, padding=kerPad, bias=True,
                               dilation=dilate)
        self.bn1 = normalization3D(inC, "bn")

        self.conv2 = nn.Conv3d(inC, inC, kernel_size=kerSize, stride=1, padding=kerPad, bias=True,
                               dilation=dilate)
        self.bn2 = normalization3D(inC, "bn")

        self.conv3 = nn.Conv3d(inC, inC, kernel_size=kerSize, stride=1, padding=kerPad, bias=True,
                               dilation=dilate)
        self.bn3 = normalization3D(inC, "bn")

        self.conv0 = nn.Conv3d(3 * inC, outC, kernel_size=1, stride=1, padding=0, bias=True,
                               dilation=dilate)
        self.bn0 = normalization3D(outC, "bn")

        self.conv = nn.Conv3d(inC, outC, kernel_size=1, stride=kerStride, padding=0, bias=True,
                              dilation=dilate)
        self.bn = normalization3D(outC, "bn")

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        c1 = self.relu(self.bn1(self.conv1(x)))
        c2 = self.relu(self.bn2(self.conv2(c1)))
        c3 = self.relu(self.bn3(self.conv3(c2)))
        # block1 = torch.cat((c1, c2), dim=1)
        block1 = torch.cat((c1, c2, c3), dim=1)
        block2 = self.relu(self.bn(self.conv(x)))
        out = torch.add(self.relu(self.bn0(self.conv0(block1))), block2)
        return out


class ResPath3D(nn.Module):
    def __init__(self, inC, outC, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1), dilate=(1, 1, 1),
                 norm="bn"):
        super(ResPath3D, self).__init__()
        self.conv1 = nn.Conv3d(inC, outC, kernel_size=kerSize, stride=kerStride, padding=kerPad, bias=True,
                               dilation=dilate)
        self.bn1 = normalization3D(outC, "bn")
        self.conv = nn.Conv3d(inC, outC, kernel_size=1, stride=kerStride, padding=0, bias=True,
                              dilation=dilate)
        self.bn = normalization3D(outC, "bn")
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        block1 = self.relu(self.bn1(self.conv1(x)))
        block2 = self.relu(self.bn(self.conv(x)))
        out = torch.add(block1, block2)
        return out


class MultiResNet3D(nn.Module):
    def __init__(self, in_channels=1, init_features=4, out_channels=1):
        super(MultiResNet3D, self).__init__()
        self.n_features = init_features
        features = init_features

        self.MultiResBlock3D1 = MultiResBlock3D(in_channels, features, kerStride=(1, 2, 2))
        self.MultiResBlock3D2 = MultiResBlock3D(features, features, kerStride=(1, 2, 2))
        self.MultiResBlock3D3 = MultiResBlock3D(features, features * 2, kerStride=(1, 2, 2))
        self.MultiResBlock3D4 = MultiResBlock3D(features * 2, features * 4, kerStride=(1, 2, 2))
        self.MultiResBlock3D5 = MultiResBlock3D(features * 4, features * 8, kerStride=(1, 2, 2))
        self.MultiResBlock3D6 = MultiResBlock3D(features * 16, features * 4)
        self.MultiResBlock3D7 = MultiResBlock3D(features * 8, features * 2)
        self.MultiResBlock3D8 = MultiResBlock3D(features * 4, features)
        self.MultiResBlock3D9 = MultiResBlock3D(features * 2, out_channels)

        self.ResPath3D1 = ResPath3D(features, features)
        self.ResPath3D2 = ResPath3D(features, features * 2)
        self.ResPath3D3 = ResPath3D(features * 2, features * 4)
        self.ResPath3D4 = ResPath3D(features * 4, features * 8)

        self.conv = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        e1 = self.MultiResBlock3D1(input)
        e2 = self.MultiResBlock3D2(e1)
        e3 = self.MultiResBlock3D3(e2)
        e4 = self.MultiResBlock3D4(e3)
        e5 = self.MultiResBlock3D5(e4)

        du4 = F.interpolate(e5, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        du4 = torch.cat((self.ResPath3D4(e4), du4), dim=1)
        d4 = self.MultiResBlock3D6(du4)

        du3 = F.interpolate(d4, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        du3 = torch.cat((self.ResPath3D3(e3), du3), dim=1)
        d3 = self.MultiResBlock3D7(du3)
        du2 = F.interpolate(d3, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        du2 = torch.cat((self.ResPath3D2(e2), du2), dim=1)
        d2 = self.MultiResBlock3D8(du2)
        du1 = F.interpolate(d2, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        du1 = torch.cat((self.ResPath3D1(e1), du1), dim=1)
        pre = self.MultiResBlock3D9(du1)
        pre = F.interpolate(pre, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        return self.relu(self.conv(pre))


class MultiResNet3D_isles(nn.Module):
    def __init__(self, in_channels=1, init_features=4, out_channels=1):
        super(MultiResNet3D_isles, self).__init__()
        self.n_features = init_features
        features = init_features

        self.MultiResBlock3D1 = MultiResBlock3D(in_channels, features, kerStride=(2, 2, 2))
        self.MultiResBlock3D2 = MultiResBlock3D(features, features, kerStride=(2, 2, 2))
        self.MultiResBlock3D3 = MultiResBlock3D(features, features * 2, kerStride=(2, 2, 2))
        self.MultiResBlock3D4 = MultiResBlock3D(features * 2, features * 4, kerStride=(1, 2, 2))
        self.MultiResBlock3D5 = MultiResBlock3D(features * 4, features * 8, kerStride=(1, 2, 2))
        self.MultiResBlock3D6 = MultiResBlock3D(features * 16, features * 4)
        self.MultiResBlock3D7 = MultiResBlock3D(features * 8, features * 2)
        self.MultiResBlock3D8 = MultiResBlock3D(features * 4, features)
        self.MultiResBlock3D9 = MultiResBlock3D(features * 2, out_channels)

        self.ResPath3D1 = ResPath3D(features, features)
        self.ResPath3D2 = ResPath3D(features, features * 2)
        self.ResPath3D3 = ResPath3D(features * 2, features * 4)
        self.ResPath3D4 = ResPath3D(features * 4, features * 8)

        self.conv = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, input):
        e1 = self.MultiResBlock3D1(input)
        e2 = self.MultiResBlock3D2(e1)
        e3 = self.MultiResBlock3D3(e2)
        e4 = self.MultiResBlock3D4(e3)
        e5 = self.MultiResBlock3D5(e4)

        du4 = F.interpolate(e5, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        du4 = torch.cat((self.ResPath3D4(e4), du4), dim=1)
        d4 = self.MultiResBlock3D6(du4)

        du3 = F.interpolate(d4, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        du3 = torch.cat((self.ResPath3D3(e3), du3), dim=1)
        d3 = self.MultiResBlock3D7(du3)
        du2 = F.interpolate(d3, scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        du2 = torch.cat((self.ResPath3D2(e2), du2), dim=1)
        d2 = self.MultiResBlock3D8(du2)
        du1 = F.interpolate(d2, scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        du1 = torch.cat((self.ResPath3D1(e1), du1), dim=1)
        pre = self.MultiResBlock3D9(du1)
        pre = F.interpolate(pre, scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        return self.relu(self.conv(pre))
