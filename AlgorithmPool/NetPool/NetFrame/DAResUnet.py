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


class PAM_Module_3D(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, out_dim):
        super(PAM_Module_3D, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
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


class DAResBlock3D(nn.Module):
    def __init__(self, inC, outC, kerSize=(3, 3, 3), kerStride=1, kerPad=1, dropoutRate=0.5, norm="bn"):
        super(DAResBlock3D, self).__init__()
        self.layerS = nn.Sequential()
        self.layerS.add_module("conv", nn.Conv3d(inC, inC, kerSize, stride=kerStride, padding=kerPad, bias=False))
        self.layerS.add_module("norm", normalization3D(inC, norm))
        self.layerS.add_module("relu", nn.RReLU(inplace=True))

        self.SMO = PAM_Module_3D(inC, outC)
        # self.SMO = SpatialAttention3D(inC, outC)

        self.layerS1 = nn.Sequential()
        self.layerS1.add_module("conv",
                                nn.Conv3d(outC, outC, kerSize, stride=kerStride, padding=kerPad, bias=False))
        self.layerS1.add_module("norm", normalization3D(outC, norm))
        self.layerS1.add_module("relu", nn.RReLU(inplace=True))

        self.layerC = nn.Sequential()
        self.layerC.add_module("conv", nn.Conv3d(inC, inC, kerSize, stride=kerStride, padding=kerPad, bias=False))
        self.layerC.add_module("norm", normalization3D(inC, norm))
        self.layerC.add_module("relu", nn.RReLU(inplace=True))

        self.CMO = CAM_Module_3D(inC)
        # self.CMO = ChannelAttention3D(inC, outC//2)

        self.layerC1 = nn.Sequential()
        self.layerC1.add_module("conv",
                                nn.Conv3d(inC, outC, kerSize, stride=kerStride, padding=kerPad, bias=False))
        self.layerC1.add_module("norm", normalization3D(outC, norm))
        self.layerC1.add_module("relu", nn.RReLU(inplace=True))

        self.layer = nn.Sequential()
        self.layer.add_module("conv",
                              nn.Conv3d(outC * 2, outC, kerSize, stride=kerStride, padding=kerPad, bias=False))
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


class resConv3D(nn.Module):
    def __init__(self, inC, outC, kerSize=(3, 3, 3), kerStride=(2, 2, 2), kerPad=(1, 1, 1), dilate=(1, 1, 1),
                 dropoutRate=0.5,
                 norm="bn"):
        super(resConv3D, self).__init__()
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

        self.layer1 = nn.Sequential()
        self.layer1.add_module("conv",
                               nn.Conv3d(inC, inC, kernel_size=1, stride=1, padding=0, bias=True,
                                         dilation=dilate))
        self.layer1.add_module("norm", normalization3D(inC, norm))
        self.layer1.add_module("relu", nn.LeakyReLU(inplace=True))

    def forward(self, x):
        y = self.layer0(x)
        z = self.layer1(x)
        out = torch.add(y, z)
        out = self.layer(out)
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
        self.layer0.add_module("conv", nn.ConvTranspose3d(inC, outC, kerSize, stride=1, padding=kerPad, bias=True,
                                                          dilation=dilate))
        self.layer0.add_module("norm", normalization3D(outC, norm))
        self.layer0.add_module("relu", nn.LeakyReLU(inplace=True))

        self.layer = nn.Sequential()
        self.layer.add_module("conv",
                              nn.ConvTranspose3d(outC, outC, kerSize, stride=kerStride, padding=kerPad, bias=True,
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


class DAResUnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(DAResUnet, self).__init__()

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


# class DAResUnet(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, init_features=4):
#         super(DAResUnet, self).__init__()
#
#         features = init_features
#         self.basicBlock = BasicBlock3D(in_channels, in_channels)
#         self.daresBlock = DAResBlock3D(in_channels, in_channels)
#
#         self.encoder1 = resConv3D(in_channels, features, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
#         self.encoder2 = resConv3D(features, features * 2, kerSize=(3, 3, 3), kerStride=(1, 1, 1), kerPad=(1, 1, 1))
#         self.encoder3 = resConv3D(features * 2, features * 4, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
#                                   kerPad=(1, 1, 1))
#         self.encoder4 = resConv3D(features * 4, features * 8, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
#                                   kerPad=(1, 1, 1))
#
#         self.bottleneck = resConv3D(features * 8, features * 16, kerSize=(3, 3, 3), kerStride=(1, 1, 1),
#                                     kerPad=(1, 1, 1))
#
#         self.upconv4 = deConv3DT(features * 16, features * 8, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
#                                  scale_factor=(1, 1, 1))
#         self.upconv3 = deConv3DT(features * 8 * 2, features * 4, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
#                                  scale_factor=(1, 2, 2))
#         self.upconv2 = deConv3DT(features * 4 * 2, features * 2, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
#                                  scale_factor=(2, 2, 2))
#         self.upconv1 = deConv3DT(features * 2 * 2, features, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
#                                  scale_factor=(1, 2, 2))
#         self.upconv0 = deConv3DT(features * 2, out_channels, kerSize=(3, 3, 3), kerPad=(1, 1, 1),
#                                  scale_factor=(2, 2, 2))
#
#         self.conv = nn.Conv3d(in_channels * 2, out_channels, kernel_size=1, stride=1, padding=0,
#                               bias=False)
#
#         self.relu = nn.Sigmoid()
#
#     def forward(self, input):
#         basicOut = self.basicBlock(input)
#         daresOut = self.daresBlock(basicOut)
#
#         enc1 = self.encoder1(input)
#         enc1 = torch.max_pool3d(enc1, kernel_size=3, padding=1, stride=(2, 2, 2))
#         enc2 = self.encoder2(enc1)
#         enc2 = torch.max_pool3d(enc2, kernel_size=3, padding=1, stride=(1, 2, 2))
#         enc3 = self.encoder3(enc2)
#         enc3 = torch.max_pool3d(enc3, kernel_size=3, padding=1, stride=(2, 2, 2))
#         enc4 = self.encoder4(enc3)
#         enc4 = torch.max_pool3d(enc4, kernel_size=3, padding=1, stride=(1, 2, 2))
#
#         bottleneck = self.bottleneck(enc4)
#
#         dec4 = self.upconv4(bottleneck)
#         dec4 = torch.cat((dec4, enc4), dim=1)
#
#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat((dec3, enc3), dim=1)
#
#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#
#         dec0 = self.upconv0(dec1)
#
#         output = self.conv(torch.cat((daresOut, dec0), dim=1))
#
#         return self.relu(output)
