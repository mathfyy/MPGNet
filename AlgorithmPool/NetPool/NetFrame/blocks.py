import torch
import torch.nn as nn
import torch.nn.functional as F

norm_dict = {
    'instance': nn.InstanceNorm3d,
    'batch': nn.BatchNorm3d
}


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input


class Normalize(nn.Module):
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                             align_corners=self.align_corners)


class PlainBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, kernel_size=3,
                 norm_key='instance', dropout_prob=None):
        super(PlainBlock, self).__init__()

        conv = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride,
                         padding=(kernel_size - 1) // 2, bias=True)

        do = Identity() if dropout_prob is None else nn.Dropout3d(p=dropout_prob, inplace=True)

        norm = norm_dict[norm_key](output_channels, eps=1e-5, affine=True)

        nonlin = nn.LeakyReLU(inplace=True)

        self.all = nn.Sequential(conv, do, norm, nonlin)

    def forward(self, x):
        return self.all(x)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, kernel_size=3,
                 norm_key='instance', dropout_prob=None):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride

        conv = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride,
                         padding=(kernel_size - 1) // 2, bias=True)

        norm = norm_dict[norm_key](output_channels, eps=1e-5, affine=True)

        do = Identity() if dropout_prob is None else nn.Dropout3d(p=dropout_prob, inplace=True)

        nonlin = nn.LeakyReLU(inplace=True)

        self.all = nn.Sequential(conv, norm, do, nonlin)

        # downsample residual
        if (input_channels != output_channels) or (stride != 1):
            self.downsample_skip = nn.Sequential(
                nn.Conv3d(input_channels, output_channels, 1, stride, bias=True),
                norm_dict[norm_key](output_channels, eps=1e-5, affine=True),
            )

    def forward(self, x):
        out = self.all(x)
        if (self.input_channels != self.output_channels) or (self.stride != 1):
            residual = self.downsample_skip(x)
        else:
            residual = x
        return residual + out


class conv_Block(nn.Module):
    def __init__(self, input_channels, output_channels, stride=(1, 1, 1), kernel_size=(1, 1, 1), pad_size=(0, 0, 0),
                 norm_key='instance', dropout_prob=None):
        super(conv_Block, self).__init__()

        conv = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride,
                         padding=pad_size, bias=True)

        do = Identity() if dropout_prob is None else nn.Dropout3d(p=dropout_prob, inplace=True)

        norm = norm_dict[norm_key](output_channels, eps=1e-5, affine=True)

        nonlin = nn.LeakyReLU(inplace=True)

        self.all = nn.Sequential(conv, do, norm, nonlin)

    def forward(self, x):
        return self.all(x)


class InceptionBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=(1, 1, 1), kernel_size=(1, 1, 1),
                 norm_key='instance', dropout_prob=None):
        super(InceptionBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride

        self.conv11 = conv_Block(input_channels, min(input_channels, output_channels), (1, 1, 1), (1, 1, 1), (0, 0, 0),
                                 norm_key,
                                 dropout_prob)
        self.conv21 = conv_Block(input_channels, min(input_channels, output_channels), (1, 1, 1), (1, 1, 1), (0, 0, 0),
                                 norm_key,
                                 dropout_prob)
        self.conv31 = conv_Block(input_channels, min(input_channels, output_channels), (1, 1, 1), (1, 1, 1), (0, 0, 0),
                                 norm_key,
                                 dropout_prob)

        self.conv12 = conv_Block(min(input_channels, output_channels), min(input_channels, output_channels),
                                 (1, stride[1], stride[2]), (1, 3, 3), (0, 1, 1),
                                 norm_key,
                                 dropout_prob)
        self.conv22 = conv_Block(min(input_channels, output_channels), min(input_channels, output_channels),
                                 (stride[0], stride[1], 1), (3, 3, 1), (1, 1, 0),
                                 norm_key,
                                 dropout_prob)
        self.conv32 = conv_Block(min(input_channels, output_channels), min(input_channels, output_channels),
                                 (stride[0], 1, stride[2]), (3, 1, 3), (1, 0, 1),
                                 norm_key,
                                 dropout_prob)

        self.conv13 = conv_Block(min(input_channels, output_channels), min(input_channels, output_channels),
                                 (stride[0], 1, 1), (3, 1, 1), (1, 0, 0), norm_key,
                                 dropout_prob)
        self.conv23 = conv_Block(min(input_channels, output_channels), min(input_channels, output_channels),
                                 (1, 1, stride[2]), (1, 1, 3), (0, 0, 1), norm_key,
                                 dropout_prob)
        self.conv33 = conv_Block(min(input_channels, output_channels), min(input_channels, output_channels),
                                 (1, stride[1], 1), (1, 3, 1), (0, 1, 0), norm_key,
                                 dropout_prob)

        self.conv = conv_Block(min(input_channels, output_channels), output_channels, (1, 1, 1), (1, 1, 1), (0, 0, 0),
                               norm_key,
                               dropout_prob)

        # downsample
        if (input_channels != output_channels) or (stride != (1, 1, 1)):
            self.downsample_skip = nn.Sequential(
                nn.Conv3d(input_channels, output_channels, 1, stride, bias=True),
                norm_dict[norm_key](output_channels, eps=1e-5, affine=True),
            )

    def forward(self, x):

        # x1=self.conv11(x)
        # x2=self.conv12(x1)
        # x3=self.conv13(x2)

        F1 = self.conv(self.conv13(self.conv12(self.conv11(x))))
        F2 = self.conv(self.conv23(self.conv22(self.conv21(x))))
        F3 = self.conv(self.conv33(self.conv32(self.conv31(x))))
        out = F1 + F2 + F3
        if (self.input_channels != self.output_channels) or (self.stride != (1, 1, 1)):
            inc = self.downsample_skip(x)
        else:
            inc = x
        return out + inc


class InceptionBlock_A(nn.Module):
    def __init__(self, input_channels, output_channels, stride=(1, 1, 1), kernel_size=(1, 1, 1),
                 norm_key='instance', dropout_prob=None):
        super(InceptionBlock_A, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride

        self.conv11 = conv_Block(input_channels, min(input_channels, output_channels), stride, (1, 1, 1), (0, 0, 0),
                                 norm_key,
                                 dropout_prob)
        self.conv21 = conv_Block(input_channels, min(input_channels, output_channels), stride, (1, 1, 1), (0, 0, 0),
                                 norm_key,
                                 dropout_prob)
        self.conv31 = conv_Block(input_channels, min(input_channels, output_channels), stride, (1, 1, 1), (0, 0, 0),
                                 norm_key,
                                 dropout_prob)

        self.conv12 = conv_Block(min(input_channels, output_channels), min(input_channels, output_channels), (1, 1, 1),
                                 (1, 3, 3), (0, 1, 1),
                                 norm_key,
                                 dropout_prob)
        self.conv22 = conv_Block(min(input_channels, output_channels), min(input_channels, output_channels), (1, 1, 1),
                                 (3, 3, 1), (1, 1, 0),
                                 norm_key,
                                 dropout_prob)
        self.conv32 = conv_Block(min(input_channels, output_channels), min(input_channels, output_channels), (1, 1, 1),
                                 (3, 1, 3), (1, 0, 1),
                                 norm_key,
                                 dropout_prob)

        self.conv13 = conv_Block(min(input_channels, output_channels), min(input_channels, output_channels), (1, 1, 1),
                                 (3, 1, 1), (1, 0, 0), norm_key,
                                 dropout_prob)
        self.conv23 = conv_Block(min(input_channels, output_channels), min(input_channels, output_channels), (1, 1, 1),
                                 (1, 1, 3), (0, 0, 1), norm_key,
                                 dropout_prob)
        self.conv33 = conv_Block(min(input_channels, output_channels), min(input_channels, output_channels), (1, 1, 1),
                                 (1, 3, 1), (0, 1, 0), norm_key,
                                 dropout_prob)

        self.conv = conv_Block(min(input_channels, output_channels), output_channels, (1, 1, 1), (1, 1, 1), (0, 0, 0),
                               norm_key,
                               dropout_prob)

        # downsample
        if (input_channels != output_channels) or (stride != (1, 1, 1)):
            self.downsample_skip = nn.Sequential(
                nn.Conv3d(input_channels, output_channels, 1, stride, bias=True),
                norm_dict[norm_key](output_channels, eps=1e-5, affine=True),
            )

    def forward(self, x):

        # x1=self.conv11(x)
        # x2=self.conv12(x1)
        # x3=self.conv13(x2)

        F1 = self.conv(self.conv13(self.conv12(self.conv11(x))))
        F2 = self.conv(self.conv23(self.conv22(self.conv21(x))))
        F3 = self.conv(self.conv33(self.conv32(self.conv31(x))))
        out = F1 + F2 + F3
        if (self.input_channels != self.output_channels) or (self.stride != (1, 1, 1)):
            inc = self.downsample_skip(x)
        else:
            inc = x
        return out + inc


# unit test
if __name__ == "__main__":
    # conv block
    # conv_layers = PlainBlock(input_channels=3, output_channels=32, kernel_size=3, dropout_prob=None, stride=2)
    # print(conv_layers)
    #
    # # residual block
    # res_block = ResidualBlock(32, 64, kernel_size=3, stride=1, norm_key='instance', dropout_prob=None)
    # print(res_block)
    x = torch.randn(1, 32, 64, 64, 64)
    # print(x.shape)
    # print(res_block(x).shape)

    incep_block = InceptionBlock(32, 64, kernel_size=(3, 3, 3), stride=2, norm_key='instance',
                                 dropout_prob=None)
    out = incep_block(x)
    print(out.shape)
