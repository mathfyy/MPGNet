import numpy as np
import torch
import torch.nn as nn

from typing import Union
from AlgorithmPool.NetPool.NetFrame.blocks_IPH import PlainBlock, ResidualBlock, InceptionBlock, Upsample


def max_min_normalize(feature):
    max_f = torch.max(feature)
    min_f = torch.min(feature)
    feature = (feature - min_f) / (max_f - min_f)
    return feature


def _normalize(feature):
    norm = feature.norm(p=2, dim=1, keepdim=True)
    feature = feature.div(norm + 1e-16)
    return feature


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


class SAB(nn.Module):
    def __init__(self, in_dim, out_dim, kerStride=(1, 1, 1)):
        super(SAB, self).__init__()

        self.layer = nn.Sequential()
        self.layer.add_module("conv", nn.Conv3d(in_dim, out_dim, 1, stride=kerStride, padding=0, bias=True))
        self.layer.add_module("norm", normalization3D(out_dim, 'bn'))
        self.layer.add_module("relu", nn.LeakyReLU(inplace=True))

        # self.CMO = CAM_Module_3D(out_dim)

        # self.layer1 = nn.Sequential()
        # self.layer1.add_module("conv", nn.Conv3d(in_dim + out_dim, out_dim, 3, stride=kerStride, padding=1, bias=True))
        # self.layer1.add_module("norm", normalization3D(out_dim, 'bn'))
        # self.layer1.add_module("relu", nn.LeakyReLU(inplace=True))

        # self.pro = nn.Sequential()
        # self.pro.add_module("norm", normalization3D(out_dim, 'bn'))
        # self.pro.add_module("relu", nn.LeakyReLU(inplace=True))

    def forward(self, enc, dec):
        # enc = torch.add(enc, self.layer(enc))
        DIFF = max_min_normalize(torch.sub(enc, torch.flip(enc, [4])))
        DIFF = self.layer(DIFF)

        # out = self.layer1(torch.cat((enc, DIFF), dim=1))
        # out = torch.mul(enc, DIFF)
        out = torch.add(enc, DIFF)

        # out = self.CMO(out)
        # out = torch.add(dec, out)
        out = torch.cat((dec, out), dim=1)
        # out = self.layer(out)
        return out


class UNetEncoder(nn.Module):
    """
        U-Net Encoder (include bottleneck)

        input_channels: #channels of input images, e.g. 4 for BraTS multimodal input
        channels_list:  #channels of every levels, e.g. [8, 16, 32, 64, 80, 80]
        block:          Type of conv blocks, choice from PlainBlock and ResidualBlock
    """

    def __init__(self, input_channels, channels_list,
                 block: Union[PlainBlock, ResidualBlock, InceptionBlock] = PlainBlock, **block_kwargs):
        super(UNetEncoder, self).__init__()

        self.input_channels = input_channels
        self.channels_list = channels_list  # last is bottleneck
        self.block_type = block
        num_upsample = len(self.channels_list) - 1

        self.levels = nn.ModuleList()
        for l, num_channels in enumerate(self.channels_list):
            in_channels = self.input_channels if l == 0 else self.channels_list[l - 1]
            out_channels = num_channels
            # first_stride = (2, 1, 1) if l == 0 else (1, 2, 2)  # level 0 don't downsample
            # first_stride = (2, 2, 2) if l % 2 == 0 else (1, 2, 2)
            if l == 0:
                first_stride = (1, 1, 1)
            elif l == 1 or l == 2:
                first_stride = (2, 2, 2)
            else:
                first_stride = (1, 2, 2)

                # 2 blocks per level
            blocks = nn.Sequential(
                block(in_channels, out_channels, stride=first_stride, **block_kwargs),
                block(out_channels, out_channels, stride=(1, 1, 1), **block_kwargs),
            )
            self.levels.append(blocks)

    def forward(self, x, return_skips=False):
        skips = []

        for s in self.levels:
            x = s(x)
            skips.append(x)

        return skips if return_skips else x


class UNetDecoder(nn.Module):
    """
        U-Net Decoder (include bottleneck)

        output_classes:   #classes of final ouput
        channels_list:    #channels of every levels in a bottom-up order, e.g. [320, 320, 256, 128, 64, 32]
        deep_supervision: Whether to use deep supervision
        ds_layer:         Last n layer for deep supervision, default set 0 for turning off
        block:            Type of conv blocks, better be consistent with encoder

        NOTE: Add sigmoid in the end WILL cause numerical unstability.
    """

    def __init__(self, output_classes, channels_list, deep_supervision=False, ds_layer=0,
                 block: Union[PlainBlock, ResidualBlock, InceptionBlock] = PlainBlock, **block_kwargs):
        super(UNetDecoder, self).__init__()

        self.output_classes = output_classes
        self.channels_list = channels_list  # first is bottleneck
        self.deep_supervision = deep_supervision
        self.block_type = block
        num_upsample = len(self.channels_list) - 1

        # decoder
        self.levels = nn.ModuleList()
        self.trans_convs = nn.ModuleList()
        for l in range(num_upsample):  # exclude bottleneck
            in_channels = self.channels_list[l]
            out_channels = self.channels_list[l + 1]
            if l >= num_upsample - 2:
                first_stride = (2, 2, 2)
                ker_size = (2, 2, 2)
            else:
                first_stride = (1, 2, 2)
                ker_size = (1, 2, 2)

            # transpose conv
            trans_conv = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=ker_size, padding=(0, 0, 0), stride=first_stride)
            self.trans_convs.append(trans_conv)

            # 2 blocks per level
            blocks = nn.Sequential(
                block(out_channels * 2, out_channels, stride=(1, 1, 1), **block_kwargs),
                block(out_channels, out_channels, stride=(1, 1, 1), **block_kwargs),
            )
            self.levels.append(blocks)

        # seg output
        self.seg_output = nn.Conv3d(
            self.channels_list[-1], self.output_classes, kernel_size=1, stride=1)

        # mid-layer deep supervision
        if (self.deep_supervision) and (ds_layer > 1):
            self.ds_layer_list = list(range(num_upsample - ds_layer, num_upsample - 1))
            self.ds = nn.ModuleList()
            for l in range(num_upsample - 1):
                if l in self.ds_layer_list:
                    in_channels = self.channels_list[l + 1]
                    up_factor = in_channels // self.channels_list[-1]
                    assert up_factor > 1  # otherwise downsample

                    if l == 0:
                        up_factor = [in_channels // self.channels_list[-1] // 2, in_channels // self.channels_list[-1],
                                     in_channels // self.channels_list[-1]]

                    ds = nn.Sequential(
                        nn.Conv3d(in_channels, self.output_classes, kernel_size=1, stride=1),
                        Upsample(scale_factor=up_factor, mode='trilinear', align_corners=False),
                    )
                else:
                    ds = None  # for easier indexing

                self.ds.append(ds)

    def forward(self, skips):
        skips = skips[::-1]  # reverse so that bottleneck is the first
        x = skips.pop(0)  # bottleneck

        ds_outputs = []
        for l, feat in enumerate(skips):
            x = self.trans_convs[l](x)  # upsample last-level feat
            x = torch.cat([feat, x], dim=1)  # concat upsampled feat and same-level skip feat
            x = self.levels[l](x)  # concated feat to conv

            if (self.training) and (self.deep_supervision) and (l in self.ds_layer_list):
                ds_outputs.append(self.ds[l](x))

        if self.training:
            return [self.seg_output(x)] + ds_outputs[::-1]  # reverse back
        else:
            return self.seg_output(x)


class UNetDecoder_MO(nn.Module):
    def __init__(self, output_classes, channels_list, deep_supervision=False, ds_layer=0,
                 block: Union[PlainBlock, ResidualBlock, InceptionBlock] = PlainBlock, **block_kwargs):
        super(UNetDecoder_MO, self).__init__()

        self.output_classes = output_classes
        self.channels_list = channels_list  # first is bottleneck
        self.deep_supervision = deep_supervision
        self.block_type = block
        num_upsample = len(self.channels_list) - 1

        # decoder
        self.levels = nn.ModuleList()
        self.trans_convs = nn.ModuleList()
        for l in range(num_upsample):  # exclude bottleneck
            in_channels = self.channels_list[l]
            out_channels = self.channels_list[l + 1]
            if l >= num_upsample - 2:
                first_stride = (2, 2, 2)
                ker_size = (2, 2, 2)
            else:
                first_stride = (1, 2, 2)
                ker_size = (1, 2, 2)

            # transpose conv
            trans_conv = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=ker_size, padding=(0, 0, 0), stride=first_stride)
            self.trans_convs.append(trans_conv)

            # 2 blocks per level
            blocks = nn.Sequential(
                block(out_channels * 2, out_channels, stride=(1, 1, 1), **block_kwargs),
                block(out_channels, out_channels, stride=(1, 1, 1), **block_kwargs),
            )
            self.levels.append(blocks)

        # seg output
        self.seg_output = nn.Conv3d(
            self.channels_list[-1], self.output_classes, kernel_size=1, stride=1)

        # mid-layer deep supervision
        if (self.deep_supervision) and (ds_layer > 1):
            self.ds_layer_list = list(range(num_upsample - ds_layer, num_upsample - 1))
            self.ds = nn.ModuleList()
            for l in range(num_upsample - 1):
                if l in self.ds_layer_list:
                    in_channels = self.channels_list[l + 1]
                    up_factor = in_channels // self.channels_list[-1]
                    assert up_factor > 1  # otherwise downsample

                    if l == 0:
                        up_factor = [in_channels // self.channels_list[-1] // 2, in_channels // self.channels_list[-1],
                                     in_channels // self.channels_list[-1]]

                    ds = nn.Sequential(
                        nn.Conv3d(in_channels, self.output_classes, kernel_size=1, stride=1),
                        Upsample(scale_factor=up_factor, mode='trilinear', align_corners=False),
                    )
                else:
                    ds = None  # for easier indexing

                self.ds.append(ds)

    def forward(self, skips):
        skips = skips[::-1]  # reverse so that bottleneck is the first
        x = skips.pop(0)  # bottleneck

        outputs = []

        ds_outputs = []
        for l, feat in enumerate(skips):
            x = self.trans_convs[l](x)  # upsample last-level feat
            x = torch.cat([feat, x], dim=1)  # concat upsampled feat and same-level skip feat
            x = self.levels[l](x)  # concated feat to conv

            outputs.append(x)

            if (self.training) and (self.deep_supervision) and (l in self.ds_layer_list):
                ds_outputs.append(self.ds[l](x))

        outputs.append(self.seg_output(x))

        return outputs



class UNetDecoder_SAB(nn.Module):
    """
        U-Net Decoder (include bottleneck)

        output_classes:   #classes of final ouput
        channels_list:    #channels of every levels in a bottom-up order, e.g. [320, 320, 256, 128, 64, 32]
        deep_supervision: Whether to use deep supervision
        ds_layer:         Last n layer for deep supervision, default set 0 for turning off
        block:            Type of conv blocks, better be consistent with encoder

        NOTE: Add sigmoid in the end WILL cause numerical unstability.
    """

    def __init__(self, output_classes, channels_list, deep_supervision=False, ds_layer=0,
                 block: Union[PlainBlock, ResidualBlock] = PlainBlock, **block_kwargs):
        super(UNetDecoder_SAB, self).__init__()

        self.output_classes = output_classes
        self.channels_list = channels_list  # first is bottleneck
        self.deep_supervision = deep_supervision
        self.block_type = block
        num_upsample = len(self.channels_list) - 1

        # decoder
        self.levels = nn.ModuleList()
        self.trans_convs = nn.ModuleList()
        self.SABs = nn.ModuleList()
        for l in range(num_upsample):  # exclude bottleneck
            in_channels = self.channels_list[l]
            out_channels = self.channels_list[l + 1]
            if l >= num_upsample - 2:
                first_stride = (2, 2, 2)
                ker_size = (2, 2, 2)
            else:
                first_stride = (1, 2, 2)
                ker_size = (1, 2, 2)

            # transpose conv
            trans_conv = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=ker_size, padding=(0, 0, 0), stride=first_stride)
            self.trans_convs.append(trans_conv)

            # 2 blocks per level
            blocks = nn.Sequential(
                block(out_channels * 2, out_channels, stride=1, **block_kwargs),
                block(out_channels, out_channels, stride=1, **block_kwargs),
            )
            self.levels.append(blocks)

            # add SAB
            SABs = SAB(out_channels, out_channels)
            self.SABs.append(SABs)

        # seg output 
        self.seg_output = nn.Conv3d(
            self.channels_list[-1], self.output_classes, kernel_size=1, stride=1)

        # mid-layer deep supervision
        if (self.deep_supervision) and (ds_layer > 1):
            self.ds_layer_list = list(range(num_upsample - ds_layer, num_upsample - 1))
            self.ds = nn.ModuleList()
            for l in range(num_upsample - 1):
                if l in self.ds_layer_list:
                    in_channels = self.channels_list[l + 1]
                    up_factor = in_channels // self.channels_list[-1]
                    assert up_factor > 1  # otherwise downsample

                    ds = nn.Sequential(
                        nn.Conv3d(in_channels, self.output_classes, kernel_size=1, stride=1),
                        Upsample(scale_factor=up_factor, mode='trilinear', align_corners=False),
                    )
                else:
                    ds = None  # for easier indexing

                self.ds.append(ds)

    def forward(self, skips):
        skips = skips[::-1]  # reverse so that bottleneck is the first
        x = skips.pop(0)  # bottleneck

        ds_outputs = []
        for l, feat in enumerate(skips):
            x = self.trans_convs[l](x)  # upsample last-level feat
            x = self.SABs[l](feat, x)
            # x = torch.cat([feat, x], dim=1)  # concat upsampled feat and same-level skip feat
            x = self.levels[l](x)  # concated feat to conv

            if (self.training) and (self.deep_supervision) and (l in self.ds_layer_list):
                ds_outputs.append(self.ds[l](x))

        if self.training:
            return [self.seg_output(x)] + ds_outputs[::-1]  # reverse back
        else:
            return self.seg_output(x)


class UNet(nn.Module):
    """
        U-Net

        input_channels:   #channels of input images, e.g. 4 for BraTS multimodal input
        output_classes:   #classes of final ouput
        channels_list:    #channels of every levels in a top-down order, e.g. [32, 64, 128, 256, 320, 320]
        block:            Type of conv blocks, choice from PlainBlock and ResidualBlock
        deep_supervision: Whether to use deep supervision in decoder
        ds_layer:         Last n layer for deep supervision, default set 0 for turning off
    """

    def __init__(self, input_channels, output_classes, channels_list, deep_supervision=False,
                 ds_layer=0, block: Union[PlainBlock, ResidualBlock] = PlainBlock, **block_kwargs):
        super(UNet, self).__init__()

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)
        self.decoder = UNetDecoder(output_classes, channels_list[::-1], block=block,
                                   deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)
        self.relu = nn.Sigmoid()
        # self.relu = nn.Softmax()

    def forward(self, x):
        out = self.decoder(self.encoder(x, return_skips=True))
        out = self.relu(out[0])
        return out


class ResUNet(nn.Module):
    """
        U-Net

        input_channels:   #channels of input images, e.g. 4 for BraTS multimodal input
        output_classes:   #classes of final ouput
        channels_list:    #channels of every levels in a top-down order, e.g. [32, 64, 128, 256, 320, 320]
        block:            Type of conv blocks, choice from PlainBlock and ResidualBlock
        deep_supervision: Whether to use deep supervision in decoder
        ds_layer:         Last n layer for deep supervision, default set 0 for turning off
    """

    def __init__(self, input_channels, output_classes, channels_list, deep_supervision=False,
                 ds_layer=0, block: Union[PlainBlock, ResidualBlock] = ResidualBlock, **block_kwargs):
        super(ResUNet, self).__init__()

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)
        self.decoder = UNetDecoder(output_classes, channels_list[::-1], block=block,
                                   deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)
        self.relu = nn.Sigmoid()
        # self.relu = nn.Softmax()

    def forward(self, x):
        out = self.decoder(self.encoder(x, return_skips=True))
        out = self.relu(out[0])
        return out


class InceptionUNet(nn.Module):
    def __init__(self, input_channels, output_classes, channels_list, deep_supervision=False,
                 ds_layer=0, block: Union[InceptionBlock] = InceptionBlock, **block_kwargs):
        super(InceptionUNet, self).__init__()

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)
        self.decoder = UNetDecoder(output_classes, channels_list[::-1], block=block,
                                   deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)
        self.relu = nn.Sigmoid()
        # self.relu = nn.Softmax()

    def forward(self, x):
        out = self.decoder(self.encoder(x, return_skips=True))
        out = self.relu(out[0])
        return out


# class UpsamplingBlock(nn.Module):
#     """
#     Upsampling Block (upsampling layer) as specified by the paper
#     This is composed of a 2d bilinear upsampling layer followed by a convolutional layer, BatchNorm layer, and ReLU activation
#     """
#
#     def __init__(self, in_channels: int, out_channels: int, size: Tuple):
#         """
#         Create the layers for the upsampling block
#         :param in_channels:   number of input features to the block
#         :param out_channels:  number of output features from this entire block
#         :param scale_factor:  tuple to determine how to scale the dimensions
#         :param residual:      residual from the opposite dense block to add before upsampling
#         """
#         super().__init__()
#         # blinear vs trilinear kernel size and padding
#         if size[0] == 2:
#             d_kernel_size = 3
#             d_padding = 1
#         else:
#             d_kernel_size = 1
#             d_padding = 0
#
#         self.upsample = nn.Upsample(
#             scale_factor=size, mode="trilinear", align_corners=True
#         )
#         self.conv = nn.Sequential(
#             nn.Conv3d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=(d_kernel_size, 3, 3),
#                 padding=(d_padding, 1, 1),
#             ),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(),
#         )
#
#     def forward(self, x, projected_residual):
#         """
#         Forward pass through the block
#         :param x:  image tensor
#         :return:   output of the forward pass
#         """
#         residual = torch.cat(
#             (self.upsample(x), self.upsample(projected_residual)),
#             dim=1,
#         )
#         return self.conv(residual)
#
#
# class DenseBlock(nn.Module):
#     """
#     Repeatable Dense block as specified by the paper
#     This is composed of a pointwise convolution followed by a depthwise separable convolution
#     After each convolution is a BatchNorm followed by a ReLU
#     Some notes on architecture based on the paper:
#       - The first block uses an input channel of 96, and the remaining input channels are 32
#       - The hidden channels is always 128
#       - The output channels is always 32
#       - The depth is always 3
#     """
#
#     def __init__(
#             self, in_channels: int, hidden_channels: int, out_channels: int, count: int
#     ):
#         """
#         Create the layers for the dense block
#         :param in_channels:      number of input features to the block
#         :param hidden_channels:  number of output features from the first convolutional layer
#         :param out_channels:     number of output features from this entire block
#         :param count:            number of times to repeat
#         """
#         super().__init__()
#
#         # First iteration takes different number of input channels and does not repeat
#         first_block = [
#             nn.Conv3d(in_channels, hidden_channels, kernel_size=(1, 1, 1)),
#             nn.BatchNorm3d(hidden_channels),
#             nn.ReLU(),
#             nn.Conv3d(
#                 hidden_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
#             ),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(),
#         ]
#
#         # Remaining repeats are identical blocks
#         repeating_block = [
#             nn.Conv3d(out_channels, hidden_channels, kernel_size=(1, 1, 1)),
#             nn.BatchNorm3d(hidden_channels),
#             nn.ReLU(),
#             nn.Conv3d(
#                 hidden_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
#             ),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(),
#         ]
#
#         self.convs = nn.Sequential(
#             *first_block,
#             *repeating_block * (count - 1),
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass through the block
#         :param x:  image tensor
#         :return:   output of the forward pass
#         """
#         return self.convs(x)
#
#
# class TransitionBlock(nn.Module):
#     """
#     Transition Block (transition layer) as specified by the paper
#     This is composed of a pointwise convolution followed by a pointwise convolution with higher stride to reduce the image size
#     We use BatchNorm and ReLU after the first convolution, but not the second
#     Some notes on architecture based on the paper:
#       - The number of channels is always 32
#       - The depth is always 3
#     """
#
#     def __init__(self, channels: int):
#         """
#         Create the layers for the transition block
#         :param channels:  number of input and output channels, which should be equal
#         """
#         super().__init__()
#         self.convs = nn.Sequential(
#             nn.Conv3d(channels, channels, kernel_size=(1, 1, 1)),
#             nn.BatchNorm3d(channels),
#             nn.ReLU(),
#             # This conv layer is analogous to H-Dense-UNet's 1x2x2 average pool
#             nn.Conv3d(channels, channels, kernel_size=(1, 1, 1), stride=(1, 2, 2)),
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass through the block
#         :param x:  image tensor
#         :return:   output of the forward pass
#         """
#         return self.convs(x)
#
#
# class DenseUNet3d(nn.Module):
#     def __init__(self):
#         """
#         Create the layers for the model
#         """
#         super().__init__()
#         # Initial Layers
#         self.conv1 = nn.Conv3d(
#             1, 96, kernel_size=(7, 7, 7), stride=2, padding=(3, 3, 3)
#         )
#         self.bn1 = nn.BatchNorm3d(96)
#         self.relu = nn.ReLU()
#         self.maxpool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1))
#
#         # Dense Layers
#         self.transition = TransitionBlock(32)
#         self.dense1 = DenseBlock(96, 128, 32, 4)
#         self.dense2 = DenseBlock(32, 128, 32, 12)
#         self.dense3 = DenseBlock(32, 128, 32, 24)
#         self.dense4 = DenseBlock(32, 32, 32, 36)
#
#         # Upsampling Layers
#         self.upsample1 = UpsamplingBlock(32 + 32, 504, size=(1, 2, 2))
#         self.upsample2 = UpsamplingBlock(504 + 32, 224, size=(1, 2, 2))
#         self.upsample3 = UpsamplingBlock(224 + 32, 192, size=(1, 2, 2))
#         self.upsample4 = UpsamplingBlock(192 + 32, 96, size=(2, 2, 2))
#         self.upsample5 = UpsamplingBlock(96 + 96, 64, size=(2, 2, 2))
#
#         # Final output layer
#         # Typo in the paper? Says stride = 0 but that's impossible
#         self.conv_classifier = nn.Conv3d(64, 3, kernel_size=1, stride=1)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass through the model
#         :param x:  image tensor
#         :return:   output of the forward pass
#         """
#         residual1 = self.relu(self.bn1(self.conv1(x)))
#         residual2 = self.dense1(self.maxpool1(residual1))
#         residual3 = self.dense2(self.transition(residual2))
#         residual4 = self.dense3(self.transition(residual3))
#         output = self.dense4(self.transition(residual4))
#
#         output = self.upsample1(output, output)
#         output = self.upsample2(output, residual4)
#         output = self.upsample3(output, residual3)
#         output = self.upsample4(output, residual2)
#         output = self.upsample5(output, residual1)
#
#         output = self.conv_classifier(output)
#
#         return output


class BasicBlock3D(nn.Module):
    def __init__(self, inC, outC, kerSize=3, kerStride=(1, 1, 1), kerPad=1, dropoutRate=0.5, norm="bn"):
        super(BasicBlock3D, self).__init__()
        self.layer = nn.Sequential()
        self.layer.add_module("conv", nn.Conv3d(inC, outC, kerSize, stride=kerStride, padding=kerPad, bias=True))
        self.layer.add_module("norm", normalization3D(outC, norm))
        self.layer.add_module("relu", nn.LeakyReLU(inplace=True))

    def forward(self, x):
        y = torch.add(x, self.layer(x))
        return y


class SpatialAttention3D(nn.Module):
    """ Spatial attention module"""

    def __init__(self, inC=2, outC=3, kernel_size=3):
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
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(ChannelAttention3D, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
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


class DAResBlock3D(nn.Module):
    def __init__(self, inC, outC, kerSize=(3, 3, 3), kerStride=1, kerPad=1, dropoutRate=0.5, norm="bn"):
        super(DAResBlock3D, self).__init__()
        self.layerS = nn.Sequential()
        self.layerS.add_module("conv", nn.Conv3d(inC, inC, kerSize, stride=kerStride, padding=kerPad, bias=True))
        self.layerS.add_module("norm", normalization3D(inC, norm))
        self.layerS.add_module("relu", nn.LeakyReLU(inplace=True))

        self.SMO = SpatialAttention3D(inC, outC)

        self.layerS1 = nn.Sequential()
        self.layerS1.add_module("conv",
                                nn.Conv3d(outC, outC, kerSize, stride=kerStride, padding=kerPad, bias=True))
        self.layerS1.add_module("norm", normalization3D(outC, norm))
        self.layerS1.add_module("relu", nn.LeakyReLU(inplace=True))

        self.layerC = nn.Sequential()
        self.layerC.add_module("conv", nn.Conv3d(inC, inC, kerSize, stride=kerStride, padding=kerPad, bias=True))
        self.layerC.add_module("norm", normalization3D(inC, norm))
        self.layerC.add_module("relu", nn.LeakyReLU(inplace=True))

        self.CMO = ChannelAttention3D(inC)

        self.layerC1 = nn.Sequential()
        self.layerC1.add_module("conv",
                                nn.Conv3d(inC, outC, kerSize, stride=kerStride, padding=kerPad, bias=True))
        self.layerC1.add_module("norm", normalization3D(outC, norm))
        self.layerC1.add_module("relu", nn.LeakyReLU(inplace=True))

        self.layer = nn.Sequential()
        self.layer.add_module("conv",
                              nn.Conv3d(outC * 2, outC, kerSize, stride=kerStride, padding=kerPad, bias=True))
        self.layer.add_module("norm", normalization3D(outC, norm))
        self.layer.add_module("relu", nn.LeakyReLU(inplace=True))

    def forward(self, x):
        sBlock = self.layerS(x)
        sBlock = self.SMO(sBlock)
        sBlock = self.layerS1(sBlock)

        cBlock = self.layerC(x)
        cBlock = self.CMO(cBlock)
        cBlock = self.layerC1(cBlock)

        block = self.layer(torch.cat((sBlock, cBlock), dim=1))

        return block


class DAResUNet(nn.Module):
    def __init__(self, input_channels, output_classes, channels_list, deep_supervision=False,
                 ds_layer=0, block: Union[PlainBlock, ResidualBlock] = PlainBlock, **block_kwargs):
        super(DAResUNet, self).__init__()

        self.basicBlock = BasicBlock3D(input_channels, input_channels)
        self.daresBlock = DAResBlock3D(input_channels, input_channels)
        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)
        self.decoder = UNetDecoder(output_classes, channels_list[::-1], block=block,
                                   deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.conv = nn.Conv3d(input_channels + output_classes, output_classes, kernel_size=1, stride=1, padding=0,
                              bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, x):
        basicOut = self.basicBlock(x)
        daresOut = self.daresBlock(basicOut)
        enc = self.encoder(x, return_skips=True)
        out = self.decoder(enc)
        out = self.relu(self.conv(torch.cat((daresOut, out[0]), dim=1)))
        return out


class brainSANet(nn.Module):
    def __init__(self, input_channels, output_classes, channels_list, deep_supervision=False,
                 ds_layer=0, block: Union[PlainBlock, ResidualBlock] = PlainBlock, **block_kwargs):
        super(brainSANet, self).__init__()
        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)

        self.global_pool = nn.AvgPool3d(1, 4, 0)
        self.linear_layer = nn.Linear(channels_list[4] * 2 * 2 * 1, 128)
        self.ReLU1 = nn.LeakyReLU(inplace=True)

        self.decoder = UNetDecoder_SAB(output_classes, channels_list[::-1], block=block,
                                       deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.relu = nn.Sigmoid()

    def forward(self, x):
        enc = self.encoder(x, return_skips=True)

        TF_l = enc[4][:, :, :, :, 0:4]
        TF_r = torch.flip(enc[4][:, :, :, :, 4:8], [4])
        TF_l_ = _normalize(self.ReLU1(self.linear_layer(torch.flatten(self.global_pool(TF_l), start_dim=1, end_dim=4))))
        TF_r_ = _normalize(self.ReLU1(self.linear_layer(torch.flatten(self.global_pool(TF_r), start_dim=1, end_dim=4))))

        out = self.decoder(enc)
        out = self.relu(out[0])

        return out, TF_l_, TF_r_


class brainSANetCS(nn.Module):
    def __init__(self, input_channels, output_classes, channels_list, deep_supervision=False,
                 ds_layer=0, block: Union[PlainBlock, ResidualBlock] = PlainBlock, **block_kwargs):
        super(brainSANetCS, self).__init__()

        self.basicBlock = BasicBlock3D(input_channels, input_channels)
        self.daresBlock = DAResBlock3D(input_channels, input_channels)
        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)

        self.global_pool = nn.AvgPool3d(1, 4, 0)
        self.linear_layer = nn.Linear(channels_list[4] * 2 * 4 * 2, 128)
        self.ReLU1 = nn.LeakyReLU(inplace=True)

        self.decoder = UNetDecoder_SAB(output_classes, channels_list[::-1], block=block,
                                       deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)
        self.conv = nn.Conv3d(input_channels + output_classes, output_classes, kernel_size=1, stride=1, padding=0,
                              bias=False)

        # self.relu = nn.Sigmoid()
        self.relu = nn.Softmax(dim=1)

    def forward(self, x):
        basicOut = self.basicBlock(x)
        daresOut = self.daresBlock(basicOut)

        enc = self.encoder(x, return_skips=True)

        TF_l = enc[4][:, :, :, :, 0:8]
        TF_r = torch.flip(enc[4][:, :, :, :, 8:16], [4])

        # temp = self.global_pool(TF_l)
        TF_l_ = _normalize(self.ReLU1(self.linear_layer(torch.flatten(self.global_pool(TF_l), start_dim=1, end_dim=4))))
        TF_r_ = _normalize(self.ReLU1(self.linear_layer(torch.flatten(self.global_pool(TF_r), start_dim=1, end_dim=4))))

        out = self.decoder(enc)
        out = self.relu(self.conv(torch.cat((daresOut, out[0]), dim=1)))

        return out, TF_l_, TF_r_


class MultiEncoderUNet(nn.Module):
    """
        Multi-encoder U-Net for Multimodal Input

        input_channels:   #channels of input images, also is the #encoders
        output_classes:   #classes of final ouput
        channels_list:    #channels of every levels of decoder in a top-down order, e.g. [32, 64, 128, 256, 320, 320]
        block:            Type of conv blocks, choice from PlainBlock and ResidualBlock
        deep_supervision: Whether to use deep supervision in decoder
        ds_layer:         Last n layer for deep supervision, default set 0 for turning off
    """

    def __init__(self, input_channels, output_classes, channels_list, deep_supervision=False,
                 ds_layer=0, block: Union[PlainBlock, ResidualBlock] = PlainBlock, **block_kwargs):
        super(MultiEncoderUNet, self).__init__()

        self.num_skips = len(channels_list)
        self.num_encoders = input_channels
        if isinstance(channels_list, list):
            channels_list = np.array(channels_list)

        # encoders
        self.encoders = nn.ModuleList()
        for _ in range(self.num_encoders):
            self.encoders.append(
                UNetEncoder(1, (channels_list // self.num_encoders), block=block, **block_kwargs))

        # all encoders shared one decoder
        self.decoder = UNetDecoder(output_classes, channels_list[::-1], block=block,
                                   deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

    def forward(self, x: torch.Tensor):
        # seperate skips for every encoder
        encoders_skips = []
        for xx, encoder in zip(x.chunk(self.num_encoders, dim=1), self.encoders):
            encoders_skips.append(encoder(xx, return_skips=True))

        # concat same-level skip of different encoders
        encoders_skips = [
            torch.cat([encoders_skips[i][j] for i in range(self.num_encoders)], dim=1)
            for j in range(self.num_skips)]

        return self.decoder(encoders_skips)


# unit tests
if __name__ == "__main__":
    block_kwargs = {
        "kernel_size": 3,
        "conv_bias": True,
        "dropout_prob": None,
        "norm_key": 'instance'
    }

    # input 
    x = torch.rand(1, 4, 128, 128, 128)
    channels = np.array([32, 64, 128, 256, 320, 320])

    # unet
    unet = UNet(4, 3, channels, deep_supervision=True, ds_layer=4, **block_kwargs)
    # print(unet)
    unet.eval()
    segs = unet(x)
    print(segs.shape)

    # multi-encoder unet
    mencoder_unet = MultiEncoderUNet(4, 3, channels, deep_supervision=True, ds_layer=4, **block_kwargs)
    # print(mencoder_unet)
    segs = mencoder_unet(x)
    for s in segs:
        print(s.shape)
