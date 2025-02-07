import torch
import torch.nn as nn
from AlgorithmPool.NetPool.NetFrame.TransBTS.Transformer import TransformerModel
from AlgorithmPool.NetPool.NetFrame.TransBTS.PositionalEncoding import FixedPositionalEncoding, \
    LearnedPositionalEncoding
from AlgorithmPool.NetPool.NetFrame.TransBTS.Unet_skipconnection import Unet
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class TransformerBTS(nn.Module):
    def __init__(
            self,
            img_dim,
            patch_dim,
            channel_list,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            dropout_rate=0.0,
            attn_dropout_rate=0.0,
            conv_patch_representation=True,
            positional_encoding_type="learned",
    ):
        super(TransformerBTS, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim[0] % patch_dim[0] == 0
        assert img_dim[1] % patch_dim[1] == 0
        assert img_dim[2] % patch_dim[2] == 0

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        # num_patches[0] = int((img_dim[0] // patch_dim[0]) ** 3)
        # num_patches[1] = int((img_dim[1] // patch_dim[1]) ** 3)
        # num_patches[2] = int((img_dim[2] // patch_dim[2]) ** 3)
        # num_patches = int((img_dim[2] // patch_dim[2]) ** 3)
        self.seq_length = [int((img_dim[0] // patch_dim[0]) ** 3), int((img_dim[1] // patch_dim[1]) ** 3),
                           int((img_dim[2] // patch_dim[2]) ** 3)]
        self.flatten_dim = channel_list[-1] * num_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,

            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        # if self.conv_patch_representation:
        #     self.conv_x = nn.Conv3d(
        #         channel_list[-1],
        #         self.embedding_dim,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1
        #     )
        self.conv_x = nn.Conv3d(
            channel_list[-1],
            self.embedding_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.norm = nn.InstanceNorm3d(num_channels, eps=1e-5, affine=True)
        self.Unet = Unet(in_channels=num_channels, base_channels=channel_list[0], num_classes=4)
        self.bn = nn.BatchNorm3d(channel_list[-1])
        self.relu = nn.ReLU(inplace=True)

    def encode(self, x):
        x = self.norm(x)
        if self.conv_patch_representation:
            # combine embedding with conv patch distribution
            x1_1, x2_1, x3_1, x4_1, x = self.Unet(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv_x(x)
            x = x.permute(0, 2, 3, 4, 1).contiguous()
            x = x.view(x.size(0), -1, self.embedding_dim)

        else:
            x1_1, x2_1, x3_1, x4_1, x = self.Unet(x)
            x = self.bn(x)
            x = self.relu(x)
            x = (
                x.unfold(2, 2, 2)
                .unfold(3, 2, 2)
                .unfold(4, 2, 2)
                .contiguous()
            )
            x = x.view(x.size(0), x.size(1), -1, 8)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.flatten_dim)
            x = self.linear_encoding(x)

        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)

        return x1_1, x2_1, x3_1, x4_1, x, intmd_x

    def decode(self, x):
        raise NotImplementedError("Should be implemented in child class!!")

    def forward(self, x, auxillary_output_layers=[1, 2, 3, 4]):

        x1_1, x2_1, x3_1, x4_1, encoder_output, intmd_encoder_outputs = self.encode(x)

        decoder_output = self.decode(
            x1_1, x2_1, x3_1, x4_1, encoder_output, intmd_encoder_outputs, auxillary_output_layers
        )

        if auxillary_output_layers is not None:
            auxillary_outputs = {}
            for i in auxillary_output_layers:
                val = str(2 * i - 1)
                _key = 'Z' + str(i)
                auxillary_outputs[_key] = intmd_encoder_outputs[val]

            return decoder_output

        return decoder_output

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim[0] / self.patch_dim[0]),
            int(self.img_dim[1] / (self.patch_dim[1])),
            int(self.img_dim[2] / (self.patch_dim[2])),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x


class BTS(TransformerBTS):
    def __init__(
            self,
            img_dim,
            patch_dim,
            channel_list,
            num_channels,
            num_classes,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            dropout_rate=0.0,
            attn_dropout_rate=0.0,
            conv_patch_representation=True,
            positional_encoding_type="learned",
    ):
        super(BTS, self).__init__(
            img_dim=img_dim,
            patch_dim=patch_dim,
            channel_list=channel_list,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            conv_patch_representation=conv_patch_representation,
            positional_encoding_type=positional_encoding_type,
        )

        self.num_classes = num_classes

        # self.Softmax = nn.Softmax(dim=1)
        self.Sigmoid = nn.Sigmoid()

        self.Enblock8_1 = EnBlock1(in_channels=self.embedding_dim, step=2)
        self.Enblock8_2 = EnBlock2(in_channels=self.embedding_dim // 2)

        self.DeUp5 = DeUp_Cat(in_channels=self.embedding_dim // 2, out_channels=self.embedding_dim // 2,
                              kernel_size=(1, 2, 2), pad=(0, 0, 0), stride=(1, 2, 2))
        self.DeBlock5 = DeBlock(in_channels=self.embedding_dim // 2)

        self.DeUp4 = DeUp_Cat(in_channels=self.embedding_dim // 2, out_channels=self.embedding_dim // 4,
                              kernel_size=(1, 2, 2), pad=(0, 0, 0), stride=(1, 2, 2))
        self.DeBlock4 = DeBlock(in_channels=self.embedding_dim // 4)

        self.DeUp3 = DeUp_Cat(in_channels=self.embedding_dim // 4, out_channels=self.embedding_dim // 8)
        self.DeBlock3 = DeBlock(in_channels=self.embedding_dim // 8)

        self.DeUp2 = DeUp_Cat(in_channels=self.embedding_dim // 8, out_channels=self.embedding_dim // 16)
        self.DeBlock2 = DeBlock(in_channels=self.embedding_dim // 16)

        self.endconv = nn.Conv3d(self.embedding_dim // 16, num_classes, kernel_size=1)

    def decode(self, x1_1, x2_1, x3_1, x4_1, x, intmd_x, intmd_layers=[1, 2, 3, 4]):
        assert intmd_layers is not None, "pass the intermediate layers for MLA"
        encoder_outputs = {}
        all_keys = []
        for i in intmd_layers:
            val = str(2 * i - 1)
            _key = 'Z' + str(i)
            all_keys.append(_key)
            encoder_outputs[_key] = intmd_x[val]
        all_keys.reverse()

        x8 = encoder_outputs[all_keys[0]]
        x8 = self._reshape_output(x8)
        x8 = self.Enblock8_1(x8)
        x8 = self.Enblock8_2(x8)

        y5 = self.DeUp5(x8, x4_1)  # (1, 64, 32, 32, 32)
        y5 = self.DeBlock5(y5)

        y4 = self.DeUp4(y5, x3_1)  # (1, 64, 32, 32, 32)
        y4 = self.DeBlock4(y4)

        y3 = self.DeUp3(y4, x2_1)  # (1, 32, 64, 64, 64)
        y3 = self.DeBlock3(y3)

        y2 = self.DeUp2(y3, x1_1)  # (1, 16, 128, 128, 128)
        y2 = self.DeBlock2(y2)

        y = self.endconv(y2)  # (1, 4, 128, 128, 128)
        y = self.Sigmoid(y)
        return y


class EnBlock1(nn.Module):
    def __init__(self, in_channels, step):
        super(EnBlock1, self).__init__()

        self.bn1 = nn.BatchNorm3d(in_channels // step)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(in_channels // step)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels // step, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels // step, in_channels // step, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)

        return x1


class EnBlock2(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock2, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1


class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 2, 2), pad=(0, 0, 0), stride=(2, 2, 2)):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad)
        self.conv3 = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        # y = y + prev
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y


class DeBlock(nn.Module):
    def __init__(self, in_channels):
        super(DeBlock, self).__init__()

        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1


def TransBTS(dataset='brats', _conv_repr=True,
             _pe_type="learned"):
    if dataset.lower() == 'brats':
        img_dim = [128, 128, 128]
        num_classes = 4
    else:
        img_dim = [24, 256, 256]
        num_classes = 3

    num_channels = 4
    patch_dim = [4, 16, 16]
    channel_list = [4 * 3, 8 * 3, 16 * 3, 32 * 3, 32 * 3]
    aux_layers = [1, 2, 3, 4]
    model = BTS(
        img_dim,
        patch_dim,
        channel_list,
        num_channels,
        num_classes,
        embedding_dim=channel_list[-1] * 2,
        num_heads=8,
        num_layers=4,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=_conv_repr,
        positional_encoding_type=_pe_type,
    )

    return aux_layers, model


if __name__ == '__main__':
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.rand((4, 4, 24, 256, 256), device=device)
        _, model = TransBTS(dataset='ydy_brats', _conv_repr=True, _pe_type="fixed")
        model.cuda()
        y = model(x)
        print(y.shape)
