import numpy as np
import torch
import torch.nn as nn
from typing import Union
import torch.optim as optim
import torch.distributed as dist
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, OptTensor, Size

import torch_scatter

# import networkx as nx
import torch_geometric.nn as pyGnn

from torch_geometric.data import Data as pygData
from torch_geometric.data import Batch as pygBatch

from scipy.spatial.distance import pdist, squareform

from AlgorithmPool.NetPool.NetFrame.blocks_IPH import PlainBlock, ResidualBlock, InceptionBlock, Upsample
from AlgorithmPool.NetPool.NetFrame.unet3D_IPH import UNetEncoder, UNetDecoder, UNetDecoder_MO
from AlgorithmPool.NetPool.NetFrame.kmax_transformer_decoder import kMaXTransformerDecoder, kMaXTransformerDecoder_new
from AlgorithmPool.NetPool.NetFrame.MedNeXt import MedNeXt, MedNeXt_MO, MedNeXt_MO_pyG


class dResUNet(nn.Module):
    def __init__(self, input_channels, output_classes, channels_list, deep_supervision=False,
                 ds_layer=(1, 1, 1), block: Union[PlainBlock, ResidualBlock] = ResidualBlock, **block_kwargs):
        super(dResUNet, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)
        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)
        # self.out_bn = nn.BatchNorm3d(output_classes, eps=1e-3, momentum=0.01)
        # self.out_bn = nn.InstanceNorm3d(input_channels, eps=1e-5)
        # self.relu = nn.Softmax(dim=1)
        self.relu = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        # tt = self.encoder(x, return_skips=True)
        out = self.decoder(self.encoder(x, return_skips=True))
        # out.append(self.relu(out[4]))
        return self.relu(out[4])


class MBANet(nn.Module):
    def __init__(self, input_channels, output_classes, channels_list, deep_supervision=False,
                 ds_layer=(1, 1, 1), block: Union[PlainBlock, ResidualBlock] = PlainBlock, **block_kwargs):
        super(MBANet, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)
        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.PA0 = SpatialAttention3D(outC=channels_list[0] // 2)
        self.CA0 = CAM_Module_3D()

        self.PA1 = SpatialAttention3D(outC=channels_list[1] // 2)
        self.CA1 = CAM_Module_3D()

        self.PA2 = SpatialAttention3D(outC=channels_list[2] // 2)
        self.CA2 = CAM_Module_3D()

        self.PA3 = SpatialAttention3D(outC=channels_list[3] // 2)
        self.CA3 = CAM_Module_3D()

        self.relu = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        enOut = self.encoder(x, return_skips=True)

        enOut[0] = torch.cat((self.PA0(enOut[0][:, 0:6, :, :, :]), self.CA0(enOut[0][:, 6:12, :, :, :])), dim=1)

        enOut[1] = torch.cat((self.PA1(enOut[1][:, 0:12, :, :, :]), self.CA1(enOut[1][:, 12:24, :, :, :])), dim=1)

        enOut[2] = torch.cat((self.PA2(enOut[2][:, 0:24, :, :, :]), self.CA2(enOut[2][:, 24:48, :, :, :])), dim=1)

        enOut[3] = torch.cat((self.PA3(enOut[3][:, 0:48, :, :, :]), self.CA3(enOut[3][:, 48:96, :, :, :])), dim=1)

        out = self.decoder(enOut)
        return self.relu(out[4])


class nnUNet(nn.Module):
    def __init__(self, input_channels, output_classes, channels_list, deep_supervision=False,
                 ds_layer=(1, 1, 1), block: Union[PlainBlock, ResidualBlock] = PlainBlock, **block_kwargs):
        super(nnUNet, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)
        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)
        # self.out_bn = nn.BatchNorm3d(output_classes, eps=1e-3, momentum=0.01)
        # self.out_bn = nn.InstanceNorm3d(input_channels, eps=1e-5)
        # self.relu = nn.Softmax(dim=1)
        self.relu = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        # tt = self.encoder(x, return_skips=True)
        out = self.decoder(self.encoder(x, return_skips=True))
        # out.append(self.relu(out[4]))
        return self.relu(out[4])


class nnUNet_pro(nn.Module):
    def __init__(self, input_channels, output_classes, channels_list, deep_supervision=False,
                 ds_layer=(1, 1, 1), block: Union[PlainBlock, ResidualBlock] = PlainBlock, **block_kwargs):
        super(nnUNet_pro, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)

        self.GRF = GlobalRelateFeature_use(4, channels_list[-1], channels_list[-1])

        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)
        self.relu = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        out_en = self.encoder(x, return_skips=True)
        brige = self.GRF(out_en[4])
        out_en[4] = brige[1]
        out = self.decoder(out_en)
        return self.relu(out[4])


class nnUNet_kMax_deeplab(nn.Module):
    def __init__(self, input_channels, output_classes, channels_list, dec_layers=(1, 1, 1),
                 in_channels=(32 * 3, 16 * 3, 8 * 3, 4 * 3), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(nnUNet_kMax_deeplab, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)
        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.transDecoder = kMaXTransformerDecoder_new(dec_layers, in_channels, output_classes - 2, num_queries,
                                                       drop_path_prob)

        self.out_conv = nn.Conv3d(32, output_classes - 1, kernel_size=1, stride=1)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        out = self.decoder(self.encoder(x, return_skips=True))
        out_Trans = self.transDecoder(out[0:3], self.relu(out[3]))
        pre = self.sigmoid(self.out_conv(out_Trans['pred_masks']))
        return pre


class PAM_Module_A(nn.Module):
    def __init__(self, in_dim):
        super(PAM_Module_A, self).__init__()
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=3, stride=(1, 1, 1),
                                    padding=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=3, stride=(1, 1, 1),
                                  padding=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width, deep = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height * deep).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height * deep)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        return attention


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


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, bias=False):
        super(GraphConvolution, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)  # xavier初始化，就是论文里的glorot初始化
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs, adj):
        # inputs: (N, n_channels), adj: sparse_matrix (N, N)
        support = torch.mm(self.dropout(inputs), self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, n_features, hidden_dim, dropout, n_classes):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(n_features, hidden_dim, dropout)
        self.gc2 = GraphConvolution(hidden_dim, n_classes, dropout)
        self.relu = nn.ReLU()

    def forward(self, inputs, adj):
        x = inputs
        x_gc1 = self.relu(self.gc1(x, adj))
        x_gc2 = self.gc2(x_gc1, adj)
        return x_gc1, x_gc2


class GraphConvolution_B(nn.Module):
    def __init__(self, batch, input_dim, output_dim, dropout, bias=False):
        super(GraphConvolution_B, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.Tensor(batch, input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)  # xavier初始化，就是论文里的glorot初始化
        if bias:
            self.bias = nn.Parameter(torch.Tensor(batch, output_dim))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs, adj):
        # inputs: (N, n_channels), adj: sparse_matrix (N, N)
        support = torch.bmm(self.dropout(inputs), self.weight)
        output = torch.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN_B_2(nn.Module):
    def __init__(self, batch, n_features, hidden_dim, dropout, n_classes):
        super(GCN_B_2, self).__init__()
        self.gc1 = GraphConvolution_B(batch, n_features, hidden_dim, dropout)
        self.gc2 = GraphConvolution_B(batch, hidden_dim, n_classes, dropout)
        self.relu = nn.ReLU()

    def forward(self, inputs, adj):
        x = inputs
        x_gc1 = self.relu(self.gc1(x, adj))
        x_gc2 = self.relu(self.gc2(x_gc1, adj))
        return x_gc2


class GCN_B_3(nn.Module):
    def __init__(self, batch, n_features, hidden_dim, dropout, n_classes):
        super(GCN_B_3, self).__init__()
        self.gc1 = GraphConvolution_B(batch, n_features, hidden_dim, dropout)
        self.gc2 = GraphConvolution_B(batch, hidden_dim, hidden_dim, dropout)
        self.gc3 = GraphConvolution_B(batch, hidden_dim, n_classes, dropout)
        self.relu = nn.ReLU()

    def forward(self, inputs, adj):
        x = inputs
        x_gc1 = self.relu(self.gc1(x, adj))
        x_gc2 = self.relu(self.gc2(x_gc1, adj))
        x_gc3 = self.relu(self.gc3(x_gc2, adj))
        return x_gc3


class GCN_B_4(nn.Module):
    def __init__(self, batch, n_features, hidden_dim, dropout, n_classes):
        super(GCN_B_4, self).__init__()
        self.gc1 = GraphConvolution_B(batch, n_features, hidden_dim, dropout)
        self.gc2 = GraphConvolution_B(batch, hidden_dim, hidden_dim, dropout)
        self.gc3 = GraphConvolution_B(batch, hidden_dim, hidden_dim, dropout)
        self.gc4 = GraphConvolution_B(batch, hidden_dim, n_classes, dropout)
        self.relu = nn.ReLU()

    def forward(self, inputs, adj):
        x = inputs
        x_gc1 = self.relu(self.gc1(x, adj))
        x_gc2 = self.relu(self.gc2(x_gc1, adj))
        x_gc3 = self.relu(self.gc3(x_gc2, adj))
        x_gc4 = self.relu(self.gc4(x_gc3, adj))
        return x_gc4


class GCN_B_5(nn.Module):
    def __init__(self, batch, n_features, hidden_dim, dropout, n_classes):
        super(GCN_B_5, self).__init__()
        self.gc1 = GraphConvolution_B(batch, n_features, hidden_dim, dropout)
        self.gc2 = GraphConvolution_B(batch, hidden_dim, hidden_dim, dropout)
        self.gc3 = GraphConvolution_B(batch, hidden_dim, hidden_dim, dropout)
        self.gc4 = GraphConvolution_B(batch, hidden_dim, hidden_dim, dropout)
        self.gc5 = GraphConvolution_B(batch, hidden_dim, n_classes, dropout)
        self.relu = nn.ReLU()

    def forward(self, inputs, adj):
        x = inputs
        x_gc1 = self.relu(self.gc1(x, adj))
        x_gc2 = self.relu(self.gc2(x_gc1, adj))
        x_gc3 = self.relu(self.gc3(x_gc2, adj))
        x_gc4 = self.relu(self.gc4(x_gc3, adj))
        x_gc5 = self.relu(self.gc4(x_gc4, adj))
        return x_gc5


class GCN_B_use(nn.Module):
    def __init__(self, batch, n_features, dropout, n_classes):
        super(GCN_B_use, self).__init__()
        self.gc1 = GraphConvolution_B(batch, n_features, n_classes, dropout)
        self.relu = nn.ReLU()

    def forward(self, inputs, adj):
        x_gc1 = self.relu(self.gc1(inputs, adj))
        return x_gc1


class GraphAttentionLayer(nn.Module):
    def __init__(self, batch, in_features, out_features, dropout, alpha, concat):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.Tensor(batch, in_features, out_features))
        self.a = nn.Parameter(torch.Tensor(batch, 2 * out_features, 1))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha, inplace=False)

    def forward(self, h, adj):
        '''
        h: (N, in_features)
        adj: sparse matrix with shape (N, N)
        '''
        Wh = torch.bmm(h, self.W)  # (N, out_features)
        Wh1 = torch.bmm(Wh, self.a[:, :self.out_features, :])  # (N, 1)
        Wh2 = torch.bmm(Wh, self.a[:, self.out_features:, :])  # (N, 1)

        # Wh1 + Wh2.T 是N*N矩阵，第i行第j列是Wh1[i]+Wh2[j]
        # 那么Wh1 + Wh2.T的第i行第j列刚好就是文中的a^T*[Whi||Whj]
        # 代表着节点i对节点j的attention
        e = self.leakyrelu(Wh1 + Wh2.transpose(1, 2))  # (N, N)
        padding = (-2 ** 31) * torch.ones_like(e)  # (N, N)
        attention = torch.where(adj > 0, e, padding)  # (N, N)
        attention = nn.functional.softmax(attention, dim=1)  # (N, N)
        # attention矩阵第i行第j列代表node_i对node_j的注意力
        # 对注意力权重也做dropout（如果经过mask之后，attention矩阵也许是高度稀疏的，这样做还有必要吗？）
        attention = nn.functional.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)  # (N, out_features)
        if self.concat:
            return nn.functional.elu(h_prime)
        else:
            return h_prime


# self.GAT = GAT(batch, channels_list[-1], channels_list[-1], nclass=channels_list[-1], dropout=0.5, alpha=0.2, nheads=1)
class GAT(nn.Module):
    def __init__(self, batch, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.MH = nn.ModuleList([
            GraphAttentionLayer(batch, nfeat, nhid, dropout, alpha, concat=True)
            for _ in range(nheads)
        ])
        # self.out_att = GraphAttentionLayer(batch, nhid * nheads, nclass, dropout, alpha, concat=False)

    def forward(self, x, adj):
        x = nn.functional.dropout(x, self.dropout, training=self.training)  # (N, nfeat)
        x = torch.cat([head(x, adj) for head in self.MH], dim=2)  # (N, nheads*nhid)
        x_1 = nn.functional.dropout(x, self.dropout, training=self.training)  # (N, nheads*nhid)
        # x_2 = nn.functional.elu(self.out_att(x_1, adj))
        return x_1


class GlobalRelateFeature(nn.Module):
    def __init__(self, in_dim, hide_dim, out_dim):
        super(GlobalRelateFeature, self).__init__()
        self.down = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=(1, 2, 2),
                              padding=1)
        self.down_norm = nn.InstanceNorm3d(in_dim, eps=1e-5, affine=True)
        self.down_leaky = nn.LeakyReLU(inplace=True)
        self.ap = PAM_Module_A(in_dim)
        self.agg = pyGnn.GCNConv(in_dim, hide_dim)
        # self.aggOut = pyGnn.GCNConv(hide_dim, out_dim)
        self.up = nn.ConvTranspose3d(hide_dim, hide_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.up_norm = nn.InstanceNorm3d(hide_dim, eps=1e-5, affine=True)
        self.up_leaky = nn.LeakyReLU(inplace=True)

    def forward(self, I, edge_index):
        x = self.down_leaky(self.down_norm(self.down(I)))
        # 对x进行下采样后，生成关联关系矩阵
        x_ap = self.ap(x)
        # x_ap进行reshape
        x_ap = x_ap.reshape([x_ap.shape[0], -1])
        x_ap = x_ap.reshape([-1, 1])
        x_node = x.permute(1, 0, 2, 3, 4).reshape([x.shape[1], -1]).permute(1, 0)
        x_agg = self.agg(x=x_node, edge_index=edge_index, edge_weight=x_ap)

        # x_out = self.aggOut(x=self.relu(x_agg), edge_index=edge_index, edge_weight=x_ap)

        # x_out = x_out.permute(1, 0).reshape([1, x.shape[0], x.shape[2], x.shape[3], x.shape[4]]).permute(1, 0, 2, 3, 4)
        # 对x_agg进行上采样
        # x_agg = torch.cat((self.norm(self.up(x_agg)), x), dim=1)
        x_agg = x_agg.permute(1, 0).reshape([x.shape[1], x.shape[0], x.shape[2], x.shape[3], x.shape[4]]).permute(1, 0,
                                                                                                                  2, 3,
                                                                                                                  4)
        x_agg = self.up_leaky(self.up_norm(self.up(x_agg)))
        return x_ap, x_agg


class GlobalRelateFeature_new(nn.Module):
    def __init__(self, batch, in_dim, hide_dim, out_dim):
        super(GlobalRelateFeature_new, self).__init__()
        # self.down = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=(1, 2, 2),
        #                       padding=1)
        # self.down_norm = nn.InstanceNorm3d(in_dim, eps=1e-5, affine=True)
        # self.down_leaky = nn.LeakyReLU(inplace=True)

        self.ap = PAM_Module_A(in_dim)
        self.agg = GCN_B_2(batch, in_dim, hide_dim, 0.5, out_dim)

        # self.up = nn.ConvTranspose3d(hide_dim, hide_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # self.up_norm = nn.InstanceNorm3d(hide_dim, eps=1e-5, affine=True)
        # self.up_leaky = nn.LeakyReLU(inplace=True)

    def forward(self, I):
        # x = self.down_leaky(self.down_norm(self.down(I)))
        x = I
        # 对x进行下采样后，生成关联关系矩阵
        x_ap = self.ap(x)
        x_node = x.reshape([x.shape[0], x.shape[1], -1]).permute(0, 2, 1)
        x_agg, x_out = self.agg(x_node, x_ap)

        x_out = x_out.permute(0, 2, 1).reshape([x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4]])

        x_agg = x_agg.permute(0, 2, 1).reshape([x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]])
        # x_agg = self.up_leaky(self.up_norm(self.up(x_agg)))
        return x_ap, x_agg, x_out


class GlobalRelateFeature_use(nn.Module):
    def __init__(self, batch, in_dim, hide_dim):
        super(GlobalRelateFeature_use, self).__init__()
        # self.down = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=(1, 2, 2),
        #                       padding=1)
        # self.down_norm = nn.InstanceNorm3d(in_dim, eps=1e-5, affine=True)
        # self.down_leaky = nn.LeakyReLU(inplace=True)

        self.ap = PAM_Module_A(in_dim)
        self.agg = GCN_B_use(batch, in_dim, 0.5, in_dim)

        # self.agg = GCN_B_2(batch, in_dim, in_dim, 0.5, in_dim)
        # self.agg = GCN_B_3(batch, in_dim, in_dim, 0.5, in_dim)
        # self.agg = GCN_B_4(batch, in_dim, in_dim, 0.5, in_dim)
        # self.agg = GCN_B_5(batch, in_dim, in_dim, 0.5, in_dim)

        # self.agg = GAT(batch, in_dim, in_dim, nclass=in_dim, dropout=0.5, alpha=0.2, nheads=1)

        # self.up = nn.ConvTranspose3d(hide_dim, hide_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # self.up_norm = nn.InstanceNorm3d(hide_dim, eps=1e-5, affine=True)
        # self.up_leaky = nn.LeakyReLU(inplace=True)

    def forward(self, I):
        # x = self.down_leaky(self.down_norm(self.down(I)))
        x = I
        # 对x进行下采样后，生成关联关系矩阵
        x_ap = self.ap(x)
        x_node = x.reshape([x.shape[0], x.shape[1], -1]).permute(0, 2, 1)
        x_agg = self.agg(x_node, x_ap)

        x_agg = x_agg.permute(0, 2, 1).reshape([x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]])
        # x_agg = self.up_leaky(self.up_norm(self.up(x_agg)))
        return x_ap, x_agg


class GlobalRelateFeature_use_CA(nn.Module):
    def __init__(self, batch, in_dim, hide_dim):
        super(GlobalRelateFeature_use_CA, self).__init__()
        # self.down = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=(1, 2, 2),
        #                       padding=1)
        # self.down_norm = nn.InstanceNorm3d(in_dim, eps=1e-5, affine=True)
        # self.down_leaky = nn.LeakyReLU(inplace=True)

        self.CA = CAM_Module_3D()

        self.ap = PAM_Module_A(in_dim)
        self.agg = GCN_B_use(batch, in_dim, 0.5, in_dim)

        # self.agg = GCN_B_2(batch, in_dim, in_dim, 0.5, in_dim)
        # self.agg = GCN_B_3(batch, in_dim, in_dim, 0.5, in_dim)
        # self.agg = GCN_B_4(batch, in_dim, in_dim, 0.5, in_dim)
        # self.agg = GCN_B_5(batch, in_dim, in_dim, 0.5, in_dim)

        # self.agg = GAT(batch, in_dim, in_dim, nclass=in_dim, dropout=0.5, alpha=0.2, nheads=1)

        # self.up = nn.ConvTranspose3d(hide_dim, hide_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # self.up_norm = nn.InstanceNorm3d(hide_dim, eps=1e-5, affine=True)
        # self.up_leaky = nn.LeakyReLU(inplace=True)

    def forward(self, I):
        # x = self.down_leaky(self.down_norm(self.down(I)))
        x = I
        # 对x进行下采样后，生成关联关系矩阵
        x = self.CA(x)
        x_ap = self.ap(x)
        x_node = x.reshape([x.shape[0], x.shape[1], -1]).permute(0, 2, 1)
        x_agg = self.agg(x_node, x_ap)

        x_agg = x_agg.permute(0, 2, 1).reshape([x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]])
        # x_agg = self.up_leaky(self.up_norm(self.up(x_agg)))
        return x_ap, x_agg


class nnUNet_kMax_deeplab_pyG0(nn.Module):
    def __init__(self, input_channels, output_classes, channels_list, dec_layers=(1, 1, 1),
                 in_channels=(32 * 3, 16 * 3, 8 * 3, 4 * 3), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(nnUNet_kMax_deeplab_pyG0, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)

        # self.GRF = GlobalRelateFeature(channels_list[-1], channels_list[-1], 1)
        self.GRF = GlobalRelateFeature_new(4, channels_list[-1], channels_list[-1], 1)

        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.transDecoder = kMaXTransformerDecoder_new(dec_layers, in_channels, output_classes - 2, num_queries,
                                                       drop_path_prob)

        self.out_conv = nn.Conv3d(32, output_classes - 1, kernel_size=1, stride=1)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pyG_B):
        x = self.norm(x)
        out_en = self.encoder(x, return_skips=True)
        # 对最底层进行相关位置特征聚合
        # brige = self.GRF(out_en[4], pyG_B.edge_index)
        brige = self.GRF(out_en[4])

        out_en[4] = brige[1]

        out = self.decoder(out_en)
        out_Trans = self.transDecoder(out[0:3], self.relu(out[3]))
        pre = self.sigmoid(self.out_conv(out_Trans['pred_masks']))
        return pre


class nnUNet_kMax_deeplab_pyG0_solo(nn.Module):
    def __init__(self, batch, input_channels, output_classes, channels_list, dec_layers=(1, 1, 1),
                 in_channels=(32 * 3, 16 * 3, 8 * 3, 4 * 3), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(nnUNet_kMax_deeplab_pyG0_solo, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)

        self.GRF = GlobalRelateFeature_use(batch, channels_list[-1], channels_list[-1])

        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.transDecoder = kMaXTransformerDecoder_new(dec_layers, in_channels, output_classes - 2, num_queries,
                                                       drop_path_prob)

        self.out_conv = nn.Conv3d(32, output_classes - 1, kernel_size=1, stride=1)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        out_en = self.encoder(x, return_skips=True)
        brige = self.GRF(out_en[4])
        out_en[4] = brige[1]
        out = self.decoder(out_en)
        out_Trans = self.transDecoder(out[0:3], self.relu(out[3]))
        pre = self.sigmoid(self.out_conv(out_Trans['pred_masks']))
        return pre


class nnUNet_kMax_deeplab_pyG0_deep(nn.Module):
    def __init__(self, batch, input_channels, output_classes, channels_list, dec_layers=(1, 1, 1),
                 in_channels=(32 * 3, 16 * 3, 8 * 3, 4 * 3), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(nnUNet_kMax_deeplab_pyG0_deep, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)

        self.GRF = GlobalRelateFeature_new(batch, channels_list[-1], channels_list[-1], 1)

        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.transDecoder = kMaXTransformerDecoder_new(dec_layers, in_channels, output_classes - 2, num_queries,
                                                       drop_path_prob)

        self.out_conv = nn.Conv3d(32, output_classes - 1, kernel_size=1, stride=1)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        out_en = self.encoder(x, return_skips=True)
        brige = self.GRF(out_en[4])
        out_en[4] = brige[1]
        out = self.decoder(out_en)
        out_Trans = self.transDecoder(out[0:3], self.relu(out[3]))
        pre = self.sigmoid(self.out_conv(out_Trans['pred_masks']))
        return brige[0], self.sigmoid(brige[2]), pre


class nnUNet_kMax_deeplab_CA0(nn.Module):
    def __init__(self, input_channels, output_classes, channels_list, dec_layers=(1, 1, 1),
                 in_channels=(32 * 3, 16 * 3, 8 * 3, 4 * 3), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(nnUNet_kMax_deeplab_CA0, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)
        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.transDecoder = kMaXTransformerDecoder_new(dec_layers, in_channels, output_classes - 2, num_queries,
                                                       drop_path_prob)

        self.PA = SpatialAttention3D(outC=32)
        self.CA_1 = CAM_Module_3D()
        self.CA_2 = CAM_Module_3D()
        self.CA_3 = CAM_Module_3D()

        # self.out_conv = nn.Conv3d(32, output_classes - 1, kernel_size=1, stride=1)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        out = self.decoder(self.encoder(x, return_skips=True))
        out_Trans = self.transDecoder(out[0:3], self.relu(out[3]))
        # 增加类之间的差异
        M = out_Trans['pred_masks']

        # 增加位置注意力，关注病灶所在位置
        M_P = self.PA(M)
        #
        # # 每个病灶区域由不同的类所组成的
        M_C_1 = self.CA_1(M_P)
        M_C_2 = self.CA_2(M_P)
        M_C_3 = self.CA_3(M_P)
        Z_1 = torch.einsum('bnhwd,bnk->bkhwd', M_C_1, out_Trans['pred_logits'][:, :, 0].unsqueeze(-1))
        Z_2 = torch.einsum('bnhwd,bnk->bkhwd', M_C_2, out_Trans['pred_logits'][:, :, 1].unsqueeze(-1))
        Z_3 = torch.einsum('bnhwd,bnk->bkhwd', M_C_3, out_Trans['pred_logits'][:, :, 2].unsqueeze(-1))

        # pre = self.sigmoid(torch.einsum('bnhwd,bnk->bkhwd', M, out_Trans['pred_logits']))

        pre = torch.cat(
            (self.sigmoid(Z_1), self.sigmoid(torch.add(Z_1, Z_2)), self.sigmoid(torch.add(torch.add(Z_1, Z_2), Z_3))),
            dim=1)
        # pre = torch.cat((Z_1, Z_2, Z_3), dim=1)
        return pre


class nnUNet_kMax_deeplab_CA0_new(nn.Module):
    def __init__(self, input_channels, output_classes, channels_list, dec_layers=(1, 1, 1),
                 in_channels=(32 * 3, 16 * 3, 8 * 3, 4 * 3), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(nnUNet_kMax_deeplab_CA0_new, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)
        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.transDecoder = kMaXTransformerDecoder_new(dec_layers, in_channels, output_classes - 2, num_queries,
                                                       drop_path_prob)

        self.CA = CAM_Module_3D()

        self.relu = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        out = self.decoder(self.encoder(x, return_skips=True))
        out_Trans = self.transDecoder(out[0:3], self.relu(out[3]))
        # 增加类之间的差异
        M = out_Trans['pred_masks']

        M_C = self.CA(M)

        Z = self.relu(torch.einsum('bnhwd,bnk->bkhwd', M_C, out_Trans['pred_logits']))
        # Z_1 = Z[:, 0, :, :, :].unsqueeze(dim=1)
        # Z_2 = Z[:, 1, :, :, :].unsqueeze(dim=1)
        # Z_3 = Z[:, 2, :, :, :].unsqueeze(dim=1)

        # 3个区域位置不交叠
        # I_Z = torch.einsum('bkhwd,bhwdk->bkk', Z, Z.permute(0, 2, 3, 4, 1))

        # 增加位置注意力，关注病灶所在位置
        # Z_1 = self.PA_1(Z[:, 0, :, :, :].unsqueeze(dim=1))
        # Z_2 = self.PA_2(Z[:, 1, :, :, :].unsqueeze(dim=1))
        # Z_3 = self.PA_3(Z[:, 2, :, :, :].unsqueeze(dim=1))

        # pre = torch.cat(
        #     (self.sigmoid(Z_1), self.sigmoid(torch.add(Z_1, Z_2)), self.sigmoid(torch.add(torch.add(Z_1, Z_2), Z_3))),
        #     dim=1)
        # pre = torch.cat((Z_1, torch.add(Z_1, Z_2), torch.add(torch.add(Z_1, Z_2), Z_3)), dim=1)
        # pre = torch.cat((Z_1, Z_2, Z_3), dim=1)
        return Z


class nnUNet_kMax_deeplab_CA0_conv(nn.Module):
    def __init__(self, input_channels, output_classes, channels_list, dec_layers=(1, 1, 1),
                 in_channels=(32 * 3, 16 * 3, 8 * 3, 4 * 3), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(nnUNet_kMax_deeplab_CA0_conv, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)
        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.transDecoder = kMaXTransformerDecoder_new(dec_layers, in_channels, output_classes - 2, num_queries,
                                                       drop_path_prob)

        self.PA = SpatialAttention3D(outC=32)
        self.CA_1 = CAM_Module_3D()
        self.CA_2 = CAM_Module_3D()
        self.CA_3 = CAM_Module_3D()

        self.out_conv1 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.out_conv2 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.out_conv3 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        out = self.decoder(self.encoder(x, return_skips=True))
        out_Trans = self.transDecoder(out[0:3], self.relu(out[3]))
        # 增加类之间的差异
        M = out_Trans['pred_masks']

        # 增加位置注意力，关注病灶所在位置
        M_P = self.PA(M)
        #
        # # 每个病灶区域由不同的类所组成的
        M_C_1 = self.CA_1(M_P)
        M_C_2 = self.CA_2(M_P)
        M_C_3 = self.CA_3(M_P)
        Z_1 = self.out_conv1(M_C_1)
        Z_2 = self.out_conv2(M_C_2)
        Z_3 = self.out_conv3(M_C_3)

        pre = torch.cat(
            (self.sigmoid(Z_1), self.sigmoid(torch.add(Z_1, Z_2)), self.sigmoid(torch.add(torch.add(Z_1, Z_2), Z_3))),
            dim=1)
        # pre = torch.cat((Z_1, Z_2, Z_3), dim=1)
        return pre


class nnUNet_kMax_deeplab_pyG0_CA0(nn.Module):
    def __init__(self, batch, input_channels, output_classes, channels_list, dec_layers=(1, 1, 1),
                 in_channels=(32 * 3, 16 * 3, 8 * 3, 4 * 3), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(nnUNet_kMax_deeplab_pyG0_CA0, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)

        self.GRF = GlobalRelateFeature_use(batch, channels_list[-1], channels_list[-1])

        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.transDecoder = kMaXTransformerDecoder_new(dec_layers, in_channels, output_classes - 2, num_queries,
                                                       drop_path_prob)

        self.PA = SpatialAttention3D(outC=32)
        self.CA_1 = CAM_Module_3D()
        self.CA_2 = CAM_Module_3D()
        self.CA_3 = CAM_Module_3D()

        # self.out_conv = nn.Conv3d(32, output_classes - 1, kernel_size=1, stride=1)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        out_en = self.encoder(x, return_skips=True)
        brige = self.GRF(out_en[4])
        out_en[4] = brige[1]
        out = self.decoder(out_en)

        out_Trans = self.transDecoder(out[0:3], self.relu(out[3]))

        # 增加位置注意力，关注病灶所在位置
        M_P = self.PA(out_Trans['pred_masks'])
        # 每个病灶区域由不同的类所组成的
        M_C_1 = self.CA_1(M_P)
        M_C_2 = self.CA_2(M_P)
        M_C_3 = self.CA_3(M_P)
        Z_1 = torch.einsum('bnhwd,bnk->bkhwd', M_C_1, out_Trans['pred_logits'][:, :, 0].unsqueeze(-1))
        Z_2 = torch.einsum('bnhwd,bnk->bkhwd', M_C_2, out_Trans['pred_logits'][:, :, 1].unsqueeze(-1))
        Z_3 = torch.einsum('bnhwd,bnk->bkhwd', M_C_3, out_Trans['pred_logits'][:, :, 2].unsqueeze(-1))

        pre = torch.cat(
            (self.sigmoid(Z_1), self.sigmoid(torch.add(Z_1, Z_2)), self.sigmoid(torch.add(torch.add(Z_1, Z_2), Z_3))),
            dim=1)
        return pre


class CFPNet(nn.Module):
    def __init__(self, batch, input_channels, output_classes, channels_list, dec_layers=(1, 1, 1),
                 in_channels=(32 * 3, 16 * 3, 8 * 3, 4 * 3), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(CFPNet, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)

        self.GRF = GlobalRelateFeature_use(batch, channels_list[-1], channels_list[-1])

        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.transDecoder = kMaXTransformerDecoder_new(dec_layers, in_channels, output_classes - 2, num_queries,
                                                       drop_path_prob)

        self.CA = CAM_Module_3D()

        self.relu = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.norm(x)
        out_en = self.encoder(x, return_skips=True)
        brige = self.GRF(out_en[4])
        out_en[4] = brige[1]
        out = self.decoder(out_en)
        out_Trans = self.transDecoder(out[0:3], self.relu(out[3]))
        M = out_Trans['pred_masks']
        M_C = self.CA(M)
        Z = self.relu(torch.einsum('bnhwd,bnk->bkhwd', M_C, out_Trans['pred_logits']))
        return Z


class CFPNet_RGE(nn.Module):
    def __init__(self, batch, input_channels, output_classes, channels_list, dec_layers=(1, 1, 1),
                 in_channels=(32 * 3, 16 * 3, 8 * 3, 4 * 3), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(CFPNet_RGE, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)

        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.transDecoder = kMaXTransformerDecoder_new(dec_layers, in_channels, output_classes - 2, num_queries,
                                                       drop_path_prob)

        self.CA = CAM_Module_3D()

        self.relu = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.norm(x)
        out_en = self.encoder(x, return_skips=True)
        out = self.decoder(out_en)
        out_Trans = self.transDecoder(out[0:3], self.relu(out[3]))
        M = out_Trans['pred_masks']
        M_C = self.CA(M)
        Z = self.relu(torch.einsum('bnhwd,bnk->bkhwd', M_C, out_Trans['pred_logits']))
        return Z


class CFPNet_PCA(nn.Module):
    def __init__(self, batch, input_channels, output_classes, channels_list, dec_layers=(1, 1, 1),
                 in_channels=(32 * 3, 16 * 3, 8 * 3, 4 * 3), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(CFPNet_PCA, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)

        self.GRF = GlobalRelateFeature_use(batch, channels_list[-1], channels_list[-1])

        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.transDecoder = kMaXTransformerDecoder_new(dec_layers, in_channels, output_classes - 2, num_queries,
                                                       drop_path_prob)

        self.PA = SpatialAttention3D(outC=32)
        # self.CA = ChannelAttention3D_old()
        # self.CA = CAM_Module_3D()

        self.out_conv = nn.Conv3d(32, output_classes - 1, kernel_size=1, stride=1)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        out_en = self.encoder(x, return_skips=True)
        brige = self.GRF(out_en[4])
        out_en[4] = brige[1]
        out = self.decoder(out_en)
        out_Trans = self.transDecoder(out[0:3], self.relu(out[3]))
        M = out_Trans['pred_masks']

        M_P = self.PA(M)
        # M_C = self.CA(M_P)

        Z = self.sigmoid(self.out_conv(M_P))
        return Z


class CFPNet_KFS(nn.Module):
    def __init__(self, batch, input_channels, output_classes, channels_list, dec_layers=(1, 1, 1),
                 in_channels=(32 * 3, 16 * 3, 8 * 3, 4 * 3), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(CFPNet_KFS, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)

        self.GRF = GlobalRelateFeature_use(batch, channels_list[-1], channels_list[-1])

        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.transDecoder = kMaXTransformerDecoder_new(dec_layers, in_channels, output_classes - 2, num_queries,
                                                       drop_path_prob)

        self.out_conv = nn.Conv3d(32, output_classes - 1, kernel_size=1, stride=1)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        out_en = self.encoder(x, return_skips=True)
        brige = self.GRF(out_en[4])
        out_en[4] = brige[1]
        out = self.decoder(out_en)
        out_Trans = self.transDecoder(out[0:3], self.relu(out[3]))
        Z = self.sigmoid(self.out_conv(out_Trans['pred_masks']))
        return Z


class MPGNet(nn.Module):
    def __init__(self, batch, input_channels, output_classes, channels_list, dec_layers=(1, 1, 1),
                 in_channels=(32 * 3, 16 * 3, 8 * 3, 4 * 3), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(MPGNet, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)

        self.GRF = GlobalRelateFeature_use_CA(batch, channels_list[-1], channels_list[-1])

        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.transDecoder = kMaXTransformerDecoder_new(dec_layers, in_channels, output_classes - 2, num_queries,
                                                       drop_path_prob)

        self.out_conv = nn.Conv3d(32, output_classes - 1, kernel_size=1, stride=1)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        out_en = self.encoder(x, return_skips=True)
        brige = self.GRF(out_en[4])
        out_en[4] = brige[1]
        out = self.decoder(out_en)
        out_Trans = self.transDecoder(out[0:3], self.relu(out[3]))
        Z = self.sigmoid(self.out_conv(out_Trans['pred_masks']))
        return Z


class CFPNet_KFS_RGA(nn.Module):
    def __init__(self, batch, input_channels, output_classes, channels_list, dec_layers=(1, 1, 1),
                 in_channels=(32 * 3, 16 * 3, 8 * 3, 4 * 3), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(CFPNet_KFS_RGA, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)

        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.transDecoder = kMaXTransformerDecoder_new(dec_layers, in_channels, output_classes - 2, num_queries,
                                                       drop_path_prob)

        self.out_conv = nn.Conv3d(32, output_classes - 1, kernel_size=1, stride=1)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        out_en = self.encoder(x, return_skips=True)
        out = self.decoder(out_en)
        out_Trans = self.transDecoder(out[0:3], self.relu(out[3]))
        Z = self.sigmoid(self.out_conv(out_Trans['pred_masks']))
        return Z


class CFPNet_KFS_ADD(nn.Module):
    def __init__(self, batch, input_channels, output_classes, channels_list, dec_layers=(1, 1, 1),
                 in_channels=(32 * 3, 16 * 3, 8 * 3, 4 * 3), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(CFPNet_KFS_ADD, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)

        self.GRF = GlobalRelateFeature_use(batch, channels_list[-1], channels_list[-1])

        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.transDecoder = kMaXTransformerDecoder_new(dec_layers, in_channels, output_classes - 2, num_queries,
                                                       drop_path_prob)

        self.out_conv = nn.Conv3d(32, output_classes - 1, kernel_size=1, stride=1)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        out_en = self.encoder(x, return_skips=True)
        brige = self.GRF(out_en[4])
        out_en[4] = brige[1]
        out = self.decoder(out_en)
        out_Trans = self.transDecoder(out[0:3], self.relu(out[3]))
        Z = self.out_conv(out_Trans['pred_masks'])

        pre = torch.cat(
            (self.sigmoid(Z[:, 0, :, :, :].unsqueeze(dim=1)),
             self.sigmoid(torch.add(Z[:, 0, :, :, :].unsqueeze(dim=1), Z[:, 1, :, :, :].unsqueeze(dim=1))),
             self.sigmoid(torch.add(torch.add(Z[:, 0, :, :, :].unsqueeze(dim=1), Z[:, 1, :, :, :].unsqueeze(dim=1)),
                                    Z[:, 2, :, :, :].unsqueeze(dim=1)))),
            dim=1)

        return pre


class CFPNet_KFS_v1(nn.Module):
    def __init__(self, batch, input_channels, output_classes, channels_list, dec_layers=(1, 1, 1),
                 in_channels=(32 * 3, 16 * 3, 8 * 3, 4 * 3), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(CFPNet_KFS_v1, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)

        self.GRF = GlobalRelateFeature_use(batch, channels_list[-1], channels_list[-1])

        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.transDecoder = kMaXTransformerDecoder_new(dec_layers, in_channels, output_classes - 2, num_queries,
                                                       drop_path_prob)
        self.relu = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        out_en = self.encoder(x, return_skips=True)
        brige = self.GRF(out_en[4])
        out_en[4] = brige[1]
        out = self.decoder(out_en)
        out_Trans = self.transDecoder(out[0:3], self.relu(out[3]))
        Z = self.relu(torch.einsum('bnhwd,bnk->bkhwd', out_Trans['pred_masks'], out_Trans['pred_logits']))
        return Z


class nnUNet_kMax_deeplab_pyG0_CA0_conv(nn.Module):
    def __init__(self, batch, input_channels, output_classes, channels_list, dec_layers=(1, 1, 1),
                 in_channels=(32 * 3, 16 * 3, 8 * 3, 4 * 3), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(nnUNet_kMax_deeplab_pyG0_CA0_conv, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)

        self.GRF = GlobalRelateFeature_use(batch, channels_list[-1], channels_list[-1])

        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.transDecoder = kMaXTransformerDecoder_new(dec_layers, in_channels, output_classes - 2, num_queries,
                                                       drop_path_prob)

        self.PA = SpatialAttention3D(outC=32)
        self.CA_1 = CAM_Module_3D()
        self.CA_2 = CAM_Module_3D()
        self.CA_3 = CAM_Module_3D()

        self.out_conv1 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.out_conv2 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.out_conv3 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        out_en = self.encoder(x, return_skips=True)
        brige = self.GRF(out_en[4])
        out_en[4] = brige[1]
        out = self.decoder(out_en)

        out_Trans = self.transDecoder(out[0:3], self.relu(out[3]))

        # 增加位置注意力，关注病灶所在位置
        M_P = self.PA(out_Trans['pred_masks'])
        # 每个病灶区域由不同的类所组成的
        M_C_1 = self.CA_1(M_P)
        M_C_2 = self.CA_2(M_P)
        M_C_3 = self.CA_3(M_P)
        Z_1 = self.out_conv1(M_C_1)
        Z_2 = self.out_conv2(M_C_2)
        Z_3 = self.out_conv3(M_C_3)

        pre = torch.cat(
            (self.sigmoid(Z_1), self.sigmoid(torch.add(Z_1, Z_2)), self.sigmoid(torch.add(torch.add(Z_1, Z_2), Z_3))),
            dim=1)
        return pre


class nnUNet_kMax_deeplab_pyG0_CA0_deep(nn.Module):
    def __init__(self, batch, input_channels, output_classes, channels_list, dec_layers=(1, 1, 1),
                 in_channels=(32 * 3, 16 * 3, 8 * 3, 4 * 3), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(nnUNet_kMax_deeplab_pyG0_CA0_deep, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)

        self.GRF = GlobalRelateFeature_new(batch, channels_list[-1], channels_list[-1], 1)

        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.transDecoder = kMaXTransformerDecoder_new(dec_layers, in_channels, output_classes - 2, num_queries,
                                                       drop_path_prob)

        self.PA = SpatialAttention3D(outC=32)
        self.CA_1 = CAM_Module_3D()
        self.CA_2 = CAM_Module_3D()
        self.CA_3 = CAM_Module_3D()

        # self.out_conv = nn.Conv3d(32, output_classes - 1, kernel_size=1, stride=1)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        out_en = self.encoder(x, return_skips=True)
        brige = self.GRF(out_en[4])
        out_en[4] = brige[1]
        out = self.decoder(out_en)

        out_Trans = self.transDecoder(out[0:3], self.relu(out[3]))

        # 增加位置注意力，关注病灶所在位置
        M_P = self.PA(out_Trans['pred_masks'])
        # 每个病灶区域由不同的类所组成的
        M_C_1 = self.CA_1(M_P)
        M_C_2 = self.CA_2(M_P)
        M_C_3 = self.CA_3(M_P)
        Z_1 = torch.einsum('bnhwd,bnk->bkhwd', M_C_1, out_Trans['pred_logits'][:, :, 0].unsqueeze(-1))
        Z_2 = torch.einsum('bnhwd,bnk->bkhwd', M_C_2, out_Trans['pred_logits'][:, :, 1].unsqueeze(-1))
        Z_3 = torch.einsum('bnhwd,bnk->bkhwd', M_C_3, out_Trans['pred_logits'][:, :, 2].unsqueeze(-1))

        pre = torch.cat(
            (self.sigmoid(Z_1), self.sigmoid(torch.add(Z_1, Z_2)), self.sigmoid(torch.add(torch.add(Z_1, Z_2), Z_3))),
            dim=1)
        return brige[0], self.sigmoid(brige[2]), pre


class nnUNet_kMax_deeplab_CA1(nn.Module):
    def __init__(self, input_channels, output_classes, channels_list,
                 dec_layers=(1, 1, 1, 1),
                 in_channels=(32, 16, 8, 4), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(nnUNet_kMax_deeplab_CA1, self).__init__()
        # 先进行实例归一化
        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)
        self.decoder = UNetDecoder_MO(output_classes, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.transDecoder = kMaXTransformerDecoder(dec_layers, in_channels, output_classes - 1, num_queries,
                                                   drop_path_prob)

        self.CA_1 = CAM_Module_3D()
        self.CA_2 = CAM_Module_3D()

        # self.CA_1 = ChannelAttention3D()
        # self.CA_2 = ChannelAttention3D()

        self.out_M_C_1 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.out_bn_1 = nn.BatchNorm3d(1, eps=1e-3, momentum=0.01)
        # self.out_bn_1 = nn.InstanceNorm3d(1, eps=1e-5)
        self.out_M_C_2 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.out_bn_2 = nn.BatchNorm3d(1, eps=1e-3, momentum=0.01)
        # self.out_bn_2 = nn.InstanceNorm3d(1, eps=1e-5)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)

        out = self.decoder(self.encoder(x, return_skips=True))

        out_Trans = self.transDecoder(out[0:4], self.relu(out[4]))

        M = out_Trans['pred_masks']

        # 对M使用通道注意力做特征分离，将获取的特征传出用于最小化矩阵内积，再通过2个conv分别预测肿瘤和背景标签(2个交叉熵)。
        # 将特征分离层与类心相乘来预测肿瘤4个分类
        M_C_1 = self.CA_1(M)
        M_C_2 = self.CA_2(M)

        # M_C_1 = torch.mul(self.CA_1(M), M)
        # M_C_2 = torch.mul(self.CA_2(M), M)

        Z = torch.einsum('bnhwd,bnk->bkhwd', M_C_1, out_Trans['pred_logits'])

        out_M_C_1 = self.sigmoid(self.out_bn_1(self.out_M_C_1(M_C_1)))
        out_M_C_2 = self.sigmoid(self.out_bn_2(self.out_M_C_2(M_C_2)))

        # out_M_C_1 = self.sigmoid(self.out_M_C_1(M_C_1))
        # out_M_C_2 = self.sigmoid(self.out_M_C_2(M_C_2))

        return M_C_1, M_C_2, out_M_C_1, out_M_C_2, self.relu(Z)


class nnUNet_kMax_deeplab_pyG1(nn.Module):
    def __init__(self, input_channels, output_classes, channels_list, poolSize, sizeWHD, dec_layers=(1, 1, 1, 1),
                 in_channels=(32, 16, 8, 4), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(nnUNet_kMax_deeplab_pyG1, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)
        self.decoder = UNetDecoder_MO(output_classes, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.avg = nn.AvgPool3d(kernel_size=poolSize)
        self.agg = pyGnn.GCNConv(4, 4)
        self.up = nn.Upsample(size=sizeWHD, mode='nearest')

        self.transDecoder = kMaXTransformerDecoder(dec_layers, in_channels, output_classes - 1, num_queries,
                                                   drop_path_prob)

        self.out_conv = nn.Conv3d(32, output_classes, kernel_size=1, stride=1)
        self.out_bn = nn.BatchNorm3d(output_classes, eps=1e-3, momentum=0.01)
        # self.out_bn = nn.InstanceNorm3d(input_channels, eps=1e-5)
        self.relu = nn.Softmax(dim=1)

    def forward(self, x, pyG_B):
        x = self.norm(x)
        batchSize = x.shape[0]
        out = self.decoder(self.encoder(x, return_skips=True))

        # 根据输入图结构，更新out[3:4]中的节点特征，采用
        # SimpleConv/GCNConv/ChebConv/GraphConv_s/GatedGraphConv/GATConv/GATv2Conv/TransformerConv/
        # TAGConv/GINEConv/ARMAConv/SGConv/SSGConv/APPNP/RGATConv/DNAConv/GMMConv/SplineConv/NNConv/
        # CGConv/LEConv/PNAConv/GENConv/GCN2Conv/WLConvContinuous_d/FAConv/PDNConv/GeneralConv/LGConv_d

        # 节点属性赋值+平均池化+图更新+上采样+加和
        new_attr_3 = self.avg(out[3]).permute(1, 0, 2, 3, 4).reshape([4, -1]).permute(1, 0)
        updata_attr_3 = self.agg(x=new_attr_3, edge_index=pyG_B.edge_index, edge_weight=pyG_B.edge_attr.squeeze())
        updata_attr_3 = self.up(updata_attr_3.permute(1, 0).reshape([4, batchSize, 6, 8, 8]).permute(1, 0, 2, 3, 4))
        out[3] = torch.add(out[3], updata_attr_3)

        new_attr_4 = self.avg(out[4]).permute(1, 0, 2, 3, 4).reshape([4, -1]).permute(1, 0)
        updata_attr_4 = self.agg(x=new_attr_4, edge_index=pyG_B.edge_index, edge_weight=pyG_B.edge_attr.squeeze())
        updata_attr_4 = self.up(updata_attr_4.permute(1, 0).reshape([4, batchSize, 6, 8, 8]).permute(1, 0, 2, 3, 4))
        out[4] = torch.add(out[4], updata_attr_4)

        out_Trans = self.transDecoder(out[0:4], self.relu(out[4]))
        pre = self.relu(self.out_bn(self.out_conv(out_Trans['pred_masks'])))
        return pre


class nnUNet_kMax_deeplab_pyG1_CA1(nn.Module):
    def __init__(self, input_channels, output_classes, channels_list, poolSize, sizeWHD,
                 dec_layers=(1, 1, 1, 1),
                 in_channels=(32, 16, 8, 4), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(nnUNet_kMax_deeplab_pyG1_CA1, self).__init__()

        self.channels_list = channels_list
        self.output_classes = output_classes
        # 先进行实例归一化
        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)
        self.decoder = UNetDecoder_MO(output_classes, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.avg = nn.AvgPool3d(kernel_size=poolSize)
        self.agg = pyGnn.GCNConv(4, 4)
        self.up = nn.Upsample(size=sizeWHD, mode='nearest')

        self.transDecoder = kMaXTransformerDecoder(dec_layers, in_channels, output_classes - 1, num_queries,
                                                   drop_path_prob)

        self.CA_1 = CAM_Module_3D()
        self.CA_2 = CAM_Module_3D()

        # self.CA_1 = ChannelAttention3D()
        # self.CA_2 = ChannelAttention3D()

        self.out_M_C_1 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.out_bn_1 = nn.BatchNorm3d(1, eps=1e-3, momentum=0.01)
        # self.out_bn_1 = nn.InstanceNorm3d(1, eps=1e-5)
        self.out_M_C_2 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.out_bn_2 = nn.BatchNorm3d(1, eps=1e-3, momentum=0.01)
        # self.out_bn_2 = nn.InstanceNorm3d(1, eps=1e-5)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pyG_B):
        x = self.norm(x)

        batchSize = x.shape[0]

        out = self.decoder(self.encoder(x, return_skips=True))

        # 根据输入图结构，更新out[3:4]中的节点特征，采用
        # SimpleConv/GCNConv/ChebConv/GraphConv_s/GatedGraphConv/GATConv/GATv2Conv/TransformerConv/
        # TAGConv/GINEConv/ARMAConv/SGConv/SSGConv/APPNP/RGATConv/DNAConv/GMMConv/SplineConv/NNConv/
        # CGConv/LEConv/PNAConv/GENConv/GCN2Conv/WLConvContinuous_d/FAConv/PDNConv/GeneralConv/LGConv_d

        # 节点属性赋值+平均池化+图更新+上采样+加和
        new_attr_3 = self.avg(out[3]).permute(1, 0, 2, 3, 4).reshape([4, -1]).permute(1, 0)
        updata_attr_3 = self.agg(x=new_attr_3, edge_index=pyG_B.edge_index, edge_weight=pyG_B.edge_attr.squeeze())
        updata_attr_3 = self.up(
            updata_attr_3.permute(1, 0).reshape([self.channels_list[0], batchSize, 6, 8, 8]).permute(1, 0, 2, 3, 4))
        out[3] = torch.add(out[3], updata_attr_3)

        new_attr_4 = self.avg(out[4]).permute(1, 0, 2, 3, 4).reshape([4, -1]).permute(1, 0)
        updata_attr_4 = self.agg(x=new_attr_4, edge_index=pyG_B.edge_index, edge_weight=pyG_B.edge_attr.squeeze())
        updata_attr_4 = self.up(
            updata_attr_4.permute(1, 0).reshape([self.output_classes, batchSize, 6, 8, 8]).permute(1, 0, 2, 3, 4))
        out[4] = torch.add(out[4], updata_attr_4)

        out_Trans = self.transDecoder(out[0:4], self.relu(out[4]))

        M = out_Trans['pred_masks']

        # 对M使用通道注意力做特征分离，将获取的特征传出用于最小化矩阵内积，再通过2个conv分别预测肿瘤和背景标签(2个交叉熵)。
        # 将特征分离层与类心相乘来预测肿瘤4个分类
        M_C_1 = self.CA_1(M)
        M_C_2 = self.CA_2(M)

        # M_C_1 = torch.mul(self.CA_1(M), M)
        # M_C_2 = torch.mul(self.CA_2(M), M)

        Z = torch.einsum('bnhwd,bnk->bkhwd', M_C_1, out_Trans['pred_logits'])

        out_M_C_1 = self.sigmoid(self.out_bn_1(self.out_M_C_1(M_C_1)))
        out_M_C_2 = self.sigmoid(self.out_bn_2(self.out_M_C_2(M_C_2)))

        # out_M_C_1 = self.sigmoid(self.out_M_C_1(M_C_1))
        # out_M_C_2 = self.sigmoid(self.out_M_C_2(M_C_2))

        return M_C_1, M_C_2, out_M_C_1, out_M_C_2, self.relu(Z)


class nnUNet_kMax_deeplab_pyG1_CA1_new(nn.Module):
    def __init__(self, input_channels, output_classes, channels_list, poolSize, sizeWHD,
                 dec_layers=(1, 1, 1),
                 in_channels=(32 * 3, 16 * 3, 16 * 3, 8 * 3), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(nnUNet_kMax_deeplab_pyG1_CA1_new, self).__init__()

        self.channels_list = channels_list
        self.output_classes = output_classes
        # 先进行实例归一化
        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)
        self.decoder = UNetDecoder_MO(output_classes - 1, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.gnnM1 = nn.ModuleList()
        self.gnnM1.append(nn.AvgPool3d(kernel_size=[12 // 6, 128 // 8, 128 // 8]))
        self.gnnM1.append(pyGnn.GCNConv(4, 4))
        self.gnnM1.append(nn.Upsample(size=[12, 128, 128], mode='nearest'))

        self.gnnM2 = nn.ModuleList()
        self.gnnM2.append(nn.AvgPool3d(kernel_size=poolSize))
        self.gnnM2.append(pyGnn.GCNConv(4, 4))
        self.gnnM2.append(nn.Upsample(size=sizeWHD, mode='nearest'))

        self.transDecoder = kMaXTransformerDecoder_new(dec_layers, in_channels, output_classes - 2, num_queries,
                                                       drop_path_prob)

        self.CA_1 = CAM_Module_3D()
        self.CA_2 = CAM_Module_3D()

        # self.CA_1 = ChannelAttention3D()
        # self.CA_2 = ChannelAttention3D()

        self.out_M_C_1 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.out_bn_1 = nn.BatchNorm3d(1, eps=1e-3, momentum=0.01)
        # self.out_bn_1 = nn.InstanceNorm3d(1, eps=1e-5)
        self.out_M_C_2 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.out_bn_2 = nn.BatchNorm3d(1, eps=1e-3, momentum=0.01)
        # self.out_bn_2 = nn.InstanceNorm3d(1, eps=1e-5)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pyG_B):
        x = self.norm(x)

        batchSize = x.shape[0]

        out = self.decoder(self.encoder(x, return_skips=True))

        # 根据输入图结构，更新out[3:4]中的节点特征，采用
        # SimpleConv/GCNConv/ChebConv/GraphConv_s/GatedGraphConv/GATConv/GATv2Conv/TransformerConv/
        # TAGConv/GINEConv/ARMAConv/SGConv/SSGConv/APPNP/RGATConv/DNAConv/GMMConv/SplineConv/NNConv/
        # CGConv/LEConv/PNAConv/GENConv/GCN2Conv/WLConvContinuous_d/FAConv/PDNConv/GeneralConv/LGConv_d

        # 节点属性赋值+平均池化+图更新+上采样+加和
        new_attr_2 = self.gnnM1[0](out[2]).permute(1, 0, 2, 3, 4).reshape([batchSize, -1]).permute(1, 0)
        updata_attr_2 = self.gnnM1[1](x=new_attr_2, edge_index=pyG_B.edge_index, edge_weight=pyG_B.edge_attr.squeeze())
        updata_attr_2 = self.gnnM1[2](
            updata_attr_2.permute(1, 0).reshape([self.channels_list[1], batchSize, 6, 8, 8]).permute(1, 0, 2, 3, 4))
        out[2] = torch.cat((out[2], updata_attr_2), dim=1)

        new_attr_3 = self.gnnM2[0](out[3]).permute(1, 0, 2, 3, 4).reshape([batchSize, -1]).permute(1, 0)
        updata_attr_3 = self.gnnM2[1](x=new_attr_3, edge_index=pyG_B.edge_index, edge_weight=pyG_B.edge_attr.squeeze())
        updata_attr_3 = self.gnnM2[2](
            updata_attr_3.permute(1, 0).reshape([self.channels_list[0], batchSize, 6, 8, 8]).permute(1, 0, 2, 3, 4))
        # out[3] = torch.add(out[3], updata_attr_3)
        out[3] = torch.cat((out[3], updata_attr_3), dim=1)
        # out[3] = updata_attr_3

        # new_attr_4 = self.avg(out[4]).permute(1, 0, 2, 3, 4).reshape([batchSize, -1]).permute(1, 0)
        # updata_attr_4 = self.agg(x=new_attr_4, edge_index=pyG_B.edge_index, edge_weight=pyG_B.edge_attr.squeeze())
        # updata_attr_4 = self.up(updata_attr_4.permute(1, 0).reshape([self.output_classes, batchSize, 6, 8, 8]).permute(1, 0, 2, 3, 4))
        # # out[4] = torch.add(out[4], updata_attr_4)
        # out[4] = torch.cat((out[4], updata_attr_4), dim=1)

        out_Trans = self.transDecoder(out[0:3], self.relu(out[3]))

        M = out_Trans['pred_masks']

        # 对M使用通道注意力做特征分离，将获取的特征传出用于最小化矩阵内积，再通过2个conv分别预测肿瘤和背景标签(2个交叉熵)。
        # 将特征分离层与类心相乘来预测肿瘤4个分类
        M_C_1 = self.CA_1(M)
        M_C_2 = self.CA_2(M)

        # M_C_1 = torch.mul(self.CA_1(M), M)
        # M_C_2 = torch.mul(self.CA_2(M), M)

        Z = torch.einsum('bnhwd,bnk->bkhwd', M_C_1, out_Trans['pred_logits'])

        out_M_C_1 = self.sigmoid(self.out_bn_1(self.out_M_C_1(M_C_1)))
        out_M_C_2 = self.sigmoid(self.out_bn_2(self.out_M_C_2(M_C_2)))

        # out_M_C_1 = self.sigmoid(self.out_M_C_1(M_C_1))
        # out_M_C_2 = self.sigmoid(self.out_M_C_2(M_C_2))

        return M_C_1, M_C_2, out_M_C_1, out_M_C_2, self.sigmoid(Z)


class nnUNet_kMax_deeplab_pyG1_CA1_new_load(nn.Module):
    def __init__(self, preNet, output_classes, channels_list, poolSize, sizeWHD,
                 dec_layers=(1, 1, 1),
                 in_channels=(32 * 3, 16 * 3, 16 * 3, 4 * 3), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(nnUNet_kMax_deeplab_pyG1_CA1_new_load, self).__init__()

        self.preModel = preNet

        self.channels_list = channels_list
        self.output_classes = output_classes

        self.gnnM1 = nn.ModuleList()
        self.gnnM1.append(nn.AvgPool3d(kernel_size=[12 // 6, 128 // 8, 128 // 8]))
        self.gnnM1.append(pyGnn.GCNConv(4, 4))
        self.gnnM1.append(nn.Upsample(size=[12, 128, 128], mode='nearest'))

        self.gnnM2 = nn.ModuleList()
        self.gnnM2.append(nn.AvgPool3d(kernel_size=poolSize))
        self.gnnM2.append(pyGnn.GCNConv(4, 4))
        self.gnnM2.append(nn.Upsample(size=sizeWHD, mode='nearest'))

        self.transDecoder = kMaXTransformerDecoder_new(dec_layers, in_channels, output_classes - 2, num_queries,
                                                       drop_path_prob)

        self.CA_1 = CAM_Module_3D()
        self.CA_2 = CAM_Module_3D()

        self.out_M_C_1 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        # self.out_bn_1 = nn.BatchNorm3d(1, eps=1e-3, momentum=0.01)
        self.out_M_C_2 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        # self.out_bn_2 = nn.BatchNorm3d(1, eps=1e-3, momentum=0.01)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pyG_B):
        batchSize = x.shape[0]
        out = self.preModel(x)

        # 节点属性赋值+平均池化+图更新+上采样+加和
        new_attr_2 = self.gnnM1[0](out[2]).permute(1, 0, 2, 3, 4).reshape([batchSize, -1]).permute(1, 0)
        updata_attr_2 = self.gnnM1[1](x=new_attr_2, edge_index=pyG_B.edge_index, edge_weight=pyG_B.edge_attr.squeeze())
        updata_attr_2 = self.gnnM1[2](
            updata_attr_2.permute(1, 0).reshape([self.channels_list[1], batchSize, 6, 8, 8]).permute(1, 0, 2, 3, 4))
        out[2] = torch.cat((out[2], updata_attr_2), dim=1)

        new_attr_3 = self.gnnM2[0](out[3]).permute(1, 0, 2, 3, 4).reshape([batchSize, -1]).permute(1, 0)
        updata_attr_3 = self.gnnM2[1](x=new_attr_3, edge_index=pyG_B.edge_index, edge_weight=pyG_B.edge_attr.squeeze())
        updata_attr_3 = self.gnnM2[2](
            updata_attr_3.permute(1, 0).reshape([self.channels_list[0], batchSize, 6, 8, 8]).permute(1, 0, 2, 3, 4))
        out[3] = torch.cat((out[3], updata_attr_3), dim=1)

        out_Trans = self.transDecoder(out[0:3], self.relu(out[3]))

        M = out_Trans['pred_masks']

        M_C_1 = self.CA_1(M)
        M_C_2 = self.CA_2(M)

        Z = torch.einsum('bnhwd,bnk->bkhwd', M_C_1, out_Trans['pred_logits'])

        out_M_C_1 = self.sigmoid(self.out_M_C_1(M_C_1))
        out_M_C_2 = self.sigmoid(self.out_M_C_2(M_C_2))

        return M_C_1, M_C_2, out_M_C_1, out_M_C_2, self.sigmoid(Z)


class MedNeXt_kMax_deeplab(nn.Module):
    def __init__(self, input_channels, output_classes, dec_layers=(1, 1, 1, 1),
                 in_channels=(32, 16, 8, 4), num_queries=32, drop_path_prob=0.2):
        super(MedNeXt_kMax_deeplab, self).__init__()

        self.en_de_coder = MedNeXt_MO(input_channels, input_channels, 3, [2, 2, 2, 2, 2, 2, 2, 2, 2],
                                      [2, 3, 4, 4, 4, 4, 4, 3, 2], output_classes)

        self.act = nn.GELU()

        self.transDecoder = kMaXTransformerDecoder(dec_layers, in_channels, output_classes - 1, num_queries,
                                                   drop_path_prob)

        self.out_conv = nn.Conv3d(32, output_classes, kernel_size=1, stride=1)
        # self.out_bn = nn.BatchNorm3d(output_classes, eps=1e-3, momentum=0.01)
        # self.out_bn = nn.InstanceNorm3d(input_channels, eps=1e-5)
        self.relu = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.en_de_coder(x)
        out_Trans = self.transDecoder(out[0:4], self.act(out[4]))
        # pre = self.relu(self.out_bn(self.out_conv(out_Trans['pred_masks'])))
        pre = self.relu(self.out_conv(out_Trans['pred_masks']))
        return pre


class MedNeXt_kMax_deeplab_CA1(nn.Module):
    def __init__(self, input_channels, output_classes, dec_layers=(1, 1, 1, 1),
                 in_channels=(32, 16, 8, 4), num_queries=32, drop_path_prob=0.2):
        super(MedNeXt_kMax_deeplab_CA1, self).__init__()

        self.en_de_coder = MedNeXt_MO(input_channels, input_channels, 3, [2, 2, 2, 2, 2, 2, 2, 2, 2],
                                      [2, 3, 4, 4, 4, 4, 4, 3, 2], output_classes)

        self.transDecoder = kMaXTransformerDecoder(dec_layers, in_channels, output_classes - 1, num_queries,
                                                   drop_path_prob)

        self.CA_1 = CAM_Module_3D()
        self.CA_2 = CAM_Module_3D()

        # self.CA_1 = ChannelAttention3D()
        # self.CA_2 = ChannelAttention3D()

        self.out_M_C_1 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.out_bn_1 = nn.BatchNorm3d(1, eps=1e-3, momentum=0.01)
        # self.out_bn_1 = nn.InstanceNorm3d(1, eps=1e-5)
        self.out_M_C_2 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.out_bn_2 = nn.BatchNorm3d(1, eps=1e-3, momentum=0.01)
        # self.out_bn_2 = nn.InstanceNorm3d(1, eps=1e-5)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.out_conv = nn.Conv3d(output_classes, output_classes, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.en_de_coder(x)
        out_Trans = self.transDecoder(out[0:4], self.relu(out[4]))

        M = out_Trans['pred_masks']

        # 对M使用通道注意力做特征分离，将获取的特征传出用于最小化矩阵内积，再通过2个conv分别预测肿瘤和背景标签(2个交叉熵)。
        # 将特征分离层与类心相乘来预测肿瘤4个分类
        M_C_1 = self.CA_1(M)
        M_C_2 = self.CA_2(M)

        # M_C_1 = torch.mul(self.CA_1(M), M)
        # M_C_2 = torch.mul(self.CA_2(M), M)

        Z = torch.einsum('bnhwd,bnk->bkhwd', M_C_1, out_Trans['pred_logits'])

        Z = self.out_conv(Z)

        # out_M_C_1 = self.sigmoid(self.out_M_C_1(M_C_1))
        # out_M_C_2 = self.sigmoid(self.out_M_C_2(M_C_2))

        out_M_C_1 = self.sigmoid(self.out_bn_1(self.out_M_C_1(M_C_1)))
        out_M_C_2 = self.sigmoid(self.out_bn_2(self.out_M_C_2(M_C_2)))

        return M_C_1, M_C_2, out_M_C_1, out_M_C_2, self.relu(Z)


class MedNeXt_kMax_deeplab_CA2(nn.Module):
    def __init__(self, input_channels, output_classes, dec_layers=(1, 1, 1, 1),
                 in_channels=(32, 16, 8, 4), num_queries=32, drop_path_prob=0.2):
        super(MedNeXt_kMax_deeplab_CA2, self).__init__()

        self.en_de_coder = MedNeXt_MO(input_channels, input_channels, 3, [2, 2, 2, 2, 2, 2, 2, 2, 2],
                                      [2, 3, 4, 4, 4, 4, 4, 3, 2], output_classes)

        self.transDecoder = kMaXTransformerDecoder(dec_layers, in_channels, output_classes - 1, num_queries,
                                                   drop_path_prob)

        self.CA = CAM_Module_3D_2O()
        # self.CA = ChannelAttention3D()
        # self.CA_1 = ChannelAttention3D()
        # self.CA_2 = ChannelAttention3D()

        self.out_M_C_1 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.out_bn_1 = nn.BatchNorm3d(1, eps=1e-3, momentum=0.01)
        # self.out_bn_1 = nn.InstanceNorm3d(1, eps=1e-5)
        self.out_M_C_2 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.out_bn_2 = nn.BatchNorm3d(1, eps=1e-3, momentum=0.01)
        # self.out_bn_2 = nn.InstanceNorm3d(1, eps=1e-5)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.en_de_coder(x)
        out_Trans = self.transDecoder(out[0:4], self.relu(out[4]))

        M = out_Trans['pred_masks']

        # 对M使用通道注意力做特征分离，将获取的特征传出用于最小化矩阵内积，再通过2个conv分别预测肿瘤和背景标签(2个交叉熵)。
        # 将特征分离层与类心相乘来预测肿瘤4个分类
        M_C_1, M_C_2 = self.CA(M)

        # M_C_1 = torch.mul(self.CA(M), M)
        # M_C_2 = torch.mul(1 - self.CA(M), M)

        Z = torch.einsum('bnhwd,bnk->bkhwd', M_C_1, out_Trans['pred_logits'])

        # out_M_C_1 = self.sigmoid(self.out_M_C_1(M_C_1))
        # out_M_C_2 = self.sigmoid(self.out_M_C_2(M_C_2))

        out_M_C_1 = self.sigmoid(self.out_bn_1(self.out_M_C_1(M_C_1)))
        out_M_C_2 = self.sigmoid(self.out_bn_2(self.out_M_C_2(M_C_2)))

        return M_C_1, M_C_2, out_M_C_1, out_M_C_2, self.relu(Z)


class MedNeXt_kMax_deeplab_CA3(nn.Module):
    def __init__(self, input_channels, output_classes, dec_layers=(1, 1, 1, 1),
                 in_channels=(32, 16, 8, 4), num_queries=32, drop_path_prob=0.2):
        super(MedNeXt_kMax_deeplab_CA3, self).__init__()

        self.en_de_coder = MedNeXt_MO(input_channels, input_channels, 3, [2, 2, 2, 2, 2, 2, 2, 2, 2],
                                      [2, 3, 4, 4, 4, 4, 4, 3, 2], output_classes)

        self.transDecoder = kMaXTransformerDecoder(dec_layers, in_channels, output_classes - 1, num_queries,
                                                   drop_path_prob)

        self.CA_1 = CAM_Module_3D()
        self.CA_2 = CAM_Module_3D()

        self.out_M_C_1 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.out_bn_1 = nn.BatchNorm3d(1, eps=1e-3, momentum=0.01)
        self.out_M_C_2 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.out_bn_2 = nn.BatchNorm3d(1, eps=1e-3, momentum=0.01)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.en_de_coder(x)
        out_Trans = self.transDecoder(out[0:4], self.relu(out[4]))

        M = out_Trans['pred_masks']

        # 对M使用通道注意力做特征分离，将获取的特征传出用于最小化矩阵内积，再通过2个conv分别预测肿瘤和背景标签(2个交叉熵)。
        # 将特征分离层与类心相乘来预测肿瘤4个分类
        M_C_1 = self.CA_1(M)
        M_C_2 = self.CA_2(M)

        Z = torch.einsum('bnhwd,bnk->bkhwd', M, out_Trans['pred_logits'])

        # out_M_C_1 = self.sigmoid(self.out_M_C_1(M_C_1))
        # out_M_C_2 = self.sigmoid(self.out_M_C_2(M_C_2))

        out_M_C_1 = self.sigmoid(self.out_bn_1(self.out_M_C_1(M_C_1)))
        out_M_C_2 = self.sigmoid(self.out_bn_2(self.out_M_C_2(M_C_2)))

        return M_C_1, M_C_2, out_M_C_1, out_M_C_2, self.relu(Z)


class MedNeXt_kMax_deeplab_pyG1(nn.Module):
    def __init__(self, input_channels, output_classes, poolSize, sizeWHD, dec_layers=(1, 1, 1, 1),
                 in_channels=(32, 16, 8, 8), num_queries=32, drop_path_prob=0.2):
        super(MedNeXt_kMax_deeplab_pyG1, self).__init__()

        self.en_de_coder = MedNeXt_MO(input_channels, input_channels, 3, [2, 2, 2, 2, 2, 2, 2, 2, 2],
                                      [2, 3, 4, 4, 4, 4, 4, 3, 2], output_classes)

        self.avg = nn.AvgPool3d(kernel_size=poolSize)
        self.agg = pyGnn.GCNConv(4, 4)
        self.up = nn.Upsample(size=sizeWHD, mode='nearest')

        self.transDecoder = kMaXTransformerDecoder(dec_layers, in_channels, output_classes - 1, num_queries,
                                                   drop_path_prob)

        self.out_conv = nn.Conv3d(32, output_classes, kernel_size=1, stride=1)
        self.out_bn = nn.BatchNorm3d(output_classes, eps=1e-3, momentum=0.01)
        # self.out_bn = nn.InstanceNorm3d(input_channels, eps=1e-5)
        self.relu = nn.Softmax(dim=1)

    def forward(self, x, pyG_B):
        batchSize = x.shape[0]
        out = self.en_de_coder(x)

        # 根据输入图结构，更新out[3:4]中的节点特征，采用
        # SimpleConv/GCNConv/ChebConv/GraphConv_s/GatedGraphConv/GATConv/GATv2Conv/TransformerConv/
        # TAGConv/GINEConv/ARMAConv/SGConv/SSGConv/APPNP/RGATConv/DNAConv/GMMConv/SplineConv/NNConv/
        # CGConv/LEConv/PNAConv/GENConv/GCN2Conv/WLConvContinuous_d/FAConv/PDNConv/GeneralConv/LGConv_d

        # 节点属性赋值+平均池化+图更新+上采样+加和
        new_attr_3 = self.avg(out[3]).permute(1, 0, 2, 3, 4).reshape([4, -1]).permute(1, 0)
        updata_attr_3 = self.agg(x=new_attr_3, edge_index=pyG_B.edge_index, edge_weight=pyG_B.edge_attr.squeeze())
        updata_attr_3 = self.up(updata_attr_3.permute(1, 0).reshape([4, batchSize, 6, 8, 8]).permute(1, 0, 2, 3, 4))
        # out[3] = torch.add(out[3], updata_attr_3)
        out[3] = torch.cat((out[3], updata_attr_3), dim=1)

        new_attr_4 = self.avg(out[4]).permute(1, 0, 2, 3, 4).reshape([4, -1]).permute(1, 0)
        updata_attr_4 = self.agg(x=new_attr_4, edge_index=pyG_B.edge_index, edge_weight=pyG_B.edge_attr.squeeze())
        updata_attr_4 = self.up(updata_attr_4.permute(1, 0).reshape([4, batchSize, 6, 8, 8]).permute(1, 0, 2, 3, 4))
        # out[4] = torch.add(out[4], updata_attr_4)
        out[4] = torch.cat((out[4], updata_attr_4), dim=1)

        out_Trans = self.transDecoder(out[0:4], self.relu(out[4]))
        pre = self.relu(self.out_bn(self.out_conv(out_Trans['pred_masks'])))
        return pre


class MedNeXt_kMax_deeplab_pyG2(nn.Module):
    def __init__(self, input_channels, output_classes, poolSize, sizeWHD,
                 dec_layers=(1, 1, 1, 1),
                 in_channels=(32, 16, 8, 4), num_queries=32, drop_path_prob=0.2):
        super(MedNeXt_kMax_deeplab_pyG2, self).__init__()

        self.en_de_coder = MedNeXt_MO_pyG(input_channels, input_channels, 3, [2, 2, 2, 2, 2, 2, 2, 2, 2],
                                          [2, 3, 4, 4, 4, 4, 4, 3, 2], output_classes, poolSize, sizeWHD)

        self.transDecoder = kMaXTransformerDecoder(dec_layers, in_channels, output_classes - 1, num_queries,
                                                   drop_path_prob)

        self.out_conv = nn.Conv3d(32, output_classes, kernel_size=1, stride=1)
        self.out_bn = nn.BatchNorm3d(output_classes, eps=1e-3, momentum=0.01)
        # self.out_bn = nn.InstanceNorm3d(input_channels, eps=1e-5)
        self.relu = nn.Softmax(dim=1)

    def forward(self, x, pyG_B):
        out = self.en_de_coder(x, pyG_B)
        out_Trans = self.transDecoder(out[0:4], self.relu(out[4]))
        pre = self.relu(self.out_bn(self.out_conv(out_Trans['pred_masks'])))
        return pre


class MedNeXt_kMax_deeplab_pyG1_CA1(nn.Module):
    def __init__(self, input_channels, output_classes, poolSize, sizeWHD,
                 dec_layers=(1, 1, 1, 1),
                 in_channels=(32, 16, 8, 8), num_queries=32, drop_path_prob=0.2):
        super(MedNeXt_kMax_deeplab_pyG1_CA1, self).__init__()

        self.en_de_coder = MedNeXt_MO(input_channels, input_channels, 3, [2, 2, 2, 2, 2, 2, 2, 2, 2],
                                      [2, 3, 4, 4, 4, 4, 4, 3, 2], output_classes)

        self.avg = nn.AvgPool3d(kernel_size=poolSize)
        self.agg = pyGnn.GCNConv(4, 4)
        self.up = nn.Upsample(size=sizeWHD, mode='nearest')

        self.transDecoder = kMaXTransformerDecoder(dec_layers, in_channels, output_classes - 1, num_queries,
                                                   drop_path_prob)

        self.CA_1 = CAM_Module_3D()
        self.CA_2 = CAM_Module_3D()

        # self.CA_1 = ChannelAttention3D()
        # self.CA_2 = ChannelAttention3D()

        self.out_M_C_1 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.out_bn_1 = nn.BatchNorm3d(1, eps=1e-3, momentum=0.01)
        # self.out_bn_1 = nn.InstanceNorm3d(1, eps=1e-5)
        self.out_M_C_2 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.out_bn_2 = nn.BatchNorm3d(1, eps=1e-3, momentum=0.01)
        # self.out_bn_2 = nn.InstanceNorm3d(1, eps=1e-5)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pyG_B):
        batchSize = x.shape[0]
        out = self.en_de_coder(x)

        # 根据输入图结构，更新out[3:4]中的节点特征，采用
        # SimpleConv/GCNConv/ChebConv/GraphConv_s/GatedGraphConv/GATConv/GATv2Conv/TransformerConv/
        # TAGConv/GINEConv/ARMAConv/SGConv/SSGConv/APPNP/RGATConv/DNAConv/GMMConv/SplineConv/NNConv/
        # CGConv/LEConv/PNAConv/GENConv/GCN2Conv/WLConvContinuous_d/FAConv/PDNConv/GeneralConv/LGConv_d

        # 节点属性赋值+平均池化+图更新+上采样+加和
        new_attr_3 = self.avg(out[3]).permute(1, 0, 2, 3, 4).reshape([4, -1]).permute(1, 0)
        updata_attr_3 = self.agg(x=new_attr_3, edge_index=pyG_B.edge_index, edge_weight=pyG_B.edge_attr.squeeze())
        updata_attr_3 = self.up(updata_attr_3.permute(1, 0).reshape([4, batchSize, 6, 8, 8]).permute(1, 0, 2, 3, 4))
        # out[3] = torch.add(out[3], updata_attr_3)
        out[3] = torch.cat((out[3], updata_attr_3), dim=1)

        new_attr_4 = self.avg(out[4]).permute(1, 0, 2, 3, 4).reshape([4, -1]).permute(1, 0)
        updata_attr_4 = self.agg(x=new_attr_4, edge_index=pyG_B.edge_index, edge_weight=pyG_B.edge_attr.squeeze())
        updata_attr_4 = self.up(updata_attr_4.permute(1, 0).reshape([4, batchSize, 6, 8, 8]).permute(1, 0, 2, 3, 4))
        # out[4] = torch.add(out[4], updata_attr_4)
        out[4] = torch.cat((out[4], updata_attr_4), dim=1)

        out_Trans = self.transDecoder(out[0:4], self.relu(out[4]))

        M = out_Trans['pred_masks']

        # 对M使用通道注意力做特征分离，将获取的特征传出用于最小化矩阵内积，再通过2个conv分别预测肿瘤和背景标签(2个交叉熵)。
        # 将特征分离层与类心相乘来预测肿瘤4个分类
        M_C_1 = self.CA_1(M)
        M_C_2 = self.CA_2(M)

        # M_C_1 = torch.mul(self.CA_1(M), M)
        # M_C_2 = torch.mul(self.CA_2(M), M)

        Z = torch.einsum('bnhwd,bnk->bkhwd', M_C_1, out_Trans['pred_logits'])

        out_M_C_1 = self.sigmoid(self.out_M_C_1(M_C_1))
        out_M_C_2 = self.sigmoid(self.out_M_C_2(M_C_2))

        # out_M_C_1 = self.sigmoid(self.out_bn_1(self.out_M_C_1(M_C_1)))
        # out_M_C_2 = self.sigmoid(self.out_bn_2(self.out_M_C_2(M_C_2)))

        return M_C_1, M_C_2, out_M_C_1, out_M_C_2, self.relu(Z)


class MedNeXt_kMax_deeplab_pyG2_CA1(nn.Module):
    def __init__(self, input_channels, output_classes, poolSize, sizeWHD,
                 dec_layers=(1, 1, 1, 1),
                 in_channels=(32, 16, 8, 4), num_queries=32, drop_path_prob=0.2):
        super(MedNeXt_kMax_deeplab_pyG2_CA1, self).__init__()

        self.en_de_coder = MedNeXt_MO_pyG(input_channels, input_channels, 3, [2, 2, 2, 2, 2, 2, 2, 2, 2],
                                          [2, 3, 4, 4, 4, 4, 4, 3, 2], output_classes, poolSize, sizeWHD)

        self.transDecoder = kMaXTransformerDecoder(dec_layers, in_channels, output_classes - 1, num_queries,
                                                   drop_path_prob)

        self.CA_1 = CAM_Module_3D()
        self.CA_2 = CAM_Module_3D()

        # self.CA_1 = ChannelAttention3D()
        # self.CA_2 = ChannelAttention3D()

        self.out_M_C_1 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.out_bn_1 = nn.BatchNorm3d(1, eps=1e-3, momentum=0.01)
        # self.out_bn_1 = nn.InstanceNorm3d(1, eps=1e-5)
        self.out_M_C_2 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.out_bn_2 = nn.BatchNorm3d(1, eps=1e-3, momentum=0.01)
        # self.out_bn_2 = nn.InstanceNorm3d(1, eps=1e-5)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pyG_B):
        out = self.en_de_coder(x, pyG_B)

        out_Trans = self.transDecoder(out[0:4], self.relu(out[4]))

        M = out_Trans['pred_masks']

        # 对M使用通道注意力做特征分离，将获取的特征传出用于最小化矩阵内积，再通过2个conv分别预测肿瘤和背景标签(2个交叉熵)。
        # 将特征分离层与类心相乘来预测肿瘤4个分类
        M_C_1 = self.CA_1(M)
        M_C_2 = self.CA_2(M)

        # M_C_1 = torch.mul(self.CA_1(M), M)
        # M_C_2 = torch.mul(self.CA_2(M), M)

        Z = torch.einsum('bnhwd,bnk->bkhwd', M_C_1, out_Trans['pred_logits'])

        # out_M_C_1 = self.sigmoid(self.out_M_C_1(M_C_1))
        # out_M_C_2 = self.sigmoid(self.out_M_C_2(M_C_2))

        out_M_C_1 = self.sigmoid(self.out_bn_1(self.out_M_C_1(M_C_1)))
        out_M_C_2 = self.sigmoid(self.out_bn_2(self.out_M_C_2(M_C_2)))

        return M_C_1, M_C_2, out_M_C_1, out_M_C_2, self.relu(Z)


class MedNeXt_kMax_deeplab_pyG2_CA2(nn.Module):
    def __init__(self, input_channels, output_classes, poolSize, sizeWHD,
                 dec_layers=(1, 1, 1, 1),
                 in_channels=(32, 16, 8, 4), num_queries=32, drop_path_prob=0.2):
        super(MedNeXt_kMax_deeplab_pyG2_CA2, self).__init__()

        self.en_de_coder = MedNeXt_MO_pyG(input_channels, input_channels, 3, [2, 2, 2, 2, 2, 2, 2, 2, 2],
                                          [2, 3, 4, 4, 4, 4, 4, 3, 2], output_classes, poolSize, sizeWHD)

        self.transDecoder = kMaXTransformerDecoder(dec_layers, in_channels, output_classes - 1, num_queries,
                                                   drop_path_prob)

        self.CA = CAM_Module_3D_2O()

        self.out_M_C_1 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.out_bn_1 = nn.BatchNorm3d(1, eps=1e-3, momentum=0.01)
        # self.out_bn_1 = nn.InstanceNorm3d(1, eps=1e-5)
        self.out_M_C_2 = nn.Conv3d(32, 1, kernel_size=1, stride=1)
        self.out_bn_2 = nn.BatchNorm3d(1, eps=1e-3, momentum=0.01)
        # self.out_bn_2 = nn.InstanceNorm3d(1, eps=1e-5)
        self.relu = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pyG_B):
        out = self.en_de_coder(x, pyG_B)

        out_Trans = self.transDecoder(out[0:4], self.relu(out[4]))

        M = out_Trans['pred_masks']

        # 对M使用通道注意力做特征分离，将获取的特征传出用于最小化矩阵内积，再通过2个conv分别预测肿瘤和背景标签(2个交叉熵)。
        # 将特征分离层与类心相乘来预测肿瘤4个分类
        M_C_1, M_C_2 = self.CA(M)

        # M_C_1 = torch.mul(self.CA_1(M), M)
        # M_C_2 = torch.mul(self.CA_2(M), M)

        Z = torch.einsum('bnhwd,bnk->bkhwd', M_C_1, out_Trans['pred_logits'])

        # out_M_C_1 = self.sigmoid(self.out_M_C_1(M_C_1))
        # out_M_C_2 = self.sigmoid(self.out_M_C_2(M_C_2))

        out_M_C_1 = self.sigmoid(self.out_bn_1(self.out_M_C_1(M_C_1)))
        out_M_C_2 = self.sigmoid(self.out_bn_2(self.out_M_C_2(M_C_2)))

        return M_C_1, M_C_2, out_M_C_1, out_M_C_2, self.relu(Z)


class nnUNet_kMax_deeplab_OOD(nn.Module):
    def __init__(self, input_channels, output_classes, channels_list, dec_layers=(1, 1, 1, 1),
                 in_channels=(32, 16, 8, 4), num_queries=32, drop_path_prob=0.2,
                 deep_supervision=False, ds_layer=4, block: Union[PlainBlock, ResidualBlock] = PlainBlock,
                 **block_kwargs):
        super(nnUNet_kMax_deeplab_OOD, self).__init__()

        self.norm = nn.InstanceNorm3d(input_channels, eps=1e-5, affine=True)

        self.encoder = UNetEncoder(input_channels, channels_list, block=block, **block_kwargs)
        self.decoder = UNetDecoder_MO(output_classes, channels_list[::-1], block=block,
                                      deep_supervision=deep_supervision, ds_layer=ds_layer, **block_kwargs)

        self.transDecoder = kMaXTransformerDecoder(dec_layers, in_channels, output_classes - 1, num_queries,
                                                   drop_path_prob)

        # self.out_conv = nn.Conv3d(32, output_classes, kernel_size=1, stride=1)
        # self.out_bn = nn.BatchNorm3d(output_classes, eps=1e-3, momentum=0.01)
        # self.out_bn = nn.InstanceNorm3d(input_channels, eps=1e-5)
        self.relu = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)

        out = self.decoder(self.encoder(x, return_skips=True))
        out_Trans = self.transDecoder(out[0:4], self.relu(out[4]))

        M = self.relu(out_Trans['pred_masks'])
        Z = torch.einsum('bnhwd,bnk->bkhwd', M, out_Trans['pred_logits'])

        M_1 = torch.mean(M[:, 0:16, :, :, :], dim=1).unsqueeze(dim=1)
        M_2 = torch.mean(M[:, 16:32, :, :, :], dim=1).unsqueeze(dim=1)
        return M_1, M_2, self.relu(Z)


class nnUNet_kMax_deeplab_OOD_load(nn.Module):
    def __init__(self, preNet, output_classes, dec_layers=(1, 1, 1, 1), in_channels=(32, 16, 8, 4), num_queries=32,
                 drop_path_prob=0.2):
        super(nnUNet_kMax_deeplab_OOD_load, self).__init__()

        self.preModel = preNet

        self.transDecoder = kMaXTransformerDecoder(dec_layers, in_channels, output_classes - 1, num_queries,
                                                   drop_path_prob)
        self.relu = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.preModel(x)
        out_Trans = self.transDecoder(out[0:4], self.relu(out[4]))

        M = self.relu(out_Trans['pred_masks'])
        Z = torch.einsum('bnhwd,bnk->bkhwd', M, out_Trans['pred_logits'])

        M_1 = torch.sum(M[:, 0:16, :, :, :], dim=1).unsqueeze(dim=1)
        M_2 = torch.sum(M[:, 16:32, :, :, :], dim=1).unsqueeze(dim=1)
        return M_1, M_2, self.relu(Z)


class ChannelAttention3D_old(nn.Module):
    def __init__(self, inC=32, reduce=4, outC=32, kernel_size=1, pool_size=(24, 256, 256)):
        super(ChannelAttention3D_old, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(pool_size)
        self.max_pool = nn.AdaptiveMaxPool3d(pool_size)

        self.fc1 = nn.Conv3d(inC, inC // reduce, kernel_size, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(inC // reduce, outC, kernel_size, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return torch.mul(self.sigmoid(out), x)


class ChannelAttention3D(nn.Module):
    def __init__(self, channel=32, ratio=2):
        super(ChannelAttention3D, self).__init__()
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv3d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention3D(nn.Module):
    def __init__(self, outC=3, kernel_size=7):
        super(SpatialAttention3D, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, outC, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_ = torch.cat([avg_out, max_out], dim=1)
        x_ = self.sigmoid(self.conv1(x_))
        return torch.mul(x_, x)


class CBAM_3D(nn.Module):
    def __init__(self, inC=4, outC=4, pool_size=(24, 256, 256)):
        super(CBAM_3D, self).__init__()

        self.channel_att = ChannelAttention3D_old(inC, outC, pool_size)
        self.spatial_att = SpatialAttention3D(outC=outC, kernel_size=7)

    def forward(self, x):
        f1 = self.channel_att(x)
        out = self.spatial_att(f1)
        return out


class CAM_Module_3D(nn.Module):
    """ Channel attention module"""

    def __init__(self, ):
        super(CAM_Module_3D, self).__init__()

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


class CAM_Module_3D_2O(nn.Module):
    """ Channel attention module"""

    def __init__(self, ):
        super(CAM_Module_3D_2O, self).__init__()

        self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma_ = nn.Parameter(torch.zeros(1))
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

        out_ = torch.bmm(1 - attention, proj_value)
        out_ = out_.view(m_batchsize, C, height, width, deep)
        out_ = self.gamma_ * out_ + x

        return out, out_


def create_graph(Input, sizeWHD, sizeG, bin, sim_thr):
    device = Input.device
    # 按照sizeG大小将input分块,生成图
    kernelSize = np.floor_divide(sizeWHD, sizeG)
    pyG_list = []
    # CropData = torch.zeros([batchSize, bin * input_channels, sizeG[0], sizeG[1], sizeG[2]])
    for i in range(Input.shape[0]):
        node_attr = []
        node_loc = []
        edge_index = []
        edge_attr = []
        for x in range(0, sizeG[0]):
            for y in range(0, sizeG[1]):
                for z in range(0, sizeG[2]):
                    #   取出当前块的数据,每个通道上计算hist，并拼接为新的向量，用于计算相似度
                    histData = []
                    for c in range(0, input_channels):
                        curData = Input[i, c, x * kernelSize[0]:(x + 1) * kernelSize[0],
                                  y * kernelSize[1]:(y + 1) * kernelSize[1], z * kernelSize[2]:(z + 1) * kernelSize[2]]
                        histData.append(torch.histc(curData.reshape((-1, 1)), bin, min=0, max=1))
                    # CropData[i, :, x, y, z] = torch.cat(histData, dim=0)
                    node_attr.append(torch.cat(histData, dim=0))
                    node_loc.append(torch.tensor([x, y, z]))
        pointDist_c = squareform(pdist(torch.stack(node_attr).to('cpu'), 'cosine'))
        pointDist_e = squareform(pdist(torch.stack(node_loc).to('cpu'), 'euclidean'))
        for m in range(0, pointDist_c.shape[0]):
            for n in range(m + 1, pointDist_c.shape[1]):
                if pointDist_c[m, n] > sim_thr:
                    edge_index.append(torch.Tensor([m, n]).long())
                    # dis = torch.pairwise_distance(node_loc[m].unsqueeze(0), node_loc[n].unsqueeze(0))
                    edge_attr.append(1 / torch.tensor(pointDist_e[m, n]))

        pyG_list.append(
            pygData(x=torch.stack(node_attr).to(device), edge_index=torch.stack(edge_index).permute(1, 0).to(device),
                    edge_attr=torch.stack(edge_attr).to(device)))
    pyG_B = pygBatch.from_data_list(pyG_list)
    return pyG_B


if __name__ == '__main__':
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    # modelPath = r'E:\code\dpNeuStroke\pyNetFrame\MPGNet\models\pretrain'
    # model = generate_model(model_type='resnet', model_depth=34,
    #                        input_W=256, input_H=256, input_D=256, resnet_shortcut='B',
    #                        no_cuda=False, gpu_id=[0],
    #                        pretrain_path=modelPath + r'\resnet_34_23dataset.pth',
    #                        nb_class=11)

    dist.init_process_group('nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)

    batchSize = 4
    input_channels = 4
    output_classes = 4
    channels_list = [4, 8, 16, 32, 32]
    sizeWHD = [24, 256, 256]

    # nnUnet
    # newNet = nnUNet(input_channels, output_classes, channels_list).to(device)

    # MedNeXt
    # base_c = input_channels * 2
    # k_size = 3
    # num_block = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    # scale = [2, 3, 4, 4, 4, 4, 4, 3, 2]
    # newNet = MedNeXt(input_channels, base_c, k_size, num_block, scale, output_classes).to(device)

    # kMax_deeplab
    # newNet = nnUNet_kMax_deeplab(input_channels, output_classes, channels_list).to(device)

    # kMax_deeplab_OOD
    # newNet = nnUNet_kMax_deeplab_OOD(input_channels, output_classes, channels_list).to(device)

    # ours RPPNet

    # 按照sizeG大小将input分块,生成图
    Input = torch.rand(size=(batchSize, input_channels, sizeWHD[0], sizeWHD[1], sizeWHD[2])).to(device)
    Label = torch.rand(size=(batchSize, output_classes, sizeWHD[0], sizeWHD[1], sizeWHD[2])).to(device)

    sizeG = [6, 8, 8]
    bin = 100
    sim_thr = -1
    pyG_B = create_graph(Input, sizeWHD, sizeG, bin, sim_thr)
    newNet = nnUNet_kMax_deeplab(input_channels, output_classes, channels_list,
                                 [sizeWHD[0] // sizeG[0], sizeWHD[1] // sizeG[1], sizeWHD[2] // sizeG[2]], sizeWHD).to(
        device)

    # 开始训练
    criterion = nn.L1Loss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, newNet.parameters()), lr=0.01)
    optimizer.zero_grad()

    out = newNet(Input, pyG_B)
    crossM = torch.sum(torch.mul(out[0], out[1]), dim=(1, 2, 3, 4)).unsqueeze(dim=1)

    # loss = criterion(out[2], Label)
    loss = criterion(crossM, torch.sum(Label - Label, dim=(1, 2, 3, 4)).unsqueeze(dim=1))
    loss.backward()
    optimizer.step()

    print("over")
