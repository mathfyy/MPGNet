import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from SpecificApplication.brainLesion import trainBrainTumor_paper

from AlgorithmPool.NetPool.NetFrame import kmax_deeplab_M_Net
from AlgorithmPool.NetPool.NetFrame.MedNeXt import MedNeXt, MedNeXt_MO
from AlgorithmPool.NetPool.NetFrame.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from AlgorithmPool.NetPool.NetFrame.UXNet_3D import UXNET

from AlgorithmPool.NetPool.NetFrame.MPEDANet import MPEDANet
from AlgorithmPool.NetPool.NetFrame.ETUNetModel import ETUNet


def trainBrainTumor(samplePath, savePath):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dist.init_process_group('nccl', init_method='tcp://localhost:23450', rank=0, world_size=1)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(1)

    input_channels = 4
    output_classes = 4  # 样本类别数
    # channels_list = [4, 8, 16, 32, 32]
    # channels_list = [4 * 2, 8 * 2, 16 * 2, 32 * 2, 32 * 2]
    channels_list = [4 * 3, 8 * 3, 16 * 3, 32 * 3, 32 * 3]
    # channels_list = [4 * 4, 8 * 4, 16 * 4, 32 * 4, 32 * 4]
    sizeWHD = [24, 256, 256]
    sizeG = [6, 8, 8]
    sim_thr = 0.75
    TVLoss_weight = 1

    val = 1
    batchSize = 4  # =4，若修改，gnn参数也要改一下
    max_epoch = 200
    lr = 3e-4 #通用
    lrStep = 3

    # 可多GPU训练的：nnUnet,MedNeXt,TransBTS,UXNET


    # MPEDANet
    # netModel = MPEDANet(in_channels=input_channels, num_classes=output_classes-1)
    # model_index = 200

    # ETUNet
    # netModel = ETUNet(input_channels, output_classes-1)
    # model_index = 200

    # TransBTS
    # _, netModel = TransBTS(dataset='ydy_brats', _conv_repr=True, _pe_type="learned")
    # model_index = 200

    # 3DUXNet
    # netModel = UXNET(in_chans=4, out_chans=3, depths=[2, 2, 2, 2], feat_size=[4 * 3, 8 * 3, 16 * 3, 32 * 3],
    #               drop_path_rate=0, layer_scale_init_value=1e-6, hidden_size=32 * 3,
    #               norm_name="instance", conv_block=True, res_block=True, spatial_dims=3)
    # model_index = 200

    # nnUNet
    # netModel = kmax_deeplab_M_Net.nnUNet(input_channels, output_classes, channels_list)
    # model_index = 200  # 独立的maxloss+tvloss

    # kMax_deeplab
    # netModel = kmax_deeplab_M_Net.nnUNet_kMax_deeplab(input_channels, output_classes, channels_list)
    # model_index = 200

    # MedNeXt
    # base_c = input_channels * 3
    # k_size = 5
    # num_block = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    # scale = [2, 3, 4, 4, 4, 4, 4, 3, 2]
    # netModel = MedNeXt_MO(input_channels, base_c, k_size, num_block, scale, output_classes - 1)
    # model_index = 200

    # dResUNet
    # netModel = kmax_deeplab_M_Net.dResUNet(input_channels, output_classes, channels_list)
    # model_index = 200

    # MBANet
    # netModel = kmax_deeplab_M_Net.MBANet(input_channels, output_classes, channels_list)
    # model_index = 200

    # CFPNet参数实验
    # TVLoss_weight = 2
    # TVLoss_weight = 1.75
    TVLoss_weight = 1.5
    # TVLoss_weight = 1.25
    # TVLoss_weight = 0.75
    # TVLoss_weight = 0.5
    # TVLoss_weight = 0.25
    # TVLoss_weight = 0

    # MPGNet
    netModel = kmax_deeplab_M_Net.MPGNet(batchSize, input_channels, output_classes, channels_list)
    model_index = 20

    # input = torch.rand(size=(batchSize, input_channels, sizeWHD[0], sizeWHD[1], sizeWHD[2]))
    # output = netModel(input)

    # if torch.cuda.device_count() > 1:
    #     netModel = torch.nn.DataParallel(netModel)
    netModel = netModel.to(device)

    # 跑5折交叉验证
    for i in range(5):
        trainBrainTumor_paper.trainnet(device, netModel, samplePath, sizeWHD, sizeG, sim_thr, savePath, batchSize,
                                   max_epoch, lr, i+1, val, model_index, TVLoss_weight)

    return
