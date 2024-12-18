import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib
import random
from torch_geometric.data import Data as pygData
from torch_geometric.data import Batch as pygBatch

from SpecificApplication.brainLesion import loadBrainData
from AlgorithmPool.FunctionalMethodPool.DataIO import writeNII
from AlgorithmPool.NetPool.LossFunction import DiceLoss, BCEDiceLoss, FocalLoss, ContrastLoss, TVLoss, MLoss
from AlgorithmPool.NetPool.EvalFunction import IOU
from AlgorithmPool.NetPool.EvalFunction import Dice_

import numpy as np


# 自动调节学习率
def adjust_learning_rate(optimizer, epoch, ilr, adjustEpoch):
    """Sets the learning rate to the initial LR decayed by 2 every 50 epochs"""
    #    print('ori lr',optimizer.param_groups[0]['lr'])
    for param_group in optimizer.param_groups:
        if param_group['lr'] > 3e-8:
            lr = ilr * (0.1 ** (epoch // adjustEpoch))
            param_group['lr'] = lr


def create_graph(pointDist, sizeG, sim_thr):
    device = pointDist.device
    pyG_list = []
    for i in range(pointDist.shape[0]):
        edge_index = []
        edge_attr = []
        node_attr = torch.ones([sizeG[0] * sizeG[1] * sizeG[2], 1]).to(device)
        curMaxSim = torch.max(pointDist[i, 0, :, :])
        for m in range(0, pointDist.shape[2]):
            if curMaxSim > 0:
                for n in range(m + 1, pointDist.shape[3]):
                    if pointDist[i, 0, m, n] > sim_thr * curMaxSim:
                        edge_index.append(torch.Tensor([m, n]).long().to(device))
                        edge_attr.append(pointDist[i, 2, m, n])
            else:
                edge_index.append(torch.Tensor([m, m]).long().to(device))
                edge_attr.append(pointDist[i, 2, m, m])

        pyG_list.append(pygData(x=node_attr.to(device), edge_index=torch.stack(edge_index).permute(1, 0).to(device),
                                edge_attr=torch.stack(edge_attr).to(device)))
    pyG_B = pygBatch.from_data_list(pyG_list)
    return pyG_B


def create_graph_new(device, pointDist, sizeG, sim_thr):
    # device = pointDist.device
    pyG_list = []
    for i in range(pointDist.shape[0]):
        edge_index = []
        edge_attr = []
        node_attr = torch.ones([sizeG[0] * sizeG[1] * sizeG[2], 1]).to(device)
        curMaxSim = torch.max(pointDist[i, 0, :, :])
        for m in range(0, pointDist.shape[2]):
            if curMaxSim > 0:
                for n in range(m + 1, pointDist.shape[3]):
                    edge_index.append(torch.Tensor([m, n]).long().to(device))
                    edge_attr.append(pointDist[i, 0, m, n])
            else:
                edge_index.append(torch.Tensor([m, m]).long().to(device))
                edge_attr.append(pointDist[i, 0, m, m])

        pyG_list.append(pygData(x=node_attr.to(device), edge_index=torch.stack(edge_index).permute(1, 0).to(device),
                                edge_attr=torch.stack(edge_attr).to(device)))
    pyG_B = pygBatch.from_data_list(pyG_list)
    return pyG_B


def create_graph_index(device, batch, sizeG):
    pyG_B = []

    # pyG_list = []
    # node_attr = torch.ones([sizeG[0] * sizeG[1] * sizeG[2], 1]).to(device)
    # for i in range(batch):
    #     edge_index = []
    #     edge_attr = []
    #     for m in range(0, sizeG[0] * sizeG[1] * sizeG[2]):
    #         for n in range(0, sizeG[0] * sizeG[1] * sizeG[2]):
    #             edge_index.append(torch.Tensor([m, n]).long().to(device))
    #             edge_attr.append(torch.Tensor([1.0]).to(device))
    #
    #     pyG_list.append(pygData(x=node_attr, edge_index=torch.stack(edge_index).permute(1, 0),
    #                             edge_attr=torch.stack(edge_attr)))
    # pyG_B = pygBatch.from_data_list(pyG_list)
    return pyG_B


def ev_predict_new(label, output_use, log, thr=0.5, volume_o_thr=8 * 8 * 6, volume_l_thr=6 * 6 * 4):
    pre_all = torch.zeros([output_use.shape[0], 4])
    rec_all = torch.zeros([output_use.shape[0], 4])
    dsc_all = torch.zeros([output_use.shape[0], 4])
    iou_all = torch.zeros([output_use.shape[0], 4])
    for b in range(output_use.shape[0]):
        for i in range(output_use.shape[1]):
            cur_outputL = torch.unsqueeze(output_use[b, i, :, :, :], dim=0)
            cur_outputL = torch.unsqueeze(cur_outputL, dim=0)
            cur_label = torch.unsqueeze(label[b, i, :, :, :], dim=0)
            cur_label = torch.unsqueeze(cur_label, dim=0)

            # 当预测结果体积小于阈值时，置为0
            volume_o = torch.sum(torch.mul(cur_outputL > thr, torch.ones_like(cur_outputL)), dim=(0, 1, 2, 3, 4))
            if volume_o <= volume_o_thr:
                cur_outputL = torch.zeros_like(cur_outputL)

            # 标签异常处理:当标签体积小于阈值时，置为0
            volume_l = torch.sum(torch.mul(cur_label > thr, torch.ones_like(cur_label)), dim=(0, 1, 2, 3, 4))
            if volume_l <= volume_l_thr:
                cur_label = torch.zeros_like(cur_label)

            precision, accuracy, recall, FPR_x, fpRate, dsc, iou, f1, mcc, preV, labelV, diffV = Dice_.ev(cur_outputL,
                                                                                                         cur_label,
                                                                                                         compare_rule='>',
                                                                                                         thr=thr)
            # if dsc < 0.1:
            #     print("volume_o:%f volume_l:%f" % (volume_o, volume_l))
            #     log.write("volume_o:%f volume_l:%f\n" % (volume_o, volume_l))

            pre_all[b, i] = precision
            rec_all[b, i] = recall
            dsc_all[b, i] = dsc
            iou_all[b, i] = iou

    return torch.mean(pre_all, dim=0), torch.mean(rec_all, dim=0), torch.mean(dsc_all, dim=0), torch.mean(iou_all,
                                                                                                          dim=0)


def ev_predict(label, output_use):
    label = torch.nn.functional.one_hot(torch.squeeze(label, dim=1).long(), output_use.shape[1]).permute(0, 4,
                                                                                                         1,
                                                                                                         2,
                                                                                                         3).to(
        torch.float32)

    outputL = torch.argmax(output_use, dim=1).to(torch.float32)
    outputL = torch.nn.functional.one_hot(torch.squeeze(outputL, dim=1).long(), output_use.shape[1]).permute(0,
                                                                                                             4,
                                                                                                             1,
                                                                                                             2,
                                                                                                             3).to(
        torch.float32)
    pre = []
    rec = []
    dsc = []
    iou = []
    for i in range(outputL.shape[1]):
        cur_outputL = torch.unsqueeze(outputL[:, i, :, :, :], dim=1)
        cur_label = torch.unsqueeze(label[:, i, :, :, :], dim=1)

        precision, accuracy, recall, FPR_x, fpRate, dsc_, iou_, f1, mcc, preV, labelV, diffV = Dice.ev(cur_outputL,
                                                                                                       cur_label)
        pre.append(precision)
        rec.append(recall)
        dsc.append(dsc_)
        iou.append(iou_)

    return pre, rec, dsc, iou


def ev_predict_old(label, output_use):
    label = torch.nn.functional.one_hot(torch.squeeze(label, dim=1).long(), output_use.shape[1]).permute(0, 4,
                                                                                                         1,
                                                                                                         2,
                                                                                                         3).to(
        torch.float32)

    outputL = torch.argmax(output_use, dim=1).to(torch.float32)
    outputL = torch.nn.functional.one_hot(torch.squeeze(outputL, dim=1).long(), output_use.shape[1]).permute(0,
                                                                                                             4,
                                                                                                             1,
                                                                                                             2,
                                                                                                             3).to(
        torch.float32)
    accP = Dice.Dice(outputL, label, '>')
    accN = Dice.Dice(outputL, label, '<=')
    return accP, accN


def trainnet(device, net, samplePath, sizeWHD, sizeG, sim_thr, savePath, batchSize, max_epoch, lr, fold_num, val,
             model_index, TVLoss_weight):
    random.seed(1234)
    np.random.seed(1234)

    thr = 0.5
    factor = 0.2
    lr_scheduler_eps = 1e-3
    lr_scheduler_patience = 30
    initial_lr = lr
    weight_decay = 3e-5
    batchSize_t = 4

    # 定义优化器
    if model_index == 5 or model_index == 91:
        for name, parameter in net.named_parameters():
            if 'preModel' in name:
                parameter.requires_grad = False
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), initial_lr,
                                        weight_decay=weight_decay, amsgrad=True)
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor,
                                                           patience=lr_scheduler_patience,
                                                           verbose=True, threshold=lr_scheduler_eps,
                                                           threshold_mode="abs")
    else:
        # optimizer = optim.AdamW(net.parameters(), lr, weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(net.parameters(), initial_lr, weight_decay=weight_decay, amsgrad=True)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor,
                                                   patience=lr_scheduler_patience,
                                                   verbose=True, threshold=lr_scheduler_eps,
                                                   threshold_mode="abs")

    if model_index == 22:
        pyG_B = create_graph_index(device, batchSize, sizeG)

    # 定义损失
    # criterionED = BCEDiceLoss.DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
    criterionED = BCEDiceLoss.BCEDiceLoss()
    criterionL1 = nn.L1Loss()
    # criterionL1 = nn.MSELoss()
    criterionC = nn.BCELoss()

    # criterionTV = TVLoss.TVLoss_2d_edge(TVLoss_weight)
    # criterionTV = TVLoss.TVLoss_3d_edge(TVLoss_weight)
    # criterionTV = TVLoss.TVLoss_2d(TVLoss_weight)
    criterionTV = TVLoss.TVLoss_3d(TVLoss_weight)

    # 增加loss的比较
    if model_index == 201:
        criterion = MLoss.DiceLoss()
    if model_index == 202:
        criterion = MLoss.DiceLossSquared()
    if model_index == 203:
        criterion = MLoss.generalized_dice_loss()
    if model_index == 204:
        criterion = FocalLoss.DiceFocalLoss()
    if model_index == 205:
        criterion = FocalLoss.ComboLoss()
    if model_index == 206:
        criterion = MLoss.focal_tversky_loss()
    if model_index == 207:
        criterion = MLoss.tversky_loss()
    if model_index == 208:
        criterion = MLoss.LogCoshDiceLoss()
    if model_index == 209:
        criterion = FocalLoss.OnlyFocalLoss()

    # 切换多个损失函数，进行对比
    # Dice loss
    # Dice loss squared
    # Generalized dice loss
    # Dice focal loss
    # Combo loss
    # Focal tversky loss【】
    # Tversky loss
    # Log-cosh dice loss
    # Focal loss

    # # 加载数据
    # MyDatasetFYY_TCGA_Txt
    train_data = loadBrainData.MyDatasetFYY_TCGA_Txt_skull(path=samplePath + 'train',
                                                     txt_name='_name' + str(fold_num) + '.txt',
                                                     transform=transforms.ToTensor(),
                                                     target_transform=transforms.ToTensor())

    val_loader = []
    if val == 1:
        val_data = loadBrainData.MyDatasetFYY_TCGA_Txt_skull(path=samplePath + 'train',
                                                       txt_name='_val_name' + str(fold_num) + '.txt',
                                                       transform=transforms.ToTensor(),
                                                       target_transform=transforms.ToTensor())
        val_loader = DataLoader(dataset=val_data, batch_size=batchSize_t, num_workers=1, pin_memory=True,
                                drop_last=True)

    test_data = loadBrainData.MyDatasetFYY_TCGA_py(path=samplePath + 'test', transform=transforms.ToTensor(),
                                                   target_transform=transforms.ToTensor())

    # skull
    # train_data = loadBrainData.MyDatasetFYY_TCGA_skull(path=samplePath + 'train', transform=transforms.ToTensor(),
    #                                                      target_transform=transforms.ToTensor())
    # test_data = loadBrainData.MyDatasetFYY_TCGA_skull_t(path=samplePath + 'test', transform=transforms.ToTensor(),
    #                                                     target_transform=transforms.ToTensor())

    test_loader = DataLoader(dataset=test_data, batch_size=batchSize_t, num_workers=1, pin_memory=True, drop_last=True)


    # 按照epoch进行训练
    Loss_list = []
    Accuracy_list = []

    acc_train_pre = 0.0
    acc_train_rec = 0.0
    acc_train_dsc = 0.0
    acc_train_iou = 0.0
    acc_test_pre = 0.0
    acc_test_rec = 0.0
    acc_test_dsc = 0.0
    acc_test_iou = 0.0
    acc_val_pre = 0.0
    acc_val_rec = 0.0
    acc_val_dsc = 0.0
    acc_val_iou = 0.0

    log = open(savePath + str(fold_num) + "_log.txt", "w")
    maxIterNum = 0
    for epoch in range(0, int(max_epoch)):
        # adjust_learning_rate(optimizer, epoch, lr, 5)  # 自动调整学习率

        train_loader = DataLoader(dataset=train_data, batch_size=batchSize, shuffle=True, num_workers=1,
                                  pin_memory=True, drop_last=True)

        net.train()
        train_loss = 0.0

        pre_0 = 0.0
        pre_1 = 0.0
        pre_2 = 0.0
        pre_3 = 0.0
        rec_0 = 0.0
        rec_1 = 0.0
        rec_2 = 0.0
        rec_3 = 0.0
        dsc_0 = 0.0
        dsc_1 = 0.0
        dsc_2 = 0.0
        dsc_3 = 0.0
        iou_0 = 0.0
        iou_1 = 0.0
        iou_2 = 0.0
        iou_3 = 0.0

        for i, data in enumerate(train_loader):
            torch.cuda.empty_cache()

            # 输入数据,在网络中instanceNorm归一化
            input = torch.unsqueeze(data[0].to(device, non_blocking=True), dim=-4).transpose(3, 4)
            input = torch.cat((input[:, :, 0:sizeWHD[0], :, :], input[:, :, sizeWHD[0]:2 * sizeWHD[0], :, :],
                               input[:, :, 2 * sizeWHD[0]:3 * sizeWHD[0], :, :],
                               input[:, :, 3 * sizeWHD[0]:4 * sizeWHD[0], :, :]), dim=1)

            # 标签为4的，重置为3
            label = torch.unsqueeze(data[2].to(device, non_blocking=True), dim=-4).transpose(3, 4)
            label[label == 4] = 3
            label = torch.mul(label <= 3, label)

            if model_index == 4 or model_index == 6 or model_index == 9 or model_index == 91:
                pyData = data[1][:, 0:4, :, :].to(device, non_blocking=True)

            # 随机翻转
            ts = np.random.random()
            if ts > 0.5:
                input = torch.flip(input, [4])
                label = torch.flip(label, [4])
                if model_index == 4 or model_index == 6 or model_index == 9 or model_index == 91:
                    pyData = data[1][:, 4:8, :, :].to(device, non_blocking=True)

            # ET:增强区，TC:坏死+增强区，WT:坏死+增强+水肿区

            # brats_t: ET=3,TC=3+2;WT=1+2+3
            # label_ET = torch.mul(label == 3, torch.ones_like(label))
            # label_TC = torch.mul(label > 1, torch.ones_like(label))
            # label_WT = torch.mul(label > 0, torch.ones_like(label))
            # label_new = torch.cat((label_ET, label_TC, label_WT), dim=1)

            # brats_2021&ydy: ET=3,TC=3+1;WT=1+2+3
            label_ET = torch.mul(label == 3, torch.ones_like(label))
            label_TC = torch.mul(label == 1, torch.ones_like(label)) + label_ET
            label_WT = torch.mul(label > 0, torch.ones_like(label))
            label_new = torch.cat((label_ET, label_TC, label_WT), dim=1)

            optimizer.zero_grad()

            if model_index == 4 or model_index == 6 or model_index == 9 or model_index == 91:
                net = net.module if hasattr(net, 'module') else net

            if model_index == 0:
                output = net(input)
                output_use = output[5]
                loss = criterionED(output_use, label_new)

                # output_ET = torch.unsqueeze(output[5][:, 0, :, :, :], dim=1)
                # output_TC = torch.unsqueeze(output[5][:, 1, :, :, :], dim=1)
                # output_WT = torch.unsqueeze(output[5][:, 2, :, :, :], dim=1)
                # loss = criterionED(output_ET, label_ET) + criterionED(output_TC, label_TC) + criterionED(output_WT,
                #                                                                                          label_WT)
                # loss = criterionED(output_use, label)
                loss_cross = loss
                loss_C1 = loss
                loss_C2 = loss
                loss_ed = loss

            if model_index == 1:
                output = net(input)
                output_use = output[4]
                loss = criterionED(output[4], label_new)
                loss_cross = loss
                loss_C1 = loss
                loss_C2 = loss
                loss_ed = loss

            if model_index == 2:
                output_use = net(input)
                loss = criterionED(output_use, label_new)
                loss_cross = loss
                loss_C1 = loss
                loss_C2 = loss
                loss_ed = loss

            if model_index == 20:
                output_use = net(input)

                label_NET = torch.mul(label == 1, torch.ones_like(label))
                label_ED = torch.mul(label == 2, torch.ones_like(label))
                label_ones = torch.cat((label_ET, label_NET, label_ED), dim=1)

                loss_ed = criterionED(output_use, label_ones)

                output_ET = output_use[:, 0, :, :, :].unsqueeze(dim=1)
                output_TC = torch.max(output_use[:, 0, :, :, :], output_use[:, 1, :, :, :]).unsqueeze(dim=1)
                output_WT = torch.max(torch.max(output_use[:, 0, :, :, :], output_use[:, 1, :, :, :]),
                                      output_use[:, 2, :, :, :]).unsqueeze(dim=1)
                output_use = torch.cat((output_ET, output_TC, output_WT), dim=1)

                if epoch >= 0:
                    outputPre = torch.mul(output_use > thr, torch.ones_like(output_use)).to(torch.float32)
                    ET_t1ceV = torch.mul(outputPre[:, 0, :, :, :], input[:, 1, :, :, :])
                    ET_flV = torch.mul(outputPre[:, 0, :, :, :], input[:, 2, :, :, :])

                    NET_V = torch.mul(outputPre[:, 1, :, :, :] - outputPre[:, 0, :, :, :] > thr,
                                      torch.ones_like(outputPre[:, 0, :, :, :])).to(torch.float32)
                    NET_t1ceV = torch.mul(NET_V, input[:, 1, :, :, :])
                    NET_flV = torch.mul(NET_V, input[:, 2, :, :, :])

                    ED_V = torch.mul(outputPre[:, 2, :, :, :] - outputPre[:, 1, :, :, :] > thr,
                                     torch.ones_like(outputPre[:, 0, :, :, :])).to(torch.float32)
                    ED_t1ceV = torch.mul(ED_V, input[:, 1, :, :, :])
                    ED_flV = torch.mul(ED_V, input[:, 2, :, :, :])

                    ET_t1ceV = torch.clamp(ET_t1ceV, min=0, max=1)
                    ET_flV = torch.clamp(ET_flV, min=0, max=1)
                    NET_t1ceV = torch.clamp(NET_t1ceV, min=0, max=1)
                    NET_flV = torch.clamp(NET_flV, min=0, max=1)
                    ED_t1ceV = torch.clamp(ED_t1ceV, min=0, max=1)
                    ED_flV = torch.clamp(ED_flV, min=0, max=1)

                    # 2d_edge
                    # loss_cross = (criterionTV(ET_t1ceV, outputPre[:, 0, :, :, :]) + criterionTV(ET_flV,
                    #                                                                             outputPre[:, 0, :, :,
                    #                                                                             :]) + criterionTV(
                    #     NET_t1ceV, NET_V) + criterionTV(NET_flV, NET_V) + criterionTV(ED_t1ceV, ED_V) + criterionTV(
                    #     ED_flV, ED_V))
                    # 3d_edge
                    # loss_cross = (criterionTV(torch.unsqueeze(ET_t1ceV, dim=1),
                    #                           torch.unsqueeze(outputPre[:, 0, :, :, :], dim=1)) + criterionTV(
                    #     torch.unsqueeze(ET_flV, dim=1),
                    #     torch.unsqueeze(outputPre[:, 0, :, :,
                    #                     :], dim=1)) + criterionTV(
                    #     torch.unsqueeze(NET_t1ceV, dim=1), torch.unsqueeze(NET_V, dim=1)) + criterionTV(
                    #     torch.unsqueeze(NET_flV, dim=1), torch.unsqueeze(NET_V, dim=1)) + criterionTV(
                    #     torch.unsqueeze(ED_t1ceV, dim=1), torch.unsqueeze(ED_V, dim=1)) + criterionTV(
                    #     torch.unsqueeze(ED_flV, dim=1), torch.unsqueeze(ED_V, dim=1)))
                    # 2d
                    # loss_cross = (criterionTV(ET_t1ceV) + criterionTV(ET_flV) + criterionTV(
                    #     NET_t1ceV) + criterionTV(NET_flV) + criterionTV(ED_t1ceV) + criterionTV(
                    #     ED_flV))
                    # 3d
                    loss_cross = (criterionTV(torch.unsqueeze(ET_t1ceV, dim=1)) + criterionTV(
                        torch.unsqueeze(ET_flV, dim=1)) + criterionTV(
                        torch.unsqueeze(NET_t1ceV, dim=1)) + criterionTV(torch.unsqueeze(NET_flV, dim=1)) + criterionTV(
                        torch.unsqueeze(ED_t1ceV, dim=1)) + criterionTV(
                        torch.unsqueeze(ED_flV, dim=1)))
                else:
                    loss_cross = 0
                loss = loss_ed + loss_cross
                loss_C1 = loss
                loss_C2 = loss

            if model_index == 200:
                output_use = net(input)

                label_NET = torch.mul(label == 1, torch.ones_like(label))
                label_ED = torch.mul(label == 2, torch.ones_like(label))
                label_ones = torch.cat((label_ET, label_NET, label_ED), dim=1)

                loss_ed = criterionED(output_use, label_ones)

                output_ET = output_use[:, 0, :, :, :].unsqueeze(dim=1)
                output_TC = torch.max(output_use[:, 0, :, :, :], output_use[:, 1, :, :, :]).unsqueeze(dim=1)
                output_WT = torch.max(torch.max(output_use[:, 0, :, :, :], output_use[:, 1, :, :, :]),
                                      output_use[:, 2, :, :, :]).unsqueeze(dim=1)
                output_use = torch.cat((output_ET, output_TC, output_WT), dim=1)

                loss = loss_ed
                loss_cross = loss
                loss_C1 = loss
                loss_C2 = loss

            if model_index > 200 and model_index < 210:
                output_use = net(input)

                label_NET = torch.mul(label == 1, torch.ones_like(label))
                label_ED = torch.mul(label == 2, torch.ones_like(label))
                label_ones = torch.cat((label_ET, label_NET, label_ED), dim=1)

                loss = criterion(output_use, label_ones)

                output_ET = output_use[:, 0, :, :, :].unsqueeze(dim=1)
                output_TC = torch.max(output_use[:, 0, :, :, :], output_use[:, 1, :, :, :]).unsqueeze(dim=1)
                output_WT = torch.max(torch.max(output_use[:, 0, :, :, :], output_use[:, 1, :, :, :]),
                                      output_use[:, 2, :, :, :]).unsqueeze(dim=1)
                output_use = torch.cat((output_ET, output_TC, output_WT), dim=1)

                loss_ed = loss
                loss_cross = loss
                loss_C1 = loss
                loss_C2 = loss

            if model_index == 21:
                output_use = net(input)
                # 类内方差最小
                if epoch >= 0:
                    outputPre = torch.mul(output_use > thr, torch.ones_like(output_use)).to(torch.float32)
                    ET_t1ceV = torch.mul(outputPre[:, 0, :, :, :], input[:, 1, :, :, :])
                    ET_flV = torch.mul(outputPre[:, 0, :, :, :], input[:, 2, :, :, :])

                    NET_V = torch.mul(outputPre[:, 1, :, :, :] - outputPre[:, 0, :, :, :] > thr,
                                      torch.ones_like(outputPre[:, 0, :, :, :])).to(torch.float32)
                    NET_t1ceV = torch.mul(NET_V, input[:, 1, :, :, :])
                    NET_flV = torch.mul(NET_V, input[:, 2, :, :, :])

                    ED_V = torch.mul(outputPre[:, 2, :, :, :] - outputPre[:, 1, :, :, :] > thr,
                                     torch.ones_like(outputPre[:, 0, :, :, :])).to(torch.float32)
                    ED_t1ceV = torch.mul(ED_V, input[:, 1, :, :, :])
                    ED_flV = torch.mul(ED_V, input[:, 2, :, :, :])

                    ET_t1ceV = torch.clamp(ET_t1ceV, min=0, max=1)
                    ET_flV = torch.clamp(ET_flV, min=0, max=1)
                    NET_t1ceV = torch.clamp(NET_t1ceV, min=0, max=1)
                    NET_flV = torch.clamp(NET_flV, min=0, max=1)
                    ED_t1ceV = torch.clamp(ED_t1ceV, min=0, max=1)
                    ED_flV = torch.clamp(ED_flV, min=0, max=1)

                    # 2d_edge
                    # loss_cross = (criterionTV(ET_t1ceV, outputPre[:, 0, :, :, :]) + criterionTV(ET_flV,
                    #                                                                             outputPre[:, 0, :, :,
                    #                                                                             :]) + criterionTV(
                    #     NET_t1ceV, NET_V) + criterionTV(NET_flV, NET_V) + criterionTV(ED_t1ceV, ED_V) + criterionTV(
                    #     ED_flV, ED_V))
                    # 3d_edge
                    # loss_cross = (criterionTV(torch.unsqueeze(ET_t1ceV, dim=1),
                    #                           torch.unsqueeze(outputPre[:, 0, :, :, :], dim=1)) + criterionTV(
                    #     torch.unsqueeze(ET_flV, dim=1),
                    #     torch.unsqueeze(outputPre[:, 0, :, :,
                    #                     :], dim=1)) + criterionTV(
                    #     torch.unsqueeze(NET_t1ceV, dim=1), torch.unsqueeze(NET_V, dim=1)) + criterionTV(
                    #     torch.unsqueeze(NET_flV, dim=1), torch.unsqueeze(NET_V, dim=1)) + criterionTV(
                    #     torch.unsqueeze(ED_t1ceV, dim=1), torch.unsqueeze(ED_V, dim=1)) + criterionTV(
                    #     torch.unsqueeze(ED_flV, dim=1), torch.unsqueeze(ED_V, dim=1)))
                    # 2d
                    # loss_cross = (criterionTV(ET_t1ceV) + criterionTV(ET_flV) + criterionTV(
                    #     NET_t1ceV) + criterionTV(NET_flV) + criterionTV(ED_t1ceV) + criterionTV(
                    #     ED_flV))
                    # 3d
                    loss_cross = (criterionTV(torch.unsqueeze(ET_t1ceV, dim=1)) + criterionTV(
                        torch.unsqueeze(ET_flV, dim=1)) + criterionTV(
                        torch.unsqueeze(NET_t1ceV, dim=1)) + criterionTV(torch.unsqueeze(NET_flV, dim=1)) + criterionTV(
                        torch.unsqueeze(ED_t1ceV, dim=1)) + criterionTV(
                        torch.unsqueeze(ED_flV, dim=1)))
                    # loss_cross = (criterionTV(torch.unsqueeze(ET_t1ceV, dim=1)) / ET_t1ceV.sum(
                    #     dim=(0, 1, 2, 3)) + criterionTV(
                    #     torch.unsqueeze(ET_flV, dim=1)) / ET_flV.sum(
                    #     dim=(0, 1, 2, 3)) + criterionTV(
                    #     torch.unsqueeze(NET_t1ceV, dim=1)) / NET_t1ceV.sum(
                    #     dim=(0, 1, 2, 3)) + criterionTV(torch.unsqueeze(NET_flV, dim=1)) / NET_flV.sum(
                    #     dim=(0, 1, 2, 3)) + criterionTV(
                    #     torch.unsqueeze(ED_t1ceV, dim=1)) / ED_t1ceV.sum(
                    #     dim=(0, 1, 2, 3)) + criterionTV(
                    #     torch.unsqueeze(ED_flV, dim=1)) / ED_flV.sum(
                    #     dim=(0, 1, 2, 3)))
                else:
                    loss_cross = 0
                loss_ed = criterionED(output_use, label_new)
                loss = loss_ed + loss_cross
                loss_C1 = loss
                loss_C2 = loss

            if model_index == 22:
                output_use = net(input, pyG_B)
                loss = criterionED(output_use, label_new)
                loss_cross = loss
                loss_C1 = loss
                loss_C2 = loss
                loss_ed = loss

            if model_index == 23:
                output = net(input)
                output_use = output[2]
                if epoch >= 0:
                    outputPre = torch.mul(output_use > thr, torch.ones_like(output_use)).to(torch.float32)
                    ET_t1ceV = torch.mul(outputPre[:, 0, :, :, :], input[:, 1, :, :, :])
                    ET_flV = torch.mul(outputPre[:, 0, :, :, :], input[:, 2, :, :, :])

                    NET_V = torch.mul(outputPre[:, 1, :, :, :] - outputPre[:, 0, :, :, :] > thr,
                                      torch.ones_like(outputPre[:, 0, :, :, :])).to(torch.float32)
                    NET_t1ceV = torch.mul(NET_V, input[:, 1, :, :, :])
                    NET_flV = torch.mul(NET_V, input[:, 2, :, :, :])

                    ED_V = torch.mul(outputPre[:, 2, :, :, :] - outputPre[:, 1, :, :, :] > thr,
                                     torch.ones_like(outputPre[:, 0, :, :, :])).to(torch.float32)
                    ED_t1ceV = torch.mul(ED_V, input[:, 1, :, :, :])
                    ED_flV = torch.mul(ED_V, input[:, 2, :, :, :])

                    ET_t1ceV = torch.clamp(ET_t1ceV, min=0, max=1)
                    ET_flV = torch.clamp(ET_flV, min=0, max=1)
                    NET_t1ceV = torch.clamp(NET_t1ceV, min=0, max=1)
                    NET_flV = torch.clamp(NET_flV, min=0, max=1)
                    ED_t1ceV = torch.clamp(ED_t1ceV, min=0, max=1)
                    ED_flV = torch.clamp(ED_flV, min=0, max=1)

                    # edge
                    # loss_cross = (criterionTV(ET_t1ceV, outputPre[:, 0, :, :, :]) + criterionTV(ET_flV,
                    #                                                                             outputPre[:, 0, :, :,
                    #                                                                             :]) + criterionTV(
                    #     NET_t1ceV, NET_V) + criterionTV(NET_flV, NET_V) + criterionTV(ED_t1ceV, ED_V) + criterionTV(
                    #     ED_flV, ED_V))
                    # 2d
                    # loss_cross = (criterionTV(ET_t1ceV) + criterionTV(ET_flV) + criterionTV(
                    #     NET_t1ceV) + criterionTV(NET_flV) + criterionTV(ED_t1ceV) + criterionTV(
                    #     ED_flV))
                    # 3d
                    loss_cross = (criterionTV(torch.unsqueeze(ET_t1ceV, dim=1)) + criterionTV(
                        torch.unsqueeze(ET_flV, dim=1)) + criterionTV(
                        torch.unsqueeze(NET_t1ceV, dim=1)) + criterionTV(torch.unsqueeze(NET_flV, dim=1)) + criterionTV(
                        torch.unsqueeze(ED_t1ceV, dim=1)) + criterionTV(
                        torch.unsqueeze(ED_flV, dim=1)))
                else:
                    loss_cross = 0
                loss_ed = criterionED(output_use, label_new)
                # 对label进行下采样
                # label_ = F.interpolate(label_WT, scale_factor=(4, 32, 32),mode='nearest')
                label_ = F.max_pool3d(label_WT, kernel_size=(4, 16, 16), stride=(4, 16, 16))
                # writeNII.writeArrayToNii(torch.squeeze(label_[0,:,:,:]), savePath, 'label_.nii')
                # input_ = F.max_pool3d(input[:, 0, :, :, :], kernel_size=(4, 32, 32), stride=(4, 32, 32))
                # writeNII.writeArrayToNii(torch.squeeze(input_[0,:,:,:]), savePath, 'input_.nii')

                loss_C1 = criterionL1(output[1], label_)
                loss_C2 = loss_C1
                loss = loss_ed + loss_cross + loss_C1

            if model_index == 24:
                output = net(input)
                output_use = output[2]
                loss_ed = criterionED(output_use, label_new)
                label_ = F.max_pool3d(label_WT, kernel_size=(4, 16, 16), stride=(4, 16, 16))
                loss_C1 = criterionL1(output[1], label_)
                loss_cross = loss_C1
                loss_C2 = loss_C1
                loss = loss_ed + loss_C1

            if model_index == 25:
                Z = net(input)
                Z_1 = Z[:, 0, :, :, :].unsqueeze(dim=1)
                Z_2 = Z[:, 1, :, :, :].unsqueeze(dim=1)
                Z_3 = Z[:, 2, :, :, :].unsqueeze(dim=1)
                output_use = torch.cat((Z_1, torch.max(Z_1, Z_2), torch.max(torch.max(Z_1, Z_2), Z_3)), dim=1)

                # I_Z = torch.einsum('bkl,blk->bkk', output_use, output_use.permute(0, 2, 3, 4, 1))
                I_Z = torch.bmm(Z.view(Z.shape[0], Z.shape[1], -1),
                                Z.view(Z.shape[0], Z.shape[1], -1).permute(0, 2, 1)) / (24 * 256 * 256)

                label_NET = torch.mul(label == 1, torch.ones_like(label))
                label_ED = torch.mul(label == 2, torch.ones_like(label))
                label_ones = torch.cat((label_ET, label_NET, label_ED), dim=1)
                I_L = torch.bmm(label_ones.view(label_ones.shape[0], label_ones.shape[1], -1),
                                label_ones.view(label_ones.shape[0], label_ones.shape[1], -1).permute(
                                    0, 2, 1)) / (24 * 256 * 256)
                loss_C1 = criterionL1(I_Z, I_L)

                # 类内方差最小
                if epoch >= 0:
                    outputPre = torch.mul(output_use > thr, torch.ones_like(output_use)).to(torch.float32)
                    ET_t1ceV = torch.mul(outputPre[:, 0, :, :, :], input[:, 1, :, :, :])
                    ET_flV = torch.mul(outputPre[:, 0, :, :, :], input[:, 2, :, :, :])

                    NET_V = torch.mul(outputPre[:, 1, :, :, :] - outputPre[:, 0, :, :, :] > thr,
                                      torch.ones_like(outputPre[:, 0, :, :, :])).to(torch.float32)
                    NET_t1ceV = torch.mul(NET_V, input[:, 1, :, :, :])
                    NET_flV = torch.mul(NET_V, input[:, 2, :, :, :])

                    ED_V = torch.mul(outputPre[:, 2, :, :, :] - outputPre[:, 1, :, :, :] > thr,
                                     torch.ones_like(outputPre[:, 0, :, :, :])).to(torch.float32)
                    ED_t1ceV = torch.mul(ED_V, input[:, 1, :, :, :])
                    ED_flV = torch.mul(ED_V, input[:, 2, :, :, :])

                    ET_t1ceV = torch.clamp(ET_t1ceV, min=0, max=1)
                    ET_flV = torch.clamp(ET_flV, min=0, max=1)
                    NET_t1ceV = torch.clamp(NET_t1ceV, min=0, max=1)
                    NET_flV = torch.clamp(NET_flV, min=0, max=1)
                    ED_t1ceV = torch.clamp(ED_t1ceV, min=0, max=1)
                    ED_flV = torch.clamp(ED_flV, min=0, max=1)

                    # 2d_edge
                    # loss_cross = (criterionTV(ET_t1ceV, outputPre[:, 0, :, :, :]) + criterionTV(ET_flV,
                    #                                                                             outputPre[:, 0, :, :,
                    #                                                                             :]) + criterionTV(
                    #     NET_t1ceV, NET_V) + criterionTV(NET_flV, NET_V) + criterionTV(ED_t1ceV, ED_V) + criterionTV(
                    #     ED_flV, ED_V))
                    # 3d_edge
                    # loss_cross = (criterionTV(torch.unsqueeze(ET_t1ceV, dim=1),
                    #                           torch.unsqueeze(outputPre[:, 0, :, :, :], dim=1)) + criterionTV(
                    #     torch.unsqueeze(ET_flV, dim=1),
                    #     torch.unsqueeze(outputPre[:, 0, :, :,
                    #                     :], dim=1)) + criterionTV(
                    #     torch.unsqueeze(NET_t1ceV, dim=1), torch.unsqueeze(NET_V, dim=1)) + criterionTV(
                    #     torch.unsqueeze(NET_flV, dim=1), torch.unsqueeze(NET_V, dim=1)) + criterionTV(
                    #     torch.unsqueeze(ED_t1ceV, dim=1), torch.unsqueeze(ED_V, dim=1)) + criterionTV(
                    #     torch.unsqueeze(ED_flV, dim=1), torch.unsqueeze(ED_V, dim=1)))
                    # 2d
                    # loss_cross = (criterionTV(ET_t1ceV) + criterionTV(ET_flV) + criterionTV(
                    #     NET_t1ceV) + criterionTV(NET_flV) + criterionTV(ED_t1ceV) + criterionTV(
                    #     ED_flV))
                    # 3d
                    loss_cross = (criterionTV(torch.unsqueeze(ET_t1ceV, dim=1)) + criterionTV(
                        torch.unsqueeze(ET_flV, dim=1)) + criterionTV(
                        torch.unsqueeze(NET_t1ceV, dim=1)) + criterionTV(torch.unsqueeze(NET_flV, dim=1)) + criterionTV(
                        torch.unsqueeze(ED_t1ceV, dim=1)) + criterionTV(
                        torch.unsqueeze(ED_flV, dim=1)))
                else:
                    loss_cross = 0
                loss_ed = criterionED(output_use, label_new)
                loss = loss_ed + loss_cross + loss_C1
                # loss_C1 = loss
                loss_C2 = loss

            if model_index == 26:
                Z = net(input)
                Z_1 = Z[:, 0, :, :, :].unsqueeze(dim=1)
                Z_2 = Z[:, 1, :, :, :].unsqueeze(dim=1)
                Z_3 = Z[:, 2, :, :, :].unsqueeze(dim=1)
                output_use = torch.cat((Z_1, torch.max(Z_1, Z_2), torch.max(torch.max(Z_1, Z_2), Z_3)), dim=1)
                I_Z = torch.bmm(Z.view(Z.shape[0], Z.shape[1], -1),
                                Z.view(Z.shape[0], Z.shape[1], -1).permute(0, 2, 1)) / (24 * 256 * 256)

                label_NET = torch.mul(label == 1, torch.ones_like(label))
                label_ED = torch.mul(label == 2, torch.ones_like(label))
                label_ones = torch.cat((label_ET, label_NET, label_ED), dim=1)
                I_L = torch.bmm(label_ones.view(label_ones.shape[0], label_ones.shape[1], -1),
                                label_ones.view(label_ones.shape[0], label_ones.shape[1], -1).permute(
                                    0, 2, 1)) / (24 * 256 * 256)
                loss_C1 = criterionL1(I_Z, I_L)
                loss_ed = criterionED(output_use, label_new)
                loss = loss_ed + loss_C1
                loss_cross = loss
                loss_C2 = loss

            if model_index == 27:
                Z = net(input)
                Z_1 = Z[:, 0, :, :, :].unsqueeze(dim=1)
                Z_2 = Z[:, 1, :, :, :].unsqueeze(dim=1)
                Z_3 = Z[:, 2, :, :, :].unsqueeze(dim=1)
                output_use = torch.cat((Z_1, torch.max(Z_1, Z_2), torch.max(torch.max(Z_1, Z_2), Z_3)), dim=1)

                # 类内方差最小
                if epoch >= 0:
                    outputPre = torch.mul(output_use > thr, torch.ones_like(output_use)).to(torch.float32)
                    ET_t1ceV = torch.mul(outputPre[:, 0, :, :, :], input[:, 1, :, :, :])
                    ET_flV = torch.mul(outputPre[:, 0, :, :, :], input[:, 2, :, :, :])

                    NET_V = torch.mul(outputPre[:, 1, :, :, :] - outputPre[:, 0, :, :, :] > thr,
                                      torch.ones_like(outputPre[:, 0, :, :, :])).to(torch.float32)
                    NET_t1ceV = torch.mul(NET_V, input[:, 1, :, :, :])
                    NET_flV = torch.mul(NET_V, input[:, 2, :, :, :])

                    ED_V = torch.mul(outputPre[:, 2, :, :, :] - outputPre[:, 1, :, :, :] > thr,
                                     torch.ones_like(outputPre[:, 0, :, :, :])).to(torch.float32)
                    ED_t1ceV = torch.mul(ED_V, input[:, 1, :, :, :])
                    ED_flV = torch.mul(ED_V, input[:, 2, :, :, :])

                    ET_t1ceV = torch.clamp(ET_t1ceV, min=0, max=1)
                    ET_flV = torch.clamp(ET_flV, min=0, max=1)
                    NET_t1ceV = torch.clamp(NET_t1ceV, min=0, max=1)
                    NET_flV = torch.clamp(NET_flV, min=0, max=1)
                    ED_t1ceV = torch.clamp(ED_t1ceV, min=0, max=1)
                    ED_flV = torch.clamp(ED_flV, min=0, max=1)

                    # 2d_edge
                    # loss_cross = (criterionTV(ET_t1ceV, outputPre[:, 0, :, :, :]) + criterionTV(ET_flV,
                    #                                                                             outputPre[:, 0, :, :,
                    #                                                                             :]) + criterionTV(
                    #     NET_t1ceV, NET_V) + criterionTV(NET_flV, NET_V) + criterionTV(ED_t1ceV, ED_V) + criterionTV(
                    #     ED_flV, ED_V))
                    # 3d_edge
                    # loss_cross = (criterionTV(torch.unsqueeze(ET_t1ceV, dim=1),
                    #                           torch.unsqueeze(outputPre[:, 0, :, :, :], dim=1)) + criterionTV(
                    #     torch.unsqueeze(ET_flV, dim=1),
                    #     torch.unsqueeze(outputPre[:, 0, :, :,
                    #                     :], dim=1)) + criterionTV(
                    #     torch.unsqueeze(NET_t1ceV, dim=1), torch.unsqueeze(NET_V, dim=1)) + criterionTV(
                    #     torch.unsqueeze(NET_flV, dim=1), torch.unsqueeze(NET_V, dim=1)) + criterionTV(
                    #     torch.unsqueeze(ED_t1ceV, dim=1), torch.unsqueeze(ED_V, dim=1)) + criterionTV(
                    #     torch.unsqueeze(ED_flV, dim=1), torch.unsqueeze(ED_V, dim=1)))
                    # 2d
                    # loss_cross = (criterionTV(ET_t1ceV) + criterionTV(ET_flV) + criterionTV(
                    #     NET_t1ceV) + criterionTV(NET_flV) + criterionTV(ED_t1ceV) + criterionTV(
                    #     ED_flV))
                    # 3d
                    loss_cross = (criterionTV(torch.unsqueeze(ET_t1ceV, dim=1)) + criterionTV(
                        torch.unsqueeze(ET_flV, dim=1)) + criterionTV(
                        torch.unsqueeze(NET_t1ceV, dim=1)) + criterionTV(torch.unsqueeze(NET_flV, dim=1)) + criterionTV(
                        torch.unsqueeze(ED_t1ceV, dim=1)) + criterionTV(
                        torch.unsqueeze(ED_flV, dim=1)))

                    # loss_cross = (criterionTV(torch.unsqueeze(ET_t1ceV, dim=1)) + criterionTV(
                    #     torch.unsqueeze(NET_t1ceV, dim=1)) + criterionTV(torch.unsqueeze(ED_flV, dim=1)))
                else:
                    loss_cross = 0
                loss_ed = criterionED(output_use, label_new)
                loss = loss_ed + loss_cross
                loss_C1 = loss
                loss_C2 = loss

            if model_index == 28:
                Z = net(input)
                Z_1 = Z[:, 0, :, :, :].unsqueeze(dim=1)
                Z_2 = Z[:, 1, :, :, :].unsqueeze(dim=1)
                Z_3 = Z[:, 2, :, :, :].unsqueeze(dim=1)
                output_use = torch.cat((Z_1, torch.max(Z_1, Z_2), torch.max(torch.max(Z_1, Z_2), Z_3)), dim=1)

                loss_ed = criterionED(output_use, label_new)
                loss = loss_ed
                loss_cross = loss
                loss_C1 = loss
                loss_C2 = loss

            if model_index == 29:
                output_use = net(input)
                loss_ed = criterionED(output_use, label_new)
                loss = loss_ed
                loss_cross = loss
                loss_C1 = loss
                loss_C2 = loss

            if model_index == 290:
                output_use = net(input)

                # 类内方差最小
                if epoch >= 0:
                    outputPre = torch.mul(output_use > thr, torch.ones_like(output_use)).to(torch.float32)
                    ET_t1ceV = torch.mul(outputPre[:, 0, :, :, :], input[:, 1, :, :, :])
                    ET_flV = torch.mul(outputPre[:, 0, :, :, :], input[:, 2, :, :, :])

                    NET_V = torch.mul(outputPre[:, 1, :, :, :] - outputPre[:, 0, :, :, :] > thr,
                                      torch.ones_like(outputPre[:, 0, :, :, :])).to(torch.float32)
                    NET_t1ceV = torch.mul(NET_V, input[:, 1, :, :, :])
                    NET_flV = torch.mul(NET_V, input[:, 2, :, :, :])

                    ED_V = torch.mul(outputPre[:, 2, :, :, :] - outputPre[:, 1, :, :, :] > thr,
                                     torch.ones_like(outputPre[:, 0, :, :, :])).to(torch.float32)
                    ED_t1ceV = torch.mul(ED_V, input[:, 1, :, :, :])
                    ED_flV = torch.mul(ED_V, input[:, 2, :, :, :])

                    ET_t1ceV = torch.clamp(ET_t1ceV, min=0, max=1)
                    ET_flV = torch.clamp(ET_flV, min=0, max=1)
                    NET_t1ceV = torch.clamp(NET_t1ceV, min=0, max=1)
                    NET_flV = torch.clamp(NET_flV, min=0, max=1)
                    ED_t1ceV = torch.clamp(ED_t1ceV, min=0, max=1)
                    ED_flV = torch.clamp(ED_flV, min=0, max=1)

                    # 2d_edge
                    # loss_cross = (criterionTV(ET_t1ceV, outputPre[:, 0, :, :, :]) + criterionTV(ET_flV,
                    #                                                                             outputPre[:, 0, :, :,
                    #                                                                             :]) + criterionTV(
                    #     NET_t1ceV, NET_V) + criterionTV(NET_flV, NET_V) + criterionTV(ED_t1ceV, ED_V) + criterionTV(
                    #     ED_flV, ED_V))
                    # 3d_edge
                    # loss_cross = (criterionTV(torch.unsqueeze(ET_t1ceV, dim=1),
                    #                           torch.unsqueeze(outputPre[:, 0, :, :, :], dim=1)) + criterionTV(
                    #     torch.unsqueeze(ET_flV, dim=1),
                    #     torch.unsqueeze(outputPre[:, 0, :, :,
                    #                     :], dim=1)) + criterionTV(
                    #     torch.unsqueeze(NET_t1ceV, dim=1), torch.unsqueeze(NET_V, dim=1)) + criterionTV(
                    #     torch.unsqueeze(NET_flV, dim=1), torch.unsqueeze(NET_V, dim=1)) + criterionTV(
                    #     torch.unsqueeze(ED_t1ceV, dim=1), torch.unsqueeze(ED_V, dim=1)) + criterionTV(
                    #     torch.unsqueeze(ED_flV, dim=1), torch.unsqueeze(ED_V, dim=1)))
                    # 2d
                    # loss_cross = (criterionTV(ET_t1ceV) + criterionTV(ET_flV) + criterionTV(
                    #     NET_t1ceV) + criterionTV(NET_flV) + criterionTV(ED_t1ceV) + criterionTV(
                    #     ED_flV))
                    # 3d
                    loss_cross = (criterionTV(torch.unsqueeze(ET_t1ceV, dim=1)) + criterionTV(
                        torch.unsqueeze(ET_flV, dim=1)) + criterionTV(
                        torch.unsqueeze(NET_t1ceV, dim=1)) + criterionTV(torch.unsqueeze(NET_flV, dim=1)) + criterionTV(
                        torch.unsqueeze(ED_t1ceV, dim=1)) + criterionTV(
                        torch.unsqueeze(ED_flV, dim=1)))

                    # loss_cross = (criterionTV(torch.unsqueeze(ET_t1ceV, dim=1)) + criterionTV(
                    #     torch.unsqueeze(NET_t1ceV, dim=1)) + criterionTV(torch.unsqueeze(ED_flV, dim=1)))
                else:
                    loss_cross = 0
                loss_ed = criterionED(output_use, label_new)
                loss = loss_ed + loss_cross
                loss_C1 = loss
                loss_C2 = loss

            if model_index == 3 or model_index == 5:
                output = net(input)
                output_use = output[2]
                loss_ce = criterionC(output[0].squeeze(),
                                     torch.mul(label > 0, torch.ones_like(label)).squeeze()) + criterionC(
                    output[1].squeeze(),
                    torch.mul(label == 0, torch.ones_like(label)).squeeze())

                loss = loss_ce + criterionED(output[2], label_new)
                loss_cross = loss_ce
                loss_C1 = loss_ce
                loss_C2 = loss_ce
                loss_ed = loss - loss_C1

            if model_index == 4:
                pyG_B = create_graph_new(device, pyData, sizeG, sim_thr)  # 读取图数据转化为输入图谱
                output = net(input, pyG_B)
                output_use = output[4]
                crossM = torch.sum(torch.mul(output[0], output[1]), dim=(1, 2, 3, 4)).unsqueeze(dim=1)
                loss_cross = criterionL1(crossM / (sizeWHD[0] * sizeWHD[1] * sizeWHD[2]),
                                         torch.sum(label - label, dim=(1, 2, 3, 4)).unsqueeze(dim=1))
                loss_ce = criterionC(output[2].squeeze(),
                                     torch.mul(label > 0, torch.ones_like(label)).squeeze()) + criterionC(
                    output[3].squeeze(),
                    torch.mul(label == 0, torch.ones_like(label)).squeeze())
                loss_ed = criterionED(output[4], label_new)
                loss = 100 * loss_cross + 2 * loss_ce + loss_ed
                loss_C1 = loss_ce
                loss_C2 = loss_ce

            if model_index == 6:
                pyG_B = create_graph_new(device, pyData, sizeG, sim_thr)  # 读取图数据转化为输入图谱
                # ttt=pyData[0, 0, :, :]
                # xxx = torch.max(torch.max(ttt))
                output_use = net(input, pyG_B)
                loss = criterionED(output_use, label_new)
                loss_cross = loss
                loss_C1 = loss
                loss_C2 = loss
                loss_ed = loss

            if model_index == 7:
                output = net(input)
                output_use = output[4]
                crossM = torch.mean(torch.matmul(output[0], output[1]), dim=(1, 2, 3, 4))
                loss_cross = criterionL1(crossM, torch.zeros_like(crossM))
                # crossM = torch.sum(torch.mul(output[0], output[1]), dim=(1, 2, 3, 4)).unsqueeze(dim=1)
                # loss_cross = criterionC(crossM,
                #                          torch.sum(label - label, dim=(1, 2, 3, 4)).unsqueeze(dim=1))
                # 误识别部分加强学习
                O_1_FP = torch.logical_xor(output[2].squeeze() > thr,
                                           torch.mul(output[2].squeeze() > thr, label.squeeze() > 0))

                loss_C1 = criterionC(output[2].squeeze(),
                                     torch.mul(label > 0, torch.ones_like(label)).squeeze()) + criterionC(
                    torch.mul(output[2].squeeze(), O_1_FP), label.squeeze() - label.squeeze())

                O_2_FP = torch.logical_xor(output[3].squeeze() > thr,
                                           torch.mul(output[3].squeeze() > thr, label.squeeze() == 0))
                loss_C2 = criterionC(output[3].squeeze(),
                                     torch.mul(label == 0, torch.ones_like(label)).squeeze()) + criterionC(
                    torch.mul(output[3].squeeze(), O_2_FP), label.squeeze() - label.squeeze())

                loss_ed = criterionED(output[4], label_new)
                loss = loss_cross + (loss_C1 + loss_C2) + loss_ed

            if model_index == 8:
                output = net(input)
                output_use = output[4]
                crossM = torch.sum(torch.mul(output[0], output[1]), dim=(1, 2, 3, 4)).unsqueeze(dim=1)
                loss_cross = criterionL1(crossM / (sizeWHD[0] * sizeWHD[1] * sizeWHD[2]),
                                         torch.sum(label - label, dim=(1, 2, 3, 4)).unsqueeze(dim=1))
                loss_ce = criterionC(output[2].squeeze(),
                                     torch.mul(label > 0, torch.ones_like(label)).squeeze()) + criterionC(
                    output[3].squeeze(),
                    torch.mul(label == 0, torch.ones_like(label)).squeeze())
                loss_ed = criterionED(output[4], label_new)
                loss = 100 * loss_cross + 2 * loss_ce + loss_ed
                loss_C1 = loss_ce
                loss_C2 = loss_ce

            if model_index == 9 or model_index == 91:
                pyG_B = create_graph_new(device, pyData, sizeG, sim_thr)  # 读取图数据转化为输入图谱
                output = net(input, pyG_B)
                output_use = output[4]
                crossM = torch.mean(torch.matmul(output[0], output[1]), dim=(1, 2, 3, 4))
                loss_cross = criterionL1(crossM, torch.zeros_like(crossM))

                # crossM = torch.sum(torch.mul(output[0], output[1]), dim=(1, 2, 3, 4)).unsqueeze(dim=1)
                # loss_cross = criterionC(crossM / (sizeWHD[0] * sizeWHD[1] * sizeWHD[2]),
                #                         torch.sum(label - label, dim=(1, 2, 3, 4)).unsqueeze(dim=1))
                # 误识别部分加强学习
                O_1_FP = torch.logical_xor(output[2].squeeze() > thr,
                                           torch.mul(output[2].squeeze() > thr, label.squeeze() > 0))

                loss_C1 = criterionC(output[2].squeeze(),
                                     torch.mul(label > 0, torch.ones_like(label)).squeeze()) + criterionC(
                    torch.mul(output[2].squeeze(), O_1_FP), label.squeeze() - label.squeeze())

                O_2_FP = torch.logical_xor(output[3].squeeze() > thr,
                                           torch.mul(output[3].squeeze() > thr, label.squeeze() == 0))
                loss_C2 = criterionC(output[3].squeeze(),
                                     torch.mul(label == 0, torch.ones_like(label)).squeeze()) + criterionC(
                    torch.mul(output[3].squeeze(), O_2_FP), label.squeeze() - label.squeeze())

                loss_ed = criterionED(output[4], label_new)
                loss = loss_cross + (loss_C1 + loss_C2) + loss_ed

            print(
                "fold:%d  epoch:%d loss:%f loss_cross:%f loss_C1:%f loss_C2:%f loss_ed:%f" % (fold_num,
                                                                                              epoch, loss, loss_cross,
                                                                                              loss_C1, loss_C2,
                                                                                              loss_ed))
            log.write(
                "fold:%d  epoch:%d loss:%f  loss_cross:%f loss_C1:%f loss_C2:%f loss_ed:%f\n" % (fold_num,
                                                                                                 epoch, loss,
                                                                                                 loss_cross, loss_C1,
                                                                                                 loss_C2, loss_ed))

            loss.backward()
            optimizer.step()

            if (model_index == 5 or model_index == 91) and epoch == 49:
                for name, parameter in net.named_parameters():
                    if 'preModel' in name:
                        parameter.requires_grad = True

            #
            train_loss += loss.detach() / batchSize

            pre, rec, dsc, iou = ev_predict_new(label_new, output_use, log, thr, 0, 0)
            # pre, rec, dsc, iou = ev_predict(label, output_use)
            print("fold:%d  pre:%f, %f, %f, %f rec:%f, %f, %f, %f dsc:%f, %f, %f, %f iou:%f, %f, %f, %f" % (fold_num,
                                                                                                            pre[0],
                                                                                                            pre[1],
                                                                                                            pre[2],
                                                                                                            pre[3],
                                                                                                            rec[0],
                                                                                                            rec[1],
                                                                                                            rec[2],
                                                                                                            rec[3],
                                                                                                            dsc[0],
                                                                                                            dsc[1],
                                                                                                            dsc[2],
                                                                                                            dsc[3],
                                                                                                            iou[0],
                                                                                                            iou[1],
                                                                                                            iou[2],
                                                                                                            iou[3]))
            log.write(
                "fold:%d  pre:%f, %f, %f, %f rec:%f, %f, %f, %f dsc:%f, %f, %f, %f iou:%f, %f, %f, %f\n" % (fold_num,
                                                                                                            pre[0],
                                                                                                            pre[1],
                                                                                                            pre[2],
                                                                                                            pre[3],
                                                                                                            rec[0],
                                                                                                            rec[1],
                                                                                                            rec[2],
                                                                                                            rec[3],
                                                                                                            dsc[0],
                                                                                                            dsc[1],
                                                                                                            dsc[2],
                                                                                                            dsc[3],
                                                                                                            iou[0],
                                                                                                            iou[1],
                                                                                                            iou[2],
                                                                                                            iou[3]))
            pre_0 += pre[0] * batchSize
            pre_1 += pre[1] * batchSize
            pre_2 += pre[2] * batchSize
            pre_3 += pre[3] * batchSize
            rec_0 += rec[0] * batchSize
            rec_1 += rec[1] * batchSize
            rec_2 += rec[2] * batchSize
            rec_3 += rec[3] * batchSize
            dsc_0 += dsc[0] * batchSize
            dsc_1 += dsc[1] * batchSize
            dsc_2 += dsc[2] * batchSize
            dsc_3 += dsc[3] * batchSize
            iou_0 += iou[0] * batchSize
            iou_1 += iou[1] * batchSize
            iou_2 += iou[2] * batchSize
            iou_3 += iou[3] * batchSize

        temp1 = train_loss * batchSize / train_data.__len__()
        Loss_list.append(temp1.detach().cpu().numpy())
        acc_pos = dsc_1 + dsc_2 + dsc_3
        temp2 = 100 * acc_pos / train_data.__len__()
        Accuracy_list.append(temp2)

        scheduler.step(np.mean(Loss_list))
        maxIterNum = maxIterNum + 1

        print(
            "fold:%d  epoch:%d Allloss:%f Allpre:%f, %f, %f, %f Allrec:%f, %f, %f, %f Alldsc:%f, %f, %f, %f Alliou:%f, %f, %f, %f" % (
            fold_num,
            epoch, train_loss * batchSize / train_data.__len__(),
            pre_0 / train_data.__len__(), pre_1 / train_data.__len__(), pre_2 / train_data.__len__(),
            pre_3 / train_data.__len__(),
            rec_0 / train_data.__len__(), rec_1 / train_data.__len__(), rec_2 / train_data.__len__(),
            rec_3 / train_data.__len__(),
            dsc_0 / train_data.__len__(), dsc_1 / train_data.__len__(), dsc_2 / train_data.__len__(),
            dsc_3 / train_data.__len__(),
            iou_0 / train_data.__len__(), iou_1 / train_data.__len__(), iou_2 / train_data.__len__(),
            iou_3 / train_data.__len__()))
        log.write(
            "fold:%d  epoch:%d Allloss:%f Allpre:%f, %f, %f, %f Allrec:%f, %f, %f, %f Alldsc:%f, %f, %f, %f Alliou:%f, %f, %f, %f\n" % (
            fold_num,
            epoch, train_loss * batchSize / train_data.__len__(),
            pre_0 / train_data.__len__(), pre_1 / train_data.__len__(), pre_2 / train_data.__len__(),
            pre_3 / train_data.__len__(),
            rec_0 / train_data.__len__(), rec_1 / train_data.__len__(), rec_2 / train_data.__len__(),
            rec_3 / train_data.__len__(),
            dsc_0 / train_data.__len__(), dsc_1 / train_data.__len__(), dsc_2 / train_data.__len__(),
            dsc_3 / train_data.__len__(),
            iou_0 / train_data.__len__(), iou_1 / train_data.__len__(), iou_2 / train_data.__len__(),
            iou_3 / train_data.__len__()))

        torch.cuda.empty_cache()

        # 测试
        net.eval()
        if val == 1:
            val_pre_0 = 0.0
            val_pre_1 = 0.0
            val_pre_2 = 0.0
            val_pre_3 = 0.0
            val_rec_0 = 0.0
            val_rec_1 = 0.0
            val_rec_2 = 0.0
            val_rec_3 = 0.0
            val_dsc_0 = 0.0
            val_dsc_1 = 0.0
            val_dsc_2 = 0.0
            val_dsc_3 = 0.0
            val_iou_0 = 0.0
            val_iou_1 = 0.0
            val_iou_2 = 0.0
            val_iou_3 = 0.0

            for i, data in enumerate(val_loader):
                # 输入数据,在网络中instanceNorm归一化
                input = torch.unsqueeze(data[0].to(device, non_blocking=True), dim=-4).transpose(3, 4)
                input = torch.cat((input[:, :, 0:sizeWHD[0], :, :], input[:, :, sizeWHD[0]:2 * sizeWHD[0], :, :],
                                   input[:, :, 2 * sizeWHD[0]:3 * sizeWHD[0], :, :],
                                   input[:, :, 3 * sizeWHD[0]:4 * sizeWHD[0], :, :]), dim=1)

                # 标签为4的，重置为3
                label = torch.unsqueeze(data[2].to(device, non_blocking=True), dim=-4).transpose(3, 4)
                label[label == 4] = 3
                label = torch.mul(label <= 3, label)

                # ET:增强区，TC:坏死+增强区，WT:坏死+增强+水肿区

                # brats_t: ET=3,TC=3+2;WT=1+2+3
                # label_ET = torch.mul(label == 3, torch.ones_like(label))
                # label_TC = torch.mul(label > 1, torch.ones_like(label))
                # label_WT = torch.mul(label > 0, torch.ones_like(label))
                # label_new = torch.cat((label_ET, label_TC, label_WT), dim=1)

                # brats_2021&ydy: ET=3,TC=3+1;WT=1+2+3
                label_ET = torch.mul(label == 3, torch.ones_like(label))
                label_TC = torch.mul(label == 1, torch.ones_like(label)) + label_ET
                label_WT = torch.mul(label > 0, torch.ones_like(label))
                label_new = torch.cat((label_ET, label_TC, label_WT), dim=1)

                with torch.no_grad():
                    if model_index == 0:
                        output = net(input)
                        output_use = output[5]

                    if model_index == 1 or model_index == 7 or model_index == 8:
                        output = net(input)
                        output_use = output[4]

                    if model_index == 2 or model_index == 21 or model_index == 29 or model_index == 290:
                        output_use = net(input)

                    if model_index == 22:
                        output_use = net(input, pyG_B)

                    if model_index == 23 or model_index == 24:
                        output = net(input)
                        output_use = output[2]

                    if model_index == 25 or model_index == 26 or model_index == 27 or model_index == 28:
                        Z = net(input)
                        Z_1 = Z[:, 0, :, :, :].unsqueeze(dim=1)
                        Z_2 = Z[:, 1, :, :, :].unsqueeze(dim=1)
                        Z_3 = Z[:, 2, :, :, :].unsqueeze(dim=1)
                        output_use = torch.cat((Z_1, torch.max(Z_1, Z_2), torch.max(torch.max(Z_1, Z_2), Z_3)), dim=1)

                    if model_index == 20 or model_index == 200:
                        output_use = net(input)
                        output_ET = output_use[:, 0, :, :, :].unsqueeze(dim=1)
                        output_TC = torch.max(output_use[:, 0, :, :, :], output_use[:, 1, :, :, :]).unsqueeze(dim=1)
                        output_WT = torch.max(torch.max(output_use[:, 0, :, :, :], output_use[:, 1, :, :, :]),
                                              output_use[:, 2, :, :, :]).unsqueeze(dim=1)
                        output_use = torch.cat((output_ET, output_TC, output_WT), dim=1)

                    if model_index == 3 or model_index == 5:
                        output = net(input)
                        output_use = output[2]

                    if model_index == 4 or model_index == 9 or model_index == 91:
                        pyData = data[1][:, 0:4, :, :].to(device, non_blocking=True)
                        pyG_B = create_graph_new(device, pyData, sizeG, sim_thr)
                        output = net(input, pyG_B)
                        output_use = output[4]

                    if model_index == 6:
                        pyData = data[1][:, 0:4, :, :].to(device, non_blocking=True)
                        pyG_B = create_graph_new(device, pyData, sizeG, sim_thr)
                        output_use = net(input, pyG_B)

                # 评估网络预测结果
                val_pre, val_rec, val_dsc, val_iou = ev_predict_new(label_new, output_use, log, thr, 0, 0)
                # val_pre, val_rec, val_dsc, val_iou = ev_predict(label, output_use)
                print(
                    "fold:%d, pre:%f, %f, %f, %f rec:%f, %f, %f, %f dsc:%f, %f, %f, %f iou:%f, %f, %f, %f" % (fold_num,
                                                                                                              val_pre[
                                                                                                                  0],
                                                                                                              val_pre[
                                                                                                                  1],
                                                                                                              val_pre[
                                                                                                                  2],
                                                                                                              val_pre[
                                                                                                                  3],
                                                                                                              val_rec[
                                                                                                                  0],
                                                                                                              val_rec[
                                                                                                                  1],
                                                                                                              val_rec[
                                                                                                                  2],
                                                                                                              val_rec[
                                                                                                                  3],
                                                                                                              val_dsc[
                                                                                                                  0],
                                                                                                              val_dsc[
                                                                                                                  1],
                                                                                                              val_dsc[
                                                                                                                  2],
                                                                                                              val_dsc[
                                                                                                                  3],
                                                                                                              val_iou[
                                                                                                                  0],
                                                                                                              val_iou[
                                                                                                                  1],
                                                                                                              val_iou[
                                                                                                                  2],
                                                                                                              val_iou[
                                                                                                                  3]))
                log.write("fold:%d, pre:%f, %f, %f, %f rec:%f, %f, %f, %f dsc:%f, %f, %f, %f iou:%f, %f, %f, %f\n" % (
                fold_num,
                val_pre[0], val_pre[1], val_pre[2], val_pre[3], val_rec[0], val_rec[1], val_rec[2], val_rec[3],
                val_dsc[0],
                val_dsc[1], val_dsc[2], val_dsc[3], val_iou[0], val_iou[1], val_iou[2], val_iou[3]))
                val_pre_0 += val_pre[0] * batchSize_t
                val_pre_1 += val_pre[1] * batchSize_t
                val_pre_2 += val_pre[2] * batchSize_t
                val_pre_3 += val_pre[3] * batchSize_t
                val_rec_0 += val_rec[0] * batchSize_t
                val_rec_1 += val_rec[1] * batchSize_t
                val_rec_2 += val_rec[2] * batchSize_t
                val_rec_3 += val_rec[3] * batchSize_t
                val_dsc_0 += val_dsc[0] * batchSize_t
                val_dsc_1 += val_dsc[1] * batchSize_t
                val_dsc_2 += val_dsc[2] * batchSize_t
                val_dsc_3 += val_dsc[3] * batchSize_t
                val_iou_0 += val_iou[0] * batchSize_t
                val_iou_1 += val_iou[1] * batchSize_t
                val_iou_2 += val_iou[2] * batchSize_t
                val_iou_3 += val_iou[3] * batchSize_t

            print(
                "fold:%d epoch:%d All_val_pre:%f, %f, %f, %f All_val_rec:%f, %f, %f, %f All_val_dsc:%f, %f, %f, %f All_val_iou:%f, %f, %f, %f" % (
                fold_num,
                epoch,
                val_pre_0 / val_data.__len__(), val_pre_1 / val_data.__len__(), val_pre_2 / val_data.__len__(),
                val_pre_3 / val_data.__len__(),
                val_rec_0 / val_data.__len__(), val_rec_1 / val_data.__len__(), val_rec_2 / val_data.__len__(),
                val_rec_3 / val_data.__len__(),
                val_dsc_0 / val_data.__len__(), val_dsc_1 / val_data.__len__(), val_dsc_2 / val_data.__len__(),
                val_dsc_3 / val_data.__len__(),
                val_iou_0 / val_data.__len__(), val_iou_1 / val_data.__len__(), val_iou_2 / val_data.__len__(),
                val_iou_3 / val_data.__len__()))
            log.write(
                "fold:%d epoch:%d All_val_pre:%f, %f, %f, %f All_val_rec:%f, %f, %f, %f All_val_dsc:%f, %f, %f, %f All_val_iou:%f, %f, %f, %f\n" % (
                fold_num,
                epoch,
                val_pre_0 / val_data.__len__(), val_pre_1 / val_data.__len__(), val_pre_2 / val_data.__len__(),
                val_pre_3 / val_data.__len__(),
                val_rec_0 / val_data.__len__(), val_rec_1 / val_data.__len__(), val_rec_2 / val_data.__len__(),
                val_rec_3 / val_data.__len__(),
                val_dsc_0 / val_data.__len__(), val_dsc_1 / val_data.__len__(), val_dsc_2 / val_data.__len__(),
                val_dsc_3 / val_data.__len__(),
                val_iou_0 / val_data.__len__(), val_iou_1 / val_data.__len__(), val_iou_2 / val_data.__len__(),
                val_iou_3 / val_data.__len__()))

            if (val_dsc_0 + val_dsc_1 + val_dsc_2 + val_dsc_3) >= acc_val_dsc:
                torch.save(net, savePath + str(fold_num) + '_net_val.pkl')
                acc_train_pre = pre_0 + pre_1 + pre_2 + pre_3
                acc_train_rec = rec_0 + rec_1 + rec_2 + rec_3
                acc_train_dsc = dsc_0 + dsc_1 + dsc_2 + dsc_3
                acc_train_iou = iou_0 + iou_1 + iou_2 + iou_3
                acc_val_pre = val_pre_0 + val_pre_1 + val_pre_2 + val_pre_3
                acc_val_rec = val_rec_0 + val_rec_1 + val_rec_2 + val_rec_3
                acc_val_dsc = val_dsc_0 + val_dsc_1 + val_dsc_2 + val_dsc_3
                acc_val_iou = val_iou_0 + val_iou_1 + val_iou_2 + val_iou_3

        test_pre_0 = 0.0
        test_pre_1 = 0.0
        test_pre_2 = 0.0
        test_pre_3 = 0.0
        test_rec_0 = 0.0
        test_rec_1 = 0.0
        test_rec_2 = 0.0
        test_rec_3 = 0.0
        test_dsc_0 = 0.0
        test_dsc_1 = 0.0
        test_dsc_2 = 0.0
        test_dsc_3 = 0.0
        test_iou_0 = 0.0
        test_iou_1 = 0.0
        test_iou_2 = 0.0
        test_iou_3 = 0.0

        for i, data in enumerate(test_loader):
            # 输入数据,在网络中instanceNorm归一化
            input = torch.unsqueeze(data[0].to(device, non_blocking=True), dim=-4).transpose(3, 4)
            input = torch.cat((input[:, :, 0:sizeWHD[0], :, :], input[:, :, sizeWHD[0]:2 * sizeWHD[0], :, :],
                               input[:, :, 2 * sizeWHD[0]:3 * sizeWHD[0], :, :],
                               input[:, :, 3 * sizeWHD[0]:4 * sizeWHD[0], :, :]), dim=1)

            # 标签为4的，重置为3
            label = torch.unsqueeze(data[2].to(device, non_blocking=True), dim=-4).transpose(3, 4)
            label[label == 4] = 3
            label = torch.mul(label <= 3, label)

            # ET:增强区，TC:坏死+增强区，WT:坏死+增强+水肿区

            # brats_t: ET=3,TC=3+2;WT=1+2+3
            # label_ET = torch.mul(label == 3, torch.ones_like(label))
            # label_TC = torch.mul(label > 1, torch.ones_like(label))
            # label_WT = torch.mul(label > 0, torch.ones_like(label))
            # label_new = torch.cat((label_ET, label_TC, label_WT), dim=1)

            # brats_2021&ydy: ET=3,TC=3+1;WT=1+2+3
            label_ET = torch.mul(label == 3, torch.ones_like(label))
            label_TC = torch.mul(label == 1, torch.ones_like(label)) + label_ET
            label_WT = torch.mul(label > 0, torch.ones_like(label))
            label_new = torch.cat((label_ET, label_TC, label_WT), dim=1)

            with torch.no_grad():
                if model_index == 0:
                    output = net(input)
                    output_use = output[5]

                if model_index == 1 or model_index == 7 or model_index == 8:
                    output = net(input)
                    output_use = output[4]

                if model_index == 2 or model_index == 21 or model_index == 29 or model_index == 290:
                    output_use = net(input)

                if model_index == 22:
                    output_use = net(input, pyG_B)

                if model_index == 23 or model_index == 24:
                    output = net(input)
                    output_use = output[2]

                if model_index == 25 or model_index == 26 or model_index == 27 or model_index == 28:
                    Z = net(input)
                    Z_1 = Z[:, 0, :, :, :].unsqueeze(dim=1)
                    Z_2 = Z[:, 1, :, :, :].unsqueeze(dim=1)
                    Z_3 = Z[:, 2, :, :, :].unsqueeze(dim=1)
                    output_use = torch.cat((Z_1, torch.max(Z_1, Z_2), torch.max(torch.max(Z_1, Z_2), Z_3)), dim=1)

                if model_index == 20 or model_index == 200:
                    output_use = net(input)
                    output_ET = output_use[:, 0, :, :, :].unsqueeze(dim=1)
                    output_TC = torch.max(output_use[:, 0, :, :, :], output_use[:, 1, :, :, :]).unsqueeze(dim=1)
                    output_WT = torch.max(torch.max(output_use[:, 0, :, :, :], output_use[:, 1, :, :, :]),
                                          output_use[:, 2, :, :, :]).unsqueeze(dim=1)
                    output_use = torch.cat((output_ET, output_TC, output_WT), dim=1)

                if model_index == 3 or model_index == 5:
                    output = net(input)
                    output_use = output[2]

                if model_index == 4 or model_index == 9 or model_index == 91:
                    pyData = data[1][:, 0:4, :, :].to(device, non_blocking=True)
                    pyG_B = create_graph_new(device, pyData, sizeG, sim_thr)
                    output = net(input, pyG_B)
                    output_use = output[4]

                if model_index == 6:
                    pyData = data[1][:, 0:4, :, :].to(device, non_blocking=True)
                    pyG_B = create_graph_new(device, pyData, sizeG, sim_thr)
                    output_use = net(input, pyG_B)

            # 评估网络预测结果
            test_pre, test_rec, test_dsc, test_iou = ev_predict_new(label_new, output_use, log, thr, 0, 0)
            # test_pre, test_rec, test_dsc, test_iou = ev_predict(label, output_use)
            print("fold:%d, pre:%f, %f, %f, %f rec:%f, %f, %f, %f dsc:%f, %f, %f, %f iou:%f, %f, %f, %f" % (fold_num,
                                                                                                            test_pre[0],
                                                                                                            test_pre[1],
                                                                                                            test_pre[2],
                                                                                                            test_pre[3],
                                                                                                            test_rec[0],
                                                                                                            test_rec[1],
                                                                                                            test_rec[2],
                                                                                                            test_rec[3],
                                                                                                            test_dsc[0],
                                                                                                            test_dsc[1],
                                                                                                            test_dsc[2],
                                                                                                            test_dsc[3],
                                                                                                            test_iou[0],
                                                                                                            test_iou[1],
                                                                                                            test_iou[2],
                                                                                                            test_iou[
                                                                                                                3]))
            log.write(
                "fold:%d, pre:%f, %f, %f, %f rec:%f, %f, %f, %f dsc:%f, %f, %f, %f iou:%f, %f, %f, %f\n" % (fold_num,
                                                                                                            test_pre[0],
                                                                                                            test_pre[1],
                                                                                                            test_pre[2],
                                                                                                            test_pre[3],
                                                                                                            test_rec[0],
                                                                                                            test_rec[1],
                                                                                                            test_rec[2],
                                                                                                            test_rec[3],
                                                                                                            test_dsc[0],
                                                                                                            test_dsc[1],
                                                                                                            test_dsc[2],
                                                                                                            test_dsc[3],
                                                                                                            test_iou[0],
                                                                                                            test_iou[1],
                                                                                                            test_iou[2],
                                                                                                            test_iou[
                                                                                                                3]))
            test_pre_0 += test_pre[0] * batchSize_t
            test_pre_1 += test_pre[1] * batchSize_t
            test_pre_2 += test_pre[2] * batchSize_t
            test_pre_3 += test_pre[3] * batchSize_t
            test_rec_0 += test_rec[0] * batchSize_t
            test_rec_1 += test_rec[1] * batchSize_t
            test_rec_2 += test_rec[2] * batchSize_t
            test_rec_3 += test_rec[3] * batchSize_t
            test_dsc_0 += test_dsc[0] * batchSize_t
            test_dsc_1 += test_dsc[1] * batchSize_t
            test_dsc_2 += test_dsc[2] * batchSize_t
            test_dsc_3 += test_dsc[3] * batchSize_t
            test_iou_0 += test_iou[0] * batchSize_t
            test_iou_1 += test_iou[1] * batchSize_t
            test_iou_2 += test_iou[2] * batchSize_t
            test_iou_3 += test_iou[3] * batchSize_t

        print(
            "fold:%d epoch:%d All_test_pre:%f, %f, %f, %f All_test_rec:%f, %f, %f, %f All_test_dsc:%f, %f, %f, %f All_test_iou:%f, %f, %f, %f" % (
            fold_num,
            epoch,
            test_pre_0 / test_data.__len__(), test_pre_1 / test_data.__len__(), test_pre_2 / test_data.__len__(),
            test_pre_3 / test_data.__len__(),
            test_rec_0 / test_data.__len__(), test_rec_1 / test_data.__len__(), test_rec_2 / test_data.__len__(),
            test_rec_3 / test_data.__len__(),
            test_dsc_0 / test_data.__len__(), test_dsc_1 / test_data.__len__(), test_dsc_2 / test_data.__len__(),
            test_dsc_3 / test_data.__len__(),
            test_iou_0 / test_data.__len__(), test_iou_1 / test_data.__len__(), test_iou_2 / test_data.__len__(),
            test_iou_3 / test_data.__len__()))
        log.write(
            "fold:%d epoch:%d All_test_pre:%f, %f, %f, %f All_test_rec:%f, %f, %f, %f All_test_dsc:%f, %f, %f, %f All_test_iou:%f, %f, %f, %f\n" % (
            fold_num,
            epoch,
            test_pre_0 / test_data.__len__(), test_pre_1 / test_data.__len__(), test_pre_2 / test_data.__len__(),
            test_pre_3 / test_data.__len__(),
            test_rec_0 / test_data.__len__(), test_rec_1 / test_data.__len__(), test_rec_2 / test_data.__len__(),
            test_rec_3 / test_data.__len__(),
            test_dsc_0 / test_data.__len__(), test_dsc_1 / test_data.__len__(), test_dsc_2 / test_data.__len__(),
            test_dsc_3 / test_data.__len__(),
            test_iou_0 / test_data.__len__(), test_iou_1 / test_data.__len__(), test_iou_2 / test_data.__len__(),
            test_iou_3 / test_data.__len__()))

        if (test_dsc_0 + test_dsc_1 + test_dsc_2 + test_dsc_3) >= acc_test_dsc:
            torch.save(net, savePath + str(fold_num) + '_net_test.pkl')
            acc_train_pre = pre_0 + pre_1 + pre_2 + pre_3
            acc_train_rec = rec_0 + rec_1 + rec_2 + rec_3
            acc_train_dsc = dsc_0 + dsc_1 + dsc_2 + dsc_3
            acc_train_iou = iou_0 + iou_1 + iou_2 + iou_3
            acc_test_pre = test_pre_0 + test_pre_1 + test_pre_2 + test_pre_3
            acc_test_rec = test_rec_0 + test_rec_1 + test_rec_2 + test_rec_3
            acc_test_dsc = test_dsc_0 + test_dsc_1 + test_dsc_2 + test_dsc_3
            acc_test_iou = test_iou_0 + test_iou_1 + test_iou_2 + test_iou_3

        torch.save(net, savePath + str(fold_num) + '_net_end.pkl')

        # 调整学习率
        # if (epoch + 1) % int(max_epoch / lrStep) == 0:
        #     lr *= 0.1
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr

        torch.cuda.empty_cache()

    print(
        "fold:%d train_pre:%f train_rec:%f train_dsc:%f train_iou:%f test_pre:%f test_rec:%f test_dsc:%f test_iou:%f" % (
        fold_num,
        acc_train_pre / 4, acc_train_rec / 4, acc_train_dsc / 4, acc_train_iou / 4, acc_test_pre / 4, acc_test_rec / 4,
        acc_test_dsc / 4, acc_test_iou / 4))
    log.write(
        "fold:%d train_pre:%f train_rec:%f train_dsc:%f train_iou:%f test_pre:%f test_rec:%f test_dsc:%f test_iou:%f\n" % (
        fold_num,
        acc_train_pre / 4, acc_train_rec / 4, acc_train_dsc / 4, acc_train_iou / 4, acc_test_pre / 4,
        acc_test_rec / 4,
        acc_test_dsc / 4, acc_test_iou / 4))
    if val == 1:
        print(
            "fold:%d train_pre:%f train_rec:%f train_dsc:%f train_iou:%f val_pre:%f val_rec:%f val_dsc:%f val_iou:%f" % (
            fold_num,
            acc_train_pre / 4, acc_train_rec / 4, acc_train_dsc / 4, acc_train_iou / 4, acc_val_pre / 4,
            acc_val_rec / 4,
            acc_val_dsc / 4, acc_val_iou / 4))
        log.write(
            "fold:%d train_pre:%f train_rec:%f train_dsc:%f train_iou:%f val_pre:%f val_rec:%f val_dsc:%f val_iou:%f\n" % (
            fold_num,
            acc_train_pre / 4, acc_train_rec / 4, acc_train_dsc / 4, acc_train_iou / 4, acc_val_pre / 4,
            acc_val_rec / 4,
            acc_val_dsc / 4, acc_val_iou / 4))

    # matplotlib.use('Qt5Agg')
    # x1 = range(0, maxIterNum)
    # x2 = range(0, maxIterNum)
    # y1 = Accuracy_list
    # y2 = Loss_list
    # plt.subplot(2, 1, 1)
    # plt.plot(x1, y1, 'o-')
    # plt.title('Test accuracy vs.epoches')
    # plt.ylabel('Test accuracy')
    # plt.subplot(2, 1, 2)
    # plt.plot(x2, y2, '.-')
    # plt.xlabel('Test loss vs.epoches')
    # plt.ylabel('Test loss')
    # plt.show()
    # plt.savefig(savePath + 'accuracy_loss.jpg')

    return
