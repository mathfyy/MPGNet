import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch_geometric.data import Batch as pygBatch
from torch_geometric.data import Data as pygData

# import surface_distance as surfdist
from AlgorithmPool.FunctionalMethodPool.DataIO import writeNII
from AlgorithmPool.NetPool.EvalFunction import Dice
from SpecificApplication.brainLesion import loadBrainData

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def find_all(sub, s):
    index_list = []
    index = s.find(sub)
    while index != -1:
        index_list.append(index)
        index = s.find(sub, index + 1)

    if len(index_list) > 0:
        return index_list
    else:
        return -1


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


def create_graph_new(pointDist, sizeG, sim_thr):
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
                    edge_index.append(torch.Tensor([m, n]).long().to(device))
                    edge_attr.append(pointDist[i, 0, m, n])
            else:
                edge_index.append(torch.Tensor([m, m]).long().to(device))
                edge_attr.append(pointDist[i, 0, m, m])

        pyG_list.append(pygData(x=node_attr.to(device), edge_index=torch.stack(edge_index).permute(1, 0).to(device),
                                edge_attr=torch.stack(edge_attr).to(device)))
    pyG_B = pygBatch.from_data_list(pyG_list)
    return pyG_B


# def ev_predict_new(label, output_use, log, thr=0.5):
#     pre_all = torch.zeros([output_use.shape[0], 4])
#     rec_all = torch.zeros([output_use.shape[0], 4])
#     dsc_all = torch.zeros([output_use.shape[0], 4])
#     iou_all = torch.zeros([output_use.shape[0], 4])
#     hd_dist_95_all = torch.zeros([output_use.shape[0], 4])
#     for b in range(output_use.shape[0]):
#         for i in range(output_use.shape[1]):
#             cur_outputL = torch.unsqueeze(output_use[b, i, :, :, :], dim=0)
#             cur_outputL = torch.unsqueeze(cur_outputL, dim=0)
#             cur_label = torch.unsqueeze(label[b, i, :, :, :], dim=0)
#             cur_label = torch.unsqueeze(cur_label, dim=0)
#
#             # 当预测结果体积小于阈值时，置为0
#             volume_o = torch.sum(torch.mul(cur_outputL > thr, torch.ones_like(cur_outputL)), dim=(0, 1, 2, 3, 4))
#             if volume_o <= 8 * 8 * 6:
#                 # if volume_o <= 3:
#                 cur_outputL = torch.zeros_like(cur_outputL)
#
#             # 标签异常处理:当标签体积小于阈值时，置为0
#             volume_l = torch.sum(torch.mul(cur_label > thr, torch.ones_like(cur_label)), dim=(0, 1, 2, 3, 4))
#             if volume_l <= 6 * 6 * 4:
#                 cur_label = torch.zeros_like(cur_label)
#
#             precision, accuracy, recall, FPR_x, fpRate, dsc, iou, f1, mcc, preV, labelV, diffV = Dice.ev(cur_outputL,
#                                                                                                          cur_label,
#                                                                                                          compare_rule='>',
#                                                                                                          thr=thr)
#             # if torch.sum(cur_outputL, dim=(0, 1, 2, 3, 4)) > 0 or torch.sum(cur_label, dim=(0, 1, 2, 3, 4)) > 0:
#             #     if torch.is_tensor(cur_outputL):
#             #         cur_outputL = torch.squeeze(cur_outputL.cpu().detach()).numpy().astype(bool)
#             #     if torch.is_tensor(cur_label):
#             #         cur_label = torch.squeeze(cur_label.cpu().detach()).numpy().astype(bool)
#             #     surface_distances = surfdist.compute_surface_distances(cur_label, cur_outputL,
#             #                                                            spacing_mm=(1.0, 1.0, 1.0))
#             #     hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 100-95)
#             # else:
#             #     hd_dist_95 = 0
#
#             # if dsc < 0.1:
#             #     print("volume_o:%f volume_l:%f" % (volume_o, volume_l))
#             #     log.write("volume_o:%f volume_l:%f\n" % (volume_o, volume_l))
#
#             pre_all[b, i] = precision
#             rec_all[b, i] = recall
#             dsc_all[b, i] = dsc
#             iou_all[b, i] = iou
#             # hd_dist_95_all[b, i] = hd_dist_95
#
#     return torch.mean(pre_all, dim=0), torch.mean(rec_all, dim=0), torch.mean(dsc_all, dim=0), torch.mean(
#         iou_all, dim=0)
def ev_predict_new(label, output_use, log, thr=0.5):
    dsc_all = torch.zeros([output_use.shape[0], 4])
    iou_all = torch.zeros([output_use.shape[0], 4])
    spe_all = torch.zeros([output_use.shape[0], 4])
    rec_all = torch.zeros([output_use.shape[0], 4])
    S_meature_all = torch.zeros([output_use.shape[0], 4])
    F_meature_all = torch.zeros([output_use.shape[0], 4])
    E_meature_all = torch.zeros([output_use.shape[0], 4])
    M_D_all = torch.zeros([output_use.shape[0], 4])
    # hd_dist_95_all = torch.zeros([output_use.shape[0], 4])
    for b in range(output_use.shape[0]):
        for i in range(output_use.shape[1]):
            cur_outputL = torch.unsqueeze(output_use[b, i, :, :, :], dim=0)
            cur_outputL = torch.unsqueeze(cur_outputL, dim=0)
            cur_label = torch.unsqueeze(label[b, i, :, :, :], dim=0)
            cur_label = torch.unsqueeze(cur_label, dim=0)

            # 当预测结果体积小于阈值时，置为0
            volume_o = torch.sum(torch.mul(cur_outputL > thr, torch.ones_like(cur_outputL)), dim=(0, 1, 2, 3, 4))
            if volume_o <= 8 * 8 * 6:
                # if volume_o <= 3:
                cur_outputL = torch.zeros_like(cur_outputL)

            # 标签异常处理:当标签体积小于阈值时，置为0
            volume_l = torch.sum(torch.mul(cur_label > thr, torch.ones_like(cur_label)), dim=(0, 1, 2, 3, 4))
            if volume_l <= 6 * 6 * 4:
                cur_label = torch.zeros_like(cur_label)

            dsc, iou, specificity, recall, S_meature, F_meature, E_meature, M_D = Dice.ev(cur_outputL, cur_label,
                                                                                          compare_rule='>',
                                                                                          thr=thr)
            # if torch.sum(cur_outputL, dim=(0, 1, 2, 3, 4)) > 0 or torch.sum(cur_label, dim=(0, 1, 2, 3, 4)) > 0:
            #     if torch.is_tensor(cur_outputL):
            #         cur_outputL = torch.squeeze(cur_outputL.cpu().detach()).numpy().astype(bool)
            #     if torch.is_tensor(cur_label):
            #         cur_label = torch.squeeze(cur_label.cpu().detach()).numpy().astype(bool)
            #     surface_distances = surfdist.compute_surface_distances(cur_label, cur_outputL,
            #                                                            spacing_mm=(1.0, 1.0, 1.0))
            #     hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 100-95)
            # else:
            #     hd_dist_95 = 0

            # if dsc < 0.1:
            #     print("volume_o:%f volume_l:%f" % (volume_o, volume_l))
            #     log.write("volume_o:%f volume_l:%f\n" % (volume_o, volume_l))

            dsc_all[b, i] = dsc
            iou_all[b, i] = iou
            spe_all[b, i] = specificity
            rec_all[b, i] = recall
            S_meature_all[b, i] = S_meature
            F_meature_all[b, i] = F_meature
            E_meature_all[b, i] = E_meature
            M_D_all[b, i] = M_D
            # hd_dist_95_all[b, i] = hd_dist_95
    return torch.mean(dsc_all, dim=0), torch.mean(iou_all, dim=0), torch.mean(spe_all, dim=0), torch.mean(rec_all,
                                                                                                          dim=0), torch.mean(
        S_meature_all, dim=0), torch.mean(F_meature_all, dim=0), torch.mean(E_meature_all, dim=0), torch.mean(M_D_all,
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


def testBrain(samplePath, savePath, sizeWHD, model_index, sizeG, sim_thr, fold_num):
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1)

    thr = 0.5

    batchSize = 4
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    net = torch.load(savePath + str(fold_num) + '_net_val.pkl')
    # net = torch.load(savePath + str(fold_num) + '_net_end.pkl')
    # net = torch.load(savePath + 'net_test.pkl')

    # use_skull
    # net = torch.load(savePath + 'net_test.pkl')

    # if torch.cuda.device_count() > 1:
    #     net = torch.nn.DataParallel(net)
    net = net.to(device)

    net.eval()

    # from thop import profile
    # data=torch.randn([4, 4, 24, 256, 256])
    # flops, params = profile(net, inputs=(data,))
    # print(flops / 1e9, params / 1e6)

    # dataset = 'train'
    # dataset = 'test'
    # dataset = 'test_brats_skull_108'
    # dataset = 'test_30'
    dataset = 'test_sj'
    savePath = savePath + '/' + dataset

    # 是否测试头骨的
    # savePath = savePath + '_skull'

    if os.path.exists(savePath) is False:
        os.makedirs(savePath)

    # test_data = loadBrainData.MyDatasetFYY_TCGA(path=samplePath + dataset, transform=transforms.ToTensor(),
    #                                             target_transform=transforms.ToTensor())

    # test_data = loadBrainData.MyDatasetFYY_TCGA_py(path=samplePath + dataset, transform=transforms.ToTensor(),
    #                                                target_transform=transforms.ToTensor())

    test_data = loadBrainData.MyDatasetFYY_TCGA_skull_t(path=samplePath + dataset, transform=transforms.ToTensor(),
                                                   target_transform=transforms.ToTensor())

    test_loader = DataLoader(dataset=test_data, batch_size=batchSize, num_workers=1, pin_memory=True, drop_last=True)

    # 测试
    log = open(savePath + str(fold_num) + "_log_add.txt", "w")
    test_dsc_0 = 0.0
    test_dsc_1 = 0.0
    test_dsc_2 = 0.0
    test_dsc_3 = 0.0
    test_iou_0 = 0.0
    test_iou_1 = 0.0
    test_iou_2 = 0.0
    test_iou_3 = 0.0
    test_spe_0 = 0.0
    test_spe_1 = 0.0
    test_spe_2 = 0.0
    test_spe_3 = 0.0
    test_rec_0 = 0.0
    test_rec_1 = 0.0
    test_rec_2 = 0.0
    test_rec_3 = 0.0
    test_S_0 = 0.0
    test_S_1 = 0.0
    test_S_2 = 0.0
    test_S_3 = 0.0
    test_F_0 = 0.0
    test_F_1 = 0.0
    test_F_2 = 0.0
    test_F_3 = 0.0
    test_E_0 = 0.0
    test_E_1 = 0.0
    test_E_2 = 0.0
    test_E_3 = 0.0
    test_M_0 = 0.0
    test_M_1 = 0.0
    test_M_2 = 0.0
    test_M_3 = 0.0
    for i, data in enumerate(test_loader):
        input = torch.unsqueeze(data[0].to(device, non_blocking=True), dim=-4).transpose(3, 4)
        input = torch.cat((input[:, :, 0:sizeWHD[0], :, :], input[:, :, sizeWHD[0]:2 * sizeWHD[0], :, :],
                           input[:, :, 2 * sizeWHD[0]:3 * sizeWHD[0], :, :],
                           input[:, :, 3 * sizeWHD[0]:4 * sizeWHD[0], :, :]), dim=1)

        label = torch.unsqueeze(data[2].to(device, non_blocking=True), dim=-4).transpose(3, 4)
        label[label == 4] = 3
        label = torch.mul(label <= 3, label)

        # skull mask
        # skullMask = torch.unsqueeze(data[1].to(device, non_blocking=True), dim=-4).transpose(3, 4)

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
                pyG_B = []
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

                # output_ET = torch.mul(output_ET, data[1].to(device, non_blocking=True).unsqueeze(dim=1))
                # output_TC = torch.mul(output_TC, data[1].to(device, non_blocking=True).unsqueeze(dim=1))
                # output_WT = torch.mul(output_WT, data[1].to(device, non_blocking=True).unsqueeze(dim=1))

                output_use = torch.cat((output_ET, output_TC, output_WT), dim=1)

            if model_index == 3 or model_index == 5:
                output = net(input)
                output_use = output[2]

            if model_index == 4 or model_index == 9 or model_index == 91:
                pyData = data[1][:, 0:4, :, :].to(device, non_blocking=True)
                pyG_B = create_graph_new(pyData, sizeG, sim_thr)
                output = net(input, pyG_B)
                output_use = output[4]

            if model_index == 6:
                pyData = data[1][:, 0:4, :, :].to(device, non_blocking=True)
                pyG_B = create_graph_new(pyData, sizeG, sim_thr)
                output_use = net(input, pyG_B)

        # skull mask
        # for c in range(label_new.shape[1]):
        #     output_use[:, c, :, :, :] = torch.mul(output_use[:, c, :, :, :], skullMask.squeeze(dim=1))
        #     label_new[:, c, :, :, :] = torch.mul(label_new[:, c, :, :, :], 1 - skullMask.squeeze(dim=1))

        # all
        outputPre = torch.mul(output_use > thr, torch.ones_like(output_use))

        for b in range(batchSize):
            test_dsc, test_iou, test_spe, test_rec, test_S, test_F, test_E, test_M = ev_predict_new(
                torch.unsqueeze(label_new[b, :, :, :, :], dim=0),
                torch.unsqueeze(output_use[b, :, :, :, :], dim=0),
                log, thr)
            # print("pre:%f, %f, %f, %f rec:%f, %f, %f, %f dsc:%f, %f, %f, %f iou:%f, %f, %f, %f" % (
            #     test_pre[0], test_pre[1], test_pre[2], test_pre[3], test_rec[0], test_rec[1], test_rec[2], test_rec[3],
            #     test_dsc[0],
            #     test_dsc[1], test_dsc[2], test_dsc[3], test_iou[0], test_iou[1], test_iou[2], test_iou[3]))
            # log.write("pre:%f, %f, %f, %f rec:%f, %f, %f, %f dsc:%f, %f, %f, %f iou:%f, %f, %f, %f\n" % (
            #     test_pre[0], test_pre[1], test_pre[2], test_pre[3], test_rec[0], test_rec[1], test_rec[2], test_rec[3],
            #     test_dsc[0],
            #     test_dsc[1], test_dsc[2], test_dsc[3], test_iou[0], test_iou[1], test_iou[2], test_iou[3]))
            test_dsc_0 += test_dsc[0]*100
            test_dsc_1 += test_dsc[1]*100
            test_dsc_2 += test_dsc[2]*100
            test_dsc_3 += (test_dsc[0]*100+test_dsc[1]*100+test_dsc[2]*100)/3
            test_iou_0 += test_iou[0]*100
            test_iou_1 += test_iou[1]*100
            test_iou_2 += test_iou[2]*100
            test_iou_3 += (test_iou[0]*100+test_iou[1]*100+test_iou[2]*100)/3
            test_spe_0 += test_spe[0]*100
            test_spe_1 += test_spe[1]*100
            test_spe_2 += test_spe[2]*100
            test_spe_3 += (test_spe[0]*100+test_spe[1]*100+test_spe[2]*100)/3
            test_rec_0 += test_rec[0]*100
            test_rec_1 += test_rec[1]*100
            test_rec_2 += test_rec[2]*100
            test_rec_3 += (test_rec[0]*100+test_rec[1]*100+test_rec[2]*100)/3
            test_S_0 += test_S[0]*100
            test_S_1 += test_S[1]*100
            test_S_2 += test_S[2]*100
            test_S_3 += (test_S[0]*100+test_S[1]*100+test_S[2]*100)/3
            test_F_0 += test_F[0]*100
            test_F_1 += test_F[1]*100
            test_F_2 += test_F[2]*100
            test_F_3 += (test_F[0]*100+test_F[1]*100+test_F[2]*100)/3
            test_E_0 += test_E[0]*100
            test_E_1 += test_E[1]*100
            test_E_2 += test_E[2]*100
            test_E_3 += (test_E[0]*100+test_E[1]*100+test_E[2]*100)/3
            test_M_0 += test_M[0]*100
            test_M_1 += test_M[1]*100
            test_M_2 += test_M[2]*100
            test_M_3 += (test_M[0]*100+test_M[1]*100+test_M[2]*100)/3

            # tt=torch.unsqueeze(outputPre[b, :, :, :, :], dim=0)
            outputPreL = torch.sum(torch.unsqueeze(outputPre[b, :, :, :, :], dim=0), dim=1)
            name = test_data.datas[4 * i + b][0]
            saveName = name[name.find(dataset) + len(dataset) + 1:len(name)]
            # print("name:%s" % (saveName))
            # log.write("name:%s\n" % (saveName))
            # print("name:%s dsc:%f, %f, %f, %f" % (saveName, test_dsc[0], test_dsc[1], test_dsc[2], test_dsc[3]))
            log.write("name:%s dsc:%f, %f, %f, %f\n" % (saveName, test_dsc[0], test_dsc[1], test_dsc[2], test_dsc[3]))
            strlist = find_all('_', saveName)
            # writeNII.writeArrayToNii(torch.squeeze(outputPreL), savePath,
            #                          str(fold_num) + '_' + saveName[0:strlist[strlist.__len__() - 1]] + '_seg.nii')
    print(
        "fold_num:%d dsc, iou, spe, rec, S, F, E, M: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f" % (
            fold_num,
            test_dsc_0 / test_data.__len__(), test_dsc_1 / test_data.__len__(), test_dsc_2 / test_data.__len__(),
            test_dsc_3 / test_data.__len__(),
            test_iou_0 / test_data.__len__(), test_iou_1 / test_data.__len__(), test_iou_2 / test_data.__len__(),
            test_iou_3 / test_data.__len__(),
            test_spe_0 / test_data.__len__(), test_spe_1 / test_data.__len__(), test_spe_2 / test_data.__len__(),
            test_spe_3 / test_data.__len__(),
            test_rec_0 / test_data.__len__(), test_rec_1 / test_data.__len__(), test_rec_2 / test_data.__len__(),
            test_rec_3 / test_data.__len__(),
            test_S_0 / test_data.__len__(), test_S_1 / test_data.__len__(), test_S_2 / test_data.__len__(),
            test_S_3 / test_data.__len__(),
            test_F_0 / test_data.__len__(), test_F_1 / test_data.__len__(), test_F_2 / test_data.__len__(),
            test_F_3 / test_data.__len__(),
            test_E_0 / test_data.__len__(), test_E_1 / test_data.__len__(), test_E_2 / test_data.__len__(),
            test_E_3 / test_data.__len__(),
            test_M_0 / test_data.__len__(), test_M_1 / test_data.__len__(), test_M_2 / test_data.__len__(),
            test_M_3 / test_data.__len__()))
    log.write(
        "fold_num:%d dsc, iou, spe, rec, S, F, E, M: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n" % (
            fold_num,
            test_dsc_0 / test_data.__len__(), test_dsc_1 / test_data.__len__(), test_dsc_2 / test_data.__len__(),
            test_dsc_3 / test_data.__len__(),
            test_iou_0 / test_data.__len__(), test_iou_1 / test_data.__len__(), test_iou_2 / test_data.__len__(),
            test_iou_3 / test_data.__len__(),
            test_spe_0 / test_data.__len__(), test_spe_1 / test_data.__len__(), test_spe_2 / test_data.__len__(),
            test_spe_3 / test_data.__len__(),
            test_rec_0 / test_data.__len__(), test_rec_1 / test_data.__len__(), test_rec_2 / test_data.__len__(),
            test_rec_3 / test_data.__len__(),
            test_S_0 / test_data.__len__(), test_S_1 / test_data.__len__(), test_S_2 / test_data.__len__(),
            test_S_3 / test_data.__len__(),
            test_F_0 / test_data.__len__(), test_F_1 / test_data.__len__(), test_F_2 / test_data.__len__(),
            test_F_3 / test_data.__len__(),
            test_E_0 / test_data.__len__(), test_E_1 / test_data.__len__(), test_E_2 / test_data.__len__(),
            test_E_3 / test_data.__len__(),
            test_M_0 / test_data.__len__(), test_M_1 / test_data.__len__(), test_M_2 / test_data.__len__(),
            test_M_3 / test_data.__len__()))
    return


def savePreResult(samplePath, savePath, sizeWHD, model_index, sizeG, sim_thr):
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1)

    thr = 0.5

    batchSize = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.load(savePath + 'net_test.pkl')

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net = net.to(device)

    net.eval()

    dataset = 'test'
    savePath = savePath + '/' + dataset

    if os.path.exists(savePath) is False:
        os.makedirs(savePath)

    # test_data = loadBrainData.MyDatasetFYY_TCGA_skull_t(path=samplePath + dataset, transform=transforms.ToTensor(),
    #                                                     target_transform=transforms.ToTensor())

    test_data = loadBrainData.MyDatasetFYY_TCGA_pre(path=samplePath + dataset, transform=transforms.ToTensor(),
                                                    target_transform=transforms.ToTensor())

    test_loader = DataLoader(dataset=test_data, batch_size=batchSize, num_workers=1, pin_memory=True, drop_last=True)

    # 测试
    for i, data in enumerate(test_loader):
        input = torch.unsqueeze(data[0].to(device, non_blocking=True), dim=-4).transpose(3, 4)

        # input = torch.unsqueeze(data.to(device, non_blocking=True), dim=-4).transpose(3, 4)

        input = torch.cat((input[:, :, 0:sizeWHD[0], :, :], input[:, :, sizeWHD[0]:2 * sizeWHD[0], :, :],
                           input[:, :, 2 * sizeWHD[0]:3 * sizeWHD[0], :, :],
                           input[:, :, 3 * sizeWHD[0]:4 * sizeWHD[0], :, :]), dim=1)

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
                pyG_B = []
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
                pyG_B = create_graph_new(pyData, sizeG, sim_thr)
                output = net(input, pyG_B)
                output_use = output[4]

            if model_index == 6:
                pyData = data[1][:, 0:4, :, :].to(device, non_blocking=True)
                pyG_B = create_graph_new(pyData, sizeG, sim_thr)
                output_use = net(input, pyG_B)

        # 按照batchsize来保存结果
        # outputPreL = torch.argmax(output_use, dim=1).to(torch.float32)

        outputPre = torch.mul(output_use > thr, torch.ones_like(output_use))

        output_ET = 3 * (outputPre[:, 0, :, :, :])
        output_NET = outputPre[:, 1, :, :, :] - outputPre[:, 0, :, :, :]
        output_ED = 2 * (outputPre[:, 2, :, :, :] - outputPre[:, 1, :, :, :])
        outputPreL = torch.cat((output_ET.unsqueeze(dim=1), output_NET.unsqueeze(dim=1), output_ED.unsqueeze(dim=1)),
                               dim=1)
        outputPreL = torch.sum(outputPreL, dim=1)

        for b in range(batchSize):
            name = test_data.datas[4 * i + b][0]
            saveName = name[name.find(dataset) + len(dataset) + 1:len(name)]
            print("name:%s" % (saveName))
            strlist = find_all('_', saveName)
            writeNII.writeArrayToNii(torch.squeeze(outputPreL[b, :, :, :]), savePath,
                                     saveName[0:strlist[strlist.__len__() - 1]] + '_seg.nii')

    return


if __name__ == '__main__':
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True

    inputSize = [24, 256, 256]
    sizeG = [6, 8, 8]
    sim_thr = 0.75

    samplePath = r'/data1/Data/GBM/BrainTumor-pipeline/img/'
    savePath = r"/data1/Data/GBM/BrainTumor-pipeline/cross_ret/KMaX-DeepLab-20-1.5/"

    model_index = 20

    for i in range(5):
        testBrain(samplePath, savePath, inputSize, model_index, sizeG, sim_thr, i + 1)

    # testBrain(samplePath, savePath, inputSize, model_index, sizeG, sim_thr, 1)
    # savePreResult(samplePath, savePath, inputSize, model_index, sizeG, sim_thr)

    print("over")
