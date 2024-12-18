import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from SpecificApplication.brainLesion import loadBrainData
from AlgorithmPool.FunctionalMethodPool.DataIO import writeNII

import matplotlib
import matplotlib.pyplot as plt
# import torch.nn as nn
# from torch.nn import functional as F
# from torchvision import transforms
# import numpy as np
# from PIL import Image
# from collections import OrderedDict
# # import cv2


def get_activation(name, model, input, activation):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


if __name__ == '__main__':
    samplePath = r'E:\data\bloodData\view\data/'
    dataset = 'data/'
    netPath = r"E:/data/bloodData/view/net.pkl"
    savePath = r'E:\data\bloodData\view/'

    # matplotlib.use('Qt5Agg')
    # x = list(range(5))
    # y = list(range(5))
    # plt.figure()
    # plt.plot(x, y)
    # plt.show()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load(netPath).to(device, non_blocking=True)
    # print(model)

    test_data = loadBrainData.MyDatasetFYY_NCCT(path=samplePath, transform=transforms.ToTensor(),
                                                target_transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=1, pin_memory=True, drop_last=True)

    model.eval()

    for i, data in enumerate(test_loader):
        name = test_data.datas[i][0]
        saveName = name[name.find(dataset) + len(dataset):len(name)]

        input = torch.unsqueeze(data[0].to(device, non_blocking=True), dim=-4).transpose(3, 4)
        label = torch.unsqueeze(data[1].to(device, non_blocking=True), dim=-4).transpose(3, 4)

        activation = {}
        model.CP0.register_forward_hook(get_activation('layer0', model, input, activation))
        model.CP1.register_forward_hook(get_activation('layer1', model, input, activation))
        model.CP2.register_forward_hook(get_activation('layer2', model, input, activation))
        model.CP3.register_forward_hook(get_activation('layer3', model, input, activation))
        model.CP4.register_forward_hook(get_activation('layer4', model, input, activation))
        model.conv.register_forward_hook(get_activation('out', model, input, activation))

        model.encoder1.register_forward_hook(get_activation('encoder1', model, input, activation))
        _ = model(input)

        layer0 = activation['layer0'].permute(0, 1, 2, 3, 4)
        layer1 = activation['layer1'].permute(0, 1, 3, 4, 2)
        layer2 = activation['layer2'].permute(0, 1, 3, 4, 2)
        layer3 = activation['layer3'].permute(0, 1, 3, 4, 2)
        layer4 = activation['layer4'].permute(0, 1, 3, 4, 2)
        out = activation['out'].permute(0, 1, 2, 3, 4)

        encoder1 = activation['encoder1']
        enc1FLIP = torch.flip(encoder1, [4])
        enc1DIFF = torch.abs(torch.sub(encoder1, enc1FLIP))
        writeNII.writeArrayToNii(torch.squeeze(enc1DIFF.permute(0, 1, 3, 4, 2)), savePath, saveName + '_' + str(i) + '_encoder1.nii')
        writeNII.writeArrayToNii(torch.squeeze(encoder1.permute(0, 1, 3, 4, 2)), savePath,
                                 saveName + '_' + str(i) + '_encoder1_1.nii')
        writeNII.writeArrayToNii(torch.squeeze(enc1FLIP.permute(0, 1, 3, 4, 2)), savePath,
                                 saveName + '_' + str(i) + '_encoder1_2.nii')

        writeNII.writeArrayToNii(torch.squeeze(layer0), savePath, saveName + '_' + str(i) + '_layer0.nii')
        writeNII.writeArrayToNii(torch.squeeze(layer1), savePath, saveName + '_' + str(i) + '_layer1.nii')
        writeNII.writeArrayToNii(torch.squeeze(layer2), savePath, saveName + '_' + str(i) + '_layer2.nii')
        writeNII.writeArrayToNii(torch.squeeze(layer3), savePath, saveName + '_' + str(i) + '_layer3.nii')
        writeNII.writeArrayToNii(torch.squeeze(layer4), savePath, saveName + '_' + str(i) + '_layer4.nii')
        writeNII.writeArrayToNii(torch.squeeze(out), savePath, saveName + '_' + str(i) + '_out.nii')

        print("over one")

    print("over")
