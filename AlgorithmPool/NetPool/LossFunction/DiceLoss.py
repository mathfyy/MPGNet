# -*- coding: utf-8 -*-
"""
Created on Mon May 17 16:11:25 2021

@author: bao.yang
"""
import torch.nn as nn
import torch


# Dice loss
class DiceLossClass(nn.Module):
    def __init__(self):
        super(DiceLossClass, self).__init__()

    def forward(self, output, y_true):
        n_classes = output.shape[1]
        smooth = 1e-5
        dice_loss = 0
        for c in range(0, n_classes):
            # 计算dice loss
            pred_flat = output[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc = (2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)  # dsc系数
            dice_loss += 1 - dsc
        return dice_loss


# Dice loss
def DiceLoss(output, y_true):
    '''
     inputs:
         output [batch, num_class,*]  one-hot code
         y_true [batch,num_class,*]   one-hot code
    '''
    n_classes = output.shape[1]
    smooth = 1e-5
    dice_loss = 0
    for c in range(0, n_classes):
        # 计算dice loss
        pred_flat = output[:, c].reshape(-1)
        true_flat = y_true[:, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        dsc = (2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)  # dsc系数
        dice_loss += 1 - dsc
    return dice_loss


def DiceLossBase(output, y_true, compare_rule='>', thr=0.5):
    '''
     inputs:
         output [batch, n_classes,*]
         y_true [batch,n_classes,*]
    '''
    n_classes = output.shape[1]
    smooth = 1e-5
    dice_loss = 0
    for c in range(0, n_classes):
        pred_flat = output[:, c] > float(thr)
        true_flat = y_true[:, c] > float(thr)
        if compare_rule is not '>':
            pred_flat = output[:, c] <= float(thr)
            true_flat = y_true[:, c] <= float(thr)

        intersection = (pred_flat * true_flat).sum()
        dsc = (2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)
        dice_loss += 1 - dsc
    return dice_loss


# 带权重的Dice loss
def Weighted_DiceLoss(output, y_true, class_weights):
    '''
     inputs:
         output [batch, num_class,*]  one-hot code
         y_true [batch,num_class,*]   one-hot code
    '''
    n_classes = output.shape[1]
    smooth = 1e-5
    dice_loss = 0
    for c in range(0, n_classes):
        # 计算dice loss
        w = class_weights[c] / class_weights.sum()
        pred_flat = output[:, c].reshape(-1)
        true_flat = y_true[:, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        dsc = (2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)  # dsc系数
        dice_loss += w * (1 - dsc)
    return dice_loss



class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        input = torch.softmax(input, dim=1)

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss/C


class Focal_loss(nn.Module):
    def __init__(self, alpha=(0.1, 0.4, 0.2, 0.3), gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(Focal_loss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        alpha = self.alpha[target].to(device)  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.reshape(8, 4, 24,256,256))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        # logpt = logpt.reshape(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


class Hybrid_loss(nn.Module):
    def __init__(self, lambdaa=1, classes=4):
        super(Hybrid_loss, self).__init__()
        self.dice_loss = MulticlassDiceLoss()
        self.focal_loss = Focal_loss()
        self.lambdaa = lambdaa
        self.classes = classes

    def forward(self, pred, true):
        loss1 = self.dice_loss(pred, true)
        loss2 = self.focal_loss(pred, true)
        total_loss = loss1 + self.lambdaa*loss2
        # total_loss = total_loss*self.classes
        return total_loss
