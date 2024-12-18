# -*- coding: utf-8 -*-
"""
Created on Mon May 17 16:57:02 2021

@author: bao.yang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import math


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, y_true, compare_rule='>', thr=0.95):
        smooth = 1e-9
        num = output.size(1)
        dice_loss = 0
        for c in range(0, num):
            pred_flat = output[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc = (2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)  # dsc系数
            dice_loss += 1 - dsc
        dice_loss = dice_loss / num
        return dice_loss


class DiceLossSquared(nn.Module):
    def __init__(self):
        super(DiceLossSquared, self).__init__()

    def forward(self, output, y_true, compare_rule='>', thr=0.95):
        smooth = 1e-9
        num = output.size(1)
        dice_loss = 0
        for c in range(0, num):
            pred_flat = output[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc = (2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)  # dsc系数
            dice_loss += 1 - dsc
        dice_loss = dice_loss / num
        return dice_loss ** 2


class generalized_dice_loss(nn.Module):
    def __init__(self):
        super(generalized_dice_loss, self).__init__()

    def forward(self, pred, target):
        epsilon = 1e-9
        wei = torch.sum(target, axis=[0, 2, 3, 4])  # (n_class,)
        wei = 1 / (wei ** 2 + epsilon)
        intersection = torch.sum(wei * torch.sum(pred * target, axis=[0, 2, 3, 4]))
        union = torch.sum(wei * torch.sum(pred + target, axis=[0, 2, 3, 4]))
        gldice_loss = 1 - (2. * intersection) / (union + epsilon)
        return gldice_loss


class tversky_loss(nn.Module):
    def __init__(self):
        super(tversky_loss, self).__init__()

    def forward(self, output, y_true, compare_rule='>', thr=0.95, beta=0.3, weights=None):
        smooth = 1e-9
        num = output.size(1)
        loss = 0.0
        for c in range(0, num):
            prob = output[:, c].reshape(-1)
            ref = y_true[:, c].reshape(-1)
            alpha = 1.0 - beta
            tp = (ref * prob).sum()
            fp = ((1 - ref) * prob).sum()
            fn = (ref * (1 - prob)).sum()
            tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
            loss = loss + (1 - tversky)
        return loss / num


class focal_tversky_loss(nn.Module):
    def __init__(self):
        super(focal_tversky_loss, self).__init__()

    def forward(self, output, y_true, compare_rule='>', thr=0.95, beta=0.7, weights=None):
        smooth = 1e-9
        gamma = 0.75
        num = output.size(1)
        loss = 0.0
        for c in range(0, num):
            prob = output[:, c].reshape(-1)
            ref = y_true[:, c].reshape(-1)
            alpha = 1.0 - beta
            tp = (ref * prob).sum()
            fp = ((1 - ref) * prob).sum()
            fn = (ref * (1 - prob)).sum()
            tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
            loss = loss + pow((1 - tversky), gamma)
        return loss / num


class LogCoshDiceLoss(nn.Module):
    def __init__(self):
        super(LogCoshDiceLoss, self).__init__()

    def forward(self, output, y_true, compare_rule='>', thr=0.95):
        smooth = 1e-9
        num = output.size(1)
        dice_loss = 0.0
        for c in range(0, num):
            pred_flat = output[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc = 1-(2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)  # dsc系数
            dice_loss += torch.log((torch.exp(dsc) + torch.exp(-dsc)) / 2.0)
        return dice_loss / num