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


# 普通输入的二元交叉熵和DiceLoss结合的损失函数
class NormalBCEDiceLoss(nn.Module):
    '''
    input [batch, *] 
    y_true [batch, *] 
    '''

    def __init__(self):
        super(NormalBCEDiceLoss, self).__init__()

    def forward(self, input, y_true):
        bce = F.binary_cross_entropy_with_logits(input, y_true)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = y_true.size(0)
        input = input.reshape(num, -1)
        y_true = y_true.reshape(num, -1)
        intersection = (input * y_true)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + y_true.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


# 输入数据为onehot编码时的BCEDiceLoss
class BCEDiceLoss(nn.Module):
    '''
     inputs:
         output [batch, num_class,*]  one-hot code
         y_true [batch,num_class,*]   one-hot code
    '''

    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, output, y_true, compare_rule='>', thr=0.95):
        smooth = 1
        num = output.size(1)
        loss = 0
        # loss_l = 0
        dice_loss = 0
        for c in range(0, num):
            # 计算交叉熵损失
            loss += F.binary_cross_entropy(output[:, c], y_true[:, c])
            # 计算L2损失
            # loss_l += F.mse_loss(output[:, c], y_true[:, c])
            # 计算dice loss
            # pred_flat = output[:, c] > float(thr)
            # true_flat = y_true[:, c] > float(thr)
            # if compare_rule != '>':
            #     pred_flat = output[:, c] <= float(thr)
            #     true_flat = y_true[:, c] <= float(thr)

            pred_flat = output[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc = (2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)  # dsc系数
            dice_loss += 1 - dsc
        return (loss + dice_loss) / num


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.reshape(N, -1)
        targets_flat = targets.reshape(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - dice_eff.sum() / N

        loss_BCE = F.binary_cross_entropy(input_flat, targets_flat)

        return loss + loss_BCE


class MultiClassDiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(MultiClassDiceLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.kwargs = kwargs

    def forward(self, input, target):
        """
            input tesor of shape = (N, C, H, W)
            target tensor of shape = (N, H, W)
        """
        # 先将 target 进行 one-hot 处理，转换为 (N, C, H, W)
        nclass = input.shape[1]
        target = torch.squeeze(target)
        target = torch.nn.functional.one_hot(target.long(), nclass)
        target = target.permute(0, 4, 1, 2, 3).to(torch.float32)

        assert input.shape == target.shape, "predict & target shape do not match"

        binaryDiceLoss = BinaryDiceLoss()
        total_loss = 0

        # 归一化输出
        # logits = F.softmax(input, dim=1)
        logits = input
        C = target.shape[1]

        # 遍历 channel，得到每个类别的二分类 DiceLoss
        for i in range(C):
            dice_loss = binaryDiceLoss(logits[:, i], target[:, i])
            total_loss += dice_loss

        # 每个类别的平均 dice_loss
        return total_loss / C


class BCEDiceLoss_algin(nn.Module):
    '''
     inputs:
         output [batch, num_class,*]  one-hot code
         y_true [batch,num_class,*]   one-hot code
    '''

    def __init__(self):
        super(BCEDiceLoss_algin, self).__init__()

    def forward(self, output, y_true, compare_rule='>', thr=0.5):
        smooth = 1e-5
        num = output.size(1)
        loss = 0
        loss_l = 0
        dice_loss = 0
        for c in range(0, num):
            # 计算交叉熵损失
            loss += F.l1_loss(output[:, c], y_true[:, c])
            # 计算L2损失
            loss_l += F.mse_loss(output[:, c], y_true[:, c])
            # 计算dice loss
            # pred_flat = output[:, c] > float(thr)
            # true_flat = y_true[:, c] > float(thr)
            # if compare_rule is not '>':
            #     pred_flat = output[:, c] <= float(thr)
            #     true_flat = y_true[:, c] <= float(thr)

            pred_flat = output[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc = (2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)  # dsc系数
            dice_loss += 1 - dsc
        return loss + dice_loss + loss_l


# 输入为onehot编码且带权重的BCEDiceLoss
class WeightedBCEDiceLoss(nn.Module):
    '''
     inputs:
         output [batch, num_class,*]  one-hot code
         y_true [batch,num_class,*]   one-hot code
    '''

    def __init__(self):
        super(WeightedBCEDiceLoss, self).__init__()

    def forward(self, output, y_true, class_weights):
        smooth = 1e-5
        num = output.size(1)
        bce = 0
        dice_loss = 0
        for c in range(0, num):
            w = class_weights[c] / class_weights.sum()
            # 计算交叉熵损失
            bce += F.binary_cross_entropy(output[:, c], y_true[:, c])
            # 计算dice loss
            pred_flat = output[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc = (2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)  # dsc系数
            dice_loss += 1 - dsc
        return w * (0.5 * bce + dice_loss)


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x, device=net_output.device)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class MCCLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_mcc=False, do_bg=True, smooth=0.0):
        """
        based on matthews correlation coefficient
        https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

        Does not work. Really unstable. F this.
        """
        super(MCCLoss, self).__init__()

        self.smooth = smooth
        self.do_bg = do_bg
        self.batch_mcc = batch_mcc
        self.apply_nonlin = apply_nonlin

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        voxels = np.prod(shp_x[2:])

        if self.batch_mcc:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, tn = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)
        tp /= voxels
        fp /= voxels
        fn /= voxels
        tn /= voxels

        nominator = tp * tn - fp * fn + self.smooth
        denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5 + self.smooth

        mcc = nominator / denominator

        if not self.do_bg:
            if self.batch_mcc:
                mcc = mcc[1:]
            else:
                mcc = mcc[:, 1:]
        mcc = mcc.mean()

        return -mcc


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


class SoftDiceLossSquared(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        squares the terms in the denominator as proposed by Milletari et al.
        """
        super(SoftDiceLossSquared, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(x.shape, y.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y = y.long()
                y_onehot = torch.zeros(shp_x)
                if x.device.type == "cuda":
                    y_onehot = y_onehot.cuda(x.device.index)
                y_onehot.scatter_(1, y, 1).float()

        intersect = x * y_onehot
        # values in the denominator get smoothed
        denominator = x ** 2 + y_onehot ** 2

        # aggregation was previously done in get_tp_fp_fn, but needs to be done here now (needs to be done after
        # squaring)
        intersect = sum_tensor(intersect, axes, False) + self.smooth
        denominator = sum_tensor(denominator, axes, False) + self.smooth

        dc = 2 * intersect / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


softmax_helper = lambda x: F.softmax(x, 1)


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        self.ignore_label = ignore_label

        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son")  # reserved for other stuff (later)
        return result
