# -*- coding: utf-8 -*-
"""
Created on Mon May 17 16:34:37 2021

@author: bao.yang
"""
import torch


# IOU
def IOU(output, y_true, compare_rule='>', thr=0.5):
    smooth = 1e-5
    '''
    input [batch, n_classes, *] one-hot code
    y_true [batch, n_classes, *] one-hot code
    '''

    if torch.is_tensor(output):
        output = output.cpu().detach().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().detach().numpy()

    iou = 0
    n_classes = output.shape[1]
    #    print('n_classes',n_classes)
    for c in range(0, n_classes):
        #        print('c',c)
        output_ = output[:, c] > float(thr)
        target_ = y_true[:, c] > float(thr)
        if compare_rule is not '>':
            output_ = output[:, c] <= float(thr)
            target_ = y_true[:, c] <= float(thr)

        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        if target_.sum() == 0 and output_.sum() == 0:
            iou += 1
        else:
            iou += (intersection + smooth) / (union + smooth)

    return iou


# 带权重的IOU
def Weighted_IOU(output, y_true, class_weights):
    smooth = 1e-5
    '''
    input [batch, n_classes, *] one-hot code
    y_true [batch, n_classes, *] one-hot code
    '''

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.data.cpu().numpy()

    iou = 0
    n_classes = output.shape[1]
    for c in range(0, n_classes):
        output_ = output[:, c] > 0.5
        y_true_ = y_true[:, c] > 0.5
        intersection = (output_ & y_true_).sum()
        union = (output_ | y_true_).sum()
        w = class_weights[c] / class_weights.sum()
        iou += w * ((intersection + smooth) / (union + smooth))
    return iou
