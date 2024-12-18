# -*- coding: utf-8 -*-
"""
Created on Mon May 17 16:50:53 2021

@author: bao.yang
"""
import torch
import numpy as np
import math
import warnings
# import cc3d

from sklearn.metrics import f1_score


# 普通输入的Dice系数
def NormalDice(output, target):
    '''
    inputs:
        output [batch, *] 
        target [batch, *] 
    '''
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
           (output.sum() + target.sum() + smooth)


# 输入为noehot的Dice系数
def Dice(output, y_true, compare_rule='>', thr=0.5):
    '''
     inputs:
         output [batch, num_class,*]   one-hot code
         y_true [batch,num_class,*]    one-hot code
    '''
    if torch.is_tensor(output):
        output = output.cpu().detach().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().detach().numpy()

    n_classes = output.shape[1]
    smooth = 1
    dsc = 0
    for c in range(0, n_classes):
        # 计算dice loss
        # pred_flat = output[:, c].reshape(-1)
        # true_flat = y_true[:, c].reshape(-1)
        pred_flat = output[:, c] > float(thr)
        true_flat = y_true[:, c] > float(thr)
        if compare_rule != '>':
            pred_flat = output[:, c] <= float(thr)
            true_flat = y_true[:, c] <= float(thr)
        intersection = (pred_flat * true_flat).sum()
        dsc += (2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)  # dsc系数
    return dsc/n_classes


def compute_dice(im1, im2, empty_value=1.0):
    if torch.is_tensor(im1):
        im1 = im1.cpu().detach().numpy()
    if torch.is_tensor(im2):
        im2 = im2.cpu().detach().numpy()

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_value

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2.0 * intersection.sum() / im_sum


def compute_absolute_volume_difference(im1, im2, voxel_size):
    """
    Computes the absolute volume difference between two masks.

    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
    voxel_size : scalar, float (ml)
        If not float, it will be converted.

    Returns
    -------
    abs_vol_diff : float, measured in ml.
        Absolute volume difference as a float.
        Maximum similarity = 0
        No similarity = inf


    Notes
    -----
    The order of inputs is irrelevant. The result will be identical if `im1` and `im2` are switched.
    """

    if torch.is_tensor(im1):
        im1 = im1.cpu().detach().numpy()
    if torch.is_tensor(im2):
        im2 = im2.cpu().detach().numpy()

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    voxel_size = voxel_size.astype(np.float)

    if im1.shape != im2.shape:
        warnings.warn(
            "Shape mismatch: ground_truth and prediction have difference shapes."
            " The absolute volume difference is computed with mismatching shape masks"
        )

    ground_truth_volume = np.sum(im1) * voxel_size
    prediction_volume = np.sum(im2) * voxel_size
    abs_vol_diff = np.abs(ground_truth_volume - prediction_volume)

    return abs_vol_diff


# def compute_absolute_lesion_difference(ground_truth, prediction, connectivity=26):
#     """
#     Computes the absolute lesion difference between two masks. The number of lesions are counted for
#     each volume, and their absolute difference is computed.
#
#     Parameters
#     ----------
#     ground_truth : array-like, bool
#         Any array of arbitrary size. If not boolean, will be converted.
#     prediction : array-like, bool
#         Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
#
#     Returns
#     -------
#     abs_les_diff : int
#         Absolute lesion difference as integer.
#         Maximum similarity = 0
#         No similarity = inf
#
#
#     Notes
#     -----
#     """
#
#     if torch.is_tensor(ground_truth):
#         ground_truth = ground_truth.cpu().detach().numpy()
#     if torch.is_tensor(prediction):
#         prediction = prediction.cpu().detach().numpy()
#
#     ground_truth = np.asarray(ground_truth).astype(np.bool)
#     prediction = np.asarray(prediction).astype(np.bool)
#
#     _, ground_truth_numb_lesion = cc3d.connected_components(ground_truth, connectivity=connectivity, return_N=True)
#     _, prediction_numb_lesion = cc3d.connected_components(prediction, connectivity=connectivity, return_N=True)
#     abs_les_diff = abs(ground_truth_numb_lesion - prediction_numb_lesion)
#
#     return abs_les_diff
#
#
# def compute_lesion_f1_score(ground_truth, prediction, empty_value=1.0, connectivity=26):
#     """
#     Computes the lesion-wise F1-score between two masks.
#
#     Parameters
#     ----------
#     ground_truth : array-like, bool
#         Any array of arbitrary size. If not boolean, will be converted.
#     prediction : array-like, bool
#         Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
#     empty_value : scalar, float.
#     connectivity : scalar, int.
#
#     Returns
#     -------
#     f1_score : float
#         Lesion-wise F1-score as float.
#         Max score = 1
#         Min score = 0
#         If both images are empty (tp + fp + fn =0) = empty_value
#
#     Notes
#     -----
#     This function computes lesion-wise score by defining true positive lesions (tp), false positive lesions (fp) and
#     false negative lesions (fn) using 3D connected-component-analysis.
#
#     tp: 3D connected-component from the ground-truth image that overlaps at least on one voxel with the prediction image.
#     fp: 3D connected-component from the prediction image that has no voxel overlapping with the ground-truth image.
#     fn: 3d connected-component from the ground-truth image that has no voxel overlapping with the prediction image.
#     """
#
#     if torch.is_tensor(ground_truth):
#         ground_truth = ground_truth.cpu().detach().numpy()
#     if torch.is_tensor(prediction):
#         prediction = prediction.cpu().detach().numpy()
#
#     ground_truth = np.asarray(ground_truth).astype(np.bool)
#     prediction = np.asarray(prediction).astype(np.bool)
#     tp = 0
#     fp = 0
#     fn = 0
#
#     # Check if ground-truth connected-components are detected or missed (tp and fn respectively).
#     intersection = np.logical_and(ground_truth, prediction)
#     labeled_ground_truth, N = cc3d.connected_components(
#         ground_truth, connectivity=connectivity, return_N=True
#     )
#
#     # Iterate over ground_truth clusters to find tp and fn.
#     # tp and fn are only computed if the ground-truth is not empty.
#     if N > 0:
#         for _, binary_cluster_image in cc3d.each(labeled_ground_truth, binary=True, in_place=True):
#             if np.logical_and(binary_cluster_image, intersection).any():
#                 tp += 1
#             else:
#                 fn += 1
#
#     # iterate over prediction clusters to find fp.
#     # fp are only computed if the prediction image is not empty.
#     labeled_prediction, N = cc3d.connected_components(
#         prediction, connectivity=connectivity, return_N=True
#     )
#     if N > 0:
#         for _, binary_cluster_image in cc3d.each(labeled_prediction, binary=True, in_place=True):
#             if not np.logical_and(binary_cluster_image, ground_truth).any():
#                 fp += 1
#
#     # Define case when both images are empty.
#     if tp + fp + fn == 0:
#         _, N = cc3d.connected_components(ground_truth, connectivity=connectivity, return_N=True)
#         if N == 0:
#             f1_score = empty_value
#     else:
#         f1_score = tp / (tp + (fp + fn) / 2)
#
#     return f1_score


def ev(output, y_true, compare_rule='>', thr=0.5):
    '''
     inputs:
         output [batch, num_class,*]   one-hot code
         y_true [batch,num_class,*]    one-hot code
    '''
    if torch.is_tensor(output):
        output = output.cpu().detach().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().detach().numpy()

    n_classes = output.shape[1]
    smooth = 1e-8
    precision = 0.0
    accuracy = 0.0
    recall = 0.0
    FPR_x = 0.0
    fpRate = 0.0
    dsc = 0.0
    iou = 0.0
    preV = 0.0
    labelV = 0.0
    f1 = 0.0
    mcc = 0.0
    diffV = 0.0
    for c in range(0, n_classes):
        # 计算dice loss
        # pred_flat = output[:, c].reshape(-1)
        # true_flat = y_true[:, c].reshape(-1)
        pred_flat = output[:, c] > float(thr)
        true_flat = y_true[:, c] > float(thr)
        if compare_rule != '>':
            pred_flat = output[:, c] <= float(thr)
            true_flat = y_true[:, c] <= float(thr)
        if pred_flat.sum() == 0 and true_flat.sum() == 0:
            pre = 1
            acc = 1
            rec = 1
            dsc_t = 1
            iou_t = 1
            FPR_t = 0
            diffV_t = 0
        else:
            # TP = (pred_flat * true_flat).sum()
            # FP = (pred_flat-pred_flat * true_flat).sum()
            # FN = (true_flat-pred_flat * true_flat).sum()
            # TN = (pred_flat + true_flat - pred_flat * true_flat) < float(thr)

            TP = np.logical_and(pred_flat, true_flat).sum()
            FP = np.logical_xor(pred_flat, pred_flat * true_flat).sum()
            FN = np.logical_xor(true_flat, pred_flat * true_flat).sum()
            TN = np.logical_or(pred_flat, true_flat).sum()

            pre = (TP) / (TP + FP + smooth)
            acc = (TP + TN) / (TP + FP + TN + FN + smooth)
            rec = (TP) / (TP + FN + smooth)
            FPR_t = 1 - (TN) / (FP + TN + smooth)
            dsc_t = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)  # dsc系数
            # dsc_t = (2. * TP) / (pred_flat.sum() + true_flat.sum() + smooth)
            iou_t = (TP + smooth) / (TP + FP + FN + smooth)
            # t1 = TP + FP
            # t2 = TP + FN
            # t3 = TN + FP
            # t4 = TN + FN
            # if t1 * t2 * t3 * t4 == 0:
            #     MCC = TP * TN - FP * FN
            # else:
            #     MCC = (TP * TN - FP * FN) / (np.sqrt(t1 * t2 * t3 * t4)+ smooth)
            diffV_t = abs(1 - (pred_flat.sum() + smooth) / (true_flat.sum() + smooth))

        precision += pre
        accuracy += acc
        recall += rec
        dsc += dsc_t
        iou += iou_t
        FPR_x += FPR_t
        fpRate += (1 - precision)
        f1_t = 2 * (precision * recall) / (precision + recall + smooth)
        f1 += f1_t
        # mcc += MCC
        preV += pred_flat.sum()
        labelV += true_flat.sum()
        diffV += diffV_t

    return precision/n_classes, accuracy/n_classes, recall/n_classes, FPR_x/n_classes, fpRate/n_classes, dsc/n_classes, iou/n_classes, f1/n_classes, mcc/n_classes, preV/n_classes, labelV/n_classes, diffV/n_classes


def evBiClass(output, y_true, compare_rule='>', thr=0.5):
    '''
     inputs:
         output [batch, num_class,*]   one-hot code
         y_true [batch,num_class,*]    one-hot code
    '''
    if torch.is_tensor(output):
        output = output.cpu().detach().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().detach().numpy()

    TP_All = 0
    FP_All = 0
    FN_All = 0
    TN_All = 0
    for n in range(0, output.shape[0]):
        pred_flat = output[n] > float(thr)
        true_flat = y_true[n] > float(thr)
        if compare_rule != '>':
            pred_flat = output[n] <= float(thr)
            true_flat = y_true[n] <= float(thr)
        TP = pred_flat > 0 and true_flat > 0
        FP = pred_flat > 0 and true_flat <= 0
        FN = pred_flat <= 0 and true_flat > 0
        TN = pred_flat <= 0 and true_flat <= 0
        TP_All += TP
        FP_All += FP
        FN_All += FN
        TN_All += TN
    return TP_All, FP_All, FN_All, TN_All


# 带权重的Dice系数
def WeightedDice(output, y_true, class_weights):
    '''
     inputs:
         output [batch, num_class,*]   one-hot code
         y_true [batch,num_class,*]    one-hot code
    '''
    n_classes = output.shape[1]
    smooth = 1e-5
    dsc = 0
    for c in range(0, n_classes):
        # 计算dice loss
        pred_flat = output[:, c].reshape(-1)
        true_flat = y_true[:, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        w = class_weights[c] / class_weights.sum()
        dsc += w * ((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth))  # dsc系数
    return dsc
