# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:12:24 2021

@author: bao.yang
"""
import SimpleITK as sitk
import numpy as np


# NII普通数据读取
def NiiRead(dir):
    itkimage = sitk.ReadImage(dir)  # W,H,D  图像
    origin = itkimage.GetOrigin()  # 原点坐标 x, y, z
    spacing = itkimage.GetSpacing()  # 像素间隔 x, y, z
    direction = itkimage.GetDirection()  # 图像方向
    if np.any(direction != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):  # 判断是否相等
        isflip = True
    else:
        isflip = False
    img_array = sitk.GetArrayFromImage(itkimage)  # D,H,W 数组
    img_array = img_array.transpose(2, 1, 0)  # W,H,D  数组
    if (isflip == True):
        img_array = img_array[:, ::-1, ::-1]  #::-1 倒序
    #    print(img_array,origin,spacing,direction)
    return img_array, origin, spacing, direction

def NiiRead2D(dir):
    itkimage = sitk.ReadImage(dir)  # W,H,D  图像
    origin = itkimage.GetOrigin()  # 原点坐标 x, y, z
    spacing = itkimage.GetSpacing()  # 像素间隔 x, y, z
    direction = itkimage.GetDirection()  # 图像方向
    img_array = sitk.GetArrayFromImage(itkimage)  # D,H,W 数组
    #    print(img_array,origin,spacing,direction)
    return img_array, origin, spacing, direction


# NII训练数据读取
def train_NIIread(dir):
    itkimage = sitk.ReadImage(dir)  # W,H,D 图像
    img_array = sitk.GetArrayFromImage(itkimage)  # D,H,W  数组
    img_array = img_array.transpose(2, 1, 0)  # W,H,D  数组
    return img_array


# if __name__=="__main__":
#    NiiRead(niidir)
