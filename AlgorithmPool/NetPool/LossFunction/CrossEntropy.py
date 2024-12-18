# -*- coding: utf-8 -*-
"""
Created on Mon May 17 16:29:51 2021

@author: bao.yang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

#二元交叉熵损失函数,模型输出做完sigmoid()后使用
def BCE(output,y_true):
    '''
     inputs:
         output [batch, *] 
         y_true [batch, *] 
    '''    
    loss = F.binary_cross_entropy(output,y_true)
    return loss

#二元交叉熵损失函数,模型输出output未做sigmoid()时使用
def BCE_logits(output,y_true):
   '''
     inputs:
         output [batch, *] 
         y_true [batch, *] 
    ''' 
   loss = F.binary_cross_entropy_with_logits(output,y_true)
   return loss

#交叉熵损失函数
def NormalCrossEntropy(output,y_true):
    '''
     inputs:
         output [batch, num_class,*] 
         y_true [batch, *] 
    '''
    criteria = nn.CrossEntropyLoss()
    loss = criteria(output,y_true)
    return loss

#输入为onehot编码的交叉熵损失函数
class CrossEntropy(nn.Module):
    '''
     inputs:
         output [batch, num_class,*]  one-hot code
         y_true [batch,num_class,*]   one-hot code
    '''
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, output, y_true):
        loss = 0
        num = output.size(1)
        for c in range(0,num):
        #计算交叉熵损失
            loss += F.binary_cross_entropy(output[:, c], y_true[:, c])
        return loss

#带权重的交叉熵损失函数
class Weighted_CrossEntropy(nn.Module):
    '''
     inputs:
         output [batch, num_class,*]  one-hot code
         y_true [batch,num_class,*]   one-hot code
         class_weights [m,n,...]
    '''
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, output, y_true, class_weights):
        loss = 0
        num = output.size(1)
        for c in range(0,num):
            w = class_weights[c]/class_weights.sum()
        #计算交叉熵损失
            loss += w*F.binary_cross_entropy(output[:, c], y_true[:, c])
        return loss