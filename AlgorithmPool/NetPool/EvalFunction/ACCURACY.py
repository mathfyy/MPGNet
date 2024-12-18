# -*- coding: utf-8 -*-
"""
Created on Sat May  8 14:39:50 2021

@author: bao.yang
"""
import torch
import numpy as np

def accuracy(output, target):
    '''
    inputs:
        output [batch, z, x, y] 
        target [batch, z, x, y] 
    '''
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    output = (np.round(output)).astype('int')
    target = target.view(-1).data.cpu().numpy()
    target = (np.round(target)).astype('int')
    (output == target).sum()

    return (output == target).sum() / len(output)